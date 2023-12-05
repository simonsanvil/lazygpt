import os, logging
import re
from typing import Union
import dotenv

import chatlm
import chatlm.context as ctx
from chatlm.clients.twilio import TwilioForWhatsappClient
from chatlm.operators.openai import gpt3, chatgpt, whisper, clip, dalle
# from chatlm import utils

logging.basicConfig(level=logging.DEBUG)
dotenv.load_dotenv(override=True)

twilio_client = chatlm.clients.TwilioForWhatsappClient(
    account_sid=os.environ.get('TWILIO_ACCOUNT_SID'),
    auth_token=os.environ.get('TWILIO_AUTH_TOKEN'), 
    from_number=os.environ.get('TWILIO_FROM_NUMBER'),
)

bckend = chatlm.backends.SqliteBackend('chat.db', create_tables=True) # this will create a local sqlite database in a file called chat.db

prompts = chatlm.prompts.get_default_templates()
prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
prompts.update(chatlm.prompts.from_dir(prompts_dir, files=['*.txt'])) # this will load all the prompts from the prompts/ directory of the form dir/{prompt_name}.txt
prompts.update(chatlm.prompts.from_env(prefix=['PROMPT_'])) # this will load all the prompts from environment variables of the form PROMPT_{PROMPT_NAME}
prompts['is_image_request'] = chatlm.prompts.Prompt(""""
I'm trying to distinguish between a text message and a request for an image generation.
Image generation requests can look like: 'generate an image of <object>' or 'can you generate an image of <object>?'
If yes, reply with 'yes', followed by the prompt for the image generation.
If no, reply with 'no'. Wrap the prompt in double quotes and the yes/no in single quotes.
Example: 
- 'can you generate an image of a dog?' -> Reply: 'yes', "a dog"
- 'can you make me a picture of a cat in pajamas?' -> Reply: 'yes', "a cat in pajamas"
- 'is it raining?' -> Reply: 'no'
The message sent was: {msg}""")


def translate_from_phone_extension(msg:str, ext:str) -> str:
    """Translate the message to the language corresponding to the country code of the phone number extension"""
    prompt = '''
    translate the message to the language corresponding to the country code of the phone number extension: {ext}.
    for example, if the phone number extension is +52 (Mexico) "hello" -> "hola". Delimit the translation with double quotes.
    The message sent was: "{msg}"
    '''.replace('\n',' ').format(ext=chat.user.phone_number[:2], msg=msg)
    msg = chatgpt([TwilioForWhatsappClient.make_message(prompt, from_='system')])
    if(match:=re.search(r'"([^"]*)"', msg)):
        msg = match.group(1)
    return msg

with chatlm.Chat('openai-whatsapp-chatbot', client=twilio_client, db_backend=bckend, prompt_templates=prompts, add_on_sent=True) as chat:
    chat.logger.setLevel(logging.DEBUG)
    # chat.logger.addHandler(logging.StreamHandler())

    @chat.handler(ctx.STARTUP, on=[ctx.startup])
    def start(chat):
        """Function called when the chat is started and the chat context is empty. It is only called once in the lifetime of the chat"""
        msg = "Hello! I'm a chatbot that uses the OpenAI-API to answer your questions and generate images for you. You can ask me anything!"
        chatgpt.model = chat.context.get('chatgpt-model', 'gpt3.5-turbo')
        if not chat.language:
            # translate the message to the language corresponding to the country code of the phone number extension
            msg = translate_from_phone_extension(msg, chat.user.phone_number[:2])
        else:
            if chat.language != 'en':
                msg = chatgpt.translate(msg, to_language=chat.language)
        chat.send_message(msg, from_='assistant')
 
    @chat.handler(on=[ctx.is_audio, ctx.message_received], level=1)
    def transcribe_audio(media:chatlm.Media) -> chatlm.prompts.Prompt:
        """To handle audio messages, we first transcribe them and then send the transcription upstream as a text"""
        transcription = whisper.transcribe(media.url, language_code=chat.language)
        chat.add_message('[The user sent an audio message. The following is a transcription of it]', from_='system')
        return '[Transcription]: {transcription}'.format(transcription=transcription)

    @chat.handler(on=[ctx.is_image, ctx.message_received], level=1)
    def describe_image(media:chatlm.Media) -> str:
        """To handle image messages, we first describe them and then send the description upstream as a text"""
        caption = clip.describe_image(media.url)
        chat.add_message('[The user sent an image. The following is a description of it]', from_='system')
        return chatlm.prompts.Prompt('[Image Description]: {caption}', caption=caption)

    @chat.contextual_handler(on=[ctx.is_text], persist=False, level=0)
    def is_shutdown(msg:str) -> bool:
        if msg.lower() in ('shutdown', 'exit', 'quit', 'bye', 'goodbye'):
            chat.logger.info("Shutting down chat")
            return True
        return False
    
    @chat.contextual_handler(on=[ctx.is_text], persist=False)
    def is_img_generation(msg:str) -> Union[bool, str]:
        prompt = chat.prompts.get('is_image_request').format(msg=msg)
        message = chat.make_message(prompt, from_='user')
        completion = chatgpt(chatgpt.make_conversation([message]))
        if completion.lower().strip().startswith('yes'):
            generation_prompt = completion[len('yes'):].strip()
            chat.set_context('is_image_request', True, persist=False) 
            # persist=False means that the context will be deleted in the next handler
            chat.set_context('img_generation_msg', generation_prompt, persist_until=ctx.message_received)
            # persist_until=ctx.message_received means that the context will be deleted after the next message
            return True
        return False
        
    @chat.handler(on=[is_img_generation], priority=chatlm.handlers.TOP_PRIORITY)
    def send_image() -> None:
        prompt = chat.context.get('img_generation_msg')
        img_url = dalle(prompt, as_url=True)
        chat.send_message_async(from_='assistant', media_type='image', media_url=img_url)

    @chat.handler(on=[ctx.is_text], priority=chatlm.handlers.LAST_PRIORITY)
    def reply(msg:str) -> chat.Response:
        """The final handler that receives any downstream text and sends the reply to the user"""
        if not chat.context.get('is_image_request', False):
            chat.add_message(msg, from_='user')
            conversation = chatgpt.make_conversation(chat.get_messages())
            reply = chatgpt(conversation)
        else:
            prompt = chat.context.get('img_generation_msg')
        chat.send_message(reply, from_='assistant') # message is automatically added to the chat history

    @chat.handler(on=[ctx.is_text, ctx.messages_count(1)], priority=chatlm.handlers.TOP_PRIORITY)
    def first_message_received(msg) -> None:
        chat.logger.info("First text message received")
        chat.set_language(gpt3.detect_language(msg))

    # @chat.handler(operators=[ctx.is_text], priority=chatlm.handlers.TOP_PRIORITY)
    # def save_message(msg) -> bool:
    #     chat.add_message(msg)
    #     return False
    
    @chat.handler(ctx.SHUTDOWN, on=[Union[is_shutdown, ctx.end_of_conversation]])
    def on_shutdown():  
        chat.logger.info("Chat has been shutdown")
        chat.backend.close()
        chat.restart_conversation()

    # chat.run(on_startup=start, on_shutdown=on_shutdown, port=5000, endpoint='/message') # run the chatbot on a local server

# we could also instance the chatbot as a flask app and run it on a server
chat = chatlm.Chat('openai-whatsapp-chatbot', client=twilio_client, db_backend=bckend, prompt_templates=prompts, add_on_sent=True)
chat.run(on_startup=start, on_shutdown=on_shutdown, port=5000, endpoint='/message') # run the chatbot on a local server