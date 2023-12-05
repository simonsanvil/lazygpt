import logging
import streamlit as st
from time import sleep
from queue import Queue, Empty
import threading
from datetime import datetime
import openai
from typing import Generator


# Dummy function for get_chatbot_response, replace it with your actual function
def get_chatgpt_response(messages, **kwargs) -> Generator[str, None, None]:
    resp = openai.ChatCompletion.create(
        messages=messages,
        stream=True,
        **kwargs
    )
    for chunk in resp:
        yield chunk['choices'][0]['delta'].get('content', '')

def check_stop_button(button, queue):
    while True:
        if button:
            queue.put(True)
            break
        sleep(0.01)  # Check the button every 0.01 seconds.

def at_add_system_msg_button_clicked(msg_area, chat_history):
    if msg_area.strip():
        logging.info("Adding system message!!: ", msg_area)
        chat_history.append(dict(role="system", content=msg_area))

def make_chat_container(messages):
    # a scrollable container to hold the chat history
    # assistant messages are in bold and italics with a dark-grey background
    scrollable_container = """
<style>
    .scrollable-container {{
        max-height: 500px;
        width: 100%;
        overflow-y: scroll;
        padding: 5px;
    }}
    .assistant-msg {{
        background-color: #333;
        color: white;
        font-weight: bold;
        font-style: italic;
    }}
    .user-msg {{
        font-style: italic;
        color: #4169e1;
    }}
</style>
<div class="scrollable-container">
    {messages_txt}
</div>
""".strip()
    messages_txt = ''
    for msg in messages:
        if msg['role'] == 'assistant':
            messages_txt += f"<strong>\n{msg['content']}\n</strong>\n<br>"#f"<div class=assistant-msg>{md.markdown(msg['content'], extensions=[CodeHiliteExtension(use_pygments=True)])}</div>\n"
        else:
            messages_txt += f"<i style='color:#4169e1'>\n{msg['content']}\n</i><br>\n"
    return scrollable_container.format(messages_txt=messages_txt)

def scroll_to_bottom():
    st.markdown("<script>scrollToBottom();</script>", unsafe_allow_html=True)

def stop_chatbot(chat_history, stop_queue, resp_chunks):
    logging.info("Stopping chatbot...")
    stop_queue.put(True)
    chat_history.append({"role": "assistant", "content": "".join(resp_chunks)})
    scroll_to_bottom()

def on_send_button_clicked(chat_area:st.empty, user_input, chat_history, model:str):
    chat_history.append({"role": "user", "content": user_input})
    chatbot_response = get_chatgpt_response(chat_history, model=model, max_tokens=500)
    stop_queue = Queue()
    resp_chunks = []
    is_stopped = False
    stop_button = st.button("Stop Chatbot", on_click=stop_chatbot, args=(chat_history, stop_queue, resp_chunks))
    while not is_stopped:
        try:
            is_stopped = stop_queue.get(timeout=0.01)
        except Empty:
            pass
        response_chunk = next(chatbot_response, None)
        # logging.info("Response chunk:", response_chunk)
        if response_chunk is None:
            logging.info("Chatbot stopped")
            break
        resp_chunks.append(response_chunk)
        # show the assistant messages in bold and italics and the user messages in normal font
        reply =  "".join(resp_chunks)
        chat_area.markdown(
            f"<p class=assistant-msg>{reply}</p>",
            unsafe_allow_html=True,
        )
    logging.info("Stopping chatbot...")
    # kill the thread
    # stop_button_thread.join()
    chat_history.append({"role": "assistant", "content": "".join(resp_chunks)})
    # clear the user input
    scroll_to_bottom()
    
    return