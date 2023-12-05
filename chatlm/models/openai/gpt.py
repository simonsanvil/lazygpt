from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import json
import logging
from numbers import Number
from typing import Any
import openai
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChatCompletionMessage,
)
from openai.types import Completion

from ml_collections import config_dict

from chatlm.models.openai.utils import num_tokens_from_messages
from ..chat_memory import ChatMemory
from ...prompting.prompt_template import PromptText
from ...lazy import LAZY_EVALUATION_PLACEHOLDER

class GPT:
    def __init__(
        self,
        client: openai.Client | openai.AsyncClient = None,
        *,
        model: str = "gpt-3.5-turbo",
        stateful: bool = True,
        lazy: bool = False,
        chat: bool = True,
        async_: bool = False,
        **gpt_kwargs,
    ):
        if client is None:
            client = openai.Client() if not async_ else openai.AsyncClient()
        self.openai_client = client
        self.model = model
        self.gpt_kwargs = gpt_kwargs
        self.stateful = stateful
        self.outputs = config_dict.ConfigDict()
        self.is_lazy = lazy or async_
        self._chat_messages = []
        self.__is_chat_model = chat
        self.__lazy_evaluations = []
        self.__in_context_manager = None
        self.__threads:dict[int|str, ChatMemory] = {0: ChatMemory()}
        self.__message_count = 0
        self.logger = logging.getLogger("chatlm.gpt")
        if async_ or isinstance(client, openai.AsyncClient):
            self.evaluate = self.evaluate_async

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__get_completion(*args, **kwds)

    def __get_completion(self, *args: Any, **kwds: Any) -> Any:
        if self.__is_chat_model:
            return self.chat_completion(*args, **kwds)

        return self.completion(*args, **kwds)
    
    @property
    def threads(self):
        return self.__threads

    @property
    def conversation(self):
        return ChatMemory(self.threads[0], default_model=self.model)

    def get_thread(self, thread_id:int=0):
        return self.threads.get(thread_id)
    
    def delete_thread(self, thread_id:int):
        for tup in self.__lazy_evaluations.copy():
            if tup[1] == thread_id:
                self.__lazy_evaluations.remove(tup)
        del self.threads[thread_id]

    def chat_completion(
        self,
        message: str | list[dict] | dict,
        role: str = "user",
        stateful: bool = None,
        call: bool = True,
        output_parser: callable = None,
        meta:dict = None,
        thread_id=0,
        return_msg_content:bool=False,
        output_key:str|list[str]=None,
        **kwargs,
    ):
        if return_msg_content and output_parser is None:
            output_parser = lambda x: x
        chat_kwargs = {**self.gpt_kwargs, **kwargs}
        meta = meta or {}
        stateful = self.stateful if stateful is None else stateful
        if thread_id in self.threads:
            chat_messages = self.threads[thread_id].copy()
        else:
            chat_messages = ChatMemory()
        if isinstance(message, dict):
            assert "content" in message.keys(), "message must contain 'content' key"
            message, role = message["content"], message.get("role", role)
        if isinstance(message, list):
            for msg in message:
                if isinstance(msg, dict):
                    assert "content" in msg.keys(), "message must contain 'content' key"
                    msg['role'] = msg.get("role", role)
                else:
                    raise ValueError("message must be a list of dictionaries")
            chat_messages += message
        else:
            # meta = message.metadata if isinstance(message, PromptText) else {}
            chat_messages.append({"content": message, "role": role})
        msg_list = [
            {"content": str(msg["content"]), "role": msg.get("role",role)} for msg in chat_messages
        ]
        chat_kwargs['model'] = chat_kwargs.get('model', self.model)
        model = chat_kwargs['model']
        if call and not self.is_lazy and model is not None:
            response = self.openai_client.chat.completions.create(messages=msg_list, **chat_kwargs)
            if output_parser is not None:
                response = output_parser(response.choices[0].text)
                
            resp_msg = response.choices[0].message.content
        elif self.is_lazy and model is not None:
            response = config_dict.FieldReference(None, field_type=ChatCompletion)
            idx_last_msg = len(chat_messages) - 1
            self.__lazy_evaluations.append(
                (idx_last_msg, thread_id, chat_kwargs, response, output_parser)
            )
            resp_msg = LAZY_EVALUATION_PLACEHOLDER
        # elif getattr(self, "__intermediate_messages", None) is None and model is None:
        #     raise ValueError("Model must be specified if not in context manager")
        else:
            response, resp_msg = None, None

        if stateful:
            for msg in chat_messages:
                if not "message_id" in msg:
                    msg["message_id"] = self.__message_count
                    self.__message_count += 1
                if not "thread_id" in msg:
                    msg["thread_id"] = thread_id
            self.threads[thread_id] = ChatMemory(chat_messages)

        if stateful and response is not None:
            assistant_msg = {
                "content": resp_msg,
                "role": "assistant",
                "response": response,
                "output_parser": output_parser if not self.is_lazy else None,
                "message_id": self.__message_count,
                **meta
            }
            self.threads[thread_id].append(assistant_msg)
            self.__message_count += 1
        if output_key is not None:
            if isinstance(output_key, list) and output_parser is None:
                raise ValueError("output_parser must be specified if output is a list")
            elif isinstance(output_key, str):
                self.outputs[output_key] = response
            else:
                raise NotImplementedError("list of strings output not yet implemented")
        return response

    def evaluate(self, async_eval: bool = False):
        # assert self.__is_chat_model and len(self.__lazy_evaluations) > 0
        print(f"Evaluating {len(self.__lazy_evaluations)} lazy evaluation(s)")
        for idx, thread_id, chat_kwargs, response_ref, func in self.__lazy_evaluations:
            msg_list = self.threads[thread_id].copy()
            msg_list = msg_list[: idx + 1]
            resp = self.openai_client.chat.completions.create(
                model=chat_kwargs.pop("model", self.model),
                messages=[{"content": str(msg["content"]), "role": msg["role"]} for msg in msg_list],
                **chat_kwargs,
            )
            resp_msg = resp.choices[0].message.content
            self.threads[thread_id][idx + 1]["content"] = resp_msg
            self.threads[thread_id][idx + 1]["output_parser"] = func
            self.threads[thread_id][idx + 1]["response"] = (
                func(resp_msg) if func is not None else resp
            )
            response_ref.set(resp, type_safe=False)
            if func is not None:
                response_ref._ops.append(
                    config_dict._Op(
                        (lambda r: func(r.choices[0].message.content)), tuple()
                    )
                )

        self.__lazy_evaluations = []

    async def __acompletion(self, *args, **kwargs):
        msgs = kwargs.get("messages", [])
        params = {k:v for k, v in kwargs.items() if k not in ["messages", "model"]}
        self.logger.debug(f"Evalualing async completion with {len(msgs)} message(s), model \"{kwargs.get('model', self.model)}\" and params {params}")
        completion = await self.openai_client.chat.completions.create(*args, **kwargs)
        self.logger.debug(f"Finished evaluating async completion")
        return completion
    
    def _eval_iter(self, threads:list[int|str]=None, pool_size:int=None, pop:bool=True):
        thread_ids = {t[1] for t in self.__lazy_evaluations}
        if threads is not None:
            thread_ids = [t for t in thread_ids if t in threads]
        pool_size = pool_size or float("inf")
        while len(self.__lazy_evaluations) > 0 and len(thread_ids) > 0:
            # TODO: make batch requests based on resolving context dependencies
            resp_batch = []
            num_lazy_evals = len(self.__lazy_evaluations)
            # only add to this batch the first lazy evaluation of each thread
            for tid in thread_ids.copy():
                if len(resp_batch) >= pool_size:
                    self.logger.warning(f"Reached pool size limit of {pool_size}")
                    break
                any_match = False
                for i, tup in enumerate(self.__lazy_evaluations.copy()):
                    if len(resp_batch) >= pool_size:
                        self.logger.warning(f"Reached pool size limit of {pool_size}")
                        break
                    if tup[1] == tid:
                        any_match = True
                        self.logger.debug(f"Found lazy evaluation for thread: {tid}")
                        if pop:
                            resp_batch.append(self.__lazy_evaluations.pop(i))
                        else:
                            resp_batch.append(self.__lazy_evaluations[i])
                        break
                if not any_match:
                    self.logger.debug(f"Thread {tid} has no lazy evaluations")
                    thread_ids.remove(tid)
            self.logger.info(f"yielding batch with {len(resp_batch)}/{num_lazy_evals} lazy evaluations")
            yield resp_batch

    async def evaluate_async(self, *, threads:list[int|str]=None, **kwargs):
        import asyncio
        self.logger.info(f"Evaluating {len(self.__lazy_evaluations)} lazy evaluation(s)")
        for resp_batch in self._eval_iter(threads=threads, **kwargs):
            tasks = []
            other_args = []
            for idx, thread_id, chat_kwargs, response_ref, func in resp_batch:
                self.logger.debug(f"Resolving lazy evaluation for thread \"{thread_id}\" and last message index: {idx+1}")
                if idx < 0:
                    continue
                msg_list =[
                    {"content": str(msg["content"]), "role": msg["role"]} 
                    for msg in self.threads[thread_id][: idx+1]
                ]
                task = self.__acompletion(
                    model=chat_kwargs.pop("model", self.model), 
                    messages=msg_list,#+[msg_in_turn], 
                    **chat_kwargs)
                tasks.append(task)
                other_args.append((idx, thread_id, response_ref, func))
            results = await asyncio.gather(*tasks)
            for i, resp in enumerate(results):
                idx, thread_id, ref, func = other_args[i]
                resp_msg = resp.choices[0].message.content
                self.threads[thread_id][idx + 1]["content"] = resp_msg
                self.threads[thread_id][idx + 1]["response"] = (
                    func(resp_msg) if func is not None else resp
                )
                ref.set(resp, type_safe=False)
                if func is not None:
                    ref._ops.append(
                        config_dict._Op(
                            (lambda r: func(r.choices[0].message.content)), tuple()
                        )
                    )
        self.logger.info("Finished evaluating all lazy evaluations")

    def merge_threads(
            self, 
            target:int|str,
            right:int|str,
            *,
            slice_right:slice|tuple=None,
        ):
        """
        Merge two threads together by copying and appending the message 
        of the `right` thread into the `target` thread.
        
        Parameters
        ----------
        target : int|str
            The id of the target thread
        right : int|str
            The id of the source thread
        slice_right : slice|tuple, optional
            The slice of the source thread to copy, by default None (copy all messages)

        Examples
        --------
        >>> gpt = GPT()
        >>> gpt.create_thread(1)
        >>> gpt.create_thread(2)
        >>> gpt("Message from thread 1", thread_id=1, model=None)
        >>> gpt("Message from thread 2", thread_id=2, model=None)
        >>> gpt.get_thread(1)
        [{'content': 'Message from thread 1', 'role': 'user'}]
        >>> gpt.merge_threads(1, 2)
        >>> gpt.get_thread(1)
        [{'content': 'Message from thread 1', 'role': 'user'}, {'content': 'Message from thread 2', 'role': 'user'}]
        >>> gpt.get_thread(2)
        [{'content': 'Message from thread 2', 'role': 'user'}]
        >>> gpt.merge_threads(2, 1, slice_right=(1,None)) # will only copy from the second message (index 1) onwards
        >>> gpt.get_thread(2)
        [{'content': 'Message from thread 2', 'role': 'user'}, {'content': 'Message from thread 2', 'role': 'user'}]
        """
        assert target in self.threads, f"Thread with id {target} does not exist"
        assert right in self.threads, f"Thread with id {right} does not exist"
        if slice_right is None:
            slice_right = slice(0, len(self.threads[right]))
        elif isinstance(slice_right, (tuple,list)):
            assert len(slice_right) == 2, "slice_right must be a tuple or list of length 2"
            slice_right = slice(*slice_right)
        if slice_right.stop == ...:
            slice_right.stop = None
        self.threads[target] += self.threads[right][slice_right]
        # if delete_right:
        #     self.delete_thread(right)

    
    
    def create_thread(self, thread_id:int|str=None, *, copy_from:int|str=None, replace_if_exists=False):
        if thread_id is not None:
            assert isinstance(thread_id, (int, str)), "thread_id must be an integer or string"
        if thread_id is None:
            thread_id = len(self.threads) + 1
        else:
            if thread_id in self.threads:
                if replace_if_exists:
                    self.delete_thread(thread_id)
                else:
                    raise ValueError(f"Thread with id {thread_id} already exists")
        if copy_from is not None:
            self.threads[thread_id] = deepcopy(self.threads[copy_from])
        else:
            self.threads[thread_id] = ChatMemory()
        return thread_id
    
    def add_thread_dependency(self, thread_id:int, dependency:int):
        assert thread_id in self.threads, f"Thread with id {thread_id} does not exist"
        assert dependency in self.threads, f"Thread with id {dependency} does not exist"
        raise NotImplementedError # TODO: implement this

    def eval(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def completion(self, prompt: str | list[str], **kwargs):
        gpt_kwargs = {**self.gpt_kwargs, **kwargs}
        response = self.openai_client.completions.create(
            model=self.model, prompt=prompt, **gpt_kwargs
        )
        return response

    @classmethod
    def from_messages(
        cls,
        openai_client: openai.Client,
        messages: list[dict],
        model: str,
        **gpt_kwargs,
    ):
        gpt = cls(openai_client, model, chat=True, stateful=True, **gpt_kwargs)
        gpt.conversation = messages
        return gpt

    def chat(self, stateful: bool = True, **kwargs):
        gpt_kwargs = {**self.gpt_kwargs, **kwargs}
        gpt = self.__class__(
            model=gpt_kwargs.pop("model", self.model),
            client=self.openai_client,
            stateful=stateful,
            **gpt_kwargs,
        )
        gpt.__is_chat_model = True
        return gpt

    def __enter__(self):
        """
        Withing the context manager the default model is set to None and lazy evaluation is enabled
        After exiting the context manager the model is set back to the previous model and evaluation is performed
        """
        self.__prev_is_lazy = self.is_lazy
        self.__no_eval = False
        self.__in_context_manager = True
        self.is_lazy = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # exceptions should be raised as normal
            return
        self.is_lazy = self.__prev_is_lazy
        self.__in_context_manager = False
        del self.__prev_is_lazy
        if not self.__no_eval:
            self.evaluate()

    def no_eval(self):
        if not self.__in_context_manager:
            raise ValueError("No eval can only be used within a context manager")
        self.__no_eval = True
        return self
    
    
    def to_json(self, path:str, remove_duplicates:bool=False, **kwargs):
        json_dict = {
            "created_at": datetime.utcnow().isoformat(),
            "thread_ids": list(self.threads.keys()),
            "default_model": self.model,
        }
        threads_json_str = config_dict.ConfigDict(
            {str(k): v.copy() for k,v in self.threads.items()}
        ).to_json_best_effort(skipkeys=True)
        threads_dict = json.loads(threads_json_str)
        json_dict["threads"] = threads_dict
        if remove_duplicates:
            msg_ids = set()
            for th in json_dict["threads"]:
                msgs = json_dict["threads"][th]
                msgs_ = []
                for msg in msgs:
                    if "message_id" in msg:
                        msg_id = msg["message_id"]
                        if msg_id in msg_ids:
                            # remove every key from msg except for message_id and thread_id
                            # continue
                            msg = {k: v for k,v in msg.items() if k in ["message_id", "thread_id"]}
                        else:
                            msg_ids.add(msg_id)
                    msgs_.append(msg)
                json_dict["threads"][th] = msgs_
        outputs_json = self.outputs.to_json_best_effort(skipkeys=True)
        json_dict["outputs"] = json.loads(outputs_json)
        with open(path, "w") as f:
            json.dump(json_dict, f, **kwargs)
    
    def count_tokens(self, thread_id:int=None, model_name:str=None):
        from .utils import num_tokens_from_messages
        if thread_id is not None:
            messages = self.threads[thread_id]
            return num_tokens_from_messages(messages, model_name or self.model)
        else:
            return {k: self.count_tokens(k) for k in self.threads.keys()}

    def print_token_counts(self, return_dict:bool=False):
        print("thread_id | num_tokens")
        print("--------- | ----------")
        token_sum = 0
        token_counts = {}
        for k in self.threads.keys():
            token_counts[k] = self.count_tokens(k)
            token_sum += token_counts[k]
            print(str(k).ljust(9), "|", token_counts[k])
        
        print("---------\nTotal:", token_sum)
        if return_dict:
            return token_counts

    @contextmanager
    def lazy(self):
        self.is_lazy = True
        try:
            yield self
        finally:
            self.is_lazy = False

    @contextmanager
    def new_thread(self, thread_id:int|str=None, *, copy_from:int=None, replace_if_exists:bool=False):
        thread_id = self.create_thread(thread_id=thread_id, copy_from=copy_from, replace_if_exists=replace_if_exists)
        try:
            if hasattr(self, "_chat_completion_old"):
                self._chat_completion_old.append(thread_id)
            else:
                self._chat_completion_old = [thread_id]
            self.chat_completion = partial(self.chat_completion, thread_id=thread_id)
            yield thread_id
        finally:
            self._chat_completion_old.pop()
            default_thread_id = self._chat_completion_old[-1] if len(self._chat_completion_old) > 0 else 0
            self.chat_completion = partial(self.chat_completion, thread_id=default_thread_id)
    
    @contextmanager
    def thread(self, thread_id:int|str=None):
        try:
            if hasattr(self, "_chat_completion_old"):
                self._chat_completion_old.append(thread_id)
            else:
                self._chat_completion_old = [thread_id]
            self.chat_completion = partial(self.chat_completion, thread_id=thread_id)
            yield thread_id
        finally:
            self._chat_completion_old.pop()
            default_thread_id = self._chat_completion_old[-1] if len(self._chat_completion_old) > 0 else 0
            self.chat_completion = partial(self.chat_completion, thread_id=default_thread_id)

    @contextmanager
    def new_params(self, **kwargs):
        writeable_params = [
            "openai_client", "model", 
            "stateful", "lazy", "chat", "async_",
        ]
        old_params = {}
        olg_gpt_kwargs = self.gpt_kwargs.copy()
        for k in kwargs.keys():
            if k in writeable_params:
                old_params[k] = getattr(self, k)
                setattr(self, k, kwargs[k])
            else:
                self.gpt_kwargs[k] = kwargs[k]
        try:
            yield self
        finally:
            for k, v in old_params.items():
                setattr(self, k, v)
            self.gpt_kwargs = olg_gpt_kwargs
        