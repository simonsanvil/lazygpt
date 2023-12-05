import copy
from datetime import datetime
from functools import wraps
import dataclasses as dc
from dataclasses import dataclass, make_dataclass, field
import types
import re
from typing import List, Any, Union
from ml_collections import config_dict

class PromptText(str):

    @property
    def metadata(self):
        if not hasattr(self, "__metadata"):
            self.__metadata = {}
        return self.__metadata

    def add_metadata(self, data:dict=None, **metadata):
        data = data or {}
        if not hasattr(self, "__metadata"):
            self.__metadata = {}
        metadata.update(data)
        self.__metadata.update(metadata)
        return self
    
    def _ipython_display_(self):
        from IPython.display import Markdown, display
        display(Markdown(self))

    # operations with strings should return PromptText
    def __add__(self, other):
        return PromptText(str(self) + str(other))
    
    def __radd__(self, other):
        return PromptText(str(other) + str(self))
    
    def __mul__(self, other):
        return PromptText(str(self) * other)
    
    def __rmul__(self, other):
        return PromptText(other * str(self))
    
    def __mod__(self, other):
        return PromptText(str(self) % other)
    
    def __getitem__(self, key):
        return PromptText(str(self)[key])
    
    def __getslice__(self, i, j):
        return PromptText(str(self)[i:j])
    

@dataclass
class PromptTemplate:
    ""
    
    prompt: str | PromptText
    input_variables: List[str] = field(default_factory=list)
    infer_input_vars: bool = False
    
    __config__ = dict()

    @property
    def config(self):
        return self.__config__ or {}

    def __post_init__(self):
        if isinstance(self.prompt, str):
            self.prompt = PromptText(self.prompt)
        if self.input_variables is False and self.infer_input_vars:
            self.input_variables = self._infer_input_variables()
        self.__config__ = config_dict.ConfigDict(self.__config__)

    def _infer_input_variables(self):
        input_vars = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", self.prompt)
        return list(set(input_vars))

    def __str__(self):
        return self.prompt
    
    def __repr__(self):
        if len(self.prompt)//2 > 20:
            prompt_head = self.prompt[:20]
            prompt_tail = self.prompt[-(20 if len(self.prompt)//2 > 20 else len(self.prompt)//2):]
            repr_prompt = prompt_head+"..."+prompt_tail
        else:
            repr_prompt = self.prompt
        return PromptText(f"{self.__class__.__name__}({repr_prompt}, input_variables={self.input_variables})")
    
    def _ipython_display_(self):
        from IPython import display as disp
        text_type = str(self.config.get("text_type", "markdown")).lower()
        if text_type in ["markdown", "md"]:
            disp_func = disp.Markdown
        elif text_type == "html":
            disp_func = disp.HTML
        elif text_type == "latex":
            disp_func = disp.Latex
        else:
            disp_func = str
        disp.display(disp_func(self.prompt))
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if len(args) == 1:
            if isinstance(args[0], dict):
                kwds.update(args[0])
            else:
                raise ValueError("only one positional argument is allowed and it must be a dict")
        elif len(args) > 1:
            raise ValueError("only one positional argument is allowed and it must be a dict")
        # for k in kwds:
        #     if k not in self.input_variables:
        #         raise ValueError(f"'{k}' not in input variable")
        defaults = self.config.get("defaults", {})
        kwds.update(defaults)
        if len(self.input_variables) > 0:
            for input_var in self.input_variables:
                if input_var not in kwds:
                    raise ValueError(f"input variable '{input_var}' is missing")

        return PromptText(self.prompt.format(*args, **kwds))
    
    def get_config(self):
        return self.__config__ or {}

    class Config:
        pass

    def to_message(
            self, inputs:dict=None, *, mapping:dict|list=None, defaults:dict=None, **input_kwargs
        ) -> dict:
        """ 
        Convert the prompt template to a message for a chatbot.

        Example:
        >>> prompt = PromptTemplate("Hello {name}!", input_variables=["name"])
        >>> prompt.to_message(
            inputs={"name": "John"}, 
            mapping={"message": "content", "role": "role"}, 
            defaults={"role": "user"})
        {'message': 'Hello John!', 'role': 'user'}
        """
        inputs = inputs or {}
        mapping = mapping or {}
        defaults = defaults or {}
        inputs.update(input_kwargs)
        msg = {
            "message": self(**inputs),
        }
        if isinstance(mapping, list):
            mapping_ = dict()
            for item in mapping:
                if isinstance(item, dict):
                    mapping_.update(item)
                else:
                    mapping_[item] = None
            mapping = mapping_
        mapping_keys = [k if v is None else v for k, v in mapping.items()]
        for k, v in mapping.items():
            if k in self.config:
                key_val = self.config[k]
            elif hasattr(self, k):
                key_val = getattr(self, k)
            elif k in defaults:
                key_val = defaults[k]
            else:
                key_val = None
            if v is not None:
                msg[v] = key_val
            else:
                msg[k] = key_val
        for k in msg.copy():
            if k not in mapping_keys:
                msg.pop(k)
        return msg
        

    def to_messages(
            self, 
            inputs:dict=None, *, 
            mapping:dict|List[Union[dict,str]]=None, 
            defaults:dict=None, 
            pre_append:List[str]|List[dict]=None, 
            append:List[str]|List[dict]=None,
            **input_kwargs
        ) -> List[dict]:
        """ 
        Convert the prompt template to a list of messages for a chatbot. You can pass additional 
        messages as a list of dictionaries in `pre_append` and `append`. Each dictionary must have a key `message`
        which is the message text. The other keys will be mapped to the keys in `mapping` (if given).

        Messages in `append` and in `config.append_messages` (if any) will be appended to the messages.
        Messaegs in `pre_append` and in `config.pre_append_messages` (if any) will be prepended to the messages.

        Example:
        >>> prompt = PromptTemplate("Hello {name}!", input_variables=["name"])
        >>> prompt.to_messages(
            inputs={"name": ["John", "Mary"]}, 
            append=[{"message": "Hello World!"}],
            mapping={"message": "content", "role": "role"}, 
            defaults={"role": "user"})
        [{'message': 'Hello John!', 'role': 'user'}, {'message': 'Hello Mary!', 'role': 'user'}, {'message': 'Hello World!', 'role': 'user'}]
        """
        inputs = inputs or {}
        mapping = mapping or ["message"]
        defaults = defaults or {}
        append = append or []
        append += self.config.get("append_messages", [])
        pre_append = pre_append or []
        pre_append += self.config.get("prepend_messages", [])
        inputs.update(input_kwargs)
        messages = pre_append or []
        if isinstance(mapping, list):
            mapping_ = dict()
            for item in mapping:
                if isinstance(item, dict):
                    mapping_.update(item)
                else:
                    mapping_[item] = item
            mapping = mapping_
        mapping_keys = [k if v is None else v for k, v in mapping.items()]
        messages.append({"message": self(**inputs)})
        if append:
            messages.extend(append)
        messages = copy.deepcopy(messages)
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                msg = {"message": msg}
                messages[i] = msg
            for k, v in mapping.items():
                if k in msg:
                    key_val = msg[k]
                    del msg[k]
                elif k in self.config:
                    key_val = self.config[k]
                elif hasattr(self, k):
                    key_val = getattr(self, k)
                elif k in defaults or k in inputs:
                    key_val = defaults.get(k, inputs.get(k))
                else:
                    continue
                if isinstance(key_val, list):
                    # if key_val is a list, add a new message for each item in the list
                    messages_at_idx = messages[:i]
                    for item in key_val:
                        msg_ = msg.copy()
                        msg_.update({(k if v is None else v): item})
                        messages_at_idx.append(msg_)
                    messages_at_idx.extend(messages[i+1:])
                    continue
                if isinstance(key_val, str):
                    key_val = key_val.format(**inputs)
                msg[k if v is None else v] = key_val
        for msg in messages:
            for k in msg.copy():
                if k not in mapping_keys:
                    msg.pop(k)
        return messages
