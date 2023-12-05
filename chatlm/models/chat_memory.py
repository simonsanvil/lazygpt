import json
from functools import partial
from numbers import Number
from copy import deepcopy


from ..lazy import LAZY_EVALUATION_PLACEHOLDER

class ChatMemory(list):


    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.meta = kwargs

    def __call__(self, msg:dict=None, *, content=None, role=None, **kwargs):
        if msg is None and (content is None or role is None):
            raise ValueError("Either msg or content and role must be specified")
        elif msg is not None and (content is not None or role is not None):
            raise ValueError("content and role cannot be specified if msg is specified")
        if msg is None:
            msg = {"content": content, "role": role}
        msg.update(kwargs)
        self.append(msg)
        return self
    
    # custom slicing should return a ChatMemory object
    def __getitem__(self, key):
        if isinstance(key, slice):
            return ChatMemory(super().__getitem__(key))
        return super().__getitem__(key)
    
    def copy(self):
        return self.__class__(deepcopy(self), **self.meta)
    
    def _ipython_display_(self):
        from IPython.display import Markdown, display
        display(Markdown(self.to_markdown()))

    def display(self, top:int=None, bottom:int=None, show_meta:bool=False, max_messages:int=None):
        mem = self.copy()
        mem.to_markdown = partial(mem.to_markdown, top=top, bottom=bottom, show_meta=show_meta, max_messages=max_messages)
        return mem
    
    def top(self, n:int=10, show_meta:bool=False):
        return self.display(top=n, show_meta=show_meta)
    
    def tail(self, n:int=10, show_meta:bool=False):
        return self.display(bottom=n, show_meta=show_meta)
    
    def to_markdown(self, max_messages:int=15, show_meta:bool=False, top:int=None, bottom:int=None):
        mem = deepcopy(self)
        s = f"<span style=\"color:DarkOrange\">Chat Memory with {len(mem)} message(s):</span>\n\n"
        if top is not None:
            if top < len(mem):
                s+= f"<span style='color:Crimson;'>Showing first {top} message(s):</span>\n\n"
                mem = mem[:top]
        elif bottom is not None:
            if bottom < len(mem):
                s+= f"<span style='color:Crimson;'>Showing last {bottom} message(s):</span>\n\n"
                mem = mem[-bottom:]
        if show_meta and len(mem.meta) > 0:
            # s+= "<span style=\"color:DarkOrange\">Meta:</span>\n\n"
            for k in mem.meta:
                v = mem.meta[k]
                if not isinstance(v, (str,bool, Number)):
                    continue
                s += f"- <tt><span style='color:DodgerBlue'>{k}:</span> {v}</tt>\n"
            s += "\n"
        else:
            for msg in mem:
                for k in msg.copy():
                    if k not in ["content", "role"]:
                        del msg[k]
        max_messages = max_messages or float("inf")
        if len(mem) == 0:
            return s
        for i, msg in enumerate(mem):
            if i >= max_messages:
                missing_messages = len(mem) - max_messages
                s+= f"<span style='color:Crimson;'><strong>and {missing_messages} more message(s)... </span></strong>\n\n"
                break
            s += f"{msg['role']}:\n\n".upper()
            for k in msg:
                if k in ["content", "role"]:
                    continue
                v = msg[k]
                if not isinstance(v, (str,bool, Number)):
                    continue
                s += f"- <span style='color:DarkOrange'>{k}:</span> {v}\n"

            if msg.get("output_parser", None) is not None:
                try:
                    msg_content = msg["output_parser"](msg["content"])
                except Exception as err:
                    print("Error parsing response from msg:", msg)
                    raise err
            else:
                msg_content = msg["content"]
            if isinstance(msg_content, str):
                if msg_content.strip().startswith(
                    ("{")
                ) and msg_content.strip().endswith(("}")):
                    try:
                        msg_content = json.loads(msg_content)
                    except Exception:
                        pass

            if isinstance(msg_content, (dict, list)):
                try:
                    json_str_quoted = json.dumps(msg_content, indent=4).replace(
                        "\n", "\n> "
                    )
                    msg_content_quoted = f"> ```json\n> {json_str_quoted}\n> ```"
                except Exception:
                    msg_content_quoted = "> " + msg["content"].replace("\n", "\n> ")
            else:
                msg_content_quoted = "> " + str(msg_content).replace("\n", "\n> ")
            s += f"{msg_content_quoted}\n\n"
        
        s = s.replace(LAZY_EVALUATION_PLACEHOLDER,f"<span style='color:OrangeRed'>{LAZY_EVALUATION_PLACEHOLDER}</span>")
        return s
