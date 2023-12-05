import re
import dataclasses as dc
from dataclasses import dataclass
from ml_collections import ConfigDict

from .prompt_template import PromptTemplate

def promptclass(cls):
    """
    Decorator to convert a class to a prompt template class.
    """
    fields = [(f.name, f.type, f) for f in dc.fields(dataclass(cls))]
    fields += [(f.name, f.type, f) for f in dc.fields(PromptTemplate)]
    # prompt_cls = make_dataclass(cls.__name__, bases=(Prompt,), fields=fields)
    prompt_cls = type(cls.__name__, (PromptTemplate,), {})
    config_dict = getattr(cls, "__config__", {})
    if getattr(cls,"Config", None):
        config_dict.update({k:v for k,v in cls.Config.__dict__.items() if not k.startswith("_")})
    prompt_cls.config = ConfigDict(config_dict)
    # prompt_cls.config = types.SimpleNamespace(**config_dict)
    prompt = getattr(cls, "__prompt__", None)
    if prompt is None:
        prompt = cls.__doc__
    assert prompt, "prompt is missing"
    input_variables = getattr(cls, "__input_variables__", config_dict.get("input_variables", []))
    prompt_cls.__doc__ = cls.__doc__.strip()
    # parse __doc__ to remove extra spaces
    prompt = re.sub(r"\n( {4}|\t)", "\n", prompt.strip())
    # parse code variables (@const) to replace with their values
    code_vars = re.findall(r"@\w+[\.\w+]*", prompt, re.MULTILINE)
    for code_var in code_vars:
        code_var = code_var.strip("@").strip()
        if code_var in globals():
            subprompt = eval(code_var)
            if isinstance(subprompt, PromptTemplate):
                subprompt = subprompt.prompt
            prompt = prompt.replace(code_var, subprompt)
    
    return prompt_cls(prompt, input_variables=input_variables)