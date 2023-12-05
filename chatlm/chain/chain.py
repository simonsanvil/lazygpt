from functools import wraps
from dataclasses import dataclass, field
from typing import List, Any, Callable

from ..prompting import PromptTemplate
from ..models.llm import LLM

@dataclass
class ChainStep:
    name: str
    prompt: PromptTemplate | str
    llm: LLM
    chain_context: dict
    output_var:str = None
    output_parser:Callable[[str], dict] = None
    context:dict = field(default_factory=dict)
    infer_input_vars:bool = True

    run_id:int = field(init=False)
    executed_at:str = field(default=None, init=False)

    def __post_init__(self):
        if isinstance(self.prompt, str):
            self.prompt = PromptTemplate(self.prompt, infer_input_vars=self.infer_input_vars)

    def __call__(self, **kwargs):
        ctx = self.context.copy()
        ctx.update(kwargs)
        result = self.llm(self.prompt(**ctx))
        if self.output_var is not None:
            self.chain_context[self.output_var] = result
        elif self.output_parser is not None:
            self.chain_context.update(self.output_parser(result))
        return result

@dataclass
class Chain:
    name: str
    llm: LLM
    context:dict = field(default_factory=dict)

    def __post_init__(self):
        self.steps = []

    @property
    def ctx(self):
        return self.context

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def doc_retrieval(self, name_id:str, func: callable, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds, **kwargs)
        return wrapper
  
    def step(
            self, 
            name:str=None, 
            *, 
            prompt: PromptTemplate|str, 
            output_var:str=None, 
            context:dict=None, 
            **kwargs
        ) -> ChainStep:
        step_num = len(self.steps)
        name = name or f"Step {step_num}"
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)
        context = context or self.context.copy()
        if prompt.input_variables and step_num > 0:
            for input_var in prompt.input_variables:
                if input_var not in context:
                    raise ValueError(f"input variable '{input_var}' not in context at step \"{name}\"")
        step = ChainStep(
            name=name, 
            prompt=prompt, 
            llm=kwargs.pop("llm", self.llm),
            context=context,
            chain_context=self.context,
            output_var=output_var,
            **kwargs
        )
        if output_var:
            self.context[output_var] = None
        self.steps.append(step)
        return step
    
        
    def func_step(self, func: callable = None, *, prompt: PromptTemplate|str=None, output_var:str=None, **kwargs):
        
        if func is None:    
            # decorator is called with arguments
            if prompt is None:
                raise ValueError("prompt must be specified when decorator is called with arguments")

            def decorator(func):
                step_kwargs = dict(context=self.context.copy(), llm=self.llm)
                step_kwargs.update(kwargs)
                step = ChainStep(
                    func.__name__, 
                    prompt=prompt, 
                    chain_context=self.context,
                    output_var=output_var, 
                    **step_kwargs)
                
                @wraps(func)
                def wrapper(*args, **kwds):
                    result = step()
                    return func(result, *args, **kwds)
                
                wrapper.name = func.__name__
                self.steps.append(wrapper)
                return wrapper
            return decorator

        # decorator is called without arguments
        @wraps(func)
        def wrapper(*args, **kwds):
            return func(self.context, *args, **kwds)

        wrapper.name = func.__name__
        self.steps.append(wrapper)
        return wrapper
  
    
    def add_to_context(self, _name:str=None, _value:Any=None, *, cache:bool=False, **kwargs):
        context_dict = {}
        if len(kwargs) > 0 and (_name is not None or _value is not None):
            raise ValueError("Either 'name' and 'value' or 'kwargs' must be specified")
        if isinstance(_name, dict) and _value is None:
            context_dict.update(_name)
        elif _name is not None and _value is not None:
            context_dict[_name] = _value
        elif (_value is None and _name is not None) or (_name is None and _value is not None):
            raise ValueError("Both 'name' and 'value' of the context variable must be specified")
        
        context_dict.update(kwargs)
        self.context.update(context_dict)

    def deploy(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self):
        return self.run()
    
    def run(self):
        for step in self.steps:
            step()