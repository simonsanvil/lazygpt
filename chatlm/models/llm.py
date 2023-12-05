from typing import List, Any, Dict, Union, Optional, Callable

class LLM:

    def __init__(self, model:callable, **llm_kwargs):
        self.model = model
        self.llm_kwargs = llm_kwargs

    def function_calling(self, funcs: List[callable]):
        return self.__class__(model=self.model, llm_kwargs=dict(function_calling=funcs, **self.llm_kwargs))
    
    def adjust_params(self, **kwargs):
        llm_kwargs = self.llm_kwargs.copy()
        llm_kwargs.update(kwargs)
        return self.__class__(model=self.model, llm_kwargs=llm_kwargs)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, prompt:str, **kwargs):
        return self.model(prompt, **self.llm_kwargs, **kwargs)