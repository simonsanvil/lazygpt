from ml_collections import FieldReference
from ml_collections.config_dict import _Op

LAZY_EVALUATION_PLACEHOLDER = "[LAZY EVALUATION - NOT YET EVALUATED]"

def ensure_variable(
        eval_on:FieldReference, 
        placeholder=None,
        *, 
        type_=None, 
        keys:list=None, 
        num_items:int=None, 
        op:callable=None, 
        type_safe:bool=True
    ):
    if not isinstance(eval_on, FieldReference):
        eval_on = FieldReference(eval_on)
    if eval_on.get() is not None:
        result = eval_on.get()
        if op is not None:
            result = op(result)
        if type_ is not None and type_safe:
            result = type_(result)
        return FieldReference(result)
    if keys is not None and type_ is None:
        type_ = dict
    elif num_items is not None and type_ is None:
        type_ = list
    elif type_ is None:
        type_ = str
    
    if placeholder is None:
        if type_ == dict:
            placeholder_val = {key:LAZY_EVALUATION_PLACEHOLDER for key in keys}
        elif type_ == list:
            placeholder_val = [LAZY_EVALUATION_PLACEHOLDER for _ in range(num_items)]
        else:
            placeholder_val = LAZY_EVALUATION_PLACEHOLDER
    else:
        placeholder_val = placeholder
    
    # get_op = _Op(lambda val: (op(eval_on.get()) if op is not None else eval_on.get()) if eval_on.get() else val, tuple())
    get_op = _Op(lambda val: ensure_variable(eval_on, type_=type_, type_safe=type_safe, op=op) if eval_on.get() is not None else val, tuple())
    return FieldReference(placeholder_val, field_type=type_, op=get_op)

