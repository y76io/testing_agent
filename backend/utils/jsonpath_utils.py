import json
from jsonpath_ng import parse

def jsonpath_get(obj, path_expr: str):
    expr = parse(path_expr)
    matches = [m.value for m in expr.find(obj)]
    return matches[0] if matches else None

def jsonpath_set(obj, path_expr: str, value):
    # Basic setter: handles paths pointing to existing fields / last item in array
    # For robust insertion, a more complete library may be needed.
    expr = parse(path_expr)
    matches = list(expr.find(obj))
    if not matches:
        return False
    for m in matches:
        context = m.context.value
        if isinstance(context, list) and isinstance(m.path.fields[0], int):
            context[m.path.fields[0]] = value
        elif isinstance(context, dict):
            key = m.path.fields[0]
            context[key] = value
        else:
            return False
    return True