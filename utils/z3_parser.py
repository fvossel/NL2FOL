from lark import Lark, Transformer, Tree, Token
from z3 import And, Or, Not, Implies, ForAll, Exists, Function, BoolSort, StringVal, StringSort, Xor, String, z3
from utils.constants import CFG_FOL
from typing import Any


class LogicTransformer(Transformer):
    def quantified(self, items):
        quantifier, variable, expr = items
        return Tree('quantified', [quantifier, variable, expr])

    def predicate(self, items):
        name = str(items[0])
        args = items[1] if len(items) > 1 else []
        return Tree('predicate', [name] + args)

    def args(self, items):
        return items

    def variable(self, items):
        return Tree('variable', [items[0]])

    def constant(self, items):
        return Tree('constant', [items[0]])

    def and_(self, items):
        return Tree('and', items)
    
    def or_(self, items):
        return Tree('or', items)

    def xor(self, items):
        return Tree('xor', items)

    def implies(self, items):
        return Tree('implies', items)

    def iff(self, items):
        return Tree('iff', items)

    def not_(self, items):
        return Tree('not', items)


def _to_z3(tree: Tree, var_map: dict=None):
    if var_map is None:
        var_map = {}
    
    if isinstance(tree, Tree):
        op = tree.data
        children = tree.children

        if op == 'quantified':
            quantifier = children[0]
            variable = str(children[1].children[0]) 
            if variable not in var_map:
                var_map[variable] = String(variable) if hasattr(z3, 'String') else StringVal(variable)
            z3_expr = _to_z3(children[2], var_map)
            if quantifier == '∀':
                return ForAll([var_map[variable]], z3_expr)
            elif quantifier == '∃':
                return Exists([var_map[variable]], z3_expr)

        elif op == 'and':
            return And(*[_to_z3(child, var_map) for child in children])

        elif op == 'or':
            return Or(*[_to_z3(child, var_map) for child in children])

        elif op == 'xor':
            return Xor(_to_z3(children[0], var_map), _to_z3(children[1], var_map))

        elif op == 'implies':
            return Implies(_to_z3(children[0], var_map), _to_z3(children[1], var_map))

        elif op == 'iff':
            return And(Implies(_to_z3(children[0], var_map), _to_z3(children[1], var_map)),
                       Implies(_to_z3(children[1], var_map), _to_z3(children[0], var_map)))

        elif op == 'not':
            return Not(_to_z3(children[0], var_map))

        elif op == 'predicate':
            name = children[0]
            args = [_to_z3(arg, var_map) for arg in children[1:]]
            if name not in var_map:
                arg_sorts = [StringSort() for _ in args]
                var_map[name] = Function(name, *arg_sorts, BoolSort())
            return var_map[name](*args)

        elif op == 'variable':
            variable = str(children[0])
            if variable not in var_map:
                var_map[variable] = String(variable) if hasattr(z3, 'String') else StringVal(variable)
            return var_map[variable]
            
        elif op == 'constant':
            value = children[0]
            if isinstance(value, Token):
                if value.type == 'CONSTANT':
                    cleaned_value = value.value.strip('"\'')
                    if cleaned_value not in var_map:
                        var_map[cleaned_value] = String(cleaned_value) if hasattr(z3, 'String') else StringVal(cleaned_value)
                    return var_map[cleaned_value]
            else:
                cleaned_value = str(value).strip('"\'')
                if cleaned_value not in var_map:
                    var_map[cleaned_value] = String(cleaned_value) if hasattr(z3, 'String') else StringVal(cleaned_value)
                return var_map[cleaned_value]

    elif isinstance(tree, str):
        return String(tree) if hasattr(z3, 'String') else StringVal(tree)

    raise ValueError(f"Unhandled expression type: {tree}")

def parse_fomula_to_z3(formula: str) -> Any:
    parser = Lark(CFG_FOL, start="start")

    tree = LogicTransformer().transform(parser.parse(formula))
    return _to_z3(tree)