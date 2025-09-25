from lark import Lark, Transformer, Tree, Token
from utils.constants import CFG_FOL

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
    
def _to_prover9(tree: Tree, bound_vars: dict=None):
    if bound_vars is None:
        bound_vars = set()
    
    if isinstance(tree, Tree):
        op = tree.data
        children = tree.children

        if op == 'quantified':
            quantifier = children[0]
            variable = str(children[1].children[0])  # Name der Variable
            new_bound_vars = bound_vars | {variable}
            inner_expr = _to_prover9(children[2], new_bound_vars)
            
            if quantifier == '∀':
                return f"all {variable} ({inner_expr})"
            elif quantifier == '∃':
                return f"exists {variable} ({inner_expr})"

        elif op == 'and':
            exprs = [_to_prover9(child, bound_vars) for child in children]
            if len(exprs) == 1:
                return exprs[0]
            return f"({' & '.join(exprs)})"

        elif op == 'or':
            exprs = [_to_prover9(child, bound_vars) for child in children]
            if len(exprs) == 1:
                return exprs[0]
            return f"({' | '.join(exprs)})"

        elif op == 'xor':
            left = _to_prover9(children[0], bound_vars)
            right = _to_prover9(children[1], bound_vars)
            # XOR als (A | B) & -(A & B)
            return f"(({left} | {right}) & -({left} & {right}))"

        elif op == 'implies':
            left = _to_prover9(children[0], bound_vars)
            right = _to_prover9(children[1], bound_vars)
            return f"({left} -> {right})"

        elif op == 'iff':
            left = _to_prover9(children[0], bound_vars)
            right = _to_prover9(children[1], bound_vars)
            return f"({left} <-> {right})"

        elif op == 'not':
            inner = _to_prover9(children[0], bound_vars)
            return f"-{inner}"

        elif op == 'predicate':
            name = children[0]
            args = [_to_prover9(arg, bound_vars) for arg in children[1:]]
            if args:
                return f"{name}({', '.join(args)})"
            else:
                return name

        elif op == 'variable':
            variable = str(children[0])
            return variable
            
        elif op == 'constant':
            value = children[0]
            if isinstance(value, Token):
                if value.type == 'CONSTANT':
                    cleaned_value = value.value.strip('"\'')
                    return cleaned_value
            else:
                cleaned_value = str(value).strip('"\'')
                return cleaned_value

    elif isinstance(tree, str):
        return tree

    raise ValueError(f"Unhandled expression type: {tree}")

def generate_prover9_input(premises: list[str], conclusion: str) -> str:
    """
    Generates a Prover9 input string from given premises and conclusion.

    Args:
        premises (list[str]): List of premise formulas in FOL.
        conclusion (str): Conclusion formula in FOL.

    Returns:
        str: Formatted Prover9 input string.
    """
    parser = Lark(CFG_FOL, start="start")
    prover9_input = []
    
    #Header section
    prover9_input.append("set(prolog_style_variables).")
    prover9_input.append("set(auto_denials).")
    prover9_input.append("clear(print_initial_clauses).")
    prover9_input.append("clear(print_kept).")
    prover9_input.append("clear(print_given).")
    prover9_input.append("")
    
    # Premises section
    prover9_input.append("formulas(assumptions).")
    for premise in premises:
        prover9_formula = _to_prover9(LogicTransformer().transform(parser.parse(premise)))
        prover9_input.append(f"  {prover9_formula}.")
    prover9_input.append("end_of_list.")
    prover9_input.append("")
    
    # Goal section
    prover9_input.append("formulas(goals).")
    prover9_goal =_to_prover9(LogicTransformer().transform(parser.parse(conclusion)))
    prover9_input.append(f"  {prover9_goal}.")
    prover9_input.append("end_of_list.")
    
    return "\n".join(prover9_input)