import subprocess
import tempfile
import os
from re import sub, findall
from Levenshtein import distance
from utils.z3_parser import parse_fomula_to_z3
from utils.prover9_parser import generate_prover9_input
from z3 import Solver, unsat, set_param, Not

def formulas_are_identical(formula1: str, formula2: str) -> bool:
    """Checks if two formulas are identical, ignoring whitespace differences."""
    cleaned_formula1 = sub(r'\s+', '', formula1)
    cleaned_formula2 = sub(r'\s+', '', formula2)
    return cleaned_formula1 == cleaned_formula2

def _map_predicates(predicate_list1: list, predicate_list2: list, max_norm_distance: float=0.6) -> list:
    """Maps predicates from predicate_list1 to predicate_list2 based on normalized Levenshtein distance."""
    mapped_list = []
    for predicate1 in predicate_list1:
        best_match = min(predicate_list2, key=lambda predicate_2: distance(predicate1, predicate_2) / max(len(predicate1), len(predicate_2)))
        norm_distance = distance(predicate1, best_match) / max(len(predicate1), len(best_match))
        
        if norm_distance <= max_norm_distance:
            mapped_list.append(best_match)
        else:
            mapped_list.append(predicate1)
    return mapped_list

def match_predicates(formula1: str, formula2: str) -> str:
    """Replaces predicates in formula1 according to the provided predicate_map."""
    matched_formula = formula1
    predicate_list1 = findall(r'\b\w+(?=\()', formula1)
    predicate_list2 = findall(r'\b\w+(?=\()', formula2)

    if len(predicate_list1) != 0 and len(predicate_list2) != 0:
        mapped_predicates = _map_predicates(predicate_list1, predicate_list2)

        for old_pred, new_pred in zip(predicate_list1, mapped_predicates):
            matched_formula = matched_formula.replace(old_pred + "(", new_pred + "(")

    return matched_formula


def formulas_are_equivalent(formula1: str, formula2: str, timeout: int=10000) -> bool:
    """Checks if two formulas are equivalent by using z3 solver by verifying the unsatisfiability
    of ¬(φ ↔ ψ), where φ is the formula produced by the LLM and ψ is the ground-truth formula.
    """

    if formulas_are_identical(formula1, formula2):
        return True # Early exit if formulas are identical for savings in computation time

    phi = parse_fomula_to_z3(formula1)
    psi = parse_fomula_to_z3(formula2)

    set_param('parallel.enable', True)
    set_param('smt.random_seed', 42)

    solver = Solver()
    solver.set("timeout", timeout)
    solver.add(Not(phi==psi))

    if solver.check() == unsat:
        return True
    else:
        return False
    

def _run_prover9(input: str, timeout: int=30):
    """Run the prover9 command line tool."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as temp_file:
        temp_file.write(input)
        temp_filename = temp_file.name

    result = subprocess.run(
        ['<PATH TO PROVER9>', '-f', temp_filename],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    os.unlink(temp_filename)
    success = "THEOREM PROVED" in result.stdout
    return success, result.stdout, result.stderr
    

def check_logical_entailment(premises: list[str], conclusion: str) ->bool:
    """Checks if a conclusion entails from the defined premises by using prover9."""

    prover9_input = generate_prover9_input(premises, conclusion)
    success, output, error = _run_prover9(prover9_input)
    
    if error:
        return False
    elif success:
        return True
    else:
        return False

    



    