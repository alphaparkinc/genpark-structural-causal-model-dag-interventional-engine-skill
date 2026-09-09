"""
Structural Causal Model DAG Interventional Engine Skill Client
Pure Python Standard Library implementation of Pearl's Structural Causal Models (SCM).
Supports causal DAG construction, structural equation assignment, observational sampling,
and graph truncation under Pearl's do-calculus interventions.
"""

from typing import Dict, List, Any, Callable, Optional, Set


class StructuralCausalModel:
    def __init__(self):
        self.variables: Set[str] = set()
        self.parents: Dict[str, List[str]] = {}
        self.equations: Dict[str, Callable[[Dict[str, Any], Any], Any]] = {}
        self.interventions: Dict[str, Any] = {}

    def add_variable(self, name: str, parents: List[str], equation: Callable[[Dict[str, Any], Any], Any]):
        self.variables.add(name)
        self.parents[name] = list(parents)
        self.equations[name] = equation

    def do(self, var_name: str, fixed_value: Any):
        """Apply Pearl's do(X = x) intervention, cutting all incoming directed edges to X."""
        self.interventions[var_name] = fixed_value

    def reset_interventions(self):
        """Remove all active interventions to restore the natural observational SCM."""
        self.interventions.clear()

    def sample(self, noise_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute structural realization for a given realization of exogenous noise."""
        state: Dict[str, Any] = {}
        remaining = set(self.variables)
        while remaining:
            progress = False
            for var in list(remaining):
                if var in self.interventions:
                    state[var] = self.interventions[var]
                    remaining.remove(var)
                    progress = True
                elif all(p in state for p in self.parents[var]):
                    parent_vals = {p: state[p] for p in self.parents[var]}
                    u_val = noise_dict.get(var, 0.0)
                    state[var] = self.equations[var](parent_vals, u_val)
                    remaining.remove(var)
                    progress = True
            if not progress and remaining:
                raise ValueError(f"Cyclic or unresolvable dependencies in SCM: {remaining}")
        return state
