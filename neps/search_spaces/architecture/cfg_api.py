from __future__ import annotations

import inspect
from typing import Literal


from .cfg_parameter import CFGParameter
from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar


def _dict_structure_to_str(
    structure: dict, primitives: str
) -> str:
    def _save_replace(string: str, _old: str, _new: str):
        while string.count(_old) > 0:
            string = string.replace(_old, _new)
        return string

    grammar = ""
    for nonterminal, productions in structure.items():
        grammar += nonterminal + " -> " + " | ".join(productions) + "\n"
    grammar = grammar.replace("(", " ")
    grammar = grammar.replace(")", "")
    grammar = grammar.replace(",", "")
    for primitive in primitives:
        grammar = _save_replace(grammar, f" {primitive} ", f' "{primitive}" ')
        grammar = _save_replace(grammar, f" {primitive}\n", f' "{primitive}"\n')
    return grammar


def CFGArchitectureParameter(**kwargs):
    """Factory function"""

    if "structure" not in kwargs:
        raise ValueError("Factory function requires structure")
    if not isinstance(kwargs["structure"], list) or len(kwargs["structure"]) == 1:
        # all good here :-)
        pass
    else:
        raise ValueError("Multiple repetitive structures are not supported")

    class _FunctionParameter(CFGParameter):
        def __init__(
            self,
            structure: Grammar
            | list[Grammar]
            | ConstrainedGrammar
            | list[ConstrainedGrammar]
            | str
            | list[str]
            | dict
            | list[dict],
            primitives: str,
            constraint_kwargs: dict | None = None,
            name: str = "CFGArchitectureParameter",
            prior_sampling_mode: Literal["mutation", "distribution"] = "mutation",
            **kwargs,
        ):
            local_vars = locals()
            self.input_kwargs = {
                args: local_vars[args]
                for args in inspect.getfullargspec(self.__init__).args  # type: ignore[misc]
                if args != "self"
            }
            self.input_kwargs.update(**kwargs)

            if isinstance(structure, list):
                raise ValueError("Multiple repetitive structures are not supported")
            else:
                if isinstance(structure, dict):
                    structure = _dict_structure_to_str(structure, primitives)

                if isinstance(structure, str):
                    if constraint_kwargs is None:
                        structure = Grammar.fromstring(structure)
                    else:
                        structure = ConstrainedGrammar.fromstring(structure)
                        structure.set_constraints(**constraint_kwargs)  # type: ignore[union-attr]

                super().__init__(
                    grammar=structure,  # type: ignore[arg-type]
                    prior_sampling_mode=prior_sampling_mode,
                    **kwargs,
                )

            self.name: str = name

    return _FunctionParameter(**kwargs)


CFGFunctionParameter = CFGArchitectureParameter
