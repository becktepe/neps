from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, ClassVar, Mapping
from typing_extensions import override, Self
from neps.utils.types import NotSet, _NotSet

import networkx as nx
import numpy as np
from nltk import Nonterminal

from ..parameter import ParameterWithPrior, MutatableParameter
from .cfg import Grammar
from .cfg_variants.constrained_cfg import ConstrainedGrammar
from .core_graph_grammar import CoreGraphGrammar
from .crossover import repetitive_search_space_crossover, simple_crossover
from .mutations import bananas_mutate, repetitive_search_space_mutation, simple_mutate


# TODO(eddiebergman): This is a halfway solution, but essentially a lot
# of things `Parameter` does, does not fit nicely with a Graph based
# parameters, in the future we probably better just have these as two seperate
# classes. For now, this class sort of captures the overlap between
# `Parameter` and Graph based parameters.
# The problem here is that the `Parameter` expects the `load_from`
# and the `.value` to be the same type, which is not the case for
# graph based parameters.
class GraphParameter(ParameterWithPrior[nx.DiGraph, str], MutatableParameter):
    # NOTE(eddiebergman): What I've managed to learn so far is that
    # these hyperparameters work mostly with strings externally,
    # i.e. setting the value through `load_from` or `set_value` should be a string.
    # At that point, the actual `.value` is a graph object created from said
    # string. This would most likely break with a few things in odd places
    # and I'm surprised it's lasted this long.
    # At serialization time, it doesn't actually serialize the .value but instead
    # relies on the string it was passed initially, I'm not actually sure if there's
    # a way to go from the graph object to the string in this code...
    # Essentially on the outside, we need to ensure we don't pass ih the graph object itself
    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]] = {"not_in_use": 1.0}
    default_confidence_choice = "not_in_use"
    has_prior: bool
    input_kwargs: dict[str, Any]

    @property
    @abstractmethod
    def id(self) -> str: ...

    # NOTE(eddiebergman): Unlike traditional parameters, it seems
    @property
    @abstractmethod
    def value(self) -> nx.DiGraph: ...

    # NOTE(eddiebergman): This is a function common to the three graph
    # parameters that is used for `load_from`
    @abstractmethod
    def create_from_id(self, value: str) -> None: ...

    # NOTE(eddiebergman): Function shared between graph parameters.
    # Used to `set_value()`
    @abstractmethod
    def reset(self) -> None: ...

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GraphGrammar):
            return NotImplemented

        return self.id == other.id

    @abstractmethod
    def compute_prior(self, normalized_value: float) -> float: ...

    @override
    def set_value(self, value: str | None) -> None:
        # NOTE(eddiebergman): Not entirely sure how this should be done
        # as previously this would have just overwritten a property method
        # `self.value = None`
        if not isinstance(value, str):
            raise ValueError(
                f"Expected a string for setting value a `GraphParameter`",
                f" got {type(value)}"
            )
        self.reset()
        self.normalized_value = value

        if value is None:
            return

        self.create_from_id(value)

    @override
    def set_default(self, default: str | None) -> None:
        # TODO(eddiebergman): Could find no mention of the word 'default' in the
        # GraphGrammers' hence... well this is all I got
        self.default = default

    @override
    def sample_value(self, *, user_priors: bool = False) -> nx.DiGraph:
        # TODO(eddiebergman): This could definitely be optimized
        # Right now it copies the entire object just to get a value out
        # of it.
        return self.sample(user_priors=user_priors).value

    @classmethod
    def serialize_value(cls, value: nx.DiGraph) -> str:
        """Functionality relying on this for GraphParameters should
        special case and use `self.id`.

        !!! warning

            Graph parameters don't directly support serialization.
            Instead they rely on holding on to the original string value
            from which they were created from.
        """
        raise NotImplementedError

    @classmethod
    def deserialize_value(cls, value: str) -> nx.DiGraph:
        """Functionality relying on this for GraphParameters should
        special case for whever this is needed...

        !!! warning

            Graph parameters don't directly support serialization.
            Instead they rely on holding on to the original string value
            from which they were created from.
        """
        raise NotImplementedError

    @override
    def load_from(self, value: str | Self) -> None:
        if isinstance(value, GraphParameter):
            value = value.id
        self.create_from_id(value)

    @abstractmethod
    def mutate(self, parent: Self | None = None, *, mutation_strategy: str = "bananas") -> Self: ...

    @abstractmethod
    def crossover(self, parent1: Self, parent2: Self | None = None) -> tuple[Self, Self]:
        ...

    def _get_non_unique_neighbors(self, num_neighbours: int) -> list[Self]:
        raise NotImplementedError

    def value_to_normalized(self, value: nx.DiGraph) -> float:
        raise NotImplementedError

    def normalized_to_value(self, normalized_value: float) -> nx.DiGraph:
        raise NotImplementedError

    @override
    def clone(self) -> Self:
        new_self =  self.__class__(**self.input_kwargs)

        # HACK(eddiebergman): It seems the subclasses all have these and
        # so we just copy over those attributes, deepcloning anything that is mutable
        if self._value is not None:
            _attrs_that_subclasses_use_to_reoresent_a_value = (
                ("_value", True),
                ("string_tree", False),
                ("string_tree_list", False),
                ("nxTree", False),
                ("_function_id", False),
            )
            for _attr, is_mutable in _attrs_that_subclasses_use_to_reoresent_a_value:
                retrieved_attr = getattr(self, _attr, NotSet)
                if retrieved_attr is NotSet:
                    continue

                if is_mutable:
                    setattr(new_self, _attr, deepcopy(retrieved_attr))
                else:
                    setattr(new_self, _attr, retrieved_attr)

        return new_self

class GraphGrammar(GraphParameter, CoreGraphGrammar):
    hp_name = "graph_grammar"

    def __init__(
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        prior: dict = None,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        new_graph_repr_func: bool = False,
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        if isinstance(grammar, list) and len(grammar) != 1:
            raise NotImplementedError("Does not support multiple grammars")

        CoreGraphGrammar.__init__(
            self,
            grammars=grammar,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )
        GraphParameter.__init__(self, value=None, default=None, is_fidelity=False)

        self.string_tree: str = ""
        self._function_id: str = ""
        self.nxTree: nx.DiGraph | None = None
        self.new_graph_repr_func = new_graph_repr_func

        if prior is not None:
            self.grammars[0].prior = prior
        self.has_prior = prior is not None

    @override
    def sample(self, *, user_priors: bool = False) -> Self:
        copy_self = self.clone()
        copy_self.reset()
        copy_self.string_tree = copy_self.grammars[0].sampler(1, user_priors=user_priors)[0]
        _ = copy_self.value  # required for checking if graph is valid!
        return copy_self

    @property
    @override
    def value(self) -> nx.DiGraph:
        if self._value is None:
            if self.new_graph_repr_func:
                self._value = self.get_graph_representation(
                    self.id,
                    self.grammars[0],
                    edge_attr=self.edge_attr,
                )
                assert isinstance(self._value, nx.DiGraph)
            else:
                _value = self.from_stringTree_to_graph_repr(
                    self.string_tree,
                    self.grammars[0],
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
                # NOTE: This asumption was not true but I don't really know
                # how to handle it otherwise, will just leave it as is for now
                #  -x- assert isinstance(_value, nx.DiGraph), _value
                self._value = _value
        return self._value

    @override
    def mutate(
        self,
        parent: GraphGrammar | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ) -> Self:
        if parent is None:
            parent = self
        parent_string_tree = parent.string_tree

        # We have 10 trials to mutate the parent,
        # if we can't mutate it in 10 trials, we raise an exception
        trials = 10
        while True:
            if mutation_strategy == "bananas":
                child_string_tree, is_same = bananas_mutate(
                    parent_string_tree=parent_string_tree,
                    grammar=self.grammars[0],
                    mutation_rate=mutation_rate,
                )
            else:
                child_string_tree, is_same = simple_mutate(
                    parent_string_tree=parent_string_tree,
                    grammar=self.grammars[0],
                )

            if is_same:
                trials -= 1
                if trials == 0:
                    raise Exception("Cannot create mutation")
            else:
                break

        return parent.create_new_instance_from_id(
            self.string_tree_to_id(child_string_tree)
        )

    @override
    def crossover(self, parent1: Self, parent2: Self | None = None) -> tuple[Self, Self]:
        if parent2 is None:
            parent2 = self
        parent1_string_tree = parent1.string_tree
        parent2_string_tree = parent2.string_tree
        children = simple_crossover(
            parent1_string_tree, parent2_string_tree, self.grammars[0]
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")

        return tuple(
            parent2.create_new_instance_from_id(self.string_tree_to_id(child))
            for child in children
        )

    @override
    def compute_prior(self, *, log: bool = True) -> float:
        return self.grammars[0].compute_prior(self.string_tree, log=log)

    @property
    def id(self) -> str:
        if self._function_id is None or self._function_id == "":
            if self.string_tree == "":
                raise ValueError("Cannot infer identifier!")
            self._function_id = self.string_tree_to_id(self.string_tree)
        return self._function_id

    @id.setter
    def id(self, value: str) -> None:
        self._function_id = value

    def create_from_id(self, identifier: str) -> None:
        self.reset()
        self._function_id = identifier
        self.id = identifier
        self.string_tree = self.id_to_string_tree(self.id)
        _ = self.value  # required for checking if graph is valid!

    @staticmethod
    def id_to_string_tree(identifier: str) -> str:
        return identifier

    @staticmethod
    def string_tree_to_id(string_tree: str) -> str:
        return string_tree

    @property
    def search_space_size(self) -> int:
        return self.grammars[0].compute_space_size

    @abstractmethod
    def create_new_instance_from_id(self, identifier: str):
        raise NotImplementedError

    def reset(self) -> None:
        self.clear_graph()
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self._function_id = ""

    def compose_functions(self, flatten_graph: bool = True) -> nx.DiGraph:
        return self._compose_functions(self.id, self.grammars[0], flatten_graph)

    def unparse_tree(self, identifier: str, as_composition: bool = True):
        return self._unparse_tree(identifier, self.grammars[0], as_composition)

    def get_dictionary(self) -> dict[str, str]:
        return {"graph_grammar": self.id}

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.grammars[0])
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )


class GraphGrammarCell(GraphGrammar):
    hp_name = "graph_grammar_cell"

    def __init__(
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        super().__init__(
            grammar,
            terminal_to_op_names,
            terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )

        self.cell = None

    def reset(self) -> None:
        super().reset()
        self.cell = None

    @abstractmethod
    def create_graph_from_string(self, child: str):
        raise NotImplementedError


class GraphGrammarRepetitive(GraphParameter, CoreGraphGrammar):
    hp_name = "graph_grammar_repetitive"

    def __init__(
        self,
        grammars: list[Grammar],
        terminal_to_op_names: dict,
        terminal_to_sublanguage_map: dict,
        number_of_repetitive_motifs: int,
        terminal_to_graph_edges: dict = None,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
    ):
        CoreGraphGrammar.__init__(
            self,
            grammars=grammars,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
        )
        GraphParameter.__init__(self, value=None, default=None, is_fidelity=False)

        self.id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph | None = None
        self._value: nx.DiGraph | None = None

        self.full_grammar = self.get_full_grammar(self.grammars)
        self.terminal_to_sublanguage_map = terminal_to_sublanguage_map
        self.number_of_repetitive_motifs = number_of_repetitive_motifs

    @override
    def mutate(
        self,
        parent: Self | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ) -> Self:
        raise NotImplementedError
        if parent is None:
            parent = self

        # bananas mutate
        if mutation_strategy == "bananas":
            inner_mutation_strategy = partial(bananas_mutate, mutation_rate=mutation_rate)
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
                inner_mutation_strategy=inner_mutation_strategy,
            )
        else:
            child_string_tree_list, is_same = repetitive_search_space_mutation(
                base_parent=parent.string_tree_list[0],
                motif_parents=parent.string_tree_list[1:],
                base_grammar=self.grammars[0],
                motif_grammars=self.grammars[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
                inner_mutation_strategy=super().mutate,
            )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        return self.create_graph_from_string(child_string_tree_list)

    @override
    def crossover(
        self,
        parent1: Self,
        parent2: Self | None = None,
    ) -> tuple[Self, Self]:
        raise NotImplementedError
        if parent2 is None:
            parent2 = self
        children = repetitive_search_space_crossover(
            base_parent=(parent1.string_tree_list[0], parent2.string_tree_list[0]),
            motif_parents=(parent1.string_tree_list[1:], parent2.string_tree_list[1:]),
            base_grammar=self.grammars[0],
            motif_grammars=self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            inner_crossover_strategy=simple_crossover,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        return [parent2.create_graph_from_string(child) for child in children]

    @override
    def sample(self, *, user_priors: bool = False) -> Self:
        copy_self = self.clone()
        copy_self.reset()
        copy_self.string_tree_list = [grammar.sampler(1)[0] for grammar in copy_self.grammars]
        copy_self.string_tree = copy_self.assemble_trees(
            copy_self.string_tree_list[0],
            copy_self.string_tree_list[1:],
            terminal_to_sublanguage_map=copy_self.terminal_to_sublanguage_map,
        )
        copy_self.id = "\n".join(copy_self.string_tree_list)
        _ = copy_self.value  # required for checking if graph is valid!
        return copy_self

    @property
    @override
    def value(self) -> nx.DiGraph:
        if self._value is None:
            _val = self.from_stringTree_to_graph_repr(
                self.string_tree,
                self.full_grammar,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
            assert isinstance(_val, nx.DiGraph)
            self._value = _val
        return self._value

    @override
    def compute_prior(self, *, log: bool = True) -> float:
        prior_probs = [
            g.compute_prior(st, log=log)
            for g, st in zip(self.grammars, self.string_tree_list)
        ]
        if log:
            return sum(prior_probs)
        else:
            return np.prod(prior_probs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GraphGrammarRepetitive):
            return NotImplemented

        return self.id == other.id

    def reset(self) -> None:
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self.id = ""

    @staticmethod
    def get_full_grammar(grammars):
        full_grammar = deepcopy(grammars[0])
        rules = full_grammar.productions()
        nonterminals = full_grammar.nonterminals
        terminals = full_grammar.terminals
        for g in grammars[1:]:
            rules.extend(g.productions())
            nonterminals.extend(g.nonterminals)
            terminals.extend(g.terminals)
        return full_grammar

    @abstractmethod
    def create_graph_from_string(self, child: list[str]):
        raise NotImplementedError

    def get_dictionary(self) -> dict[str, str]:
        return {"graph_grammar": "\n".join(self.string_tree_list)}

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.full_grammar)
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def create_from_id(self, identifier: str | list[str]) -> None:
        self.reset()
        self.string_tree_list = (
            identifier.split("\n") if isinstance(identifier, str) else identifier
        )
        self.string_tree = self.assemble_trees(
            self.string_tree_list[0],
            self.string_tree_list[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
        )
        self.id = "\n".join(self.string_tree_list)
        _ = self.value  # required for checking if graph is valid!

    @property
    def search_space_size(self) -> int:
        def recursive_worker(
            nonterminal: Nonterminal, grammar, lower_level_motifs: int = 0
        ) -> int:
            primitive_nonterminal = "OPS"
            if str(nonterminal) == primitive_nonterminal:
                return (
                    lower_level_motifs * self.number_of_repetitive_motifs
                    + len(grammar.productions(lhs=Nonterminal(primitive_nonterminal)))
                    - self.number_of_repetitive_motifs
                )
            potential_productions = grammar.productions(lhs=nonterminal)
            _possibilites = 0
            for potential_production in potential_productions:
                edges_nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in grammar.nonterminals
                ]
                possibilities_per_edge = [
                    recursive_worker(e_nonterminal, grammar, lower_level_motifs)
                    for e_nonterminal in edges_nonterminals
                ]
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        lower_level_motifs = recursive_worker(self.grammars[1].start(), self.grammars[1])
        return recursive_worker(
            self.grammars[0].start(),
            self.grammars[0],
            lower_level_motifs=lower_level_motifs,
        )


class GraphGrammarMultipleRepetitive(GraphParameter, CoreGraphGrammar):
    hp_name = "graph_grammar_multiple_repetitive"

    def __init__(
        self,
        grammars: list[Grammar] | list[ConstrainedGrammar],
        terminal_to_op_names: dict,
        terminal_to_sublanguage_map: dict,
        prior: list[dict] = None,
        terminal_to_graph_edges: dict = None,
        fixed_macro_grammar: bool = False,
        edge_attr: bool = True,
        edge_label: str = "op_name",
        zero_op: list = ["Zero", "zero"],
        identity_op: list = ["Identity", "id"],
        name: str = None,
        scope: str = None,
        **kwargs,
    ):
        def _check_mapping(macro_grammar, motif_grammars, terminal_to_sublanguage_map):
            for terminal, start_symbol in terminal_to_sublanguage_map.items():
                if terminal not in macro_grammar.terminals:
                    raise Exception(f"Terminal {terminal} not defined in macro grammar")
                if not any(
                    start_symbol == str(grammar.start()) for grammar in motif_grammars
                ):
                    raise Exception(
                        f"Start symbol {start_symbol} not defined in motif grammar"
                    )

        def _identify_macro_grammar(grammar, terminal_to_sublanguage_map):
            grammars = deepcopy(grammar)
            motif_grammars = []
            for start_symbol in terminal_to_sublanguage_map.values():
                motif_grammars += [
                    grammar
                    for grammar in grammars
                    if start_symbol == str(grammar.start())
                ]
                grammars = [
                    grammar
                    for grammar in grammars
                    if start_symbol != str(grammar.start())
                ]
            if len(grammars) != 1:
                raise Exception("Cannot identify macro grammar")
            return grammars[0], motif_grammars

        if prior is not None:
            assert len(grammars) == len(
                prior
            ), "At least one of the grammars has no prior defined!"
            for g, p in zip(grammars, prior):
                g.prior = p
        self.has_prior = prior is not None

        self.macro_grammar, grammars = _identify_macro_grammar(
            grammars, terminal_to_sublanguage_map
        )
        _check_mapping(self.macro_grammar, grammars, terminal_to_sublanguage_map)

        self.fixed_macro_grammar = fixed_macro_grammar
        if not self.fixed_macro_grammar:
            grammars.insert(0, self.macro_grammar)

        self.terminal_to_sublanguage_map = OrderedDict(terminal_to_sublanguage_map)
        if any(
            k in terminal_to_op_names for k in self.terminal_to_sublanguage_map.keys()
        ):
            raise Exception(
                f"Terminals {[k for k in self.terminal_to_sublanguage_map.keys()]} already defined in primitives mapping and cannot be used for repetitive substitutions"
            )
        self.number_of_repetitive_motifs_per_grammar = [
            sum(
                map(
                    (str(grammar.start())).__eq__,
                    self.terminal_to_sublanguage_map.values(),
                )
            )
            if str(grammar.start()) in self.terminal_to_sublanguage_map.values()
            else 1
            for grammar in grammars
        ]

        CoreGraphGrammar.__init__(
            self,
            grammars=grammars,
            terminal_to_op_names={
                **terminal_to_op_names,
                **self.terminal_to_sublanguage_map,
            },
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )
        GraphParameter.__init__(self, value=None, default=None, is_fidelity=False)

        self._function_id: str = ""
        self.string_tree: str = ""
        self.string_tree_list: list[str] = []
        self.nxTree: nx.DiGraph | None = None
        self._value: nx.DiGraph | None = None

        if self.fixed_macro_grammar:
            self.fixed_macro_string_tree = self.macro_grammar.sampler(1)[0]

        if self.fixed_macro_grammar:
            self.full_grammar = self.get_full_grammar(
                [self.macro_grammar] + self.grammars
            )
        else:
            self.full_grammar = self.get_full_grammar(self.grammars)

    @override
    def sample(self, *, user_priors: bool = False) -> Self:
        copy_self = self.clone()
        copy_self.reset()
        copy_self.string_tree_list = [
            grammar.sampler(1, user_priors=user_priors)[0]
            for grammar, number_of_motifs in zip(
                copy_self.grammars, copy_self.number_of_repetitive_motifs_per_grammar
            )
            for _ in range(number_of_motifs)
        ]
        copy_self.string_tree = copy_self.assemble_string_tree(copy_self.string_tree_list)
        _ = copy_self.value  # required for checking if graph is valid!
        return copy_self

    @property
    @override
    def value(self) -> nx.DiGraph:
        if self._value is None:
            if self.fixed_macro_grammar:
                self._value = []
                string_list_idx = 0
                for grammar, number_of_motifs in zip(
                    self.grammars, self.number_of_repetitive_motifs_per_grammar
                ):
                    for _ in range(number_of_motifs):
                        self._value.append(
                            self.from_stringTree_to_graph_repr(
                                self.string_tree_list[string_list_idx],
                                grammar,
                                valid_terminals=self.terminal_to_op_names.keys(),
                                edge_attr=self.edge_attr,
                            )
                        )
                        string_list_idx += 1
                self._value = self._value[0]  # TODO trick
            else:
                self._value = self.from_stringTree_to_graph_repr(
                    self.string_tree,
                    self.full_grammar,
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
                motif_trees = self.string_tree_list[1:]
                repetitive_mapping = {
                    replacement: motif
                    for motif, replacement in zip(
                        self.terminal_to_sublanguage_map.keys(), motif_trees
                    )
                }
                for subgraph in self._value[1].values():
                    old_node_attributes = nx.get_node_attributes(subgraph, "op_name")
                    new_node_labels = {
                        k: (repetitive_mapping[v] if v in motif_trees else v)
                        for k, v in old_node_attributes.items()
                    }
                    nx.set_node_attributes(subgraph, new_node_labels, name="op_name")
        return self._value

    @override
    def mutate(
        self,
        parent: Self | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ) -> Self:
        if parent is None:
            parent = self

        bananas_inner_mutation = partial(bananas_mutate, mutation_rate=mutation_rate)
        child_string_tree_list, is_same = repetitive_search_space_mutation(
            base_parent=self.fixed_macro_string_tree
            if self.fixed_macro_grammar
            else parent.string_tree_list[0],
            motif_parents=parent.string_tree_list
            if self.fixed_macro_grammar
            else parent.string_tree_list[1:],
            base_grammar=self.macro_grammar,
            motif_grammars=self.grammars
            if self.fixed_macro_grammar
            else self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            number_of_repetitive_motifs_per_grammar=self.number_of_repetitive_motifs_per_grammar,
            inner_mutation_strategy=bananas_inner_mutation
            if mutation_strategy == "bananas"
            else super().mutate,
            fixed_macro_parent=self.fixed_macro_grammar,
        )

        if all(is_same):
            raise ValueError("Parent is the same as child!")

        if self.fixed_macro_grammar:
            child_string_tree_list = child_string_tree_list[1:]

        return self.create_new_instance_from_id(
            self.string_tree_list_to_id(child_string_tree_list)
        )

    @override
    def crossover(
        self,
        parent1: Self,
        parent2: Self | None = None,
    ) -> tuple[Self, Self]:
        if parent2 is None:
            parent2 = self
        children = repetitive_search_space_crossover(
            base_parent=(parent1.fixed_macro_string_tree, parent2.fixed_macro_string_tree)
            if self.fixed_macro_grammar
            else (parent1.string_tree_list[0], parent2.string_tree_list[0]),
            motif_parents=(parent1.string_tree_list, parent2.string_tree_list)
            if self.fixed_macro_grammar
            else (parent1.string_tree_list[1:], parent2.string_tree_list[1:]),
            base_grammar=self.macro_grammar,
            motif_grammars=self.grammars
            if self.fixed_macro_grammar
            else self.grammars[1:],
            terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            number_of_repetitive_motifs_per_grammar=self.number_of_repetitive_motifs_per_grammar,
            inner_crossover_strategy=simple_crossover,
            fixed_macro_parent=self.fixed_macro_grammar,
            multiple_repetitive=True,
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")

        return tuple(
            parent2.create_new_instance_from_id(
                self.string_tree_list_to_id(
                    child[1:] if self.fixed_macro_grammar else child
                )
            )
            for child in children
        )

    @override
    def compute_prior(self, *, log: bool = True) -> float:
        prior_probs = [
            g.compute_prior(st, log=log)
            for g, st in zip(self.grammars, self.string_tree_list)
        ]
        if log:
            return sum(prior_probs)
        else:
            return np.prod(prior_probs)

    @property
    def id(self) -> str:
        if self._function_id is None or self._function_id == "":
            if len(self.string_tree_list) == 0:
                raise ValueError("Cannot infer identifier")
            self._function_id = self.string_tree_list_to_id(self.string_tree_list)
        return self._function_id

    @id.setter
    def id(self, value: str) -> None:
        self._function_id = value

    @staticmethod
    def id_to_string_tree_list(identifier: str) -> list[str]:
        return identifier.split("\n")

    def id_to_string_tree(self, identifier: str) -> str:
        string_tree_list = self.id_to_string_tree_list(identifier)
        return self.assemble_string_tree(string_tree_list)

    @staticmethod
    def string_tree_list_to_id(string_tree_list: list[str]) -> str:
        return "\n".join(string_tree_list)

    def string_tree_to_id(self, string_tree: str) -> str:
        raise NotImplementedError

    def assemble_string_tree(self, string_tree_list: list[str]) -> str:
        if self.fixed_macro_grammar:
            string_tree = self.assemble_trees(
                self.fixed_macro_string_tree,
                string_tree_list,
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            )
        else:
            string_tree = self.assemble_trees(
                string_tree_list[0],
                string_tree_list[1:],
                terminal_to_sublanguage_map=self.terminal_to_sublanguage_map,
            )
        return string_tree

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GraphGrammarMultipleRepetitive):
            return NotImplemented
        return self.id == other.id

    def reset(self) -> None:
        self.clear_graph()
        self.string_tree_list = []
        self.string_tree = ""
        self.nxTree = None
        self._value = None
        self._function_id = ""

    def compose_functions(self, flatten_graph: bool = True):
        return self._compose_functions(self.id, self.full_grammar, flatten_graph)

    def unparse_tree(self, identifier: str, as_composition: bool = True):
        return self._unparse_tree(identifier, self.full_grammar, as_composition)

    @staticmethod
    def get_full_grammar(grammars):
        full_grammar = deepcopy(grammars[0])
        rules = full_grammar.productions()
        nonterminals = full_grammar.nonterminals
        terminals = full_grammar.terminals
        for g in grammars[1:]:
            rules.extend(g.productions())
            nonterminals.extend(g.nonterminals)
            terminals.extend(g.terminals)
        return full_grammar

    @abstractmethod
    def create_new_instance_from_id(self, child: str):
        raise NotImplementedError

    def get_dictionary(self) -> dict[str, str]:
        return {"graph_grammar": self.id}

    def create_nx_tree(self, string_tree: str) -> nx.DiGraph:
        nxTree = self.from_stringTree_to_nxTree(string_tree, self.full_grammar)
        return self.prune_tree(
            nxTree, terminal_to_torch_map_keys=self.terminal_to_op_names.keys()
        )

    def create_from_id(self, identifier: str) -> None:
        self.reset()
        self.id = identifier
        self.string_tree_list = self.id_to_string_tree_list(self.id)
        self.string_tree = self.id_to_string_tree(self.id)
        _ = self.value  # required for checking if graph is valid!

    @property
    def search_space_size(self) -> int:
        def recursive_worker(
            nonterminal: Nonterminal, grammar, lower_level_motifs: dict = None
        ) -> int:
            if lower_level_motifs is None:
                lower_level_motifs = {}
            potential_productions = grammar.productions(lhs=nonterminal)
            _possibilites = 0
            for potential_production in potential_productions:
                edges_nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in grammar.nonterminals
                ]
                possibilities_per_edge = [
                    recursive_worker(e_nonterminal, grammar, lower_level_motifs)
                    for e_nonterminal in edges_nonterminals
                ]
                possibilities_per_edge += [
                    lower_level_motifs[str(rhs_sym)]
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in lower_level_motifs.keys()
                ]
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        if self.fixed_macro_grammar:
            if len(self.grammars) > 1:
                raise Exception(
                    "Compute space size for fixed macro only works for one repetitive level"
                )
            return np.prod(
                [
                    grammar.compute_space_size
                    for grammar, n_grammar in zip(
                        self.grammars, self.number_of_repetitive_motifs_per_grammar
                    )
                    for _ in range(n_grammar)
                ]
            )
        else:
            if len(self.grammars) > 2:
                raise Exception(
                    "Compute space size for no fixed macro only works for one repetitive level"
                )
            macro_space_size = self.grammars[0].compute_space_size
            motif_space_size = self.grammars[1].compute_space_size
            return (
                macro_space_size
                // self.number_of_repetitive_motifs_per_grammar[1]
                * motif_space_size
            )
