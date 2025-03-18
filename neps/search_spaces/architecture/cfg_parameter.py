from typing import Any, ClassVar, Mapping, Literal
from typing_extensions import override, Self

from neps.search_spaces.architecture.cfg import Grammar
from neps.search_spaces.parameter import MutatableParameter, ParameterWithPrior
from neps.search_spaces.architecture.crossover import simple_crossover
from neps.search_spaces.architecture.mutations import bananas_mutate, simple_mutate

class CFGParameter(ParameterWithPrior[str, str], MutatableParameter):
    hp_name = "CFGArchitectureParameter"

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]] = {"not_in_use": 1.0}
    default_confidence_choice = "not_in_use"
    has_prior: bool
    input_kwargs: dict[str, Any]

    def __init__(
            self,
            grammar: Grammar,
            default: str | None = None,
            prior: dict | None = None,
            prior_sampling_mode: Literal["mutation", "distribution"] = "mutation",
        ):
        ParameterWithPrior.__init__(self, value=None, default=default, is_fidelity=False)
        self.grammar = grammar
        if prior is not None:
            self.grammar.prior = prior
        self.has_prior = prior is not None
        self.prior_sampling_mode = prior_sampling_mode

    @override
    def clone(self) -> Self:
        val = self.value
        cloned = self.__class__(**self.input_kwargs)
        cloned._value = val
        return cloned

    @override
    def set_value(self, value: str | None) -> None:
        self._value = value

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CFGParameter):
            return NotImplemented

        return self.value == other.value        

    @override
    def set_default(self, default: str | None) -> None:
        self.default = default

    @classmethod
    def serialize_value(cls, value: str) -> str:
        return value
    
    @classmethod
    def deserialize_value(cls, value: str) -> str:
        return value
    
    @override
    def load_from(self, value: str | Self) -> None:
        if isinstance(value, CFGParameter):
            assert value.value is not None
            value = value.value

        self.create_from_id(value)

    def create_new_instance_from_value(self, value: str) -> Self:
        new_instance = self.clone()
        new_instance._value = value
        return new_instance
    
    def create_from_id(self, identifier: str) -> None:
        self.id = identifier
        self._value = identifier 
        self._function_id = identifier

    @override
    def sample(self, *, user_priors: bool = False) -> Self:
        copy_self = self.clone()
        
        # For prior-sampling be mutate the default configuration
        if self.prior_sampling_mode == "mutation":
            assert self.default is not None
            dummy_parent = self.create_new_instance_from_value(self.default)
            copy_self._value = self.mutate(dummy_parent).value
        else:
            copy_self._value = copy_self.grammar.sampler(1, user_priors=user_priors)[0]
        
        return copy_self
    
    def sample_value(self, *, user_priors: bool = False) -> str:
        _value = self.sample(user_priors=user_priors).value
        assert isinstance(_value, str)
        return _value
    
    @override
    def mutate(
        self,
        parent: Self | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ) -> Self:
        if parent is None:
            parent = self
        parent_string_tree = parent.value
        assert parent_string_tree is not None

        # We have 10 trials to mutate the parent,
        # if we can't mutate it in 10 trials, we raise an exception
        trials = 10
        while True:
            if mutation_strategy == "bananas":
                child_string_tree, is_same = bananas_mutate(
                    parent_string_tree=parent_string_tree,
                    grammar=self.grammar,
                    mutation_rate=mutation_rate,
                )
            else:
                child_string_tree, is_same = simple_mutate(
                    parent_string_tree=parent_string_tree,
                    grammar=self.grammar,
                )

            if is_same:
                trials -= 1
                if trials == 0:
                    raise Exception("Cannot create mutation")
            else:
                break
        
        return self.create_new_instance_from_value(child_string_tree)

    @override
    def crossover(self, parent1: Self, parent2: Self | None = None) -> tuple[Self, Self]:
        if parent2 is None:
            parent2 = self
        parent1_string_tree = parent1.value
        parent2_string_tree = parent2.value
        assert parent1_string_tree is not None
        assert parent2_string_tree is not None

        children = simple_crossover(
            parent1_string_tree, parent2_string_tree, self.grammar
        )
        if all(not c for c in children):
            raise Exception("Cannot create crossover")

        return (
            parent2.create_new_instance_from_value(children[0]),
            parent2.create_new_instance_from_value(children[1])
        )
    
    @override
    def compute_prior(self, *, log: bool = True) -> float:
        return self.grammar.compute_prior(self.value, log=log)
    
    def get_dictionary(self) -> dict[str, str]:
        return {"CFGArchitectureParameter": self.value}
    
    @property
    def search_space_size(self) -> int:
        return self.grammar.compute_space_size
    
    @property
    def id(self) -> str:
        if self._function_id is None or self._function_id == "":
            if self.value == "":
                raise ValueError("Cannot infer identifier!")
            self._function_id = self.string_tree_to_id(self.value)
        return self._function_id
    
    @staticmethod
    def id_to_string_tree(identifier: str) -> str:
        return identifier

    @staticmethod
    def string_tree_to_id(string_tree: str) -> str:
        return string_tree

    @id.setter
    def id(self, value: str) -> None:
        self._function_id = value

    def _get_non_unique_neighbors(self, num_neighbours: int) -> list[Self]:
        raise NotImplementedError

    def value_to_normalized(self, value: str) -> float:
        raise NotImplementedError

    def normalized_to_value(self, normalized_value: float) -> str:
        raise NotImplementedError