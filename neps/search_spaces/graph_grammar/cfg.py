import itertools
import sys
from collections import defaultdict, deque
from typing import Deque, Tuple

import numpy as np
from nltk import CFG
from nltk.grammar import Nonterminal


class Grammar(CFG):
    """
    Extended context free grammar (CFG) class from the NLTK python package
    We have provided functionality to sample from the CFG.
    We have included generation capability within the class (before it was an external function)
    Also allow sampling to return whole trees (not just the string of terminals)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store some extra quantities needed later
        non_unique_nonterminals = [str(prod.lhs()) for prod in self.productions()]
        self.nonterminals = list(set(non_unique_nonterminals))
        self.terminals = list(
            {str(individual) for prod in self.productions() for individual in prod.rhs()}
            - set(self.nonterminals)
        )
        # collect nonterminals that are worth swapping when doing genetic operations (i.e not those with a single production that leads to a terminal)
        self.swappable_nonterminals = list(
            {i for i in non_unique_nonterminals if non_unique_nonterminals.count(i) > 1}
        )

        self.max_sampling_level = 2

        self.convergent = False

        self.check_grammar()

    def set_convergent(self):
        self.convergent = True

    def set_unconstrained(self):
        self.convergent = False

    def check_grammar(self):
        if len(set(self.terminals).intersection(set(self.nonterminals))) > 0:
            raise Exception(
                f"Same terminal and nonterminal symbol: {set(self.terminals).intersection(set(self.nonterminals))}!"
            )
        for nt in self.nonterminals:
            if len(self.productions(Nonterminal(nt))) == 0:
                raise Exception(f"There is no production for nonterminal {nt}")

    @property
    def compute_space_size(self) -> int:
        """Computes the size of the space described by the grammar.

        Args:
            primitive_nonterminal (str, optional): The primitive nonterminal of the grammar. Defaults to "OPS".

        Returns:
            int: size of space described by grammar.
        """

        def recursive_worker(nonterminal: Nonterminal, memory_bank: dict = None) -> int:
            if memory_bank is None:
                memory_bank = {}

            potential_productions = self.productions(lhs=nonterminal)
            _possibilites = 0
            for potential_production in potential_productions:
                edges_nonterminals = [
                    rhs_sym
                    for rhs_sym in potential_production.rhs()
                    if str(rhs_sym) in self.nonterminals
                ]
                possibilities_per_edge = [
                    memory_bank[str(e_nonterminal)]
                    if str(e_nonterminal) in memory_bank.keys()
                    else recursive_worker(e_nonterminal, memory_bank)
                    for e_nonterminal in edges_nonterminals
                ]
                memory_bank.update(
                    {
                        str(e_nonterminal): possibilities_per_edge[i]
                        for i, e_nonterminal in enumerate(edges_nonterminals)
                    }
                )
                product = 1
                for p in possibilities_per_edge:
                    product *= p
                _possibilites += product
            return _possibilites

        return recursive_worker(self.start())

    def generator(self, n=1, depth=5):
        # return the first n strings generated by the CFG of a maximum depth
        sequences = []
        for sentence in self._generate(n=n, depth=depth):
            sequences.append(" ".join(sentence))
        return sequences

    def sampler_restricted(self, n, max_length=5, cfactor=0.1, min_length=0):
        # sample n unqiue sequences from the CFG
        # such that the number of terminals is between min_length and max_length
        # cfactor controls the avg length of sampled sequence (see self.sampler)
        # setting smaller cfactor can reduce number of samples required to find n of specified size

        # store in a dict fr quick look up when seeing if its a unique sample
        sequences_dict = {}
        sequences = [[]] * n
        i = 0
        while i < n:
            sample = self._convergent_sampler(symbol=self.start(), cfactor=cfactor)
            # split up words, depth and num productions
            tree = sample[0] + ")"
            # count number of terminals
            length = 0
            for t in self.terminals:
                length += tree.count(t + ")")
            # check satisfies depth restrictions
            if (length <= max_length) and (length >= min_length):
                # check not already in samples
                if tree not in sequences_dict:
                    sequences_dict[tree] = "true"
                    sequences[i] = tree
                    i += 1
        return sequences

    def sampler(
        self,
        n=1,
        start_symbol: str = None,
    ):
        # sample n sequences from the CFG
        # convergent: avoids very long sequences (we advise setting True)
        # cfactor: the factor to downweight productions (cfactor=1 returns to naive sampling strategy)
        #          smaller cfactor provides smaller sequences (on average)

        # Note that a simple recursive traversal of the grammar (setting convergent=False) where we choose
        # productions at random, often hits Python's max recursion depth as the longer a sequnce gets, the
        # less likely it is to terminate. Therefore, we set the default sampler (setting convergent=True) to
        # downweight frequent productions when traversing the grammar.
        # see https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-236grammar
        if start_symbol is None:
            start_symbol = self.start()
        else:
            start_symbol = Nonterminal(start_symbol)

        if self.convergent:
            cfactor = 0.1
            return [
                f"{self._convergent_sampler(symbol=start_symbol, cfactor=cfactor)[0]})"
                for i in range(0, n)
            ]
        else:
            return [f"{self._sampler(symbol=start_symbol)})" for i in range(0, n)]

    def _sampler(self, symbol=None):
        # simple sampler where each production is sampled uniformly from all possible productions
        # Tree choses if return tree or list of terminals
        # recursive implementation

        # init the sequence
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol
        productions = self.productions(lhs=symbol)
        # sample
        production = choice(productions)
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                tree = tree + " " + self._sampler(sym) + ")"
        return tree

    def sampler_maxMin_func(self, symbol: str = None, largest: bool = True):
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol
        productions = self.productions(lhs=symbol)
        # sample
        production = productions[-1 if largest else 0]
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                tree = tree + " " + self.sampler_maxMin_func(sym, largest=largest) + ")"
        return tree

    def _convergent_sampler(
        self, cfactor, symbol=None, pcount=defaultdict(int)
    ):  # pylint: disable=dangerous-default-value
        # sampler that down-weights the probability of selcting the same production many times
        # ensuring that the sampled trees are not 'too' long (size to be controlled by cfactor)
        #
        # recursive implementation
        #:pcount: storage for the productions used in the current branch

        # init the sequence
        tree = "(" + str(symbol)
        # init counter of tree depth and number of production rules
        depth, num_prod = 1, 1
        # collect possible productions from the starting symbol
        productions = self.productions(lhs=symbol)
        # init sampling weights
        weights = []
        # calc weights for the possible productions
        for prod in productions:
            if prod in pcount:
                # if production already occured in branch then downweight
                weights.append(cfactor ** (pcount[prod]))
            else:
                # otherwise set to be 1
                weights.append(1.0)
        # normalize weights to get probabilities
        norm = sum(weights)
        probs = [weight / norm for weight in weights]
        # sample
        production = choice(productions, probs)
        # update counts
        pcount[production] += 1
        depths = []
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                # otherwise keep generating the sequence
                recursion = self._convergent_sampler(
                    symbol=sym, cfactor=cfactor, pcount=pcount
                )
                depths.append(recursion[1])
                num_prod += recursion[2]
                tree = tree + " " + recursion[0] + ")"
        # count the maximum depth and update

        if len(depths) > 0:
            depth = max(depths) + 1
        # update counts
        pcount[production] -= 1
        return tree, depth, num_prod

    def _generate(self, start=None, depth=None, n=None):
        """
        see https://www.nltk.org/_modules/nltk/parse/generate.html
        Generates an iterator of all sentences from a CFG.

        :param grammar: The Grammar used to generate sentences.
        :param start: The Nonterminal from which to start generate sentences.
        :param depth: The maximal depth of the generated tree.
        :param n: The maximum number of sentences to return.
        :return: An iterator of lists of terminal tokens.
        """
        if not start:
            start = self.start()
        if depth is None:
            depth = sys.maxsize

        iter_prod = self._generate_all([start], depth)

        if n:
            iter_prod = itertools.islice(iter_prod, n)

        return iter_prod

    def _generate_all(self, items, depth):
        # see https://www.nltk.org/_modules/nltk/parse/generate.html
        if items:
            try:
                for frag1 in self._generate_one(items[0], depth):
                    for frag2 in self._generate_all(items[1:], depth):
                        yield frag1 + frag2
            except RuntimeError as _error:
                if _error.message == "maximum recursion depth exceeded":
                    # Helpful error message while still showing the recursion stack.
                    raise RuntimeError(
                        "The grammar has rule(s) that yield infinite recursion!!"
                    ) from _error
                else:
                    raise
        else:
            yield []

    def _generate_one(self, item, depth):
        # see https://www.nltk.org/_modules/nltk/parse/generate.html
        if depth > 0:
            if isinstance(item, Nonterminal):
                for prod in self.productions(lhs=item):
                    yield from self._generate_all(prod.rhs(), depth - 1)
            else:
                yield [item]

    @staticmethod
    def _remove_empty_spaces(child):
        while child[0] == " ":
            child = child[1:]
        while child[-1] == " ":
            child = child[:-1]
        return child

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ) -> str:
        """Grammar-based mutation, i.e., we sample a new subtree from a nonterminal
        node in the parse tree.

        Args:
            parent (str): parent of the mutation.
            subtree_index (int): index pointing to the node that is root of the subtree.
            subtree_node (str): nonterminal symbol of the node.
            patience (int, optional): Number of tries. Defaults to 50.

        Returns:
            str: mutated child from parent.
        """
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        _patience = patience
        while _patience > 0:
            # only sample subtree -> avoids full sampling of large parse trees
            new_subtree = self.sampler(1, start_symbol=subtree_node)[0]
            child = pre + new_subtree + post
            if parent != child:  # ensure that parent is really mutated
                break
            _patience -= 1

        child = self._remove_empty_spaces(child)

        return child

    def crossover(
        self,
        parent1: str,
        parent2: str,
        patience: int = 50,
        return_crossover_subtrees: bool = False,
    ):
        # randomly swap subtrees in two trees
        # if no suitiable subtree exists then return False
        subtree_node, subtree_index = self.rand_subtree(parent1)
        # chop out subtree
        pre, sub, post = self.remove_subtree(parent1, subtree_index)
        _patience = patience
        while _patience > 0:
            # sample subtree from donor
            donor_subtree_index = self.rand_subtree_fixed_head(parent2, subtree_node)
            # if no subtrees with right head node return False
            if not donor_subtree_index:
                _patience -= 1
            else:
                donor_pre, donor_sub, donor_post = self.remove_subtree(
                    parent2, donor_subtree_index
                )
                # return the two new tree
                child1 = pre + donor_sub + post
                child2 = donor_pre + sub + donor_post

                child1 = self._remove_empty_spaces(child1)
                child2 = self._remove_empty_spaces(child2)

                if return_crossover_subtrees:
                    return (
                        child1,
                        child2,
                        (pre, sub, post),
                        (donor_pre, donor_sub, donor_post),
                    )
                return child1, child2

        return False, False

    def rand_subtree(self, tree: str) -> Tuple[str, int]:
        """Helper function to choose a random subtree in a given parse tree.
        Runs a single pass through the tree (stored as string) to look for
        the location of swappable nonterminal symbols.

        Args:
            tree (str): parse tree.

        Returns:
            Tuple[str, int]: return the parent node of the subtree and its index.
        """
        split_tree = tree.split(" ")
        swappable_indices = [
            i
            for i in range(0, len(split_tree))
            if split_tree[i][1:] in self.swappable_nonterminals
        ]
        r = np.random.randint(1, len(swappable_indices))
        chosen_non_terminal = split_tree[swappable_indices[r]][1:]
        chosen_non_terminal_index = swappable_indices[r]
        return chosen_non_terminal, chosen_non_terminal_index

    @staticmethod
    def rand_subtree_fixed_head(
        tree: str, head_node: str, swappable_indices: list = None
    ) -> int:
        # helper function to choose a random subtree from a given tree with a specific head node
        # if no such subtree then return False, otherwise return the index of the subtree

        # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
        if swappable_indices is None:
            split_tree = tree.split(" ")
            swappable_indices = [
                i for i in range(0, len(split_tree)) if split_tree[i][1:] == head_node
            ]
        if not isinstance(swappable_indices, list):
            raise TypeError("Expected list for swappable indices!")
        if len(swappable_indices) == 0:
            # no such subtree
            return False
        else:
            # randomly choose one of these non-terminals
            r = (
                np.random.randint(1, len(swappable_indices))
                if len(swappable_indices) > 1
                else 0
            )
            chosen_non_terminal_index = swappable_indices[r]
            return chosen_non_terminal_index

    @staticmethod
    def remove_subtree(tree: str, index: int) -> Tuple[str, str, str]:
        """Helper functioon to remove a subtree from a parse tree
        given its index.
        E.g. '(S (S (T 2)) (ADD +) (T 1))'
        becomes '(S (S (T 2)) ', '(T 1))'  after removing (ADD +)

        Args:
            tree (str): parse tree
            index (int): index of the subtree root node

        Returns:
            Tuple[str, str, str]: part before the subtree, subtree, part past subtree
        """
        split_tree = tree.split(" ")
        pre_subtree = " ".join(split_tree[:index]) + " "
        #  get chars to the right of split
        right = " ".join(split_tree[index + 1 :])
        # remove chosen subtree
        # single pass to find the bracket matching the start of the split
        counter, current_index = 1, 0
        for char in right:
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            if counter == 0:
                break
            current_index += 1
        post_subtree = right[current_index + 1 :]
        removed = "".join(split_tree[index]) + " " + right[: current_index + 1]
        return (pre_subtree, removed, post_subtree)

    @staticmethod
    def unparse_tree(tree: str):
        string = []
        temp = ""
        # perform single pass of tree
        for char in tree:
            if char == " ":
                temp = ""
            elif char == ")":
                if temp[-1] != ")":
                    string.append(temp)
                temp += char
            else:
                temp += char
        return " ".join(string)


class DepthConstrainedGrammar(Grammar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_constraints = None

    def set_depth_constraints(self, depth_constraints):
        self.depth_constraints = depth_constraints
        if not all(k in self.nonterminals for k in self.depth_constraints.keys()):
            raise Exception(
                f"Nonterminal {set(self.depth_constraints.keys())-set(self.nonterminals)} does not exist in grammar"
            )

    @staticmethod
    def is_depth_constrained():
        return True

    def sampler(
        self,
        n=1,
        start_symbol: str = None,
        depth_information: dict = None,
    ):
        if self.depth_constraints is None:
            raise ValueError("Depth constraints are not set!")

        if start_symbol is None:
            start_symbol = self.start()
        else:
            start_symbol = Nonterminal(start_symbol)

        if depth_information is None:
            depth_information = {}
        return [
            f"{self._depth_constrained_sampler(symbol=start_symbol, depth_information=depth_information)})"
            for i in range(0, n)
        ]

    def _compute_depth_information_for_pre(self, tree: str) -> dict:
        depth_information = {nt: 0 for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for split in tree.split(" "):
            if split == "":
                continue
            elif split[0] == "(":
                q_nonterminals.append(split[1:])
                depth_information[split[1:]] += 1
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                depth_information[nt] -= 1
                split = split[:-1]
        return depth_information

    def _compute_depth_information(self, tree: str) -> tuple:
        split_tree = tree.split(" ")
        depth_information = [0] * len(split_tree)
        subtree_depth = [0] * len(split_tree)
        helper_subtree_depth = [0] * len(split_tree)
        helper_dict_depth_information = {nt: 0 for nt in self.nonterminals}
        helper_dict_subtree_depth: dict = {nt: deque() for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for i, split in enumerate(split_tree):
            if split == "":
                continue
            elif split[0] == "(":
                nt = split[1:]
                q_nonterminals.append(nt)
                depth_information[i] = helper_dict_depth_information[nt] + 1
                helper_dict_depth_information[nt] += 1
                helper_dict_subtree_depth[nt].append(i)
                for j in helper_dict_subtree_depth[nt]:
                    subtree_depth[j] = max(subtree_depth[j], helper_subtree_depth[j] + 1)
                    helper_subtree_depth[j] += 1
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                helper_dict_depth_information[nt] -= 1
                for j in helper_dict_subtree_depth[nt]:
                    helper_subtree_depth[j] -= 1
                _ = helper_dict_subtree_depth[nt].pop()
                split = split[:-1]
        return depth_information, subtree_depth

    def _compute_max_depth(self, tree: str, subtree_node: str) -> int:
        max_depth = 0
        depth_information = {nt: 0 for nt in self.nonterminals}
        q_nonterminals: Deque = deque()
        for split in tree.split(" "):
            if split == "":
                continue
            elif split[0] == "(":
                q_nonterminals.append(split[1:])
                depth_information[split[1:]] += 1
                if split[1:] == subtree_node and depth_information[split[1:]] > max_depth:
                    max_depth = depth_information[split[1:]]
                continue
            while split[-1] == ")":
                nt = q_nonterminals.pop()
                depth_information[nt] -= 1
                split = split[:-1]
        return max_depth

    def _depth_constrained_sampler(self, symbol=None, depth_information: dict = None):
        if depth_information is None:
            depth_information = {}
        # init the sequence
        tree = "(" + str(symbol)
        # collect possible productions from the starting symbol & filter if constraints are violated
        lhs = str(symbol)
        if lhs in depth_information.keys():
            depth_information[lhs] += 1
        else:
            depth_information[lhs] = 1
        if (
            lhs in self.depth_constraints.keys()
            and depth_information[lhs] >= self.depth_constraints[lhs]
        ):
            productions = [
                production
                for production in self.productions(lhs=symbol)
                if lhs
                not in [str(sym) for sym in production.rhs() if not isinstance(sym, str)]
            ]
        else:
            productions = self.productions(lhs=symbol)

        if len(productions) == 0:
            raise Exception(
                "There can be no word sampled! This is due to the grammar and/or constraints."
            )

        # sample
        production = choice(productions)
        for sym in production.rhs():
            if isinstance(sym, str):
                # if terminal then add string to sequence
                tree = tree + " " + sym
            else:
                tree = (
                    tree
                    + " "
                    + self._depth_constrained_sampler(sym, depth_information)
                    + ")"
                )
        depth_information[lhs] -= 1
        return tree

    def mutate(
        self, parent: str, subtree_index: int, subtree_node: str, patience: int = 50
    ) -> str:
        # chop out subtree
        pre, _, post = self.remove_subtree(parent, subtree_index)
        _patience = patience
        while _patience > 0:
            # only sample subtree -> avoids full sampling of large parse trees
            depth_information = self._compute_depth_information_for_pre(pre)
            new_subtree = self.sampler(
                1, start_symbol=subtree_node, depth_information=depth_information
            )[0]
            child = pre + new_subtree + post
            if parent != child:  # ensure that parent is really mutated
                break
            _patience -= 1
        child = self._remove_empty_spaces(child)
        return child

    def crossover(
        self,
        parent1: str,
        parent2: str,
        patience: int = 50,
        return_crossover_subtrees: bool = False,
    ):
        # randomly swap subtrees in two trees
        # if no suitiable subtree exists then return False
        subtree_node, subtree_index = self.rand_subtree(parent1)
        # chop out subtree
        pre, sub, post = self.remove_subtree(parent1, subtree_index)
        head_node_depth = self._compute_depth_information_for_pre(pre)[subtree_node] + 1
        sub_depth = self._compute_max_depth(sub, subtree_node)
        _patience = patience
        while _patience > 0:
            # sample subtree from donor
            donor_subtree_index = self._rand_subtree_fixed_head(
                parent2, subtree_node, head_node_depth, sub_depth=sub_depth
            )
            # if no subtrees with right head node return False
            if not donor_subtree_index:
                _patience -= 1
            else:
                donor_pre, donor_sub, donor_post = self.remove_subtree(
                    parent2, donor_subtree_index
                )
                # return the two new tree
                child1 = pre + donor_sub + post
                child2 = donor_pre + sub + donor_post
                child1 = self._remove_empty_spaces(child1)
                child2 = self._remove_empty_spaces(child2)

                if return_crossover_subtrees:
                    return (
                        child1,
                        child2,
                        (pre, sub, post),
                        (donor_pre, donor_sub, donor_post),
                    )
                return child1, child2

        return False, False

    def _rand_subtree_fixed_head(
        self,
        tree: str,
        head_node: str,
        head_node_depth: int = 0,
        sub_depth: int = 0,
    ) -> int:
        # helper function to choose a random subtree from a given tree with a specific head node
        # if no such subtree then return False, otherwise return the index of the subtree

        # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
        if head_node in self.depth_constraints:
            depth_information, subtree_depth = self._compute_depth_information(tree)
            split_tree = tree.split(" ")
            swappable_indices = [
                i
                for i in range(len(split_tree))
                if split_tree[i][1:] == head_node
                and head_node_depth - 1 + subtree_depth[i]
                <= self.depth_constraints[head_node]
                and depth_information[i] - 1 + sub_depth
                <= self.depth_constraints[head_node]
            ]
        else:
            swappable_indices = None
        return super().rand_subtree_fixed_head(
            tree=tree, head_node=head_node, swappable_indices=swappable_indices
        )


# helper function for quickly getting a single sample from multinomial with probs
def choice(options, probs=None):
    x = np.random.rand()
    if probs is None:
        # then uniform probs
        num = len(options)
        probs = [1 / num] * num
    cum = 0
    choice = -1
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            choice = i
            break
    return options[choice]
