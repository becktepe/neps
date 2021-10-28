from itertools import combinations
from typing import List

import networkx as nx

from ..hyperparameter import Hyperparameter
from ..numerical.categorical import CategoricalHyperparameter


class GraphHyperparameter(Hyperparameter):
    def __init__(self, name: str, num_nodes: int, edge_choices: List[str]):
        super().__init__(name)

        assert num_nodes > 1, "DAG has to have more than one node"
        self.num_nodes = num_nodes
        self.edge_list = list(combinations(list(range(num_nodes)), 2))
        self.edge_choices = edge_choices
        self.graph = []
        for edge_id, _ in enumerate(self.edge_list):
            self.graph.append(
                CategoricalHyperparameter(
                    name="edge_%d" % edge_id, choices=self.edge_choices
                )
            )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.num_nodes == other.num_nodes
            and self.edge_list == other.edge_list
            and self.edge_choices == other.edge_choices
        )

    def __hash__(self):
        return hash((self.name, self.num_nodes, self.edge_choices))

    def __repr__(self):
        return "Graph {}, num_nodes: {}, edge_choices: {}".format(
            self.name, self.num_nodes, self.edge_choices
        )

    def __copy__(self):
        return self.__class__(
            name=self.name, num_nodes=self.num_nodes, edge_choices=self.edge_choices
        )

    def sample(self, random_state):
        edge_labels = []
        for edge in self.graph:
            edge_labels.append(edge.sample(random_state))

        G = nx.DiGraph()
        G.add_edges_from(self.edge_list)

        edge_attribute = {}
        remove_edge_list = []

        for i, edge in enumerate(self.edge_list):
            edge_attribute[edge] = {"op_name": edge_labels[i]}
            if edge_labels[i] == "none":
                remove_edge_list.append(edge)

        nx.set_edge_attributes(G, edge_attribute)
        G.remove_edges_from(remove_edge_list)

        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != self.num_nodes - 1 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        # Assign dummy variables as node attributes:
        for i in G.nodes:
            G.nodes[i]["op_name"] = "1"
        G.graph_type = "edge_attr"

        if nx.is_empty(G):
            raise ValueError("Invalid DAG")

        return G

    def mutate(self, parent=None):
        pass

    def crossover(self, parent1, parent2=None):
        pass
