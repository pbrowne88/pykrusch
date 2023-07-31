from pygraphviz import AGraph
from pykrusch.config import NODE_COLOUR_GV
from pykrusch.dist import Dist, get_rv_dist
from pykrusch.param import (
    NamedParameter,
    UnknownParameter,
    NumericalParameter,
    DataParameter,
    NamedData,
    ComponentsParameter,
)
from pytensor.graph.basic import Variable
from pykrusch.graphviz.pgvHTML import make_rv_html, make_data_html


# THIS IS HOW YOU STOP CIRCULAR IMPORTS FOR TYPE CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pykrusch.dist import Dist


# Random Variable Handling

# Strategy thus far:
# Step 1: "Mix down" the entire PYMC graph as a pytensor function with no inputs (this preserves the parameters for each of the priors/hyperpriors)
# Step 2: for random variable in the graph, get the owner:
# - As far as I can tell, random variables will always have some kind of `_rv` as the sole owner
# - I can use this to determine the distribution shape (I think), which can then be used to follow important parameters
# Step 3: Follow the origin of the parameters until you come across a named node or an input (defined as a node with no parents, I think)
# - Sometimes, this will be a named node, in which case it's sufficient to merely find it and report it
# - Other times, this will be a
#

# for `normal_rv`, subscript 3 = location (mu), subscript 4 = scale (sigma)
# Only named nodes have a '.name' attribute

# When searching the tree, we're going to terminate at:
# 1. Another Named Node
# 2. A scalar
# 3. Some kinda mathematical operation


class RandomVar:
    def __init__(
        self,
        node,
        plot_posterior,
        posterior_data,
    ):
        self.name: str = node.name
        self.dist: Dist = get_rv_dist(node)
        self.dist.plot_posterior = plot_posterior
        self.dist.posterior_data = posterior_data

        self.shape = "plaintext"
        self.color = NODE_COLOUR_GV
        self.style = "filled"

    def rv_add_self_to_graph(self, graph: AGraph):
        # Add the node plus attributes
        graph.add_node(
            self.name,
            shape=self.shape,
            color=self.color,
            style=self.style,
            label=make_rv_html(self, graph),
        )
        for param in self.dist.params:
            self.connect_param(param=param, graph=graph)

    def connect_param(self, param, graph):
        if isinstance(param, NumericalParameter):
            pass
        elif isinstance(param, NamedParameter):
            graph.add_edge(
                param.name,
                self.name,
                tailport="s",
                headport=str(param.slot) + ":n",
                key=str(param.slot),
            )
        elif isinstance(param, NamedData):
            if not graph.has_node(param.name):
                graph.add_node(
                    param.name,
                    shape="plaintext",
                    color=NODE_COLOUR_GV,
                    style="filled",
                    label=make_data_html(param, graph),
                )
            graph.add_edge(
                param.name,
                self.name,
                tailport="s",
                headport=str(param.slot) + ":n",
                key=str(param.slot),
            )

        elif isinstance(param, ComponentsParameter):
            for component in param.flat_components:
                self.connect_param(component, graph)

        elif isinstance(param, UnknownParameter):
            pass

        elif isinstance(param, DataParameter):
            pass

        else:
            raise Exception(
                f"PyKrusch doesn't know how to handle treating {param} as a {self.dist} Random Variable"
            )


def specify_rv_nodes(
    treelist: list[Variable],
    plot_posterior: bool = False,
    posterior_data=None,
) -> list[RandomVar]:
    return [RandomVar(node, plot_posterior, posterior_data) for node in treelist]


def add_rv_nodes(rv_nodes: list[RandomVar], graph: AGraph):
    for rv in rv_nodes:
        rv.rv_add_self_to_graph(graph)
