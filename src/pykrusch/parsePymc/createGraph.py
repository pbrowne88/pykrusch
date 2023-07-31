import pygraphviz as pgv
from pykrusch.parsePymc.randomVariable import specify_rv_nodes, add_rv_nodes
from pykrusch.parsePymc.deterministicVariable import specify_f_nodes, add_f_nodes
from pymc import Model


def create_graph(model: Model, plot_posterior, posterior_data):
    """
    This function serves as a 'main' of sorts: it is resposible for calling
    the other functions needed to get all the nodes in the graph
    and then assemble them into graph format.
    """

    # Start by finding the parameters for each of the
    # free, observed, and deterministic nodes
    rv_nodes = specify_rv_nodes(model.free_RVs, plot_posterior, posterior_data)
    obs_nodes = specify_rv_nodes(
        model.observed_RVs, plot_posterior=False, posterior_data=None
    )
    f_nodes = specify_f_nodes(model.deterministics)

    # Instantiate an empty directed graph
    graph = pgv.AGraph(directed=True, strict=False)

    # Add the nodes to the graph
    add_rv_nodes(rv_nodes, graph)
    add_rv_nodes(obs_nodes, graph)
    add_f_nodes(f_nodes, graph)

    return graph
