from __future__ import annotations
from typing import TYPE_CHECKING

import click
from pykrusch.config import *

from shutil import rmtree

if TYPE_CHECKING:
    from pygraphviz import AGraph
    from pymc import Model
    from arviz import InferenceData




# region
def posterior_data_check(
    MODEL: Model,
    posterior_trace: InferenceData | None = None,
    posterior_pickle: str | None = None,
    sample_posterior: bool = False,
):
    too_many = SystemExit(
        """More than one posterior source was passed. Please pass at most
        one of posterior_data, posterior_pickle, or sample_posterior=True."""
    )

    if posterior_trace and posterior_pickle:
        raise too_many
    if posterior_trace and sample_posterior:
        raise too_many
    if posterior_pickle and sample_posterior:
        raise too_many
    if posterior_trace:
        print(" ")
        print("Using supplied trace data.")
        print(" ")
        return True, posterior_trace
    if posterior_pickle:
        import pickle

        print(" ")
        print("Opening model trace pickle.")
        print(" ")
        with open(posterior_pickle, "rb") as stream:
            return True, pickle.load(stream)
    if sample_posterior:
        print(" ")
        print("Sampling pymc model.")
        print(" ")
        import pymc as pm
        with MODEL:
            return True, pm.sample()
        
    return False, None


def model_check(
    MODEL,
    model_file,
    model_name,
):
    if not MODEL and not model_file:
        raise SystemExit(
            """Please provide either MODEL (in the form of a pymc Model object), or model_file. 
Running pykrusch from the command line requires model_file."""
        )
    if MODEL and model_file:
        raise SystemExit(
            """Cannot pass both MODEL and model_file. Please provide exactly one of MODEL, model_file."""
        )
    if model_file and not model_name:
        raise SystemExit("""Use of model_file requires model_name.""")
    if MODEL:
        return MODEL
    if model_file and model_name:
        from importlib.util import spec_from_file_location, module_from_spec
        import sys

        spec = spec_from_file_location("_MODEL_FROM_FILE", model_file)
        mod = module_from_spec(spec)
        sys.modules["_MODEL_FROM_FILE"] = mod
        spec.loader.exec_module(mod)
        return mod.__dict__[model_name]


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help="""Visualize Bayesian model structure. 
Takes a Bayesian model specified in pymc and outputs an image portraying the dependency
structure formed by the model's random and deterministic variables, including prior
distribution (and, if specified, posterior distribution).""",
)
@click.argument(
    "model_file",
    type=click.Path(exists=True)
)
@click.option(
    "-n",
    "--model_name",
    default="model",
    help="Name of the model object in model_file. Defaults to 'model'.",
)
@click.option(
    "-p",
    "--posterior_pickle",
    default=None,
    help="Path to the .pkl file containing the sampling trace (only needed for plotting posterior).",
)
@click.option(
    "-s",
    "--sample_posterior",
    is_flag=True,
    default=False,
    help="Flag for sampling the posterior from the model. Can be slow (only needed if plotting posterior).",
)
@click.option(
    "-o",
    "--outname",
    default="krusch.png",
    help="Path of the output image. Defaults to 'krusch.png'.",
)
@click.option(
    "-r",
    "--retain_figs",
    is_flag=True,
    default=False,
    help="Flag for retaining the intermediary figures (defaults to False).",
)
def main(
    model_file: str | None = None,
    model_name: str | None = None,
    posterior_pickle: str | None = None,
    sample_posterior: bool = False,
    outname: str = "krusch.png",
    retain_figs=False,
):
    

    krusch_(
        pymc_model=None,
        model_file=model_file,
        model_name=model_name,
        posterior_trace=None,
        posterior_pickle=posterior_pickle,
        sample_posterior=sample_posterior,
        outname=outname,
        retain_figs=retain_figs,
    )


# endregion


def krusch(
        pymc_model: Model,
        posterior_trace: InferenceData|None = None,
        outname: str = "krusch.png"): 
    """Produces a visualization of a `pymc`-specified Bayesian Generalized Linear Model.

    Args:

        pymc_model (Model): 
            The `pymc` `Model` object to be visualized.

        posterior_trace (InferenceData | None, optional): 
            Optional: argument specifying the `arviz` `InferenceData` object containing the
            posterior information needed to plot the posterior distribution for each viable
            variable in the model. If supplied, `pykrusch` will plot the posterior to the best
            of its ability. If this argument is not supplied (or is `None`), the posterior 
            distributions will not be plotted.

        outname (str, optional): _description_. 
            Optional: filename of the resulting visualization. Defaults to "krusch.png".
    """
    krusch_(
        pymc_model=pymc_model,
        model_file=None,
        model_name=None,
        posterior_trace=posterior_trace,
        posterior_pickle=None,
        sample_posterior=None,
        outname=outname,
        retain_figs=False,
    )



def krusch_(
    pymc_model: Model | None = None,
    model_file: str | None = None,
    model_name: str | None = None,
    posterior_trace: InferenceData | None = None,
    posterior_pickle: str | None = None,
    sample_posterior: bool = False,
    outname: str = "krusch.png",
    retain_figs=False,
):
    
    from pykrusch.parsePymc.createGraph import create_graph
    from pykrusch.figureControl import FC


    MODEL = model_check(pymc_model, model_file, model_name)

    posterior_data: InferenceData | None = None
    plot_posterior: bool = False

    # Get pymc Model object. If directly supplied as argument to
    # MODEL, simply return MODEL. Otherwise, import the file
    # indicated in model_file and grab the model identified by
    # model_name

    plot_posterior, posterior_data = posterior_data_check(
        MODEL=MODEL,
        posterior_trace=posterior_trace,
        posterior_pickle=posterior_pickle,
        sample_posterior=sample_posterior,
    )

    # This initializes the figure numbering system
    # for the temporary figures
    FC.fig_init(FC)

    ###
    # CORE LOOP: Create graph does most of the heavy lifting
    ###
    graph: AGraph = create_graph(
        MODEL,
        plot_posterior,
        posterior_data,
    )
    graph.graph_attr["dpi"] = DPI
    graph.draw(outname, prog=ENGINE)

    if not retain_figs:
        rmtree(TEMP_IMG_DIR)

