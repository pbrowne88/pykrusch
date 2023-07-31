from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import TYPE_CHECKING
from pykrusch.config import *
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from pykrusch.figureControl import MathImage
    from pykrusch.dist import Dist, DiscreteDist
    from arviz.data import InferenceData


def plot_distrogram(
    dist: Dist,
    mi: MathImage,
    plot_posterior: bool = False,
    posterior_data: InferenceData | None = None,
):
    plt.style.use("default")
    xmin = dist.xmin
    xmax = dist.xmax
    ls = np.linspace(xmin, xmax, LINSPACE_N)
    plt.plot(ls, dist.pdf(ls), linewidth=DISTROGRAM_LINE_WIDTH)

    if plot_posterior and dist.numerical and dist.scipy:
        data_vars = list(posterior_data.posterior.data_vars)

        if dist.varname in data_vars:
            vals = posterior_data.posterior[dist.varname].values
            combined_trace = np.concatenate(vals)

            if combined_trace.ndim == 1:
                range_buffer = abs(max(combined_trace) - min(combined_trace)) * 0.1

                pxmin = min(combined_trace) - range_buffer
                pxmax = max(combined_trace) + range_buffer

                pls = np.linspace(pxmin, pxmax, LINSPACE_N)

                kde = stats.gaussian_kde(combined_trace)

                plt.plot(pls, kde.evaluate(pls), linewidth=DISTROGRAM_LINE_WIDTH)

            else:
                print(f"{dist.varname} did not have exactly one dimension")

        else:
            print(f"{dist.varname} not found in posterior data variables!")

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=15)

    if (xmin - 1) < 0 and 0 < (xmax + 1) and dist.numerical and dist.scipy:
        plt.axvline(x=0, alpha=0.1, c="black", linewidth=DISTROGRAM_LINE_WIDTH)

    if not dist.numerical or not dist.scipy:
        plt.axis("off")

    plt.savefig(
        mi.filepath,
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0,
    )

    plt.close()


def plot_discrete_distrogram(
    dist: Dist | DiscreteDist,
    mi: MathImage,
    plot_posterior: bool = False,
    posterior_data: InferenceData | None = None,
):
    plt.style.use("default")
    xmin = dist.xmin
    xmax = dist.xmax + 1

    ar = np.arange(xmin, xmax)

    plt.bar(
        ar,
        dist.pmf(ar),
        linewidth=DISCRETE_LINE_WIDTH,
        color="None",
        edgecolor="tab:blue",
    )

    plt.bar([xmin], [-0.00001], color="white")

    if plot_posterior and dist.numerical and dist.scipy:
        data_vars = list(posterior_data.posterior.data_vars)

        if dist.varname in data_vars:
            combined_trace = np.concatenate(
                posterior_data.posterior[dist.varname].values
            )

            pxmin = min(combined_trace)
            pxmax = max(combined_trace) + 1
            par = np.arange(pxmin, pxmax)

            unique, counts = np.unique(combined_trace, return_counts=True)

            ardict = {}

            for i in par:
                ardict[i] = 0
                if i in unique:
                    ardict[i] += float(counts[np.where(unique == i)] / sum(counts))

            plt.bar(
                par,
                list(ardict.values()),
                linewidth=2,
                color="None",
                edgecolor="tab:orange",
                width=0.7,
            )

        else:
            print(f"{dist.varname} not found in posterior data variables!")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=15)

    if (xmin - 1) < 0 and 0 < (xmax + 1) and dist.numerical and dist.scipy:
        plt.axvline(x=0, alpha=0.1, c="black", linewidth=DISTROGRAM_LINE_WIDTH)

    if not dist.numerical or not dist.scipy:
        plt.axis("off")

    plt.savefig(
        mi.filepath,
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0,
    )

    plt.close()


def plot_unknown_distrogram(dist: Dist, mi: MathImage):
    plt.style.use("default")
    fig = plt.figure()

    ls = np.linspace(-3, 3, LINSPACE_N)
    plt.plot(
        ls,
        stats.norm.pdf(ls, loc=0, scale=1),
        linewidth=DISTROGRAM_LINE_WIDTH,
        c="white",
    )

    text = fig.text(0.5, 0.5, s="?", ha="center", va="center", size=70, c="tab:blue")
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.axis("off")
    plt.savefig(
        mi.filepath,
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0,
    )
    plt.close()
