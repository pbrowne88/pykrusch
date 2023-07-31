import matplotlib.pyplot as plt
from pykrusch.config import PARAM_CONVERSION
from re import split
from pykrusch.config import *


def string_to_latex(math_str) -> str:
    parts = []
    # Need to make sure that escaped '_' symbols aren't split:

    if LATEX_NAMES:
        for str_part in split(r"(?<!\\)_", math_str):
            if str_part in PARAM_CONVERSION:
                parts.append(PARAM_CONVERSION[str_part])
            else:
                parts.append(str_part)

        if len(parts) == 1:
            latex_str = rf"{parts[0]}"
        else:
            the_rest = "\_".join(parts[1:])
            latex_str = rf"{parts[0]}_{{{the_rest}}}"

        return latex_str

    else:
        return math_str.replace("\\", "").replace("{", "").replace("}", "")


def render_img_from_latex(latex_str: str, filepath):
    plt.style.use("default")

    plt.rcParams["mathtext.fontset"] = "cm"

    if LATEX_NAMES:
        mpl_string = rf"${latex_str}$"
    else:
        mpl_string = rf"{latex_str}".replace("\\", "").replace("{", "").replace("}", "")

    fig = plt.figure()
    fig.patch.set_facecolor(NODE_COLOUR_MPL)
    text = fig.text(0.5, 0.5, s=rf"{mpl_string}", ha="center", va="center", size=15)
    plt.savefig(
        filepath,
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0,
    )
    plt.close()
