DATA_LETTER = "X"
LATEX_NAMES = True

PARAM_CONVERSION = {
    "mu": "μ",
    "lam": "λ",
    "sigma": "σ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "nu": "ν",
    "tau": "τ",
    "pi": "π",
    "lambda": "λ",
    "theta": "θ",
}

NODE_COLOUR_GV = "grey97"
NODE_COLOUR_MPL = "#F7F7F7"

PARAM_FONT_SIZE = "35.0"
PARAM_FONT_SIZE_NUMERICAL = "30.0"
DIST_FONT_SIZE = "40.0"
DIST_FONT_SIZE_SMALL = "20.0"

IMAGE_SCALE_FACTOR = 1
DIST_IMAGE_WIDTH = 82.5
DIST_IMAGE_HEIGHT = 61.5

DISTROGRAM_LINE_WIDTH = 5
DISCRETE_LINE_WIDTH = 2


if LATEX_NAMES:
    DETERMINISTIC_TEXT_SCALE = 1
else:
    DETERMINISTIC_TEXT_SCALE = 0.7
DETERMINISTIC_TEXT_HEIGHT = 80

TEMP_IMG_DIR = ".PYKRUSCHFIGTEMP"

DPI = 300
ENGINE = "dot"

LINSPACE_N = 500
