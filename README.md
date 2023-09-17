- [`pykrusch`](#pykrusch)
	- [Example](#example)
	- [Installation Prerequisites](#installation-prerequisites)
	- [Installation](#installation)
	- [Usage](#usage)
		- [As a Function](#as-a-function)
		- [From the Terminal](#from-the-terminal)


# `pykrusch`

`pykrusch` is a package for visualizing Bayesian Generalized Linear Model structure. It is built atop [`pymc`](https://github.com/pymc-devs/pymc) -- a Python package for building and fitting Bayesian statistical models. 

This package was inspired by the diagrams John K. Kruschke developed to visualize Bayesian model structure in his book, [Doing Bayesian Data Analysis](https://sites.google.com/site/doingbayesiandataanalysis/). 


## Example

`pykrusch` is designed to produce informative, intuitive visualizations of Bayesan GLMs specified in `pymc`. It takes in a model specification such as the following:

```python
import pymc as pm
import numpy as np

x = np.random.normal(loc=0, scale=1, size=10) # Data Placeholder
y = np.random.normal(loc=2*x, scale=1, size=10) # Data Placeholder


with pm.Model() as model:

	# Priors
	α = pm.Normal("α", 1, 2)
	β = pm.Normal("β", -1, 2)
	σ = pm.Exponential("σ", 3)

	# Linear Model
	μ = pm.Deterministic("μ", α + β*x)
	
	# Likelihood
	y_ = pm.Normal("y", μ, σ, observed=y)
```

With the `pymc` model specified, users may pass the `pymc` model to `pykrusch`'s `krusch` function as the first and only argument, like so:

```python
from pykrusch import krusch

krusch(model)
```

Doing so will produce a visualization of the model in the current working directory that will, by default, be titled `krusch.png`. Using the model code specified above, `pykrusch` will produce the following image:

<img src="img/simple_model.png" width="500">



## Installation Prerequisites

The `pykrusch` package depends on `graphviz`, which must be installed before attempting to install `pykrusch`. Installation instructions for `graphviz` can be found at the [GraphViz installation instructions page.](https://pygraphviz.github.io/documentation/stable/install.html#windows-install)


## Installation

To install `pykrusch`, use the following terminal command:

```bash
pip install pkyrusch
```


## Usage

The `pykrusch` package can be used as a function from within a Python (`.py`) / iPython (`.ipynb`) file, or it can be used from the terminal.

### As a Function

To use `pykrusch` as a function, simply import the package and call the `krusch()` function, supplying the name of the `pymc` model as the first argument:

```python
from pykrusch import krusch

krusch(model)
```

`pykrusch` will create a visualization of the Bayesian GLM it is supplied with, outputting the result as `krusch.png` into the working directory. By altering the `outname` parameter, one can change the location, name, and format of the image output (at this time, `png` is the only supported image type; others may or may not function properly). For example:

```python
krusch(model, outname="img/simple_model.png")
```

By default, `pykrusch` only plots prior distributions. If one wishes to plot certain applicable posteriors alongside the priors, one can supply an `arviz` `InferenceData` object to the `posterior_trace` argument: 


```python
pykrusch.krusch(model, posterior_trace=trace)
```

The resulting is featured below, with prior distributions in blue and posterior distributions in orange:

<img src="img/simple_model_posterior.png" width="500">

### From the Terminal

To use `pykrusch` from the command line, one must supply it with the name of the Python file that contains the `pymc` model in question. 

```bash
pykrusch simple_model.py
```

Operating in this mode, `pykrusch` assumes that the name of the model in the specified file is `model`. If the name of the Bayesian model is not `model`, the model's name can be specified using the `--model-name`/`-n` argument, like so:

```bash
pykrusch simple_model.py -n simple_model
```

By default, `pykrusch` produces an output image named `krusch.png`. If you would like `pykrusch` to give the output image a different name, you can use the `--outname`/`-o` argument:

```bash
pykrusch simple_model.py -o model_image.png
```

If you would like `pykrusch` to plot the model's posterior distributions in addition to the priors, you have two options. The first option involves sampling from your model using `pm.sample()` and storing the resulting `InferenceData` object as a `.pkl` file using `pickle`. The resulting `.pkl` file can then be fed in as an argument to `pykrusch`'s `--posterior-pickle`/`-p` argument:

```bash
pykrusch simple_model.py -p posterior_pickle.pkl
```

You can also instruct `pykrusch` to sample from the model using the default `pymc` sampling configuration and extract the resulting posterior distributions for plotting. This is accomplished via the `--sample-posterior`/`-s` flag:

```bash
pykrusch simple_model.py -s
```

Note: using the default `pymc` sampling behaviour may be slow and might not produce satisfactory results for all models. The sampling may contain large numbers of divergences, fail to converge, or produce nonsensical results. Only use the `--sample-posterior`/`-s` flag if you have tested your model using `pm.sample()` with no arguments and found the results to be to your satisfaction.


