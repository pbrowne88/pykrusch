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



with model:
	trace = pm.sample()


if __name__ == "__main__":

	from pykrusch import krusch
	krusch(model, outname="../img/simple_model_posterior.png", posterior_trace=trace)