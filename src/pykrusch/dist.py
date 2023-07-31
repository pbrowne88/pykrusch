from __future__ import annotations
from pykrusch.param import (
    NumericalParameter,
    NamedParameter,
    UnknownParameter,
)
from pykrusch.param import recursive_rv_search
from pykrusch.figureControl import FC, MathImage, MathImageFound
from pykrusch.graphviz.distrogramsMPL import (
    plot_distrogram,
    plot_unknown_distrogram,
    plot_discrete_distrogram,
)
from pykrusch.graphviz.latexText import string_to_latex, render_img_from_latex
from typing import TYPE_CHECKING
from collections import namedtuple
import scipy.stats as sps
import numpy as np
import pymc as pm

np.seterr(divide="ignore")

if TYPE_CHECKING:
    from pykrusch.param import Parameter
    from pykrusch.parsePymc.randomVariable import RandomVar
    from arviz import InferenceData

_ = 0

partup = namedtuple(
    "Param",
    [
        "greek_name",
        "pos",
        "meaning",
        "op",
    ],
    defaults=[None],
)


def reciprocal(n):
    return 1.0 / n


class Dist:
    def __init__(self, owner, varname=""):
        # These attributes are to be overwritten by the specific classes
        self.needed_params: list[partup]
        self.type: str
        self.distimage: str
        self.fz: sps.rv_continuous | sps.rv_discrete
        self.numerical: bool = False
        self.plot_posterior: bool = False
        self.posterior_data: InferenceData | None = None
        self.scipy = True

        # These attributes are NOT to be overwritten
        self.owner = owner
        self.params: list[Parameter] = self.get_params()
        self.varname = varname
        self.num_params: int = len(self.params)

    def get_params(self):
        parameters = []
        for needed_param in self.needed_params:
            result = recursive_rv_search(self.owner.get_parents()[needed_param.pos])
            result.give_slot(needed_param.pos)
            result.give_greek(needed_param.greek_name)
            result.give_meaning(needed_param.meaning)

            # Apply the special operation to numerical values that get fed through
            if needed_param.op and isinstance(result, NumericalParameter):
                result.value = needed_param.op(result.value)

            parameters.append(result)

        return parameters

    @property
    def dist_name_latex(self):
        mi: MathImage = FC.fig_path(FC, symbols=self.varname)
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(string_to_latex(self.varname), mi.filepath)
        return mi.filepath

    ### TODO: include something here to change between pdf and pmf
    ### depending on whether the distribution is continuous or discrete

    @property
    def distimage(self):
        mi_distrogram: MathImage = FC.fig_path(FC)

        plot_distrogram(
            dist=self,
            mi=mi_distrogram,
            plot_posterior=self.plot_posterior,
            posterior_data=self.posterior_data,
        )
        return mi_distrogram.filepath

    def ppf(self, x):
        return self.fz.ppf(x)

    def pdf(self, x):
        return self.fz.pdf(x)

    @property
    def xmin(self):
        return self.ppf(0.01) - self.xbuffer

    @property
    def xmax(self):
        return self.ppf(0.99) + self.xbuffer

    @property
    def xbuffer(self):
        range = self.ppf(0.99) - self.ppf(0.01)
        return range * 0.1

    def __repr__(self):
        return str(self.type)

    def __str__(self):
        return self.__repr__()


class UnknownDist(Dist):
    def __init__(self, owner, varname=""):
        self.type = "unknown"
        self.numerical = False
        self.owner = owner
        self.params: list[Parameter] = self.get_params()
        self.varname = varname

        self.num_params = 1

    # need to make sure that all of these are

    def get_params(self):
        parameters = []
        for param_slot in self.owner.get_parents()[3:]:
            result = recursive_rv_search(param_slot)
            result.greek_name = "?"
            result.meaning = " "

            if isinstance(result, NamedParameter):
                parameters.append(result)

        if len(parameters) == 0:
            result = UnknownParameter()
            result.greek_name = "?"
            result.meaning = " "
            result.slot = 1

            parameters.append(result)

        return parameters

    @property
    def distimage(self):
        mi_distrogram: MathImage = FC.fig_path(FC)

        plot_unknown_distrogram(dist=self, mi=mi_distrogram)
        return mi_distrogram.filepath


class Normal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Normal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.fz = sps.norm(
                loc=self.params[0].value,
                scale=self.params[1].value,
            )
            self.numerical = True
        else:
            self.fz = sps.norm()


class Exponential(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Exponential"
        self.needed_params = [
            partup("lambda", 3, "rate"),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.fz = sps.expon(scale=1 / self.params[0].value)
            self.numerical = True
        else:
            self.fz = sps.expon(1)


class Gamma(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Gamma"
        self.needed_params = [
            partup("alpha", 3, "shape"),
            partup("beta", 4, "rate", reciprocal),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.fz = sps.gamma(
                a=self.params[0].value,
                scale=1 / self.params[1].value,
            )
            self.numerical = True
        else:
            self.fz = sps.gamma(2, 2)


class Weibull(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Weibull"
        self.needed_params = [partup("alpha", 3, "shape"), partup("beta", 4, "scale")]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.numerical = True
        else:
            self.fz = sps.weibull_min(2, 2)

    def pdf(self, x):
        if self.numerical:
            return sps.weibull_min.pdf(
                x,
                c=self.params[0].value,
                scale=self.params[1].value,
            )
        else:
            return self.fz(x)

    def ppf(self, x):
        if self.numerical:
            return sps.weibull_min.ppf(
                x,
                c=self.params[0].value,
                scale=self.params[1].value,
            )
        else:
            return self.fz(x)


class Uniform(Dist):
    def __init__(self, owner, varname=""):
        self.type = "uniform"
        self.needed_params = [partup("", 3, "lower"), partup("", 4, "upper")]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            lower = self.params[0].value
            upper = self.params[1].value
            scale = upper - lower
            self.fz = sps.uniform(loc=lower, scale=scale)
            self.numerical = True
        else:
            self.fz = sps.uniform()


class TruncatedNormal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "truncated normal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            partup("", 5, "lower"),
            partup("", 6, "upper"),
        ]
        super().__init__(owner, varname)

        fz = sps.truncnorm

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.fz = fz(
                loc=self.params[0].value,
                scale=self.params[1].value,
                a=self.params[2].value,
                b=self.params[3].value,
            )
            self.numerical = True
        else:
            self.fz = fz(
                loc=0,
                scale=1,
                a=-1,
                b=1,
            )


class Beta(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Beta"
        self.needed_params = [
            partup("alpha", 3, "shape"),
            partup("beta", 4, "shape"),
        ]
        super().__init__(owner, varname)

        fz = sps.beta

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.fz = fz(
                a=self.params[0].value,
                b=self.params[1].value,
            )
            self.numerical = True
        else:
            self.fz = fz(
                a=3,
                b=5,
            )

    @property
    def xmin(self):
        return 0

    @property
    def xmax(self):
        return 1


class Kumaraswamy(Dist):
    def __init__(self, owner, varname=""):
        self.type = "kumaraswamy"
        self.needed_params = [
            partup("alpha", 3, "shape"),
            partup("beta", 4, "shape"),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.a = self.params[0].value
            self.b = self.params[1].value
            self.numerical = True
        else:
            self.a = 3
            self.b = 5

    def pdf(self, x):
        a = self.a
        b = self.b

        return a * b * (x ** (a - 1)) * ((1 - (x**a)) ** (b - 1))

    @property
    def xmin(self):
        return 0

    @property
    def xmax(self):
        return 1


class Laplace(Dist):
    def __init__(self, owner, varname=""):
        self.type = "laplace"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("b", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.b = self.params[1].value
            #             _ = self.params[2].value,
            #             _ = self.params[3].value,

            self.numerical = True
        else:
            self.mu = 3
            self.b = 5
        #                 _ = 7,
        #                 _ = 9,

        self.fz: sps.rv_continuous = sps.laplace(
            scale=self.b,
            loc=self.mu
            # _=self._,
            # _=self._,
        )

    def ppf(self, x):
        return self.fz.ppf(x)

    def pdf(self, x):
        return self.fz.pdf(x)


class StudentT(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Student T"
        self.needed_params = [
            partup("nu", 3, "d.o.f."),
            partup("mu", 4, "location"),
            partup("sigma", 5, "scale"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.nu = self.params[0].value
            self.mu = self.params[1].value
            self.sigma = self.params[2].value
            #             _ = self.params[3].value,

            self.numerical = True
        else:
            self.nu = 3
            self.mu = 5
            self.sigma = 7
        #                 _ = 9,

        self.fz: sps.rv_continuous = sps.t

    def ppf(self, x):
        return self.fz.ppf(x, df=self.nu, loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, df=self.nu, loc=self.mu, scale=self.sigma)


class HalfStudent(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Half Student"
        self.needed_params = [
            partup("nu", 3, "d.o.f."),
            partup("sigma", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)
        self.scipy = False

        self.nu: float
        self.sigma: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.nu = self.params[0].value
            self.sigma = self.params[1].value
            self.numerical = True

        else:
            self.nu = 3
            self.sigma = 5

    def pdf(self, x):
        fz = pm.HalfStudentT.dist(nu=self.nu, sigma=self.sigma)
        out = pm.logp(fz, x).eval()
        return np.e ** (out)

    @property
    def xmin(self):
        return -0.1

    @property
    def xmax(self):
        return 5


class Cauchy(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Cauchy"
        self.needed_params = [
            partup("alpha", 3, "location"),
            partup("beta", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.a = self.params[0].value
            self.b = self.params[1].value
            #             self._ = self.params[2].value,
            #             self._ = self.params[3].value,

            self.numerical = True
        else:
            self.a = 0
            self.b = 1
        #                 self._ = 7,
        #                 self._ = 9,

        self.fz: sps.rv_continuous = sps.cauchy

    def ppf(self, x):
        return self.fz.ppf(x, loc=self.a, scale=self.b)

    def pdf(self, x):
        return self.fz.pdf(x, loc=self.a, scale=self.b)


class HalfCauchy(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Half Cauchy"
        self.needed_params = [
            partup("beta", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.beta: float
        self.__: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.beta = self.params[0].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.beta = 3
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.halfcauchy

    def ppf(self, x):
        result = self.fz.ppf(x, scale=self.beta)
        return result

    def pdf(self, x):
        return self.fz.pdf(x, scale=self.beta)


class LogNormal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "LogNormal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = (self.params[0].value,)
            self.sigma = (self.params[1].value,)
            #             self._ = self.params[3].value,

            self.numerical = True
        else:
            self.mu = (3,)
            self.sigma = (5,)
        #                 self._ = 9,

        self.fz: sps.rv_continuous = sps.lognorm

    def ppf(self, x):
        return self.fz.ppf(x, 1, loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, 1, loc=self.mu, scale=self.sigma)

    @property
    def xmin(self):
        return self.ppf(0.1) - self.xbuffer

    @property
    def xmax(self):
        return self.ppf(0.9) + self.xbuffer

    @property
    def xbuffer(self):
        range = self.ppf(0.9) - self.ppf(0.1)
        return range * 0.1


class ChiSquared(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Chi-Squared"
        self.needed_params = [
            partup("nu", 3, "d.o.f."),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.nu = self.params[0].value
            #             self._ = self.params[2].value,
            #             self._ = self.params[3].value,

            self.numerical = True
        else:
            self.nu = 3
        #                 self._ = 7,
        #                 self._ = 9,

        self.fz: sps.rv_continuous = sps.chi2(
            df=self.nu,
            #            _=self._,
            #            _=self._,
        )


class HalfNormal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Half Normal"
        self.needed_params = [
            partup("sigma", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.sigma: float
        self.__: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.sigma = self.params[0].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.sigma = 3
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.halfnorm

    def ppf(self, x):
        result = self.fz.ppf(x, scale=self.sigma)
        return result

    def pdf(self, x):
        return self.fz.pdf(x, scale=self.sigma)


class Wald(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Wald"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("lam", 4, "precision"),
            partup("alpha", 5, "shift"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.lam: float
        self.alpha: float
        # self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.lam = self.params[1].value
            self.alpha = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.lam = 5
            self.alpha = 7

    #                 self.____ = 9

    def pdf(self, x):
        mu = self.mu
        lam = self.lam
        term1 = (lam / (2 * np.pi)) ** (1 / 2)
        term2 = x ** ((-3) / 2)
        term3 = (-lam / (2 * x)) * (((x - mu) / mu) ** 2)
        return term1 * term2 * (np.e ** (term3))

    @property
    def xmin(self):
        return 0

    @property
    def xmax(self):
        return 10


class Pareto(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Pareto"
        self.needed_params = [
            partup("alpha", 3, "shape"),
            partup("m", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.alpha: float
        self.m: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.alpha = self.params[0].value
            self.m = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.alpha = 3
            self.m = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.pareto

    def ppf(self, x):
        return self.fz.ppf(x, b=self.alpha, scale=self.m)

    def pdf(self, x):
        return self.fz.pdf(x, b=self.alpha, scale=self.m)


class InverseGamma(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Inverse Gamma"
        self.needed_params = [
            partup("alpha", 3, "shape"),
            partup("beta", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.alpha: float
        self.beta: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.alpha = self.params[0].value
            self.beta = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.alpha = 3
            self.beta = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.invgamma

    def ppf(self, x):
        return self.fz.ppf(x, a=self.alpha, scale=self.beta)

    def pdf(self, x):
        return self.fz.pdf(x, a=self.alpha, scale=self.beta)


class ExGaussian(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Exponential Gaussian"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            partup("nu", 5, "exponential"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.sigma: float
        self.nu: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.sigma = self.params[1].value
            self.nu = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.sigma = 5
            self.nu = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.exponnorm

    def ppf(self, x):
        return self.fz.ppf(x, K=self.nu, loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, K=self.nu, loc=self.mu, scale=self.sigma)


class VonMises(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Von Mises"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("kappa", 4, "concentration"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.kappa: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.kappa = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.kappa = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.vonmises

    def ppf(self, x):
        return self.fz.ppf(x, kappa=self.kappa, loc=self.mu)

    def pdf(self, x):
        return self.fz.pdf(x, kappa=self.kappa, loc=self.mu)


class SkewNormal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Skew Normal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            partup("alpha", 5, "skew"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.sigma: float
        self.alpha: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.sigma = self.params[1].value
            self.alpha = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.sigma = 5
            self.alpha = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.skewnorm

    def ppf(self, x):
        return self.fz.ppf(x, a=self.alpha, loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, a=self.alpha, loc=self.mu, scale=self.sigma)


class Triangular(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Triangular"
        self.needed_params = [
            partup("", 3, "lower"),
            partup("c", 4, "mode"),
            partup("", 5, "upper"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.lower: float
        self.c: float
        self.upper: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.lower = self.params[0].value
            self.c = self.params[1].value
            self.upper = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.lower = 3
            self.c = 5
            self.upper = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.triang

    def ppf(self, x):
        c = (self.c - self.lower) / (self.upper - self.lower)
        upper = self.upper - self.lower
        result = self.fz.ppf(x, c=c, loc=self.lower, scale=upper)
        return result

    def pdf(self, x):
        c = (self.c - self.lower) / (self.upper - self.lower)
        upper = self.upper - self.lower
        result = self.fz.pdf(x, c=c, loc=self.lower, scale=upper)
        return result


class Gumbel(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Gumbel"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("beta", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.beta: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.beta = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.beta = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.gumbel_r

    def ppf(self, x):
        return self.fz.ppf(x, loc=self.mu, scale=self.beta)

    def pdf(self, x):
        return self.fz.pdf(x, loc=self.mu, scale=self.beta)


class Logistic(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Logistic"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("beta", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.s: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.s = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.s = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.logistic

    def ppf(self, x):
        return self.fz.ppf(x, loc=self.mu, scale=self.s)

    def pdf(self, x):
        return self.fz.pdf(x, loc=self.mu, scale=self.s)


class LogitNormal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Logit Normal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)
        self.scipy = False

        self.mu: float
        self.sigma: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.sigma = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 1
            self.sigma = 2

    #                 self.___ = 7
    #                 self.____ = 9

    def pdf(self, x):
        fz = pm.LogitNormal.dist(mu=self.mu, sigma=self.sigma)
        out = pm.logp(fz, x).eval()
        return np.e ** (out)

    @property
    def xmin(self):
        return -0.1

    @property
    def xmax(self):
        return 1.1

    @property
    def distimage(self):
        mi_distrogram: MathImage = FC.fig_path(FC)

        plot_unknown_distrogram(dist=self, mi=mi_distrogram)
        return mi_distrogram.filepath


class Rice(Dist):
    def __init__(self, owner, varname=""):
        self.type = "rice"
        self.needed_params = [
            partup("b", 3, "shape"),
            partup("sigma", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.b: float
        self.sigma: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.b = self.params[0].value
            self.sigma = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.b = 3
            self.sigma = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.rice

    def ppf(self, x):
        return self.fz.ppf(x, b=self.b, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, b=self.b, scale=self.sigma)


class Moyal(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Moyal"
        self.needed_params = [
            partup("mu", 3, "location"),
            partup("sigma", 4, "scale"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.sigma: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            self.sigma = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
            self.sigma = 5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.moyal

    def ppf(self, x):
        return self.fz.ppf(x, loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self.fz.pdf(x, loc=self.mu, scale=self.sigma)


class AsymmetricLaplace(Dist):
    def __init__(self, owner, varname=""):
        self.type = "Asymmetric Laplace"
        self.needed_params = [
            partup("kappa", 4, "symmetry"),
            partup("mu", 5, "location"),
            partup("b", 3, "scale"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.kappa: float
        self.mu: float
        self.b: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.kappa = self.params[0].value
            self.mu = self.params[1].value
            self.b = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.kappa = 3
            self.mu = 5
            self.b = 7
        #                 self.____ = 9

        self.fz: sps.rv_continuous = sps.laplace_asymmetric

    def ppf(self, x):
        return self.fz.ppf(x, kappa=self.kappa, loc=self.mu, scale=self.b)

    def pdf(self, x):
        return self.fz.pdf(x, kappa=self.kappa, loc=self.mu, scale=self.b)


class DiscreteDist(Dist):
    def pmf(self, x):
        return self.fz.pmf(x)

    @property
    def distimage(self):
        mi_distrogram: MathImage = FC.fig_path(FC)

        plot_discrete_distrogram(
            dist=self,
            mi=mi_distrogram,
            plot_posterior=self.plot_posterior,
            posterior_data=self.posterior_data,
        )
        return mi_distrogram.filepath

    @property
    def xmin(self):
        return self.ppf(0.01)

    @property
    def xmax(self):
        return self.ppf(0.99)


class DiscreteUniform(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Discrete Uniform"
        self.needed_params = [
            partup("", 3, "lower"),
            partup("", 4, "upper"),
        ]
        super().__init__(owner, varname)

        self.lower: float
        self.upper: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.lower = self.params[0].value
            self.upper = self.params[1].value
            self.numerical = True

        else:
            self.lower = 3
            self.upper = 5

        self.fz: sps.rv_discrete = sps.randint

    def ppf(self, x):
        return self.fz.ppf(x, low=self.lower, high=self.upper)

    def pmf(self, x):
        results = self.fz.pmf(x, low=self.lower, high=self.upper)
        return results

    @property
    def xmin(self):
        return self.lower

    @property
    def xmax(self):
        return self.upper


class Binomial(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Binomial"
        self.needed_params = [
            partup("n", 3, "trials"),
            partup("p", 4, "P(success)"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.n: float
        self.p: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.n = self.params[0].value
            self.p = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.n = 10
            self.p = 0.3
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.binom

    def ppf(self, x):
        return self.fz.ppf(x, n=self.n, p=self.p)

    def pmf(self, x):
        return self.fz.pmf(x, n=self.n, p=self.p)

    @property
    def xmin(self):
        return self.ppf(0.01)

    @property
    def xmax(self):
        return self.ppf(0.99)


class BetaBinomial(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Beta Binomial"
        self.needed_params = [
            partup("n", 3, "trials"),
            partup("alpha", 4, "shape"),
            partup("beta", 5, "shape"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.n: float
        self.alpha: float
        self.beta: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.n = self.params[0].value
            self.alpha = self.params[1].value
            self.beta = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.n = 3
            self.alpha = 0.5
            self.beta = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.betabinom

    def ppf(self, x):
        return self.fz.ppf(x, n=self.n, a=self.alpha, b=self.beta)

    def pmf(self, x):
        return self.fz.pmf(x, n=self.n, a=self.alpha, b=self.beta)


class Bernoulli(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Bernoulli"
        self.needed_params = [
            partup("p", 3, "probability"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.p: float
        self.__: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.p = self.params[0].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.p = 0.7
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.bernoulli

    def ppf(self, x):
        return self.fz.ppf(x, p=self.p)

    def pmf(self, x):
        return self.fz.pmf(x, p=self.p)


class Poisson(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Poisson"
        self.needed_params = [
            partup("mu", 3, "occurrences"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.mu: float
        self.__: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.mu = self.params[0].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.mu = 3
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.poisson

    def ppf(self, x):
        return self.fz.ppf(x, mu=self.mu)

    def pmf(self, x):
        return self.fz.pmf(x, mu=self.mu)


class NegativeBinomial(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Negative Binomial"
        self.needed_params = [
            partup("n", 3, "successes"),
            partup("p", 4, "probability"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.n: float
        self.p: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.n = self.params[0].value
            self.p = self.params[1].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.n = 3
            self.p = 0.5
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.nbinom

    def ppf(self, x):
        return self.fz.ppf(x, n=self.n, p=self.p)

    def pmf(self, x):
        return self.fz.pmf(x, n=self.n, p=self.p)


class Geometric(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Geometric"
        self.needed_params = [
            partup("p", 3, "probability"),
            #             partup("", 5, ""),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.p: float
        self.__: float
        self.___: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.p = self.params[0].value
            #             self.___ = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.p = 0.4
        #                 self.___ = 7
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.geom

    def ppf(self, x):
        return self.fz.ppf(x, p=self.p)

    def pmf(self, x):
        return self.fz.pmf(x, p=self.p)


class Hypergeometric(DiscreteDist):
    def __init__(self, owner, varname=""):
        self.type = "Hypergeometric"
        self.needed_params = [
            partup("N", 3, "population"),
            partup("k", 4, "successes"),
            partup("n", 5, "samples"),
            #             partup("", 6, ""),
        ]
        super().__init__(owner, varname)

        self.N: float
        self.k: float
        self.n: float
        self.____: float

        if all(isinstance(p, NumericalParameter) for p in self.params):
            self.N = self.params[0].value + self.params[1].value
            self.k = self.params[0].value
            self.n = self.params[2].value
            #             self.____ = self.params[3].value

            self.numerical = True
        else:
            self.N = 25
            self.k = 5
            self.n = 15
        #                 self.____ = 9

        self.fz: sps.rv_discrete = sps.hypergeom

    def ppf(self, x):
        return self.fz.ppf(x, M=self.N, n=self.k, N=self.n)

    def pmf(self, x):
        return self.fz.pmf(x, M=self.N, n=self.k, N=self.n)


dist_dict: dict = {
    "UNKNOWN": UnknownDist,
    "exponential_rv": Exponential,
    "normal_rv": Normal,
    "gamma_rv": Gamma,
    "weibull_rv": Weibull,
    "uniform_rv": Uniform,
    "truncated_normal_rv": TruncatedNormal,
    "beta_rv": Beta,
    "kumaraswamy_rv": Kumaraswamy,
    "laplace_rv": Laplace,
    "studentt_rv": StudentT,
    "cauchy_rv": Cauchy,
    "halfcauchy_rv": HalfCauchy,
    "lognormal_rv": LogNormal,
    "chisquare_rv": ChiSquared,
    "halfnormal_rv": HalfNormal,
    "wald_rv": Wald,
    "pareto_rv": Pareto,
    "invgamma_rv": InverseGamma,
    "exgaussian_rv": ExGaussian,
    "vonmises_rv": VonMises,
    "skewnormal_rv": SkewNormal,
    "triangular_rv": Triangular,
    "halfstudentt_rv": HalfStudent,
    "gumbel_rv": Gumbel,
    "logistic_rv": Logistic,
    "logit_normal_rv": LogitNormal,
    "rice_rv": Rice,
    "moyal_rv": Moyal,
    "asymmetriclaplace_rv": AsymmetricLaplace,
    "binomial_rv": Binomial,
    "beta_binomial_rv": BetaBinomial,
    "bernoulli_rv": Bernoulli,
    "poisson_rv": Poisson,
    "nbinom_rv": NegativeBinomial,
    "discrete_uniform_rv": DiscreteUniform,
    "geometric_rv": Geometric,
    "hypergeometric_rv": Hypergeometric,
}


def get_rv_dist(node: RandomVar) -> Dist:
    owner = node.owner
    dist_type = str(owner).split("{")[0]

    # Get the distribution type from the dictionary thereof:
    try:
        return dist_dict[dist_type](owner, node.name)
    except KeyError:
        print(f"PyKrusch doesn't recognize {dist_type}")
        unknown_dist = dist_dict["UNKNOWN"](owner, node.name)
        unknown_dist.type = f"{dist_type}"
        if unknown_dist.type.endswith("_rv"):
            unknown_dist.type = unknown_dist.type.rpartition("_rv")[0]
        return unknown_dist
