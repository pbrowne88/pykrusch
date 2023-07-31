from pykrusch.figureControl import FC, MathImageFound
from pykrusch.graphviz.latexText import (
    render_img_from_latex,
    string_to_latex,
)
from pykrusch.graphviz.pgvHTML import make_f_html
from pytensor.graph.basic import Variable
from pygraphviz import AGraph
from pykrusch.config import DATA_LETTER, NODE_COLOUR_GV
import itertools


class DeterministicVar:
    def __init__(self, node: Variable):
        self.name: str = str(node.name)
        self.expression = start_recursive_f_search(node)

        self.shape = "plaintext"
        self.color = NODE_COLOUR_GV
        self.style = "filled"

    def f_add_self_to_graph(self, graph: AGraph):
        img_list = self.self_img_equals()
        img_list += self.expression.math_img_list()
        self.expression.op_to_graph(graph, self.name)
        graph.add_node(
            self.name,
            shape=self.shape,
            color=self.color,
            style=self.style,
            label=make_f_html(img_list, graph, self),
        )

    def self_img_equals(self):
        self_sanitized_name = self.name.replace("_", "\_")
        latex_str = string_to_latex(self_sanitized_name)
        mi_self = FC.fig_path(FC, symbols=latex_str)
        if not isinstance(mi_self, MathImageFound):
            render_img_from_latex(latex_str, mi_self.filepath)
        latex_str_eq = string_to_latex(r"=")
        mi_equals = FC.fig_path(FC, symbols=latex_str_eq)
        if not isinstance(mi_equals, MathImageFound):
            render_img_from_latex(latex_str_eq, mi_equals.filepath)
        return [mi_self, mi_equals]


class MathOP:
    OP_ID = itertools.count()

    def __init__(self, apply_parents, single_arg=False):
        self.op = []
        self.arg1 = recursive_f_search(apply_parents[0])
        if single_arg:
            self.arg2 = None
        else:
            self.arg2 = recursive_f_search(apply_parents[1])
        self.pre = []
        self.post = []
        self.op_id = "math" + str(MathOP.OP_ID.__next__())
        self.print_arg2 = True

    def __repr__(self):
        if self.arg2:
            return str(f" [{self.arg1} {self.op} {self.arg2}] ")

        else:
            return str(f" [{self.op}{self.arg1}] ")

    def __str__(self):
        return self.__repr__()

    def op_to_graph(self, graph, f_node_name):
        self.arg1.op_to_graph(graph, f_node_name)
        if self.arg2:
            self.arg2.op_to_graph(graph, f_node_name)

    def math_img_list(self) -> list:
        img_list = []
        for i in self.pre:
            img_list.append(self.self_img(i))
        img_list += self.arg1.math_img_list()
        for ii in self.op:
            img_list.append(self.self_img(ii))
        if self.arg2 and self.print_arg2:
            img_list += self.arg2.math_img_list()
        for iii in self.post:
            img_list.append(self.self_img(iii))
        return img_list

    def self_img(self, symbols):
        latex_str = string_to_latex(symbols)
        mi = FC.fig_path(FC, symbols=latex_str)
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(latex_str, mi.filepath)
        return mi

    def latex_as_data(self):
        raise NotImplementedError

    def latex_as_subset(self):
        raise NotImplementedError

    def latex_as_variable(self):
        raise NotImplementedError


### Classes that inherit from MathOps
# region


class Add(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        self.op = ["+"]


class Subtract(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        self.op = ["-"]


class Multiply(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)


class Divide(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        self.pre = ["("]
        self.op = [")", "/", "("]
        self.post = [")"]


class Power(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)

        n = "x"

        try:
            n = self.arg2.value
            if n.is_integer():
                n = int(n)
            self.print_arg2 = False
            self.pre = ["("]
            self.post = [f")^{{{n}}}"]
            return
        except AttributeError:
            pass
        try:
            n = self.arg2.name
            self.pre = ["("]
            self.op = [")","^\wedge","("]
            self.post = [f")"]
            return
        except AttributeError:
            pass

        self.print_arg2 = False
        self.pre = ["("]
        self.post = [f")^{{{n}}}"]


class Exp(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.pre = [
            r"\mathrm{exp}",
            "(",
        ]
        self.post = [")"]


class SubSet(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        self.op = []

    # Even if the variable subset is named, we don't want
    # to create a node for it in the graph, so we only
    # ever pass the recursive function along to the first
    # node.
    def op_to_graph(self, graph, f_node_name):
        self.arg1.op_to_graph(graph, f_node_name)

    # Subset is going to have to be special
    def math_img_list(self) -> list:
        return [self.self_img()]

    def self_img(self):
        name1 = self.arg1.latex_as_variable()
        name2 = self.arg2.latex_as_subset()

        if r"_" in name1 and r"{" in name1 and r"}" in name1:
            latex_str = name1.replace(r"}", "") + rf"[{name2}]" + r"}"
        else:
            latex_str = name1 + rf"_{{[{name2}]}}"

        mi = FC.fig_path(
            FC, edges_from=[self.arg1.name], symbols=latex_str, op_id=self.arg1.op_id
        )
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(latex_str, mi.filepath)

        return mi


class Logarithm(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.pre = [
            "ln",
            "(",
        ]
        self.post = [")"]


class Logarithm2(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.pre = [
            "log_2",
            "(",
        ]
        self.post = [")"]


class Logarithm10(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.pre = [
            r"log_10",
            "(",
        ]
        self.post = [")"]


class Identity(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.op = []

    def math_img_list(self):
        return self.arg1.math_img_list()

    @property
    def value(self):
        return self.arg1.value


class Subtensor(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        subset = int(self.arg2.value)
        self.op = [rf"_[{subset}]"]
        #TODO: Consider changing how Subtensor MathOP handles arg2.
        self.arg2 = None


class Reciprocal(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.op = ["1/"]


class InverseLogit(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        self.pre = [r"\mathrm{invlogit}", "("]
        self.post = [")"]


class Dot(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents)
        self.op = ["⋅"]


class UnknownOp(MathOP):
    def __init__(self, apply_parents):
        super().__init__(apply_parents, single_arg=True)
        try:
            self.arg2 = recursive_f_search(apply_parents[1])
        except:
            pass
        self.pre = ["("]
        self.op = ["?"]
        self.post = [")"]


# TODO: Add all math operations
mathop_dict = {
    # Old Pytensor Syntax: retained for now, might no longer be useful. 
    "Elemwise{add,": {"op": Add},
    "Elemwise{mul,": {"op": Multiply},
    "Elemwise{true_div,": {"op": Divide},
    "Elemwise{pow,": {"op": Power},
    "Elemwise{exp,": {"op": Exp},
    "Elemwise{identity}": {"op": Identity},
    "InplaceDimShuffle": {"op": Identity},
    "dot": {"op": Dot},
    "Sum{": {"op": Identity},
    "Subtensor{": {"op": SubSet},
    "Elemwise{log10,": {"op": Logarithm10},
    "Elemwise{log2,": {"op": Logarithm2},
    "Elemwise{log,": {"op": Logarithm},
    "AdvancedSubtensor": {"op": SubSet},
    "Elemwise{sigmoid,": {"op": InverseLogit},

    # New Pytensor Syntax:

    "Add": {"op": Add},
    "Mul": {"op": Multiply},
    "Sub": {"op": Subtract},
    "True_div": {"op": Divide},
    "Int_div": {"op": Divide},
    "Reciprocal": {"op": Reciprocal},
    "Pow": {"op": Power},
    "Exp": {"op": Exp},
    "Identity": {"op": Identity},
    "InplaceDimShuffle": {"op": Identity},
    "dot": {"op": Dot},
    "Sum{": {"op": Identity},
    "Subtensor{": {"op": SubSet},
    "Log10": {"op": Logarithm10},
    "Log2": {"op": Logarithm2}, 
    "Log.": {"op": Logarithm}, # Period is there to differentiate from other log bases 
                               # It seems as though all OP names are followed by periods.
    "AdvancedSubtensor": {"op": SubSet},
    "Sigmoid": {"op": InverseLogit},
    "ExpandDims": {"op": Identity},


    "UNKNOWN": {"op": UnknownOp},
}


# endregion


### Other OPs that can result from a recursive f search:


class OtherOP:
    OP_ID = itertools.count()

    def __init__(self):
        self.name: str
        self.op_id = "other" + str(OtherOP.OP_ID.__next__())

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return self.__repr__()

    def op_to_graph(self, graph: AGraph, f_node_name):
        if graph.has_node(self.name):
            graph.add_edge(self.name, f_node_name, key=self.op_id)

    def self_img(self):
        raise NotImplementedError

    def math_img_list(self) -> list:
        return [self.self_img()]

    def latex_as_data(self):
        return string_to_latex(f"{DATA_LETTER}_{self.name}")

    # Replacing underscores with escaped underscores ensures
    # That no subsetting will happen in the string
    def latex_as_subset(self):
        return string_to_latex(self.name.replace("_", "\_"))

    def latex_as_variable(self):
        return string_to_latex(self.name)

    def latex_as_op(self):
        raise NotImplementedError


# region
class UnnamedDataOP(OtherOP):
    # This is a class attribute used to keep track of each instance of DataOp that has
    # been added to the graph; the effect will be purely cosmetic, but it will help differentiate
    # the various forms of data
    data_iterator = 1

    def __init__(self):
        super().__init__()
        self.name = f"{UnnamedDataOP.data_iterator}"
        UnnamedDataOP.data_iterator += 1

    def op_to_graph(self, graph, f_node_name):
        pass

    def self_img(self):
        mi = FC.fig_path(FC, symbols=self.latex_as_data(), op_id=self.op_id)
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(self.latex_as_data(), mi.filepath)
        return mi


class NamedDataOP(OtherOP):
    def __init__(self, name):
        super().__init__()
        self.name = f"{name}"

    def self_img(self):
        mi = FC.fig_path(
            FC,
            edges_from=[str(self.name)],
            symbols=self.latex_as_data(),
            op_id=self.op_id,
        )
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(self.latex_as_data(), mi.filepath)
        return mi


class NamedOP(OtherOP):
    def __init__(self, name):
        super().__init__()
        self.name = str(name)

    def self_img(self):
        mi = FC.fig_path(
            FC,
            edges_from=[str(self.name)],
            symbols=self.latex_as_variable(),
            op_id=self.op_id,
        )
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(self.latex_as_variable(), mi.filepath)
        return mi


class ScalarOP(OtherOP):
    def __init__(self, value):
        super().__init__()
        self.value = value
        if float(self.value).is_integer():
            self.name = str(int(self.value))
        else:
            self.name = str(self.value)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return self.__repr__()

    def op_to_graph(self, graph, f_node_name):
        pass

    def self_img(self):
        mi = FC.fig_path(
            FC,
            edges_from=[str(self.name)],
            symbols=self.latex_as_variable(),
            op_id=self.op_id,
        )
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(str(self.value), mi.filepath)
        return mi


# endregion


def recursive_f_search(node: Variable) -> MathOP | ScalarOP | OtherOP:
    # Get the node's number of parents; we'll need to know it later.
    lenparents = len(node.get_parents())

    # If the node has a name, it's either named data or a named random variable.
    # If it has no parents, it's almost certainly data.
    # Otherwise, it's a random variable.
    try:
        if node.name and lenparents == 0:
            return NamedDataOP(str(node.name))

        if node.name and lenparents >= 0:
            return NamedOP(str(node.name))

    except AttributeError:
        pass

    # If the node has no name and no parents, we can be certain that it's
    # unnamed data -- we merely need to decide if it's a scalar (in which
    # case it's almost certainly a parameter), or if it's unnamed data

    if lenparents == 0:
        try:
            return ScalarOP(float(node.eval()))
        except TypeError:
            return UnnamedDataOP()

    mathop_type = None

    for type in mathop_dict.keys():
        if str(node).startswith(type):
            mathop_type = mathop_dict[type]
            continue

    if not mathop_type:
        print(f"Pykrusch didn't recognize {str(node)} as a valid MathOP type")
        mathop_type = mathop_dict["UNKNOWN"]

    mathop_op = mathop_type["op"]

    expression = mathop_op(node.owner.get_parents())

    return expression


def start_recursive_f_search(node: Variable) -> MathOP | OtherOP:
    if str(node.owner).startswith("Elemwise{identity") or str(node.owner).startswith("Identity"):
        return Identity(node.owner.get_parents())
    else:
        raise Exception(
            f"""
            Pykrusch doesn't know how to handle deterministic
            variables that don't begin with an Identity operation.

            This node was named: {str(node.owner)}
            """
        )


def specify_f_nodes(treelist) -> dict[str, list]:
    return [DeterministicVar(node) for node in treelist]


def add_f_nodes(nodes: list[DeterministicVar], graph: AGraph):
    for fnode in nodes:
        fnode.f_add_self_to_graph(graph)
