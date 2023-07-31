from pytensor.graph.basic import Apply
from pykrusch.config import PARAM_CONVERSION, PARAM_FONT_SIZE, PARAM_FONT_SIZE_NUMERICAL
from pykrusch.figureControl import MathImage, FC, MathImageFound
from pykrusch.graphviz.latexText import render_img_from_latex, string_to_latex


class Parameter:
    def __init__(self):
        self.name: str
        self.meaning: str
        self.printout: str
        self.type: str

        # This gets supplied in dist.py after instantiation
        self.greek_name: str
        self.slot: int

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"({self.type}, {self.printout})"

    def give_greek(self, greek):
        if greek in PARAM_CONVERSION:
            self.greek_name = PARAM_CONVERSION[greek]
        else:
            self.greek_name = greek

    def give_slot(self, slot):
        self.slot = slot

    def give_symbol(self):
        return self.greek_name, PARAM_FONT_SIZE

    def give_meaning(self, meaning):
        self.meaning = meaning


class NumericalParameter(Parameter):
    def __init__(self, value):
        self.name = ""
        self.value = value
        self.printout = str(value)
        self.type = "NumericalParameter"

    def give_symbol(self):
        if float(self.value).is_integer():
            numerical_arg = int(float(self.value))
        else:
            numerical_arg = float(self.value)

        if self.greek_name:
            return self.greek_name + f" = {numerical_arg}", PARAM_FONT_SIZE_NUMERICAL
        else:
            return f"{numerical_arg}", PARAM_FONT_SIZE_NUMERICAL


class NamedParameter(Parameter):
    def __init__(self, name):
        self.name = name
        self.printout = str(name)
        self.type = "NamedParameter"


class NamedData(Parameter):
    def __init__(self, name):
        self.name = name
        self.printout = str(name)
        self.type = "NamedData"

        mi: MathImage = FC.fig_path(FC, symbols=self.name)
        if not isinstance(mi, MathImageFound):
            render_img_from_latex(string_to_latex(self.name), mi.filepath)
        self.mi = mi


class DataParameter(Parameter):
    def __init__(self):
        self.name = ""
        # TODO: THIS NEEDS TO BE CHANGED
        self.printout = "DATA_PARAMETER"
        self.type = "DataParameter"


class UnknownParameter(Parameter):
    def __init__(self):
        self.name = ""
        self.printout = "UNKNOWN_PARAMETER"
        self.type = "UnknownParameter"


class ComponentsParameter(Parameter):
    def __init__(self, components):
        self.components = components
        self.name = [c.name for c in self.components]
        self.type = [c.type for c in self.components]
        self.printout = str(self.name)
        self.flat_components: set()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"({self.type}, {self.printout})"

    def give_slot(self, slot):
        self.slot = slot
        for component in self.flat_components:
            component: Parameter
            component.give_slot(slot)

    def give_greek(self, greek):
        self.greek_name = greek
        if greek in PARAM_CONVERSION:
            self.greek_name = PARAM_CONVERSION[greek]
        else:
            self.greek_name = greek

    def give_meaning(self, meaning):
        self.meaning = meaning
        for component in self.flat_components:
            component: Parameter
            component.give_meaning(meaning)

    def flatten_components(self):
        out_set = set()
        for component in self.components:
            if isinstance(component, ComponentsParameter):
                out_set = out_set.union(component.flatten_components())
            else:
                out_set.add(component)

        self.flat_components = out_set

        self.name = [c.name for c in self.flat_components]
        self.type = [c.type for c in self.flat_components]
        self.printout = str(self.name)
        return out_set


def recursive_rv_search(node) -> Parameter:
    # First, attempt to retrieve the node name, if there is one, return it

    parents = node.get_parents()

    try:
        if node.name:
            if hasattr(node, "data") or len(parents) == 0:
                return NamedData(node.name)
            return NamedParameter(node.name)
    except AttributeError:
        pass

    # If we make it past the try block above, it means the node
    # has no name.
    # If an unnamed node has 0 parents, it's either a parameter or data.
    # If it's a parameter, we should be able to retrieve it using
    # `node.data.item()`. If that fails, it's unnamed data.
    # So, let's get the parents:

    # Does the node have 0 parents? If so, it's either
    # a scalar (usually a parameter in a distribution)
    # or it's data. If it's a scalar, get it and return it.
    # If it's data, we simply call it 'x' for now.
    ###
    # TODO: Consider creating a better representation for data
    # Some kind of class, perhaps?
    ###
    if len(parents) == 0:
        try:
            return NumericalParameter(node.data.item())
        except ValueError:
            return DataParameter()

    # If the node has one parent, we go deeper.
    if len(parents) == 1:
        return recursive_rv_search(parents[0])

    # If the node is a math operation, has multiple parents,
    # and all of the parents are scalars, we'll try evaluating
    # to a scalar:
    if len(parents) == 2:
        try:
            parents[0].data.item()
            parents[1].data.item()
            # If the node type is an 'Apply' pytensor node,
            # We want to evaluate the apply node's CHILD.
            if type(node) == Apply:
                return NumericalParameter(float(node.outputs[0].eval()))
            else:
                return NumericalParameter(float(node.eval()))
        except NameError:
            pass
        except ValueError:
            pass
        except AttributeError:
            pass

    # If the above doesn't work, we check to see if the node is a
    # subtensor, which means that it's probably going to split a named
    # node into more than one component.

    # TODO: Improve handling here: currently, subtensor handling
    # simply treats them as invisible.

    if len(parents) == 2 and str(node).startswith("Subtensor"):
        return recursive_rv_search(parents[0])

        pass

    # If the node has multiple parents, we switch to a list of parameters/inputs.
    # This is just for debugging purposes.

    comp_list = []
    for parent in parents:
        result = recursive_rv_search(parent)
        comp_list.append(result)

    comp = ComponentsParameter(comp_list)
    comp.flatten_components()

    print(comp.flat_components)

    return comp
