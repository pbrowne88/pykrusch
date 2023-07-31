from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING
from pygraphviz import AGraph
from pykrusch.config import *

if TYPE_CHECKING:
    from pykrusch.figureControl import MathImage
    from pykrusch.parsePymc.randomVariable import RandomVar
    from pykrusch.parsePymc.deterministicVariable import DeterministicVar
    from pykrusch.dist import Dist
    from pykrusch.param import NamedParameter, NamedData


def make_data_html(param: NamedData, graph):
    label_start = f"""<
    <TABLE BORDER="20" CELLBORDER="2" CELLSPACING="0" VALIGN="BOTTOM">
    <TR>
    """
    label_end = f"""</TR></TABLE>>"""

    label = """"""

    label += label_start

    image = Image.open(param.mi.filepath)
    tw = image.width * DETERMINISTIC_TEXT_SCALE
    th = image.height * DETERMINISTIC_TEXT_SCALE
    label += f"""<TD FIXEDSIZE="TRUE" WIDTH="{tw}" HEIGHT="{th}" PORT="{param.mi.portnum}"><IMG SCALE="BOTH" SRC="{param.mi.filepath}"></IMG></TD>"""

    label += label_end

    return label


def make_f_html(img_list: list[MathImage], graph: AGraph, f_node: DeterministicVar):
    label_start = f"""<
    <TABLE BORDER="20" CELLBORDER="2" CELLSPACING="0" VALIGN="BOTTOM">
    <TR>
    """
    label_end = f"""</TR></TABLE>>"""

    label = """"""

    label += label_start

    for img in img_list:
        image = Image.open(img.filepath)
        tw = image.width * DETERMINISTIC_TEXT_SCALE
        th = image.height * DETERMINISTIC_TEXT_SCALE
        label += f"""<TD FIXEDSIZE="TRUE" WIDTH="{tw}" HEIGHT="{th}" PORT="{img.portnum}"><IMG SCALE="BOTH" SRC="{img.filepath}"></IMG></TD>"""
        for edge_from in img.edges_from:
            if edge_from and graph.has_edge(edge_from, f_node.name, key=img.op_id):
                e = graph.get_edge(edge_from, f_node.name, key=img.op_id)
                e.attr["headport"] = str(img.portnum) + ":n"
                e.attr["tailport"] = "s"

    label += label_end

    return label


def make_rv_html(rv: RandomVar, graph: AGraph):
    dist: Dist = rv.dist
    tw = DIST_IMAGE_WIDTH * IMAGE_SCALE_FACTOR
    th = DIST_IMAGE_HEIGHT * IMAGE_SCALE_FACTOR

    param_slots = max(dist.num_params, 1)
    slot_width = tw // param_slots

    label_start = f"""<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">
                    <TR>"""

    for param in dist.params:
        param_symbol, font_size = param.give_symbol()

        label_start += f"""
        <TD WIDTH="{slot_width}" PORT="{str(param.slot)}">
            <TABLE CELLPADDING="0" BORDER="0" CELLSPACING="0">
                <TR>
                    <TD><FONT POINT-SIZE="{font_size}">{param_symbol}</FONT></TD>
                </TR>
                <TR>
                    <TD><FONT POINT-SIZE="{DIST_FONT_SIZE_SMALL}">{param.meaning}</FONT></TD>
                </TR>
            </TABLE>
        </TD>"""

    label_start += """</TR>"""

    distimage = dist.distimage

    if distimage:
        label_start += f"""
                <TR>
                    <TD COLSPAN="{param_slots}" WIDTH="{tw}" HEIGHT="{th}"><IMG SCALE="TRUE" SRC="{distimage}"></IMG></TD>
                </TR>"""

    image = Image.open(dist.dist_name_latex)
    tw = image.width * DETERMINISTIC_TEXT_SCALE
    th = image.height * DETERMINISTIC_TEXT_SCALE
    label_end = f"""
                <TR>
                    <TD COLSPAN="{param_slots}" ALIGN="center" HEIGHT="25"><FONT POINT-SIZE="{DIST_FONT_SIZE_SMALL}">{dist.type.upper()}</FONT></TD>
                    
                </TR>
                <TR>
                    <TD WIDTH="{tw}" HEIGHT="{th}" COLSPAN="{param_slots}" ALIGN="center"><IMG SCALE="TRUE" SRC="{dist.dist_name_latex}"></IMG></TD>    
                </TR>
                
                
            </TABLE>>"""

    label_start += label_end

    return label_start

    # <TR>
    #     <TD COLSPAN="{param_slots}" ALIGN="center"><FONT POINT-SIZE="{DIST_FONT_SIZE}">{rv.name}</FONT></TD>
    # </TR>
