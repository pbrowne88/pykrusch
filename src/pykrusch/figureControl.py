from os import mkdir, remove, rmdir
from pathlib import PosixPath
from pykrusch.config import *


class MathImage:
    def __init__(
        self,
        filepath: PosixPath,
        portnum: int,
        edges_from=[],
        op_id=None,
    ):
        self.filepath = filepath
        self.portnum = portnum
        self.edges_from: list = edges_from
        self.op_id = op_id


class MathImageFound(MathImage):
    def __init__(
        self,
        filepath: PosixPath,
        portnum: int,
        edges_from=[],
        op_id=None,
    ):
        self.filepath = filepath
        self.portnum = portnum
        self.edges_from: list = edges_from
        self.op_id = op_id


class FC:
    fignum = 0
    temppath = PosixPath(TEMP_IMG_DIR)

    symbol_dict = {}
    other_imgs = []

    def fig_init(fc):
        try:
            mkdir(fc.temppath)
        except FileExistsError:
            pass

    def fig_path(
        fc,
        edges_from=[],
        symbols="",
        op_id=None,
    ) -> MathImage | MathImageFound:
        fc.fignum += 1

        portnum = int(fc.fignum)

        if symbols and symbols in fc.symbol_dict:
            filepath = fc.symbol_dict[symbols]
            return MathImageFound(
                filepath=filepath, portnum=portnum, edges_from=edges_from, op_id=op_id
            )

        filepath = fc.temppath / f"{fc.fignum}.png"

        if symbols:
            fc.symbol_dict[symbols] = filepath
        else:
            fc.other_imgs.append(filepath)

        return MathImage(
            filepath=filepath, portnum=portnum, edges_from=edges_from, op_id=op_id
        )

    def fig_clean(fc):
        for f in fc.symbol_dict.values():
            try:
                remove(f)
            except FileNotFoundError:
                pass

        for f in fc.other_imgs:
            try:
                remove(f)
            except FileNotFoundError:
                pass

        try:
            rmdir(fc.temppath)
        except FileNotFoundError:
            pass
