import copy
from math import sqrt

import dace
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.symbolic import symstr

from daceml.onnx.implementation_repository import register_fpga_expansion

@register_fpga_expansion("Add")
def expansion(node, state, sdfg):
    print("add expanded FPGA")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def addop(A: atype, B: btype, C: ctype):
        C[:] = A + B

    return addop.to_sdfg()