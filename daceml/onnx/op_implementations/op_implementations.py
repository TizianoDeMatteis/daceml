import copy
from math import sqrt

import dace
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.symbolic import symstr
import numpy as np
from daceml.onnx.implementation_repository import register_pure_expansion


@register_pure_expansion("Add")
def expansion(node, state, sdfg):
    print("add expanded")
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


@register_pure_expansion("Relu")
def expansion(node, state, sdfg):
    print("relu expanded: note 1d implementation")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    xtype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    ytype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    cast_lambda = "lambda x: max(x, dace.{}(0))".format(
        xtype.dtype.to_string())

    @dace.program
    def relu(X: xtype, Y: ytype):
        Y[:] = dace.elementwise(cast_lambda, X)

    return relu.to_sdfg()


@register_pure_expansion("MatMul")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    input0_dim = len(in_edges[0].data.subset.size())
    input1_dim = len(in_edges[1].data.subset.size())


    if input0_dim == 1 and input1_dim == 2:
        # emulate np matmul
        sdfg_exp = dace.SDFG('matmulExpansion')
        nn = in_edges[0].data.subset.size()[0]
        mm = in_edges[1].data.subset.size()[1]

        N = str(nn)
        M = str(mm)
        sdfg_exp.add_array('A',
                           shape=[nn],
                           dtype=sdfg.arrays[in_edges[0].data.data].dtype)
        sdfg_exp.add_array('B',
                           shape=[nn, mm],
                           dtype=sdfg.arrays[in_edges[1].data.data].dtype)
        sdfg_exp.add_array('Y',
                           shape=[mm],
                           dtype=sdfg.arrays[out_edges[0].data.data].dtype)

        init_state = sdfg_exp.add_state()
        init_state.add_mapped_tasklet('_matmul_init',
                                      dict(i='0:{}'.format(M)), {},
                                      'out = 0',
                                      {'out': dace.Memlet.simple("Y", "i")},
                                      external_edges=True)
        state_exp = sdfg_exp.add_state_after(init_state)

        state_exp.add_mapped_tasklet(
            '_matmul_',
            dict(i='0:{}'.format(mm), j='0:{}'.format(nn)), {
                '_a': dace.Memlet.simple("A", 'j'),
                '_b': dace.Memlet.simple("B", 'j,i')
            },
            '_c = _a * _b',
            {'_c': dace.Memlet.simple("Y", 'i', wcr_str='lambda x, y: x + y')},
            external_edges=True)
        sdfg_exp.save('/tmp/matmul.sdfg')
        return sdfg_exp
