import copy
from math import sqrt
from math import erf

import dace
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.symbolic import symstr

from daceml.onnx.implementation_repository import register_pure_expansion
import numpy as np


@register_pure_expansion("Div")
def expansion(node, state, sdfg):
    print("div expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def divop(A: atype, B: btype, C: ctype):
        C[:] = A / B

    return divop.to_sdfg()

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

@register_pure_expansion("Mul")
def expansion(node, state, sdfg):
    print("mul expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def mulop(A: atype, B: btype, C: ctype):
        C[:] = A * B

    return mulop.to_sdfg()

@register_pure_expansion("Sub")
def expansion(node, state, sdfg):
    print("sub expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def subop(A: atype, B: btype, C: ctype):
        C[:] = A - B

    return subop.to_sdfg()


@register_pure_expansion("ReduceMean")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    axes = None
    keepdims = None
    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            if str(node.schema.attributes[name]) == "axes":
                axes = getattr(node, name)
            elif str(node.schema.attributes[name]) == "keepdims":
                keepdims = getattr(node, name)

    assert(axes == [-1] and keepdims ==1)

    node.validate(sdfg, state)
    sdfg_exp = dace.SDFG('reducemeanExpansion')

    in_edges = state.in_edges(node)

    mm = in_edges[0].data.subset.size()[0]
    nn = in_edges[0].data.subset.size()[1]
    gg = in_edges[0].data.subset.size()[2]

    M = str(mm)
    N = str(nn)
    G = str(gg)

    sdfg_exp.add_array('data', (mm, nn, gg), dace.float32)
    sdfg_exp.add_array('reduced', (mm, nn, 1), dace.float32)
    state_exp = sdfg_exp.add_state()

    me, mx = state_exp.add_map('outer_map', dict(i='0:' + M, j='0:' + N))

    data = state_exp.add_read('data')
    reduced = state_exp.add_access('reduced')

    redsum = state_exp.add_reduce('lambda a1, b1: a1 + b1', None, 0)
    tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)

    tmean = state_exp.add_tasklet('meantasklet', {'tsum'}, {'mean'},
                                  'mean = tsum / (%s)' % G)

    state_exp.add_edge(
        data, None, me, None,
        dace.Memlet.simple(data, '0:' + M + ', 0:' + N + ', 0:' + G))
    state_exp.add_edge(me, None, redsum, None,
                       dace.Memlet.simple(data, 'i, j, 0:' + G))
    state_exp.add_edge(redsum, None, tmp_sum, None,
                       dace.Memlet.simple(tmp_sum, '0'))
    state_exp.add_edge(tmp_sum, None, tmean, 'tsum',
                       dace.Memlet.simple(tmp_sum, '0'))
    state_exp.add_edge(tmean, 'mean', mx, None,
                       dace.Memlet.simple(reduced, 'i, j, 0'))
    state_exp.add_edge(
        mx, None, reduced, None,
        dace.Memlet.simple(reduced, '0:' + M + ', 0:' + N + ', 0'))
    sdfg_exp.fill_scope_connectors()

    return sdfg_exp



#Todo, not safe to copy directly
@register_pure_expansion("Cast")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def castop(input: atype, output: btype):
        #output[:] = int(input)
        output[:] = input

    return castop.to_sdfg()

#@register_pure_expansion("Cast")
#def expansion(node, state, sdfg):
#    node.validate(sdfg, state)
#
#    in_edges = state.in_edges(node)
#    out_edges = state.out_edges(node)
#
#    input_dim = len(in_edges[0].data.subset.size())
#    output_dim = len(out_edges[0].data.subset.size())
#
#    sdfg_exp = dace.SDFG('castExpansion')
#
#    assert(input_dim == 3 and output_dim == 3)
#
#    mm = in_edges[0].data.subset.size()[0]
#    nn = in_edges[0].data.subset.size()[1]
#    kk = in_edges[0].data.subset.size()[2]
#
#    M = str(mm)
#    N = str(nn)
#    K = str(kk)
#
#    sdfg_exp.add_array('input', (mm, nn, kk), dace.float32)
#    sdfg_exp.add_array('output', (mm, nn, kk), dace.int64)
#
#    state_exp = sdfg_exp.add_state()
#    input = state_exp.add_read('input')
#    output = state_exp.add_access('output')
#    me, mx = state_exp.add_map('outer_map',
#                               dict(i='0:' + M, j='0:' + N, k='0:' + K))
#
#    tcast = state_exp.add_tasklet('tcast', {'_a'}, {'_b'}, '_b = int(_a)')
#
#    state_exp.add_edge(
#        input, None, me, None,
#        dace.Memlet.simple(input, '0:' + M + ', 0:' + N + ', 0:' + K))
#    state_exp.add_edge(me, None, tcast, '_a',
#                       dace.Memlet.simple(input, 'i, j, k'))
#    state_exp.add_edge(tcast, '_b', mx, None,
#                       dace.Memlet.simple(output, 'i, j, k'))
#    state_exp.add_edge(
#        mx, None, output, None,
#        dace.Memlet.simple(output, '0:' + M + ', 0:' + N + ', 0:' + K))
#    sdfg_exp.fill_scope_connectors()
#    return sdfg_exp

@register_pure_expansion("Erf")
def expansion(node, state, sdfg):
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    input_dim = len(in_edges[0].data.subset.size())
    output_dim = len(out_edges[0].data.subset.size())

    sdfg_exp = dace.SDFG('erfExpansion')

    assert(input_dim == 3 and output_dim == 3)

    mm = in_edges[0].data.subset.size()[0]
    nn = in_edges[0].data.subset.size()[1]
    kk = in_edges[0].data.subset.size()[2]

    M = str(mm)
    N = str(nn)
    K = str(kk)

    sdfg_exp.add_array('input', (mm, nn, kk), dace.float32)
    sdfg_exp.add_array('output', (mm, nn, kk), dace.float32)

    state_exp = sdfg_exp.add_state()
    input = state_exp.add_read('input')
    output = state_exp.add_access('output')
    me, mx = state_exp.add_map('outer_map',
                               dict(i='0:' + M, j='0:' + N, k='0:' + K))

    terf = state_exp.add_tasklet('terf', {'_a'}, {'_b'}, '_b = erf(_a)')

    state_exp.add_edge(
        input, None, me, None,
        dace.Memlet.simple(input, '0:' + M + ', 0:' + N + ', 0:' + K))
    state_exp.add_edge(me, None, terf, '_a',
                       dace.Memlet.simple(input, 'i, j, k'))
    state_exp.add_edge(terf, '_b', mx, None,
                       dace.Memlet.simple(output, 'i, j, k'))
    state_exp.add_edge(
        mx, None, output, None,
        dace.Memlet.simple(output, '0:' + M + ', 0:' + N + ', 0:' + K))
    sdfg_exp.fill_scope_connectors()
    return sdfg_exp



@register_pure_expansion("Sqrt")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def sqrtop(X: atype, Y: btype):
        # Y[:] = X ** dace.float32(0.5)
        Y[:] = sqrt(X)

    return sqrtop.to_sdfg()

@register_pure_expansion("Pow")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def powop(X: atype, Y: btype, Z: ctype):
        Z[:] = X ** Y

    return powop.to_sdfg()

@register_pure_expansion("Softmax")
def expansion(node, state, sdfg):
    node.validate(sdfg, state)

    axis = None
    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            if str(node.schema.attributes[name]) == "axis":
                axis = getattr(node, name)
    assert(axis == 3)
    in_edges = state.in_edges(node)
    ii = in_edges[0].data.subset.size()[0]
    jj = in_edges[0].data.subset.size()[1]
    kk = in_edges[0].data.subset.size()[2]
    ll = in_edges[0].data.subset.size()[3]
    I = str(ii)
    J = str(jj)
    K = str(kk)
    L = str(ll)
    sdfg_exp = dace.SDFG('softmaxExpansion')
    sdfg_exp.add_array('input', (ii, jj, kk, ll), dace.float32)
    sdfg_exp.add_array('output', (ii, jj, kk, ll), dace.float32)
    state_exp = sdfg_exp.add_state()
    ome, omx = state_exp.add_map('outer_map',
                                 dict(i='0:' + I, j='0:' + J, k='0:' + K))
    ime, imx = state_exp.add_map('inner_map', dict(l='0:' + L))

    # tmp_max = dace.define_local([1], dtype=dace.float32)
    # tmp_sum = dace.define_local([1], dtype=dace.float32)
    tmp_max = state_exp.add_transient('tmp_max', (1, ), dace.float32)
    tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)
    tmp_out = state_exp.add_transient('tmp_out', (ii, jj, kk, ll),
                                      dace.float32)
    input = state_exp.add_read('input')
    output = state_exp.add_access('output')

    red1 = state_exp.add_reduce('lambda a1, b1: max(a1, b1)', None, 0)
    texp1 = state_exp.add_tasklet('tasklet1', {'a2', 'b2'}, {'c2'},
                                  'c2 = exp(a2-b2)')

    state_exp.add_edge(
        input, None, ome, None,
        dace.Memlet.simple(input,
                           '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))
    state_exp.add_edge(ome, None, red1, None,
                       dace.Memlet.simple(input, 'i, j, k, 0:' + L))
    state_exp.add_edge(red1, None, tmp_max, None,
                       dace.Memlet.simple(tmp_max, '0'))

    state_exp.add_edge(ome, None, ime, None,
                       dace.Memlet.simple(input, 'i, j, k, 0:' + L))
    state_exp.add_edge(tmp_max, None, ime, None,
                       dace.Memlet.simple(tmp_max, '0'))

    state_exp.add_edge(ime, None, texp1, "a2",
                       dace.Memlet.simple(input, 'i, j, k, l'))
    state_exp.add_edge(ime, None, texp1, "b2",
                       dace.Memlet.simple(tmp_max, '0'))
    state_exp.add_edge(texp1, "c2", imx, None,
                       dace.Memlet.simple(tmp_out, 'i, j, k, l'))
    state_exp.add_edge(imx, None, omx, None,
                       dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
    state_exp.add_edge(
        omx, None, tmp_out, None,
        dace.Memlet.simple(tmp_out,
                           '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))

    ome1, omx1 = state_exp.add_map('outer_map1',
                                   dict(i='0:' + I, j='0:' + J, k='0:' + K))
    ime1, imx1 = state_exp.add_map('inner_map1', dict(l='0:' + L))
    red2 = state_exp.add_reduce('lambda a3, b3: a3 + b3', None, 0)
    texp2 = state_exp.add_tasklet('tasklet2', {'a4', 'b4'}, {'c4'},
                                  'c4 = a4 / b4')

    state_exp.add_edge(
        tmp_out, None, ome1, None,
        dace.Memlet.simple(tmp_out,
                           '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))
    state_exp.add_edge(ome1, None, red2, None,
                       dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
    state_exp.add_edge(red2, None, tmp_sum, None,
                       dace.Memlet.simple(tmp_sum, '0'))

    state_exp.add_edge(ome1, None, ime1, None,
                       dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
    state_exp.add_edge(tmp_sum, None, ime1, None,
                       dace.Memlet.simple(tmp_sum, '0'))

    state_exp.add_edge(ime1, None, texp2, "a4",
                       dace.Memlet.simple(tmp_out, 'i, j, k, l'))
    state_exp.add_edge(ime1, None, texp2, "b4",
                       dace.Memlet.simple(tmp_sum, '0'))
    state_exp.add_edge(texp2, "c4", imx1, None,
                       dace.Memlet.simple(output, 'i, j, k, l'))
    state_exp.add_edge(imx1, None, omx1, None,
                       dace.Memlet.simple(output, 'i, j, k, 0:' + L))
    state_exp.add_edge(
        omx1, None, output, None,
        dace.Memlet.simple(output,
                           '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))

    sdfg_exp.fill_scope_connectors()

    return sdfg_exp



@register_pure_expansion("Reshape")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)
    
    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    input_dim = len(in_edges[0].data.subset.size())
    output_dim = len(out_edges[0].data.subset.size())
    sdfg_exp = dace.SDFG('ReshapeExpansion')

    if input_dim == 4 and output_dim == 3:
        ii = in_edges[0].data.subset.size()[0]
        jj = in_edges[0].data.subset.size()[1]
        kk = in_edges[0].data.subset.size()[2]
        ll = in_edges[0].data.subset.size()[3]

        rr = in_edges[1].data.subset.size()[0]
        
        I = str(ii)
        J = str(jj)
        K = str(kk)
        L = str(ll)

        R = str(rr)

        mm = out_edges[0].data.subset.size()[0]
        nn = out_edges[0].data.subset.size()[1]
        pp = out_edges[0].data.subset.size()[2]

        M = str(mm)
        N = str(nn)
        P = str(pp)

        sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
        sdfg_exp.add_array('shape', (rr,), dace.float32)
        sdfg_exp.add_array('reshaped', (mm, nn, pp), dace.float32)

        state_exp = sdfg_exp.add_state()

        task1 = state_exp.add_tasklet('iden', {'_a', '_dummy'}, {'_b'}, '_b = _a')

        data = state_exp.add_read('data')
        shape = state_exp.add_read('shape')
        reshaped = state_exp.add_access('reshaped')

        me1, mx1 = state_exp.add_map('map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
        state_exp.add_edge(data, None, me1, None, dace.Memlet.simple(data, '0:'+I+', 0:'+J+', 0:'+K+', 0:'+L))
        state_exp.add_edge(shape, None, me1, None, dace.Memlet.simple(shape, '0:'+R))
        state_exp.add_edge(me1, None, task1, '_a', dace.Memlet.simple(data, 'i, j, k, l'))
        state_exp.add_edge(me1, None, task1, '_dummy', dace.Memlet.simple(shape, '0'))
        state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(reshaped, 'int((i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)/({3}*{4})), int((i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)%({3}*{4})/{4}), (i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)%({3}*{4})%{4}'.format(J, K, L, N, P)))
        #state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(reshaped, 'i, j, k*{0}+l'.format(L)))
        state_exp.add_edge(mx1, None, reshaped, None, dace.Memlet.simple(reshaped, '0:'+M+', 0:'+N+', 0:'+P))

        sdfg_exp.fill_scope_connectors()

    elif input_dim == 3 and output_dim == 4:
        ii = in_edges[0].data.subset.size()[0]
        jj = in_edges[0].data.subset.size()[1]
        kk = in_edges[0].data.subset.size()[2]

        rr = in_edges[1].data.subset.size()[0]
        
        I = str(ii)
        J = str(jj)
        K = str(kk)

        R = str(rr)

        mm = out_edges[0].data.subset.size()[0]
        nn = out_edges[0].data.subset.size()[1]
        pp = out_edges[0].data.subset.size()[2]
        qq = out_edges[0].data.subset.size()[3]

        M = str(mm)
        N = str(nn)
        P = str(pp)
        Q = str(qq)

        sdfg_exp.add_array('data', (ii, jj, kk), dace.float32)
        sdfg_exp.add_array('shape', (rr,), dace.float32)
        sdfg_exp.add_array('reshaped', (mm, nn, pp, qq), dace.float32)

        state_exp = sdfg_exp.add_state()

        task1 = state_exp.add_tasklet('iden', {'_a', '_dummy'}, {'_b'}, '_b = _a')

        data = state_exp.add_read('data')
        shape = state_exp.add_read('shape')
        reshaped = state_exp.add_access('reshaped')

        me1, mx1 = state_exp.add_map('map1', dict(i='0:' + I, j='0:' + J, k='0:' + K))
        state_exp.add_edge(data, None, me1, None, dace.Memlet.simple(data, '0:'+I+', 0:'+J+', 0:'+K))
        state_exp.add_edge(shape, None, me1, None, dace.Memlet.simple(shape, '0:'+R))
        state_exp.add_edge(me1, None, task1, '_a', dace.Memlet.simple(data, 'i, j, k'))
        state_exp.add_edge(me1, None, task1, '_dummy', dace.Memlet.simple(shape, '0'))
        #state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(reshaped, 'i, j, int(k/{0}), k%{0}'.format(Q)))
        state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(reshaped, 'int((i*{0}*{1}+j*{1}+k)/({2}*{3}*{4})), int(((i*{0}*{1}+j*{1}+k)%({2}*{3}*{4}))/({3}*{4})), int(((i*{0}*{1}+j*{1}+k)%({2}*{3}*{4})%({3}*{4}))/{4}), (i*{0}*{1}+j*{1}+k)%({2}*{3}*{4})%({3}*{4})%{4}'.format(J, K, N, P, Q)))
        state_exp.add_edge(mx1, None, reshaped, None, dace.Memlet.simple(reshaped, '0:'+M+', 0:'+N+', 0:'+P+', 0:'+Q))

        sdfg_exp.fill_scope_connectors()
    return sdfg_exp


@register_pure_expansion("MatMul")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    input0_dim = len(in_edges[0].data.subset.size())
    input1_dim = len(in_edges[1].data.subset.size())

    if input0_dim == 4 and input1_dim == 4:

        sdfg_exp = dace.SDFG('matmulExpansion')
        mm = in_edges[0].data.subset.size()[0]
        nn = in_edges[0].data.subset.size()[1]
        ii = in_edges[0].data.subset.size()[2]
        kk = in_edges[0].data.subset.size()[3]
        jj = in_edges[1].data.subset.size()[3]

        M = str(mm)
        N = str(nn)
        I = str(ii)
        K = str(kk)
        J = str(jj)

        sdfg_exp.add_array('A', (mm, nn, ii, kk), dace.float32)
        sdfg_exp.add_array('B', (mm, nn, kk, jj), dace.float32)
        sdfg_exp.add_array('Y', (mm, nn, ii, jj), dace.float32)

        init_state = sdfg_exp.add_state()
        init_state.add_mapped_tasklet(
            'batched_matmul_init', {
                '_o%d' % i: '0:%s' % symstr(d)
                for i, d in enumerate((mm, nn, ii, jj))
            }, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    'Y', ','.join(
                        ['_o%d' % i for i in range(len((mm, nn, ii, jj)))]))
            },
            external_edges=True)

        state_exp = sdfg_exp.add_state_after(init_state)

        state_exp.add_mapped_tasklet(
            '_BatchedBatchedMatMult_',
            {'__i%d' % i: '0:%s' % s
             for i, s in enumerate([M, N, I, J, K])}, {
                 '_a': dace.Memlet.simple("A", ('__i0, __i1, __i2, __i4')),
                 '_b': dace.Memlet.simple("B", ('__i0, __i1, __i4, __i3'))
             },
            '_c = _a * _b', {
                '_c':
                dace.Memlet.simple("Y",
                                   '__i0, __i1, __i2, __i3',
                                   wcr_str='lambda x, y: x + y')
            },
            external_edges=True)
        return sdfg_exp
    elif input0_dim == 3 and input1_dim == 2:
        sdfg_exp = dace.SDFG('matmulExpansion')
        mm = in_edges[0].data.subset.size()[0]
        ii = in_edges[0].data.subset.size()[1]
        kk = in_edges[0].data.subset.size()[2]
        jj = in_edges[1].data.subset.size()[1]

        M = str(mm)
        I = str(ii)
        K = str(kk)
        J = str(jj)

        sdfg_exp.add_array('A', (mm, ii, kk), dace.float32)
        sdfg_exp.add_array('B', (kk, jj), dace.float32)
        sdfg_exp.add_array('Y', (mm, ii, jj), dace.float32)

        init_state = sdfg_exp.add_state()
        init_state.add_mapped_tasklet(
            'batched_matmul_init',
            {'_o%d' % i: '0:%s' % symstr(d)
             for i, d in enumerate((mm, ii, jj))}, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    'Y', ','.join(['_o%d' % i for i in range(len((mm, ii, jj)))]))
            },
            external_edges=True)

        state_exp = sdfg_exp.add_state_after(init_state)

        state_exp.add_mapped_tasklet(
            '_BatchedBatchedMatMult_',
            {'__i%d' % i: '0:%s' % s
             for i, s in enumerate([M, I, J, K])}, {
                 '_a': dace.Memlet.simple("A", ('__i0, __i1, __i3')),
                 '_b': dace.Memlet.simple("B", ('__i3, __i2'))
             },
            '_c = _a * _b', {
                '_c':
                dace.Memlet.simple(
                    "Y", '__i0, __i1, __i2', wcr_str='lambda x, y: x + y')
            },
            external_edges=True)
        return sdfg_exp
    else:
        print("Unsupported dimensions for MatMul")


@register_pure_expansion("Transpose")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)
    perm = None
    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            if str(node.schema.attributes[name]) == "perm":
                perm = getattr(node, name)
    
    in_edges = state.in_edges(node)
    input_dim = len(in_edges[0].data.subset.size())

    sdfg_exp = dace.SDFG('TransposeExpansion')

    assert(input_dim == 4)

    ii = in_edges[0].data.subset.size()[0]
    jj = in_edges[0].data.subset.size()[1]
    kk = in_edges[0].data.subset.size()[2]
    ll = in_edges[0].data.subset.size()[3]
    I = str(ii)
    J = str(jj)
    K = str(kk)
    L = str(ll)

    if perm == [0, 2, 1, 3]:
        sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
        sdfg_exp.add_array('transposed', (ii, kk, jj, ll), dace.float32)

        state_exp = sdfg_exp.add_state()

        task1 = state_exp.add_tasklet('iden', {'_a'}, {'_b'}, '_b = _a')

        data = state_exp.add_read('data')
        transposed = state_exp.add_access('transposed')

        me1, mx1 = state_exp.add_map('map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
        state_exp.add_edge(data, None, me1, None, dace.Memlet.simple(data, '0:'+I+', 0:'+J+', 0:'+K+', 0:'+L))
        state_exp.add_edge(me1, None, task1, '_a', dace.Memlet.simple(data, 'i, j, k, l'))
        state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(transposed, 'i, k, j, l'))
        state_exp.add_edge(mx1, None, transposed, None, dace.Memlet.simple(transposed, '0:'+I+', 0:'+K+', 0:'+J+', 0:'+L))
        sdfg_exp.fill_scope_connectors()

    elif perm == [0, 2, 3, 1]:
        sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
        sdfg_exp.add_array('transposed', (ii, kk, ll, jj), dace.float32)

        state_exp = sdfg_exp.add_state()

        task1 = state_exp.add_tasklet('iden', {'_a'}, {'_b'}, '_b = _a')

        data = state_exp.add_read('data')
        transposed = state_exp.add_access('transposed')

        me1, mx1 = state_exp.add_map('map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
        state_exp.add_edge(data, None, me1, None, dace.Memlet.simple(data, '0:'+I+', 0:'+J+', 0:'+K+', 0:'+L))
        state_exp.add_edge(me1, None, task1, '_a', dace.Memlet.simple(data, 'i, j, k, l'))
        state_exp.add_edge(task1, '_b', mx1, None, dace.Memlet.simple(transposed, 'i, k, l, j'))
        state_exp.add_edge(mx1, None, transposed, None, dace.Memlet.simple(transposed, '0:'+I+', 0:'+K+', 0:'+L+', 0:'+J))
        sdfg_exp.fill_scope_connectors()

    return sdfg_exp
