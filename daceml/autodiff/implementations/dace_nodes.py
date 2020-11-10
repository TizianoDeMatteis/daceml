import typing

from dace import SDFGState, SDFG, detect_reduction_type, Memlet
import dace.dtypes as dtypes
import dace.sdfg.nodes as nd
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node
import dace.libraries.standard.nodes

from daceml.autodiff.backward_pass_generator import AutoDiffException
from daceml.autodiff.backward_implementation_abc import BackwardImplementation
from daceml.util.utils import in_edge_with_name, in_desc_with_name, out_desc_with_name, out_edge_with_name


@autoregister_params(node_type=dace.libraries.standard.nodes.Reduce)
class ReverseReduce(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        reduction_type = detect_reduction_type(node.wcr)
        if reduction_type is not dtypes.ReductionType.Sum:
            return False

        return True

    @staticmethod
    def backward(
        forward_node: Node, forward_state: SDFGState, forward_sdfg: SDFG,
        backward_state: SDFGState, backward_sdfg: SDFG,
        given_gradients: typing.Dict[typing.Optional[str],
                                     typing.Optional[str]],
        required_gradients: typing.Dict[typing.Optional[str],
                                        typing.Optional[str]]
    ) -> typing.Union[Node, SDFG]:

        reduction_type = detect_reduction_type(forward_node.wcr)

        if len(given_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one output edge"
                .format(forward_node))

        if len(required_gradients) != 1:
            raise AutoDiffException(
                "recieved invalid SDFG: reduce node {} should have exactly one input edge"
                .format(forward_node))

        input_name = next(iter(required_gradients))
        in_edge = in_edge_with_name(forward_node, forward_state, input_name)
        in_desc = in_desc_with_name(forward_node, forward_state, forward_sdfg,
                                    input_name)

        output_name = next(iter(given_gradients))
        out_edge = out_edge_with_name(forward_node, forward_state, output_name)
        out_desc = out_desc_with_name(forward_node, forward_state,
                                      forward_sdfg, output_name)

        all_axes: typing.List[int] = list(range(len(in_desc.shape)))
        reduce_axes: typing.List[
            int] = all_axes if forward_node.axes is None else forward_node.axes
        non_reduce_axes: typing.List[int] = [
            i for i in all_axes if i not in reduce_axes
        ]

        if reduction_type is dtypes.ReductionType.Sum:
            # in this case, we need to simply scatter the grad across the axes that were reduced

            sdfg = SDFG("_reverse_" + str(reduction_type).replace(".", "_") +
                        "_")
            state = sdfg.add_state()

            rev_input_conn_name = given_gradients[output_name]
            rev_output_conn_name = required_gradients[input_name]

            _, rev_input_arr = sdfg.add_array(rev_input_conn_name,
                                              shape=out_desc.shape,
                                              dtype=out_desc.dtype)
            _, rev_output_arr = sdfg.add_array(rev_output_conn_name,
                                               shape=in_desc.shape,
                                               dtype=in_desc.dtype)

            state.add_mapped_tasklet(
                "_distribute_grad_" + str(reduction_type).replace(".", "_") +
                "_", {
                    "i" + str(i): "0:{}".format(shape)
                    for i, shape in enumerate(in_desc.shape)
                }, {
                    "__in":
                    Memlet.simple(
                        rev_input_conn_name,
                        "0" if forward_node.axes is None else ",".join(
                            "i" + str(i) for i in non_reduce_axes))
                },
                "__out = __in", {
                    "__out":
                    Memlet.simple(rev_output_conn_name, ",".join(
                        "i" + str(i) for i in all_axes))
                },
                external_edges=True)

            return backward_state.add_nested_sdfg(sdfg, None,
                                                  {rev_input_conn_name},
                                                  {rev_output_conn_name})
        else:
            raise AutoDiffException(
                "Unsupported reduction type '{}'".format(reduction_type))
