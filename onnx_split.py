import onnx
import onnx_graphsurgeon

def tensor_copy(tensor):
    if isinstance(tensor, onnx_graphsurgeon.Variable):
        return onnx_graphsurgeon.Variable(tensor.name + "__copy", tensor.dtype, tensor.shape)
    return onnx_graphsurgeon.Constant(tensor.name + "__copy", tensor._values, tensor.data_location)

def main():
    model = onnx.load("resnet50-quant-optimized.onnx")
    graph = onnx_graphsurgeon.import_onnx(model)
    cur_id = 0
    for k, var in graph.tensors().items():
        var.name = var.name + f"__{cur_id}"
        cur_id += 1
    graph.toposort()
    for node in graph.nodes:

        if node.op == "Relu":
            if len(node.outputs[0].outputs) != 1:
                continue

            if node.o().op != "QuantizeLinear":
                continue
            q_node:onnx_graphsurgeon.Node = node.o()
            dq_node = q_node.o()
            if len(dq_node.outputs[0].outputs) < 2:
                continue
            o = dq_node.outputs[0]
            consumer = dq_node.o()


            new_q_out = tensor_copy(q_node.outputs[0])
            graph.layer([q_node.inputs[0], q_node.inputs[1], q_node.inputs[2]], [new_q_out], op=q_node.op, name=q_node.name + "__copy", attrs=q_node.attrs)
            new_dq_out = tensor_copy(dq_node.outputs[0])
            graph.layer([new_q_out, dq_node.inputs[1], dq_node.inputs[2]], [new_dq_out], op=dq_node.op, name=dq_node.name + "__copy", attrs=dq_node.attrs)
            
            


            new_dq_out.outputs = [o.outputs[1]]
            o.outputs = [o.outputs[0]]
            consumer.inputs = [consumer.inputs[1], consumer.inputs[0]]
            # consumer.inputs=[o.outputs[0], con_other_input]
    # graph.cleanup(True, True, True)
    model = onnx_graphsurgeon.export_onnx(graph)
    onnx.save(model, "resnet50-quant-cleaned.onnx")

if __name__ == "__main__":
    main()
