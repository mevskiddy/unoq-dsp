"""Small independently executed TFLite fixtures, with no TensorFlow dependency."""
import flatbuffers
import numpy as np
import tflite


def single_op(builtin, option_type, options, tensors, op_inputs, op_output, version=2):
    """Tensors: (shape, type, data-or-None, scale-or-None, zero). Input is #0."""
    b = flatbuffers.Builder(4096)
    def table(name, fields):
        getattr(tflite, name+"Start")(b)
        for key, value in fields.items():
            getattr(tflite, name+"Add"+key)(b, value)
        return getattr(tflite, name+"End")(b)
    def vec(values, dtype):
        return b.CreateNumpyVector(np.asarray(values, dtype=dtype))
    def offsets(name, values):
        getattr(tflite, name)(b, len(values))
        for value in reversed(values):
            b.PrependUOffsetTRelative(value)
        return b.EndVector()
    buffers = [table("Buffer", {})]
    ts = []
    for i, (shape, typ, values, scale, zero) in enumerate(tensors):
        buffer = 0
        if values is not None:
            raw = b.CreateByteVector(values.tobytes())
            buffer = len(buffers)
            buffers.append(table("Buffer", {"Data": raw}))
        fields = {"Name": b.CreateString(f"t{i}"), "Shape": vec(shape, "i4"), "Type": typ, "Buffer": buffer}
        if scale is not None:
            sv, zv = vec([scale], "f4"), vec([zero], "i8")
            fields["Quantization"] = table("QuantizationParameters", {"Scale": sv, "ZeroPoint": zv})
        ts.append(table("Tensor", fields))
    opt = table(option_type, options) if option_type else None
    iv, ov = vec(op_inputs, "i4"), vec([op_output], "i4")
    fields = {"Inputs": iv, "Outputs": ov}
    if option_type:fields.update(BuiltinOptionsType=getattr(tflite.BuiltinOptions, option_type), BuiltinOptions=opt)
    op = table("Operator", fields)
    tv = offsets("SubGraphStartTensorsVector", ts)
    ops = offsets("SubGraphStartOperatorsVector", [op])
    iv, ov = vec([0], "i4"), vec([op_output], "i4")
    graph = table("SubGraph", {"Tensors": tv, "Operators": ops, "Inputs": iv, "Outputs": ov})
    code = table("OperatorCode", {"BuiltinCode": builtin, "DeprecatedBuiltinCode": min(builtin, 127), "Version": version})
    gv = offsets("ModelStartSubgraphsVector", [graph])
    cv = offsets("ModelStartOperatorCodesVector", [code])
    bv = offsets("ModelStartBuffersVector", buffers)
    model = table("Model", {"Version": 3, "Subgraphs": gv, "OperatorCodes": cv, "Buffers": bv})
    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())
