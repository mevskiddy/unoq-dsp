"""Read the documented MediaPipe fixed-anchor detector metadata subset.

Schema sources: google-ai-edge/mediapipe, mediapipe/tasks/metadata/
metadata_schema.fbs (M001) and object_detector_metadata_schema.fbs (V001).
Unknown detector contracts are rejected; anchors/preprocessing are never guessed.
"""
import base64
from pathlib import Path
import struct
import zipfile

from flatbuffers.table import Table
from flatbuffers import number_types as N
import numpy as np
import tflite

from .errors import UnsupportedModel


def root(data, identifier):
    if len(data) < 8 or data[4:8] != identifier:
        raise UnsupportedModel("Unknown task metadata schema")
    return Table(data, struct.unpack_from("<I", data)[0])


def field(t, index):
    return t.Offset(4+2*index)


def sub(t, index):
    offset = field(t, index)
    if not offset:
        raise UnsupportedModel(f"Required metadata table {index} is absent")
    return Table(t.Bytes, t.Indirect(t.Pos+offset))


def tables(t, index, maximum=100000):
    offset = field(t, index)
    if not offset:
        return []
    count = t.VectorLen(offset)
    if count > maximum:
        raise UnsupportedModel("Metadata vector exceeds parser limit")
    return [Table(t.Bytes, t.Indirect(t.Vector(offset)+4*i)) for i in range(count)]


def scalar(t, index, dtype, default=0):
    offset = field(t, index)
    return t.Get(dtype, t.Pos+offset) if offset else default


def string(t, index):
    offset = field(t, index)
    value = t.String(t.Pos+offset) if offset else b""
    if len(value) > 4096:
        raise UnsupportedModel("Metadata string exceeds parser limit")
    return value.decode("utf8")


def floats(t, index):
    offset = field(t, index)
    if not offset or not 1 <= t.VectorLen(offset) <= 3:
        raise UnsupportedModel("Expected one or three normalization values")
    return [t.Get(N.Float32Flags, t.Vector(offset)+4*i) for i in range(t.VectorLen(offset))]


def task_metadata(path):
    """Return a supported task contract or None when no detector metadata exists."""
    with Path(path).open('rb') as f:data=f.read(16*1024*1024+1)
    if len(data)>16*1024*1024:raise UnsupportedModel('Metadata source exceeds model parser limit')
    model = tflite.Model.GetRootAsModel(data, 0)
    raw = None
    for i in range(model.MetadataLength()):
        metadata = model.Metadata(i)
        if metadata.Name() == b"TFLITE_METADATA":
            raw = model.Buffers(metadata.Buffer()).DataAsNumpy().tobytes()
    if raw is None:
        return None
    try:
        if len(raw) > 8*1024*1024:
            raise UnsupportedModel("Task metadata exceeds 8 MiB")
        graphs = tables(root(raw, b"M001"), 3, 1)
        if len(graphs) != 1:
            raise UnsupportedModel("Task metadata requires one subgraph")
        graph = graphs[0]
        custom = next((c for c in tables(graph, 9, 32) if string(c, 0) == "DETECTOR_METADATA"), None)
        if custom is None:
            return None
        offset = field(custom, 1)
        start, length = custom.Vector(offset), custom.VectorLen(offset)
        detector = root(bytes(custom.Bytes[start:start+length]), b"V001")
        decoding = sub(detector, 2)
        classes = scalar(decoding, 0, N.Int32Flags)
        count = scalar(decoding, 1, N.Int32Flags)
        if (not 1 <= classes <= 4096 or not 1 <= count <= 100000
                or scalar(decoding, 2, N.Int32Flags) != 4
                or scalar(decoding, 4, N.Int32Flags) != 0):
            raise UnsupportedModel("Unsupported detector dimensions or keypoints")
        entries = tables(sub(sub(detector, 1), 0), 0)
        if len(entries) != count:
            raise UnsupportedModel("Anchor count does not match detector outputs")
        anchors = np.array([[scalar(a, i, N.Float32Flags) for i in range(4)] for a in entries], dtype="<f4")
        if not np.all(np.isfinite(anchors)) or np.any(anchors[:, 2:] <= 0):
            raise UnsupportedModel("Invalid anchor dimensions")
        scales = [scalar(decoding, i, N.Float32Flags) for i in (7, 6, 9, 8)]
        if not all(np.isfinite(v) and v > 0 for v in scales):
            raise UnsupportedModel("Invalid detector decoding scales")
        inputs, outputs = tables(graph, 2, 1), tables(graph, 3, 32)
        if len(inputs) != 1 or len(outputs) != 2:
            raise UnsupportedModel("Raw SSD task metadata requires one input and two outputs")
        units = tables(inputs[0], 4, 8)
        if len(units) != 1 or scalar(units[0], 0, N.Uint8Flags) != 1:
            raise UnsupportedModel("Detector requires explicit normalization metadata")
        normalization = sub(units[0], 1)
        mean, std = floats(normalization, 0), floats(normalization, 1)
        if not all(np.isfinite(v) for v in mean+std) or any(v<=0 for v in std):
            raise UnsupportedModel("Invalid image normalization")
        output_names = [string(t, 0) for t in outputs]
        if sorted(output_names) != ["location", "score"]:
            raise UnsupportedModel("Unrecognized detector output metadata names")
        for output in outputs:
            if any(scalar(unit,0,N.Uint8Flags)!=3 for unit in tables(output,4,8)):
                raise UnsupportedModel("Detector output calibration/process units are not implemented")
        g = model.Subgraphs(0)
        names = {name: g.Tensors(g.Outputs(i)).Name().decode() for i, name in enumerate(output_names)}
        if tuple(g.Tensors(g.Outputs(output_names.index("score"))).ShapeAsNumpy()) != (1, count, classes):
            raise UnsupportedModel("Score shape disagrees with task metadata")
        if tuple(g.Tensors(g.Outputs(output_names.index("location"))).ShapeAsNumpy()) != (1, count, 4):
            raise UnsupportedModel("Box shape disagrees with task metadata")
        labels = []
        for associated in tables(outputs[output_names.index("score")], 6, 32):
            if scalar(associated, 2, N.Int8Flags) == 2 and not string(associated, 3):
                with zipfile.ZipFile(path) as archive:
                    info = archive.getinfo(string(associated, 0))
                    if info.file_size > 1024*1024:
                        raise UnsupportedModel("Label file exceeds 1 MiB")
                    labels = archive.read(info).decode("utf8").splitlines()
                break
        if labels and len(labels) != classes:
            raise UnsupportedModel("Label count disagrees with class count")
        return {"kind": "ssd_fixed_anchors", "schema": 1, "num_classes": classes, "num_boxes": count,
                "anchors_xywh_f32_base64": base64.b64encode(anchors.tobytes()).decode("ascii"),
                "scales_yxhw": scales, "exponential_size": bool(scalar(decoding, 10, N.BoolFlags)),
                "sigmoid_scores": bool(scalar(decoding, 11, N.BoolFlags)),
                "boxes_output": names["location"], "scores_output": names["score"],
                "normalization": {"mean": mean, "std": std}, "labels": labels,
                "postprocessing_device": "cpu", "neural_graph_device": "dsp"}
    except (IndexError, ValueError, TypeError, KeyError, struct.error, zipfile.BadZipFile) as exc:
        if isinstance(exc, UnsupportedModel):
            raise
        raise UnsupportedModel(f"Invalid detector metadata: {exc}") from exc
