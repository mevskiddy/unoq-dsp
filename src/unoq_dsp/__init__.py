"""Compile once; execute explicitly on the UNO Q DSP, without CPU fallback."""
import hashlib
import struct
from pathlib import Path

from .bundle import CompiledModel, load
from .errors import DSPError, InvalidBundle, UnsupportedModel
from .transport import HardwareSession

__version__ = "0.1.0rc1"


def compile_model(path, *, dsp_threads=2, reuse_memory=True):
    from .tflite import lower
    try:
        plan, metadata = lower(path, dsp_threads=dsp_threads, reuse_memory=reuse_memory)
    except (ValueError,TypeError,IndexError,AttributeError,struct.error) as exc:
        if isinstance(exc,UnsupportedModel):raise
        raise UnsupportedModel(f"Invalid or unsupported TFLite graph: {exc}") from exc
    metadata.update(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    source_format="tflite", compiler_version=__version__)
    from .metadata import task_metadata
    task = task_metadata(path)
    if task is not None:
        metadata['task'] = task
    return CompiledModel(plan, metadata)


class Session(HardwareSession):
    def __init__(self, model, *, library=None, pipeline="rpc"):
        if not isinstance(model, CompiledModel):
            path = Path(model)
            model = load(path) if path.suffix == ".uqmodel" else compile_model(path)
        self._open_plan(model.plan, model.metadata, library, pipeline)


__all__ = ["compile_model", "load", "CompiledModel", "Session", "DSPError", "InvalidBundle", "UnsupportedModel"]

from .vision import ImageClassifier, ObjectDetector
__all__ += ["ImageClassifier", "ObjectDetector"]
