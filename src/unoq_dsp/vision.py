"""Image APIs: DSP neural inference, explicit host image/box processing."""
import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time

import numpy as np

from . import CompiledModel, Session, compile_model, load
from .errors import UnsupportedModel


def resolve(model):
    if isinstance(model, CompiledModel):
        return model
    return load(model) if Path(model).suffix == ".uqmodel" else compile_model(model)


@lru_cache(maxsize=64)
def image_lookup(dtype, mean, std, scale, zero):
    # Image channels have a finite byte domain too: prepare normalization and
    # quantization once, instead of allocating several float images per frame.
    values=(np.arange(256,dtype="f4")[:,None]-np.asarray(mean,dtype="f4"))/np.asarray(std,dtype="f4")/np.float32(scale)
    values=np.where(values>=0,np.floor(values+.5),np.ceil(values-.5))+np.float32(zero)
    bounds=np.iinfo(dtype)
    result=np.broadcast_to(np.clip(values,bounds.min,bounds.max),(256,3)).astype(dtype)
    result.flags.writeable=False
    return result


def prepare_image(image, spec, normalization):
    from PIL import Image
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an H×W×3 uint8 RGB image")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Pass a Pillow image or uint8 RGB array")
    shape = tuple(spec["shape"])
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise UnsupportedModel("Image API requires batch-one NHWC RGB input")
    mean = np.asarray(normalization["mean"], dtype="f4")
    std = np.asarray(normalization["std"], dtype="f4")
    if mean.size not in (1,3) or std.size not in (1,3) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std<=0):
        raise ValueError("Expected finite RGB normalization with positive standard deviations")
    pixels = np.asarray(image.convert("RGB").resize((shape[2], shape[1]), Image.Resampling.BILINEAR), dtype="u1")
    q = spec["quantization"]
    table=image_lookup(np.dtype(spec["dtype"]).str,tuple(float(v) for v in mean.reshape(-1)),
                       tuple(float(v) for v in std.reshape(-1)),q["scale"],q["zero_point"])
    return table[pixels,np.arange(3)][None],image.size


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    score: float
    # Normalized x_min, y_min, x_max, y_max in the original image.
    box: tuple


class SSDDecoder:
    def __init__(self, task):
        if task.get("kind") != "ssd_fixed_anchors" or task.get("schema") != 1:
            raise UnsupportedModel("No supported fixed-anchor detector task metadata")
        self.task = task
        self.count, self.classes = task["num_boxes"], task["num_classes"]
        if type(self.count) is not int or type(self.classes) is not int or not 1<=self.count<=100000 or not 1<=self.classes<=4096:
            raise UnsupportedModel("Invalid detector dimensions")
        encoded = task["anchors_xywh_f32_base64"]
        if not isinstance(encoded,str) or len(encoded)>((self.count*16+2)//3)*4:
            raise UnsupportedModel("Invalid anchor payload length")
        raw = base64.b64decode(encoded,validate=True)
        if len(raw)!=self.count*16:
            raise UnsupportedModel("Invalid anchor count")
        self.anchors=np.frombuffer(raw,dtype="<f4").reshape(self.count,4)
        self.scales=np.asarray(task["scales_yxhw"],dtype="f4")
        if self.scales.shape!=(4,) or not np.all(np.isfinite(self.scales)) or np.any(self.scales<=0) or not np.all(np.isfinite(self.anchors)) or np.any(self.anchors[:,2:]<=0):
            raise UnsupportedModel("Invalid detector anchors or scales")
        self.labels=task.get("labels",[])
        if self.labels and len(self.labels)!=self.classes:
            raise UnsupportedModel("Label count does not match detector classes")

    def decode(self, outputs, *, score_threshold=.5, iou_threshold=.5, max_results=20, class_aware=False):
        if not 0<=score_threshold<=1 or not 0<=iou_threshold<=1 or type(max_results) is not int or not 1<=max_results<=1000:
            raise ValueError("Invalid confidence, IoU or result limit")
        raw=np.asarray(outputs[self.task["boxes_output"]])
        scores=np.asarray(outputs[self.task["scores_output"]])
        if raw.shape!=(1,self.count,4) or scores.shape!=(1,self.count,self.classes):
            raise ValueError("Detector outputs do not match metadata")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scores)):
            raise ValueError("Nonfinite detector output")
        raw,scores=raw[0],scores[0]
        classes=np.argmax(scores,axis=1)
        confidence=scores[np.arange(self.count),classes]
        if self.task["sigmoid_scores"]:
            confidence=1/(1+np.exp(-np.clip(confidence,-80,80)))
        selected=np.flatnonzero(confidence>=score_threshold)
        if not selected.size:
            return []
        anchors=self.anchors[selected]
        values=raw[selected]/self.scales
        cy=values[:,0]*anchors[:,3]+anchors[:,1]
        cx=values[:,1]*anchors[:,2]+anchors[:,0]
        if self.task["exponential_size"]:
            with np.errstate(over="raise",invalid="raise"):
                try: sizes=np.exp(values[:,2:])
                except FloatingPointError as exc:raise ValueError("Nonfinite decoded box dimensions") from exc
        else:sizes=values[:,2:]
        height,width=sizes[:,0]*anchors[:,3],sizes[:,1]*anchors[:,2]
        boxes=np.stack((cx-width/2,cy-height/2,cx+width/2,cy+height/2),axis=1)
        valid=np.flatnonzero((height>0)&(width>0)&np.all(np.isfinite(boxes),axis=1))
        order=valid[np.argsort(-confidence[selected[valid]],kind="stable")]
        areas=width*height
        keep=[]
        while len(order) and len(keep)<max_results:
            best=int(order[0]);keep.append(best);rest=order[1:]
            lower=np.maximum(boxes[best,:2],boxes[rest,:2]);upper=np.minimum(boxes[best,2:],boxes[rest,2:])
            intersection=np.prod(np.maximum(upper-lower,0),axis=1)
            union=areas[best]+areas[rest]-intersection
            overlap=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
            suppress=overlap>iou_threshold
            if class_aware:suppress &= classes[selected[rest]]==classes[selected[best]]
            order=rest[~suppress]
        result=[]
        for i in keep:
            source=selected[i];category=int(classes[source]);box=np.clip(boxes[i],0,1)
            if box[2]<=box[0] or box[3]<=box[1]:continue
            result.append(Detection(category,self.labels[category] if self.labels else str(category),float(confidence[source]),tuple(float(v) for v in box)))
        return result


class ObjectDetector:
    """A model-driven detector; image normalization/anchors come from metadata.

    Bundled profiles select stretch or letterbox resizing. Neural operators
    execute on DSP. Image preparation and box decoding execute on Linux;
    fixed-anchor SSD uses NMS, while the one-to-one YOLO head uses top-k.
    """
    def __init__(self, model, *, library=None, pipeline="rpc"):
        model=resolve(model)
        task=model.metadata.get('task',{})
        self.profile=task.get('preprocessing')
        if task.get('kind')=='yolo_one_to_one':
            from .modern import YOLODecoder
            self.decoder=YOLODecoder(task,model.metadata.get('outputs',[]),model.metadata['input']['shape'])
        else:self.decoder=SSDDecoder(task)
        self.session=Session(model,library=library,pipeline=pipeline)
        self.last_timings={}

    def detect(self,image,**options):
        start=time.perf_counter_ns()
        if self.profile:
            from .modern import prepare_profile
            x,geometry=prepare_profile(image,self.session.metadata['input'],self.profile)
            options=dict(options,geometry=geometry)
        else:x,_=prepare_image(image,self.session.metadata["input"],self.decoder.task["normalization"])
        prepared=time.perf_counter_ns()
        outputs=self.session.run(x)
        inferred=time.perf_counter_ns()
        result=self.decoder.decode(outputs,**options)
        end=time.perf_counter_ns()
        self.last_timings={"preprocess_ms":(prepared-start)/1e6,"inference_and_transfer_ms":(inferred-prepared)/1e6,
                           "postprocess_ms":(end-inferred)/1e6,"total_ms":(end-start)/1e6}
        return result

    def close(self):self.session.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class ImageClassifier:
    """Explicit normalization when classification metadata is absent; no guessing."""
    def __init__(self,model,*,mean=None,std=None,labels=(),library=None,pipeline="rpc"):
        model=resolve(model)
        task=model.metadata.get('task',{})
        self.profile=task.get('preprocessing') if task.get('kind')=='image_classification' else None
        if self.profile:
            if mean is not None or std is not None:raise ValueError('Bundled preprocessing is explicit; do not override normalization')
            if not labels:labels=task.get('labels',())
        elif mean is None or std is None:raise ValueError('Provide mean/std or a bundle with classification preprocessing')
        self.softmax=bool(task.get('apply_softmax',False))
        outputs=model.metadata.get("outputs",[model.metadata["output"]])
        shape=outputs[0]["shape"]
        if len(outputs)!=1 or len(shape)<2 or shape[0]!=1 or int(np.prod(shape))!=shape[-1]:
            raise UnsupportedModel("Classifier requires one class-vector output")
        if labels and len(labels)!=shape[-1]:raise ValueError("Label count does not match model output")
        self.labels=list(labels);self.normalization={"mean":mean,"std":std}
        self.output_spec=outputs[0]
        self.session=Session(model,library=library,pipeline=pipeline)
        self.last_timings={}

    def classify(self,image,*,top_k=5):
        if type(top_k) is not int or top_k<1:raise ValueError("top_k must be positive")
        start=time.perf_counter_ns()
        if self.profile:
            from .modern import prepare_profile
            x,_=prepare_profile(image,self.session.metadata['input'],self.profile)
        else:x,_=prepare_image(image,self.session.metadata["input"],self.normalization)
        prepared=time.perf_counter_ns()
        result=self.session.run(x).reshape(-1)
        inferred=time.perf_counter_ns()
        if self.output_spec.get("dequantize"):
            scores=result
        else:
            q=self.output_spec["quantization"]
            scores=(result.astype("f4")-q["zero_point"])*q["scale"]
        if self.softmax:
            scores=np.exp(scores-scores.max());scores/=scores.sum()
        end=time.perf_counter_ns()
        self.last_timings={'preprocess_ms':(prepared-start)/1e6,'inference_and_transfer_ms':(inferred-prepared)/1e6,'postprocess_ms':(end-inferred)/1e6,'total_ms':(end-start)/1e6}
        return [{"class_id":int(i),"label":self.labels[i] if self.labels else str(i),"score":float(scores[i])}
                for i in np.argsort(-scores,kind="stable")[:top_k]]

    def close(self):self.session.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
