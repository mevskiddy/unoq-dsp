"""Explicit preprocessing and one-to-one box decoding for compiled vision tasks.

Only image conversion and output interpretation run on the host. All learned
operators, including detector attention, remain in the compiled DSP graph.
"""
from functools import lru_cache
import numpy as np
from .errors import UnsupportedModel

@lru_cache(maxsize=32)
def profile_lookup(mean,std,scale,zero,dtype):
    # Preserve the export's float32 operation order and ties-to-even rounding
    # over the complete byte domain, then reuse the exact table for each image.
    values=(np.arange(256,dtype='f4')[:,None]/np.float32(255)-np.asarray(mean,dtype='f4'))/np.asarray(std,dtype='f4')
    values=np.rint(values/np.float32(scale))+zero
    bounds=np.iinfo(dtype);table=np.clip(values,bounds.min,bounds.max).astype(dtype)
    table.flags.writeable=False
    return table


def prepare_profile(image,spec,profile):
    from PIL import Image
    if isinstance(image,np.ndarray):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected H×W×3 uint8 RGB')
        image=Image.fromarray(image)
    if not isinstance(image,Image.Image):raise TypeError('Expected Pillow image or RGB array')
    if image.mode!='RGB':image=image.convert('RGB')
    width,height=image.size
    if not width or not height:raise ValueError('Image dimensions must be positive')
    shape=tuple(spec['shape'])
    if len(shape)!=4 or shape[0]!=1 or shape[3]!=3:raise UnsupportedModel('Expected batch-one NHWC RGB')
    oh,ow=shape[1:3];geometry=None
    if profile['resize']=='center_crop':
        crop=float(profile['crop_pct'])
        if oh!=ow or not 0<crop<=1:raise UnsupportedModel('Invalid square center crop')
        side=int(oh/crop)
        size=(side,int(side*height/width)) if width<height else (int(side*width/height),side)
        image=image.resize(size,Image.Resampling.BICUBIC)
        left=round((size[0]-ow)/2);top=round((size[1]-oh)/2)
        pixels=np.asarray(image.crop((left,top,left+ow,top+oh)),dtype='u1')
    elif profile['resize']=='letterbox':
        try:import cv2
        except ImportError as exc:raise UnsupportedModel('Letterbox preprocessing requires the vision extra (OpenCV)') from exc
        ratio=min(ow/width,oh/height);nw,nh=round(width*ratio),round(height*ratio)
        if not nw or not nh:raise ValueError('Image aspect ratio is too extreme for this letterbox size')
        left=round((ow-nw)/2-.1);top=round((oh-nh)/2-.1)
        resized=cv2.resize(np.asarray(image),(nw,nh),interpolation=cv2.INTER_LINEAR)
        pixels=np.full((oh,ow,3),114,'u1');pixels[top:top+nh,left:left+nw]=resized
        geometry={'ratio':ratio,'left':left,'top':top,'width':width,'height':height}
    else:raise UnsupportedModel('Unknown image resize profile')
    mean=np.asarray(profile['mean'],dtype='f4');std=np.asarray(profile['std'],dtype='f4')
    if mean.shape!=(3,) or std.shape!=(3,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std<=0):raise UnsupportedModel('Invalid RGB normalization')
    q=spec['quantization'];dtype=np.dtype(spec['dtype'])
    table=profile_lookup(tuple(float(v) for v in mean),tuple(float(v) for v in std),q['scale'],q['zero_point'],dtype.str)
    return table[pixels,np.arange(3)][None],geometry


class YOLODecoder:
    """Raw distance-to-grid boxes and logits from a one-to-one head; no NMS."""
    def __init__(self,task,outputs,input_shape):
        if task.get('kind')!='yolo_one_to_one' or task.get('schema')!=1:raise UnsupportedModel('Invalid YOLO task contract')
        self.task=task;self.labels=task.get('labels',[]);self.classes=task.get('num_classes')
        if type(self.classes) is not int or not 1<=self.classes<=4096 or len(self.labels)!=self.classes:raise UnsupportedModel('Invalid detector labels')
        self.strides=task.get('strides');self.names=task.get('output_order')
        if self.strides!=[8,16,32] or not isinstance(self.names,list) or len(self.names)!=6 or len(set(self.names))!=6:raise UnsupportedModel('Invalid detector head configuration')
        self.height,self.width=input_shape[1:3]
        self.specs={o['name']:o for o in outputs};self.anchors=[]
        for i,stride in enumerate(self.strides):
            h,w=self.height//stride,self.width//stride
            if self.height%stride or self.width%stride:raise UnsupportedModel('Input must be divisible by detector strides')
            for j,c in ((0,4),(1,self.classes)):
                spec=self.specs.get(self.names[2*i+j])
                if spec is None or tuple(spec['shape'])!=(1,h,w,c) or np.dtype(spec['dtype'])!=np.int8:raise UnsupportedModel('Detector output shape/type mismatch')
            yy,xx=np.mgrid[:h,:w];self.anchors.append(np.stack((xx+.5,yy+.5),axis=-1).astype('f4'))
    def decode(self,outputs,*,geometry,score_threshold=.25,max_results=20,iou_threshold=None):
        from .vision import Detection
        if iou_threshold is not None:raise ValueError('The one-to-one YOLO head does not use NMS; omit iou_threshold')
        if not 0<=score_threshold<=1 or type(max_results) is not int or not 1<=max_results<=300:raise ValueError('Invalid confidence or result limit')
        boxes=[];scores=[]
        for i,stride in enumerate(self.strides):
            values=[]
            for name in self.names[2*i:2*i+2]:
                spec=self.specs[name];value=np.asarray(outputs[name])
                if value.shape!=tuple(spec['shape']) or value.dtype!=np.int8:raise ValueError('Unexpected raw detector output')
                q=spec['quantization'];values.append((value[0].astype('f4')-q['zero_point'])*q['scale'])
            dist,logits=values;anchors=self.anchors[i]
            boxes.append(np.concatenate((anchors-dist[:,:,:2],anchors+dist[:,:,2:]),axis=-1).reshape(-1,4)*stride)
            scores.append(1/(1+np.exp(-np.clip(logits.reshape(-1,self.classes),-80,80))))
        boxes=np.concatenate(boxes);scores=np.concatenate(scores)
        # Match upstream's two-stage top-k, allowing multiple classes per anchor.
        anchor_ids=np.argsort(-scores.max(1),kind='stable')[:min(300,len(scores))]
        candidates=scores[anchor_ids].reshape(-1);order=np.argsort(-candidates,kind='stable')[:max_results]
        result=[]
        for j in order:
            confidence=float(candidates[j])
            if confidence<score_threshold:break
            category=int(j%self.classes);box=boxes[anchor_ids[j//self.classes]].copy()
            box[[0,2]]=(box[[0,2]]-geometry['left'])/(geometry['ratio']*geometry['width'])
            box[[1,3]]=(box[[1,3]]-geometry['top'])/(geometry['ratio']*geometry['height'])
            box=np.clip(box,0,1)
            if not np.all(np.isfinite(box)) or box[2]<=box[0] or box[3]<=box[1]:continue
            result.append(Detection(category,self.labels[category],confidence,tuple(float(x) for x in box)))
        return result
