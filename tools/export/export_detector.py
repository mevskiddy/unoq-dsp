# SPDX-License-Identifier: AGPL-3.0-only
# Ultralytics YOLO26 adaptation: raw one-to-one output export, November 2026.
"""YOLO26-Nano one-to-one learned graph with six raw NHWC outputs.

Weights and learned operations are unchanged. Grid decoding, sigmoid and top-k
are an explicit host postprocessing contract. INT8 calibration is exploratory;
published 640px accuracy does not apply to this 320px quantized export.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import types
import cv2
import numpy as np
import torch
import litert_torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ai_edge_litert.interpreter import Interpreter
from calibration import calibrate_minmax
from ai_edge_quantizer import quantizer,calibrator,recipe

ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
size=320
model=YOLO(str(ROOT/'models/yolo26n.pt')).model.eval()
head=model.model[-1]
head.end2end=True
model.fuse(verbose=False)
letterbox=LetterBox(new_shape=(size,size),auto=False)
files=sorted((ROOT/'local-artifacts/calibration/coco128/images/train2017').glob('*.jpg'))
if len(files)!=128:raise RuntimeError(f'Expected 128 calibration images, got {len(files)}')
def image(path):
    pixels=letterbox(image=cv2.imread(str(path)))[:,:,::-1].copy()
    return pixels[None].astype('f4')/255
sample=torch.from_numpy(image(files[0]))
with torch.no_grad():
    _,original=model(sample.permute(0,3,1,2))
    original=original['one2one']

def raw_head(self,x):
    outputs=[]
    for i in range(self.nl):
        outputs.extend((self.one2one_cv2[i](x[i]).permute(0,2,3,1),
                        self.one2one_cv3[i](x[i]).permute(0,2,3,1)))
    return tuple(outputs)
head.forward=types.MethodType(raw_head,head)
wrapped=litert_torch.to_channel_last_io(model,args=[0]).eval()
with torch.no_grad():
    raw=wrapped(sample)
    boxes=torch.cat([raw[i].permute(0,3,1,2).reshape(1,4,-1) for i in (0,2,4)],dim=2)
    scores=torch.cat([raw[i].permute(0,3,1,2).reshape(1,80,-1) for i in (1,3,5)],dim=2)
    torch.testing.assert_close(boxes,original['boxes'],rtol=0,atol=0)
    torch.testing.assert_close(scores,original['scores'],rtol=0,atol=0)
fp=ROOT/'models/yolo26n-320-raw-fp32.tflite'
ip=ROOT/'models/yolo26n-320-raw-minmax-int8.tflite'
print('Raw learned heads exactly match original fused PyTorch model',flush=True)
if not fp.exists():litert_torch.convert(wrapped,(sample,)).export(str(fp))
ref=Interpreter(model_path=str(fp));signatures=ref.get_signature_list()
signature=next(iter(signatures));input_name=signatures[signature]['inputs'][0]
qt=quantizer.Quantizer(str(fp));qt.load_quantization_recipe(recipe.static_wi8_ai8())
print('Calibrating static INT8 with',len(files),'images',flush=True)
result=calibrate_minmax(qt,fp,{signature:[{input_name:image(path)} for path in files]},ROOT/'models/yolo26n-320-calibration-minmax.json')
qt.quantize(calibration_result=result).export_model(str(ip),overwrite=True)
report={'calibration_reduction':'global observed min/max, not moving average','architecture':'YOLO26n','input_size':size,'head':'one-to-one, raw learned outputs',
        'output_order':['boxes_s8','logits_s8','boxes_s16','logits_s16','boxes_s32','logits_s32'],
        'strides':[8,16,32],'labels':model.names,'calibration_count':len(files),
        'calibration_dataset':'COCO128; no held-out COCO accuracy claim',
        'postprocessing':'grid-distance decoding, sigmoid, top-k on host; no NMS for one-to-one head',
        'versions':{n:importlib.metadata.version(n) for n in ['torch','ultralytics','litert-torch','ai-edge-quantizer','ai-edge-litert']},
        'checkpoint_sha256':hashlib.sha256((ROOT/'models/yolo26n.pt').read_bytes()).hexdigest(),
        'float_sha256':hashlib.sha256(fp.read_bytes()).hexdigest(),
        'int8_sha256':hashlib.sha256(ip.read_bytes()).hexdigest()}
(ROOT/'models/yolo26n-320-minmax-export.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved',ip,flush=True)
