"""Reproducible exploratory MobileNetV4 export; no held-out accuracy claim.

Calibration uses a deterministic, class-balanced Imagenette training subset.
This ten-class subset is not enough to certify full ImageNet quantized accuracy.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import timm
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from calibration import calibrate_minmax
from ai_edge_quantizer import calibrator, quantizer, recipe

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'local-artifacts/calibration/imagenette2-160'
OUT=ROOT/'models'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--calibration',choices=('moving-average','minmax'),default='moving-average',
                    help='The selected classifier uses moving-average; minmax is an experimental alternate.')
args=parser.parse_args()
variant='-minmax' if args.calibration=='minmax' else ''
torch.set_num_threads(4)
name='mobilenetv4_conv_small.e2400_r224_in1k'
model=timm.create_model(name,pretrained=True).eval()
config=timm.data.resolve_model_data_config(model)
transform=timm.data.create_transform(**config,is_training=False)
wrapped=litert_torch.to_channel_last_io(model,args=[0]).eval()
sample=torch.zeros(1,224,224,3)
float_path=OUT/'mobilenetv4-small-fp32.tflite'
int_path=OUT/f'mobilenetv4-small{variant}-int8.tflite'
print('Exporting',name,flush=True)
if not float_path.exists():
    litert_torch.convert(wrapped,(sample,)).export(str(float_path))
runtime=Interpreter(model_path=str(float_path))
signatures=runtime.get_signature_list()
signature=next(iter(signatures));input_name=signatures[signature]['inputs'][0]
paths=[]
for folder in sorted((DATA/'train').iterdir()):
    if folder.is_dir():paths.extend(sorted(folder.glob('*.JPEG'))[:20])
if len(paths)!=200:raise RuntimeError(f'Expected 200 calibration images, found {len(paths)}')
samples=[]
for path in paths:
    with Image.open(path) as im:x=transform(im.convert('RGB'))[None].permute(0,2,3,1).contiguous().numpy()
    samples.append({input_name:x})
qt=quantizer.Quantizer(str(float_path))
qt.load_quantization_recipe(recipe.static_wi8_ai8())
print('Calibrating',len(samples),'training images',flush=True)
if args.calibration=='minmax':
    result=calibrate_minmax(qt,float_path,{signature:samples},OUT/'mobilenetv4-small-calibration-minmax.json')
else:
    result=qt.calibrate({signature:samples},num_threads=4,
                        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED)
qt.quantize(calibration_result=result).export_model(str(int_path),overwrite=True)
report={'calibration_reduction':args.calibration,'architecture':name,'preprocessing':config,'calibration_count':len(paths),
        'calibration_files':[str(p.relative_to(DATA)) for p in paths],
        'calibration_scope':'Imagenette ten-class training subset; not ImageNet accuracy certification',
        'source_model':model.pretrained_cfg,
        'versions':{n:importlib.metadata.version(n) for n in ['torch','timm','litert-torch','ai-edge-quantizer','ai-edge-litert']},
        'float_sha256':hashlib.sha256(float_path.read_bytes()).hexdigest(),
        'int8_sha256':hashlib.sha256(int_path.read_bytes()).hexdigest()}
(OUT/f'mobilenetv4-small{variant}-export.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print('Saved',int_path,flush=True)
