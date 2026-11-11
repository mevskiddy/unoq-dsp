"""Package named model profiles; weights are unchanged by packaging."""
import json
from pathlib import Path
from unoq_dsp import compile_model
root=Path(__file__).resolve().parents[2]
classifier=compile_model(root/'models/mobilenetv4-small-int8.tflite')
labels=json.loads((root/'models/imagenet-labels.json').read_text())
config=json.loads((root/'models/mobilenetv4-small-export.json').read_text())['preprocessing']
classifier.metadata['task']={'kind':'image_classification','schema':1,'labels':labels,'apply_softmax':True,'preprocessing':{'resize':'center_crop','crop_pct':config['crop_pct'],'mean':config['mean'],'std':config['std']},'quality_scope':'Exploratory 200-image Imagenette calibration, 200-image held-out agreement check; full ImageNet accuracy not certified'}
classifier.save(root/'models/mobilenetv4-small.uqmodel')
model=compile_model(root/'models/yolo26n-320-raw-minmax-int8.tflite')
config=json.loads((root/'models/yolo26n-320-minmax-export.json').read_text())
model.metadata['task']={'kind':'yolo_one_to_one','schema':1,'num_classes':80,'strides':[8,16,32],'labels':[config['labels'][str(i)] for i in range(80)],'output_order':[f'serving_default_output_{i}_output' for i in range(6)],'preprocessing':{'resize':'letterbox','mean':[0,0,0],'std':[1,1,1]},'quality_scope':'COCO128 calibration, two upstream demo-image smoke checks; held-out COCO mAP not certified'}
model.save(root/'models/yolo26n-320.uqmodel')
print('Packaged MobileNetV4 and YOLO26 profiles')
