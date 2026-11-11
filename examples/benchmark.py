"""Usage: ./run python examples/benchmark.py photo.jpg"""
import json
from pathlib import Path
import statistics
import sys
import time
from PIL import Image, ImageOps
from unoq_dsp import Session, load
from unoq_dsp.modern import prepare_profile
root = Path(__file__).resolve().parents[1]
with Image.open(sys.argv[1]) as source:
    image = ImageOps.exif_transpose(source).convert('RGB')
report = {}
for name in ('mobilenetv4-small', 'yolo26n-320'):
    bundle = load(root/'models'/f'{name}.uqmodel')
    x, _ = prepare_profile(image, bundle.metadata['input'], bundle.metadata['task']['preprocessing'])
    with Session(bundle) as model:
        for _ in range(5): model.run(x)
        elapsed, cpu = [], []
        for _ in range(30):
            start, work = time.perf_counter_ns(), time.process_time_ns()
            model.run(x)
            cpu.append((time.process_time_ns()-work)/1e6)
            elapsed.append((time.perf_counter_ns()-start)/1e6)
    report[name] = {'inference_median_ms': statistics.median(elapsed),
                    'client_process_cpu_median_ms': statistics.median(cpu),
                    'iterations': 30, 'transport': 'blocking RPC'}
print(json.dumps(report, indent=2))
