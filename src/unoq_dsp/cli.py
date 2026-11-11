"""Offline inspection/compilation and explicit on-board inference commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import compile_model, Session, ObjectDetector, ImageClassifier
from .bundle import json_value
from .compatibility import inspect_model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="unoq-dsp")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Report model coverage without loading the DSP")
    p.add_argument("model")
    p.add_argument("--output")
    p = sub.add_parser("compile", help="Prepare supported TFLite graphs for DSP execution")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)
    for name in ("infer", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--input", required=True, help="Already-preprocessed .npy input")
        p.add_argument("--library")
        p.add_argument("--pipeline", choices=("mailbox", "rpc"), default="rpc")
        if name == "infer":
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--iterations", type=int, default=100)
            p.add_argument("--warmup", type=int, default=10)
    for name in ("detect", "classify"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("image")
        p.add_argument("--library")
        p.add_argument("--pipeline", choices=("mailbox", "rpc"), default="rpc")
        if name == "detect":
            p.add_argument("--score-threshold",type=float,default=.5)
            p.add_argument("--iou-threshold",type=float,help="SSD NMS overlap threshold; omit for one-to-one YOLO")
            p.add_argument("--max-results",type=int,default=20)
        else:
            p.add_argument("--mean",type=float,nargs="+",help="Required only without bundled preprocessing")
            p.add_argument("--std",type=float,nargs="+",help="Required only without bundled preprocessing")
            p.add_argument("--labels",help="One label per line")
            p.add_argument("--top-k",type=int,default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_model(args.model)
            text = json.dumps(report, indent=2, allow_nan=False)+"\n"
            if args.output:
                Path(args.output).write_text(text)
            print(text, end="")
            return 0 if report["compilable"] else 2
        if args.command == "compile":
            model = compile_model(args.model)
            model.save(args.output)
            print(json.dumps({"output": args.output, "plan_bytes": len(model.plan),
                              "input": model.metadata["input"], "output_tensor": model.metadata["output"]}, default=json_value))
            return 0
        if args.command in ("detect", "classify"):
            from PIL import Image, ImageOps
            with Image.open(args.image) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            if args.command == "detect":
                with ObjectDetector(args.model,library=args.library,pipeline=args.pipeline) as detector:
                    options={"score_threshold":args.score_threshold,"max_results":args.max_results}
                    if args.iou_threshold is not None:options["iou_threshold"]=args.iou_threshold
                    result=detector.detect(image,**options)
                    print(json.dumps({"detections":[asdict(d) for d in result],"timings":detector.last_timings,"box_format":"normalized_xyxy"},indent=2))
            else:
                labels=Path(args.labels).read_text().splitlines() if args.labels else []
                with ImageClassifier(args.model,mean=args.mean,std=args.std,labels=labels,library=args.library,pipeline=args.pipeline) as classifier:
                    print(json.dumps(classifier.classify(image,top_k=args.top_k),indent=2))
            return 0
        x = np.load(args.input, allow_pickle=False)
        if args.command == "benchmark" and not (1 <= args.iterations <= 100000 and 0 <= args.warmup <= 1000):
            raise ValueError("Invalid benchmark iteration count")
        with Session(args.model, library=args.library, pipeline=args.pipeline) as session:
            if args.command == "infer":
                result = session.run(x)
                if isinstance(result, dict):
                    if Path(args.output).suffix != ".npz":
                        raise ValueError("Use .npz output for multiple-output models")
                    np.savez(args.output, **result)
                else:
                    np.save(args.output, result)
                return 0
            for _ in range(args.warmup):
                session.run(x)
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                session.run(x)
                samples.append((time.perf_counter_ns()-start)/1e6)
        print(json.dumps({"samples": len(samples), "median_ms": float(np.median(samples)),
                          "p95_ms": float(np.percentile(samples, 95)),
                          "throughput_per_second": 1000/float(np.mean(samples)),
                          "includes": "Python session, input/output transfer, DSP graph",
                          "excludes": "model loading, image preprocessing, detection postprocessing",
                          "pipeline": args.pipeline}, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"unoq-dsp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
