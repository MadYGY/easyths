"""
ONNX Inference Script for Captcha Recognition.

All parameters are read from ONNX metadata — no config needed.

Usage:
    python infer_onnx.py --model model.onnx --image captcha.png
    python infer_onnx.py --model model.onnx --dir data/test
    python infer_onnx.py --model model.onnx --benchmark

Author: noimank (康康)
Email: noimank@163.com
"""
import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description="ONNX Captcha Recognition")
    p.add_argument("--model", type=str, default="onnx_model/captcha_ocr.onnx")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--dir", type=str, default=None)
    p.add_argument("--output", type=str, default="outputs/onnx_infer")
    p.add_argument("--providers", type=str, default="CPUExecutionProvider")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    return p.parse_args()


class ONNXCaptchaRecognizer:
    """ONNX captcha recognizer. All parameters from model metadata."""

    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        meta = {p.key: p.value for p in onnx.load(model_path).metadata_props}

        self.character = meta.get("character", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.img_h = int(meta.get("img_h", 64))
        self.img_w = int(meta.get("img_w", 256))
        self.nc = int(meta.get("nc", 1))
        self.blank = len(self.character)

        prov = 'CUDAExecutionProvider' if 'cuda' in provider.lower() else 'CPUExecutionProvider'
        self.session = ort.InferenceSession(model_path, providers=[prov])
        self.input_name = self.session.get_inputs()[0].name

        print(f"Model: {model_path} | {self.img_h}x{self.img_w} nc={self.nc} | {len(self.character)} chars")

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        tensor = self._preprocess(image)
        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        latency = (time.perf_counter() - t0) * 1000
        return self._decode(output.argmax(axis=1)[0].tolist()), latency

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.nc == 1:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            img = cv2.resize(img, (self.img_w, self.img_h)).astype(np.float32) / 255.0
            return img[np.newaxis, np.newaxis, ...]    # (1, 1, H, W)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (self.img_w, self.img_h)).astype(np.float32) / 255.0
            return np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    def _decode(self, indices: list) -> str:
        chars, prev = [], -1
        for idx in indices:
            if idx != prev and idx != self.blank and idx < len(self.character):
                chars.append(self.character[idx])
            prev = idx
        return ''.join(chars)


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}"); return

    rec = ONNXCaptchaRecognizer(str(model_path), args.providers)

    if args.benchmark:
        times = []
        for i in range(args.warmup + args.iterations):
            img = np.random.randint(0, 255, (rec.img_h, rec.img_w, 3), dtype=np.uint8)
            _, lat = rec.recognize(img)
            if i >= args.warmup:
                times.append(lat)
        avg, std = np.mean(times), np.std(times)
        print(f"Benchmark: {avg:.2f}ms +/-{std:.2f}  FPS={1000/avg:.1f}")
        with open(out_dir / "benchmark.json", 'w') as f:
            json.dump({'avg_ms': avg, 'std_ms': std, 'fps': 1000/avg}, f, indent=2)
        return

    sources = []
    if args.image:
        p = Path(args.image)
        if p.exists(): sources = [p]
    elif args.dir:
        d = Path(args.dir)
        if d.exists():
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                sources.extend(d.glob(ext))
    else:
        d = Path("data/test")
        if d.exists():
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                sources.extend(d.glob(ext))

    if not sources:
        print("No images found, running benchmark")
        args.benchmark = True
        return main.__wrapped__() if hasattr(main, '__wrapped__') else None

    results, total_lat = [], 0.0
    for p in sources:
        img = cv2.imread(str(p))
        if img is None: continue
        text, lat = rec.recognize(img)
        total_lat += lat
        results.append({'image': str(p), 'prediction': text, 'latency_ms': lat})
        print(f"  {p.name}: {text} ({lat:.1f}ms)")

    avg_lat = total_lat / len(results) if results else 0
    print(f"\n{len(results)} images | avg {avg_lat:.2f}ms | {1000/avg_lat:.1f} FPS" if avg_lat > 0 else "")

    with open(out_dir / "inference_results.json", 'w', encoding='utf-8') as f:
        json.dump({'model': str(model_path), 'total': len(results),
                   'avg_latency_ms': avg_lat, 'results': results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
