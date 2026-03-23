"""
Export Captcha Recognition Model to ONNX.

Inference parameters (character, img_h, img_w, nc) are embedded into ONNX metadata.

Usage:
    python export_onnx.py
    python export_onnx.py --model outputs/best_model.pt --output onnx_model

Author: noimank (康康)
Email: noimank@163.com
"""
import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
logging.getLogger('torch.onnx').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import argparse
import yaml
import torch
import onnx
import onnxruntime as ort

from models.crnn import CRNN, DEFAULT_CHARSET


def parse_args():
    p = argparse.ArgumentParser(description="Export Captcha Model to ONNX")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--model", type=str, default="outputs/best_model.pt")
    p.add_argument("--output", type=str, default="onnx_model")
    p.add_argument("--name", type=str, default="captcha_ocr.onnx")
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    g = config.get('Global', {})
    a = config.get('Architecture', {})
    character = g.get('character', DEFAULT_CHARSET)
    img_h, img_w = g.get('img_h', 64), g.get('img_w', 256)
    nc = a.get('backbone', {}).get('nc', 1)
    hidden_size = a.get('head', {}).get('hidden_dim', 128)

    model = CRNN(character=character, img_h=img_h, img_w=img_w,
                 nc=nc, hidden_size=hidden_size)

    print(f"\nExport ONNX | {img_h}x{img_w} nc={nc} | charset={len(model.character)} "
          f"| classes={model.num_classes} | opset={args.opset}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}"); sys.exit(1)

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name

    dummy = torch.randn(1, nc, img_h, img_w)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )

    # Embed metadata
    onnx_model = onnx.load(str(out_path))
    for k, v in [("character", model.character), ("img_h", str(img_h)),
                 ("img_w", str(img_w)), ("nc", str(nc))]:
        m = onnx_model.metadata_props.add()
        m.key, m.value = k, v
    onnx.save(onnx_model, str(out_path))

    # Verify
    sess = ort.InferenceSession(str(out_path))
    inp = sess.get_inputs()[0]
    ort_out = sess.run(None, {inp.name: dummy.numpy()})[0]

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    match = np.allclose(pt_out, ort_out, rtol=1e-4, atol=1e-4)
    print(f"Input: {inp.shape}  Output: {list(ort_out.shape)}")
    print(f"Verify: {'PASS' if match else 'WARN (slight diff)'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
