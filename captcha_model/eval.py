"""
Captcha Recognition Evaluation Script.

Usage:
    python eval.py                                  # Default config
    python eval.py --model outputs/best_model.pt    # Specific model
    python eval.py --test_dir data/test             # Specific test dir

Author: noimank (康康)
Email: noimank@163.com
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from models.crnn import CRNN, DEFAULT_CHARSET
from models.loss import CaptchaDataset


def parse_args():
    p = argparse.ArgumentParser(description="Captcha Recognition Evaluation")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--model", type=str, default="outputs/best_model.pt")
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--output", type=str, default="outputs/eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    max_len = max(b[1].shape[0] for b in batch)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, (_, lbl, ln) in enumerate(batch):
        labels[i, :ln] = lbl
        lengths[i] = ln
    return images, labels, lengths


def evaluate_dataset(model, loader, device, character):
    model.eval()
    blank = model.blank_idx
    total_chars = correct_chars = correct_seqs = total = 0
    total_latency = 0.0
    errors = []

    with torch.no_grad():
        for imgs, labels, lengths in loader:
            imgs = imgs.to(device)
            bs = imgs.shape[0]

            t0 = time.perf_counter()
            logits = model(imgs)
            total_latency += (time.perf_counter() - t0) * 1000

            preds = logits.argmax(dim=1).cpu().numpy()

            for i in range(bs):
                label_idx = labels[i, :lengths[i]].tolist()
                label_str = ''.join(character[j] for j in label_idx)

                # CTC greedy decode
                chars, prev = [], -1
                for idx in preds[i]:
                    idx = int(idx)
                    if idx != prev and idx != blank and idx < len(character):
                        chars.append(character[idx])
                    prev = idx
                pred_str = ''.join(chars)

                total += 1
                total_chars += len(label_str)
                correct_chars += sum(p == l for p, l in zip(pred_str, label_str))
                if pred_str == label_str:
                    correct_seqs += 1
                else:
                    errors.append({'prediction': pred_str, 'ground_truth': label_str})

    return {
        'char_acc': correct_chars / max(total_chars, 1),
        'seq_acc': correct_seqs / max(total, 1),
        'avg_latency_ms': total_latency / max(total, 1),
        'total_samples': total,
        'correct_samples': correct_seqs,
        'errors': errors,
    }


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
    test_dir = args.test_dir or config.get('Eval', {}).get('test_dir', 'data/test')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = CRNN(character=character, img_h=img_h, img_w=img_w,
                 nc=nc, hidden_size=hidden_size).to(device)

    print(f"\nEval | model={args.model} | test={test_dir}")
    print(f"     | {img_h}x{img_w} nc={nc} | charset={len(model.character)} | classes={model.num_classes}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}"); sys.exit(1)
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}"); sys.exit(1)

    model_data = torch.load(model_path, map_location=device, weights_only=True)
    if model_data.get("model_state_dict"):
        model.load_state_dict(model_data["model_state_dict"])
    else:
        model.load_state_dict(model_data)

    test_ds = CaptchaDataset(test_dir, character=character,
                             img_h=img_h, img_w=img_w, nc=nc, augment=False)
    test_ds.training = False
    print(f"Samples: {len(test_ds)}")

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0, collate_fn=collate_fn)

    t0 = datetime.now()
    results = evaluate_dataset(model, loader, device, model.character)
    eval_time = (datetime.now() - t0).total_seconds()

    print(f"\nCharAcc: {results['char_acc']:.4f}  SeqAcc: {results['seq_acc']:.4f}")
    print(f"Latency: {results['avg_latency_ms']:.2f}ms  Total: {results['total_samples']}  Time: {eval_time:.1f}s")

    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results['errors'][:20]:
            print(f"  GT='{e['ground_truth']}' -> Pred='{e['prediction']}'")
        if len(results['errors']) > 20:
            print(f"  ... +{len(results['errors'])-20} more")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", 'w', encoding='utf-8') as f:
        json.dump({**results, 'model': str(model_path), 'test_dir': test_dir,
                   'eval_time_s': eval_time, 'timestamp': datetime.now().isoformat()},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
