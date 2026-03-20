"""
Standalone captcha recognition inference using only ONNX model.

All configuration (charset, input size, normalization) is embedded
directly in the ONNX model's metadata — no config file needed.

Usage:
    python -m captcha_model.onnx_infer model.onnx captcha.png

    # Or import directly:
    from captcha_model.onnx_infer import CaptchaRecognizer
    model = CaptchaRecognizer("model.onnx")
    text = model.predict("captcha.png")
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from captcha_model.ctc_decoder import greedy_decode, beam_search_decode


class CaptchaRecognizer:
    """
    Self-contained captcha recognizer — loads everything from ONNX metadata.

    Requires only:
        - onnxruntime
        - numpy
        - Pillow
    """

    def __init__(self, model_path: str, decode_method: str = "greedy", beam_width: int = 10):
        """
        Args:
            model_path: Path to ONNX model file (with embedded metadata).
            decode_method: CTC decoding method, "greedy" or "beam_search".
            beam_width: Beam width for beam search decoding.
        """
        self.model_path = model_path
        self.decode_method = decode_method
        self.beam_width = beam_width

        self._load_metadata()
        self._init_session()

    def _load_metadata(self) -> None:
        """Extract all configuration from ONNX model metadata."""
        import onnxruntime as ort
        from onnx import load as load_onnx

        onnx_model = load_onnx(self.model_path)
        meta = onnx_model.metadata_props
        m = {entry.key: entry.value for entry in meta}

        self.charset: str = m["charset"]
        self.input_height: int = int(m["input_height"])
        self.input_width: int = int(m["input_width"])
        self.channels: int = int(m["channels"])
        self.downsampling: int = int(m.get("downsampling", "8"))
        self.seq_len: int = self.input_width // self.downsampling

        # Normalization params
        self.mean: List[float] = json.loads(m.get("mean", "[0.5, 0.5, 0.5]"))
        self.std: List[float] = json.loads(m.get("std", "[0.5, 0.5, 0.5]"))

        # Character mapping
        self.charset_size = len(self.charset)
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.charset)}
        self.idx_to_char[0] = ""  # CTC blank token

        self.num_classes = self.charset_size + 1  # +1 for CTC blank

    def _init_session(self) -> None:
        """Initialize ONNX Runtime session."""
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32) / 255.0

        if self.channels == 3:
            mean = np.array(self.mean, dtype=np.float32)
            std = np.array(self.std, dtype=np.float32)
            image_array = (image_array - mean) / std
            image_array = np.transpose(image_array, (2, 0, 1))
        else:
            image_array = np.expand_dims(image_array, 0)

        return image_array

    def _greedy_decode(self, probs: np.ndarray) -> List[int]:
        """Greedy CTC decoding."""
        return greedy_decode(probs, blank=0)

    def _beam_search_decode(self, probs: np.ndarray) -> List[int]:
        """Beam search CTC decoding with log-space computation."""
        return beam_search_decode(probs, blank=0, beam_width=self.beam_width)

    def _decode(self, probs: np.ndarray) -> str:
        """Decode probability matrix to text."""
        if self.decode_method == "beam_search":
            indices = self._beam_search_decode(probs)
        else:
            indices = self._greedy_decode(probs)
        return "".join(self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char)

    def predict(
        self,
        image_path: Union[str, "Image.Image"],
        return_confidence: bool = False,
    ) -> Union[str, Tuple[str, List[float]]]:
        """
        Recognize captcha text from an image.

        Args:
            image_path: Path to image file or PIL Image object.
            return_confidence: If True, return (text, per_char_confidences).

        Returns:
            Predicted text, or (text, confidences) if return_confidence=True.
        """
        if isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = Image.open(image_path)

        img_array = self._preprocess(image)
        img_array = np.expand_dims(img_array, 0)  # Add batch dim

        probs: np.ndarray = self.session.run(None, {"input": img_array})[0]
        probs = probs.squeeze(1)  # (seq_len, num_classes)

        text = self._decode(probs)

        if return_confidence:
            confidences = self._get_confidences(probs, text)
            return text, confidences

        return text

    def _get_confidences(self, probs: np.ndarray, text: str) -> List[float]:
        """Calculate per-character confidence scores."""
        best_path = np.argmax(probs, axis=1)
        confidences = []
        prev = None
        char_idx = 0

        for t, token in enumerate(best_path):
            if token != prev:
                if token != 0 and char_idx < len(text):
                    confidences.append(float(probs[t, token]))
                    char_idx += 1
            prev = token

        return confidences

    def predict_batch(
        self,
        image_paths: List[Union[str, "Image.Image"]],
        return_confidence: bool = False,
    ) -> List[Union[str, Tuple[str, List[float]]]]:
        """Recognize captcha text from multiple images using batch inference."""
        if not image_paths:
            return []

        # Preprocess all images
        images = []
        for p in image_paths:
            if isinstance(p, Image.Image):
                img = p
            else:
                img = Image.open(p)
            images.append(self._preprocess(img))

        batch = np.stack(images, axis=0)
        probs: np.ndarray = self.session.run(None, {"input": batch})[0]
        # Output shape: (T, B, C)

        results = []
        for i in range(probs.shape[1]):
            sample_probs = probs[:, i, :]
            text = self._decode(sample_probs)
            if return_confidence:
                confidences = self._get_confidences(sample_probs, text)
                results.append((text, confidences))
            else:
                results.append(text)
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone ONNX captcha recognition")
    parser.add_argument("model", type=str, help="Path to ONNX model file")
    parser.add_argument("images", type=str, nargs="+", help="Path(s) to captcha image(s)")
    parser.add_argument(
        "--decode",
        type=str,
        default="greedy",
        choices=["greedy", "beam_search"],
        help="CTC decoding method",
    )
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        help="Show per-character confidence scores",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    recognizer = CaptchaRecognizer(args.model, decode_method=args.decode)
    print(f"  charset:    {recognizer.charset}")
    print(f"  input size: {recognizer.input_width}x{recognizer.input_height}")
    print(f"  classes:    {recognizer.num_classes} (including blank)")
    print()

    for image_path in args.images:
        if not Path(image_path).exists():
            print(f"[ERROR] Image not found: {image_path}")
            continue

        if args.show_confidence:
            text, confidences = recognizer.predict(image_path, return_confidence=True)
            print(f"Image: {image_path}")
            print(f"  Prediction: {text}")
            if confidences:
                print("  Confidence per character:")
                for i, (char, conf) in enumerate(zip(text, confidences)):
                    print(f"    [{i}] '{char}' = {conf:.4f}")
            print()
        else:
            text = recognizer.predict(image_path)
            print(f"{Path(image_path).name}: {text}")


if __name__ == "__main__":
    main()
