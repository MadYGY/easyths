"""
Inference script for captcha recognition with CTC support.

Usage:
    # Single image (uses default model outputs/best_model.pth)
    python -m captcha_model.infer --image captcha.png

    # Multiple images
    python -m captcha_model.infer --image img1.png img2.png

    # Use ONNX model (auto-detected by .onnx suffix)
    python -m captcha_model.infer --model outputs/best_model.onnx --image captcha.png

    # Get confidence scores
    python -m captcha_model.infer --image captcha.png --show_confidence
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from captcha_model.utils import load_config, get_device, load_state_dict_from_path
from captcha_model.ctc_decoder import greedy_decode, beam_search_decode


class CaptchaPredictor:
    """
    Unified predictor for captcha recognition supporting both PyTorch and ONNX models.

    Model type is auto-detected by file suffix (.pth or .onnx).
    """

    def __init__(
        self,
        model_path: str,
        config: Dict,
        decode_method: str = "greedy",
    ):
        self.config = config
        self.charset = config["charset"]
        # CTC blank at index 0, characters at indices 1 to charset_size
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.charset)}
        self.idx_to_char[0] = ""  # blank token
        self.decode_method = decode_method
        self.model_path = model_path

        # Auto-detect model type by suffix
        self.use_onnx = model_path.endswith(".onnx")

        if self.use_onnx:
            self._init_onnx(model_path)
        else:
            self._init_pytorch(model_path)

    def _init_pytorch(self, model_path: str) -> None:
        """Initialize PyTorch model."""
        import torch
        from captcha_model.model import CaptchaRecognizer

        self.device = get_device()
        model_config = self.config.get("model", {})
        charset_size = len(self.charset)

        self.model = CaptchaRecognizer(
            charset_size=charset_size,
            in_channels=self.config["image"]["channels"],
            dropout=model_config.get("dropout", 0.2),
            hidden_size=model_config.get("hidden_size", 384),
            num_tcn_layers=model_config.get("num_tcn_layers", 4),
        )

        state_dict = load_state_dict_from_path(model_path, self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _init_onnx(self, model_path: str) -> None:
        """Initialize ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX runtime requires: pip install onnxruntime")

        self.ort_session = ort.InferenceSession(model_path)
        self.device = "cpu"

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for model input.

        Args:
            image_path: Path to the input image.

        Returns:
            Preprocessed image array (C, H, W).
        """
        image = Image.open(image_path).convert("RGB")
        channels = self.config["image"]["channels"]

        # Convert to grayscale if needed
        if channels == 1:
            image = image.convert("L")

        # Resize to model input size
        target_width = self.config["model"]["input_width"]
        target_height = self.config["model"]["input_height"]
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)

        # Convert to numpy and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Apply normalization from config
        norm_config = self.config.get("preprocessing", {}).get("normalization", {})
        mean = np.array(norm_config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.array(norm_config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)

        if channels == 3:
            image_array = (image_array - mean) / std
            # Transpose to (C, H, W)
            image_array = np.transpose(image_array, (2, 0, 1))
        else:
            image_array = (image_array - mean[0]) / std[0]
            image_array = np.expand_dims(image_array, 0)

        return image_array

    def _decode_output(self, indices: List[int]) -> str:
        """Decode model output indices to string."""
        return "".join(self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char)

    def predict(
        self,
        image_path: str,
        return_confidence: bool = False,
    ) -> Union[str, Tuple[str, List[float]]]:
        """Predict captcha text from an image."""
        image = self._preprocess_image(image_path)

        if self.use_onnx:
            text, confidences = self._predict_onnx(image, return_confidence)
        else:
            text, confidences = self._predict_pytorch(image, return_confidence)

        if return_confidence:
            return text, confidences
        return text

    def _predict_pytorch(
        self,
        image: np.ndarray,
        return_confidence: bool,
    ) -> Tuple[str, Optional[List[float]]]:
        """Run PyTorch inference."""
        import torch

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_output = self.model(image_tensor)
            # Model output: (T, B, C), take batch index 0 -> (T, C)
            probs = torch.exp(log_output)[:, 0, :].cpu().numpy()

            if self.decode_method == "beam_search":
                indices = beam_search_decode(probs, blank=0, beam_width=10)
            else:
                indices = greedy_decode(probs, blank=0)

            text = self._decode_output(indices)

            if return_confidence:
                confidences = self._get_confidences(probs, indices)
                return text, confidences

            return text, None

    def _predict_onnx(
        self,
        image: np.ndarray,
        return_confidence: bool,
    ) -> Tuple[str, Optional[List[float]]]:
        """Run ONNX inference."""
        probs = self.ort_session.run(None, {"input": np.expand_dims(image, 0)})[0]
        # ONNX output: (T, B, C), take batch index 0 -> (T, C)
        probs = probs[:, 0, :]

        if self.decode_method == "beam_search":
            indices = beam_search_decode(probs, blank=0, beam_width=10)
        else:
            indices = greedy_decode(probs, blank=0)

        text = self._decode_output(indices)

        if return_confidence:
            confidences = self._get_confidences(probs, indices)
            return text, confidences

        return text, None

    def _get_confidences(self, probs: np.ndarray, indices: List[int]) -> List[float]:
        """
        Calculate per-character confidence scores.

        Args:
            probs: Probability matrix (seq_len, num_classes).
            indices: Decoded character indices (after CTC collapsing).

        Returns:
            List of confidence scores for each decoded character.
        """
        best_path = np.argmax(probs, axis=1)
        confidences = []
        prev = None
        conf_idx = 0

        for t, token in enumerate(best_path):
            if token != prev:
                if token != 0 and conf_idx < len(indices):
                    confidences.append(float(probs[t, token]))
                    conf_idx += 1
            prev = token

        return confidences

    def predict_batch(
        self,
        image_paths: List[str],
        return_confidence: bool = False,
    ) -> List[Union[str, Tuple[str, List[float]]]]:
        """Predict captcha text for multiple images using true batch inference."""
        if not image_paths:
            return []

        images = np.stack([self._preprocess_image(p) for p in image_paths], axis=0)

        if self.use_onnx:
            probs = self.ort_session.run(None, {"input": images})[0]
            # (T, B, C) -> per-sample decode
            results = []
            for i in range(probs.shape[1]):
                sample_probs = probs[:, i, :]
                if self.decode_method == "beam_search":
                    indices = beam_search_decode(sample_probs, blank=0, beam_width=10)
                else:
                    indices = greedy_decode(sample_probs, blank=0)
                text = self._decode_output(indices)
                if return_confidence:
                    confidences = self._get_confidences(sample_probs, indices)
                    results.append((text, confidences))
                else:
                    results.append(text)
            return results
        else:
            import torch
            images_tensor = torch.from_numpy(images).to(self.device)
            with torch.no_grad():
                log_output = self.model(images_tensor)
                probs = torch.exp(log_output).cpu().numpy()  # (T, B, C)

            results = []
            for i in range(probs.shape[1]):
                sample_probs = probs[:, i, :]
                if self.decode_method == "beam_search":
                    indices = beam_search_decode(sample_probs, blank=0, beam_width=10)
                else:
                    indices = greedy_decode(sample_probs, blank=0)
                text = self._decode_output(indices)
                if return_confidence:
                    confidences = self._get_confidences(sample_probs, indices)
                    results.append((text, confidences))
                else:
                    results.append(text)
            return results


def main():
    parser = argparse.ArgumentParser(description="Captcha recognition inference")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/best_model.pth",
        help="Path to model file (.pth or .onnx), auto-detected by suffix",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to captcha image(s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
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
        help="Show confidence scores for each character",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading model: {args.model}")
    predictor = CaptchaPredictor(args.model, config, decode_method=args.decode)

    print(f"Model type: {'ONNX' if predictor.use_onnx else 'PyTorch'}")
    print(f"Predicting {len(args.image)} image(s)...\n")

    for image_path in args.image:
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            continue

        if args.show_confidence:
            result = predictor.predict(image_path, return_confidence=True)
            text, confidences = result
            print(f"Image: {image_path}")
            print(f"  Prediction: {text}")
            if confidences:
                print("  Confidence per character:")
                for i, (char, conf) in enumerate(zip(text, confidences)):
                    print(f"    Position {i}: '{char}' ({conf:.4f})")
            print()
        else:
            text = predictor.predict(image_path)
            print(f"{image_path}: {text}")


if __name__ == "__main__":
    main()
