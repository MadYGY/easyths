import functools
import json
from pathlib import Path

import cv2
import ddddocr
import joblib
import numpy as np
import structlog
from PIL import Image

from .config import project_config_instance
from .screen_capture import get_mss_instance

DEFAULT_IMAGE_SIZE = 20
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "captcha_svm.joblib"
DEFAULT_CAPTCHA_WIDTH = 82
DEFAULT_CAPTCHA_HEIGHT = 34
DEFAULT_HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
REQUIRED_MODEL_KEYS = {
    "svm",
    "scaler",
    "labels",
    "image_size",
    "hog_params",
    "target_width",
    "target_height",
    "threshold",
    "version",
}


@functools.lru_cache(maxsize=1)
def _get_ddddocr_instance() -> ddddocr.DdddOcr:
    """获取 ddddocr 实例。"""
    ocr = ddddocr.DdddOcr(show_ad=False, beta=True)
    ocr.set_ranges("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return ocr


@functools.lru_cache(maxsize=1)
def _get_svm_model() -> dict[str, object] | None:
    if not DEFAULT_MODEL_PATH.exists():
        return None

    bundle = joblib.load(DEFAULT_MODEL_PATH)
    missing_keys = REQUIRED_MODEL_KEYS.difference(bundle)
    if missing_keys:
        raise ValueError(
            f"captcha svm model missing keys: {', '.join(sorted(missing_keys))}"
        )
    return bundle


def _get_svm_meta() -> dict[str, object]:
    model = _get_svm_model()
    if model is None:
        return {}
    return {
        "image_size": int(model.get("image_size", DEFAULT_IMAGE_SIZE)),
        "hog_params": dict(model.get("hog_params", DEFAULT_HOG_PARAMS)),
        "target_width": int(model.get("target_width", DEFAULT_CAPTCHA_WIDTH)),
        "target_height": int(model.get("target_height", DEFAULT_CAPTCHA_HEIGHT)),
        "threshold": model.get("threshold"),
    }


def _slot_bounds(width: int, slots: int = 4) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, width, slots + 1, dtype=int)
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(slots)
    ]


def _binarize(gray: np.ndarray, threshold: int | None) -> np.ndarray:
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _remove_tiny_components(binary: np.ndarray, min_area: int = 2) -> np.ndarray:
    if min_area <= 1:
        return binary

    component_input = (binary > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        component_input,
        connectivity=8,
    )
    if component_count <= 1:
        return binary

    cleaned = np.zeros_like(binary)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == component_index] = 255
    return cleaned


def _crop_to_foreground(image_bgr: np.ndarray, threshold: int | None, padding: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray, threshold)
    binary = _remove_tiny_components(binary, min_area=2)
    points = cv2.findNonZero(binary)
    if points is None:
        return image_bgr

    x, y, width, height = cv2.boundingRect(points)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_bgr.shape[1], x + width + padding)
    bottom = min(image_bgr.shape[0], y + height + padding)
    cropped = image_bgr[top:bottom, left:right]
    if cropped.size == 0:
        return image_bgr
    return cropped


def _fit_to_canvas(image_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return image_bgr

    source_height, source_width = image_bgr.shape[:2]
    if source_width == target_width and source_height == target_height:
        return image_bgr

    scale = min(
        target_width / max(source_width, 1),
        target_height / max(source_height, 1),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _normalize_slot(slot_bgr: np.ndarray, size: int, threshold: int | None) -> np.ndarray:
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray, threshold)
    binary = _remove_tiny_components(binary, min_area=2)

    points = cv2.findNonZero(binary)
    cropped = binary
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        cropped = binary[y : y + height, x : x + width]

    height, width = cropped.shape
    scale = min((size - 2) / max(width, 1), (size - 2) / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    offset_x = (size - new_width) // 2
    offset_y = (size - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return 255 - canvas


@functools.lru_cache(maxsize=1)
def _hog_descriptor(size: int, hog_params_json: str) -> cv2.HOGDescriptor:
    hog_params = json.loads(hog_params_json)
    cell_width, cell_height = hog_params["pixels_per_cell"]
    block_cells_x, block_cells_y = hog_params["cells_per_block"]
    block_size = (cell_width * block_cells_x, cell_height * block_cells_y)
    cell_size = (cell_width, cell_height)
    return cv2.HOGDescriptor(
        _winSize=(size, size),
        _blockSize=block_size,
        _blockStride=cell_size,
        _cellSize=cell_size,
        _nbins=hog_params["orientations"],
    )


def _extract_hog_features(
    image: np.ndarray,
    size: int,
    hog_params: dict[str, object],
    threshold: int | None,
) -> np.ndarray:
    if image.ndim == 2:
        normalized = _normalize_slot(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            size=size,
            threshold=threshold,
        )
    else:
        normalized = _normalize_slot(image, size=size, threshold=threshold)
    descriptor = _hog_descriptor(size, json.dumps(hog_params, sort_keys=True))
    features = descriptor.compute(normalized)
    if features is None:
        raise ValueError("failed to compute captcha HOG features")
    return features.reshape(-1).astype(np.float32)


def _predict_svm(model: dict[str, object], sample_vectors: np.ndarray) -> list[str]:
    scaler = model["scaler"]
    svm = model["svm"]
    transformed = scaler.transform(sample_vectors)
    predictions = svm.predict(transformed)
    return [str(value) for value in predictions]


def _grab_control_image(captcha_control) -> Image.Image:
    if captcha_control is None:
        raise ValueError("captcha_control is None")

    rect = captcha_control.element_info.rectangle
    monitor = {
        "top": rect.top,
        "left": rect.left,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }
    screenshot = get_mss_instance().grab(monitor)
    return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


class CaptchaOCR:
    def __init__(self):
        self.logger = structlog.get_logger(__name__)

    def _recognize_with_svm(self, image: Image.Image) -> str:
        model = _get_svm_model()
        if model is None:
            raise FileNotFoundError(f"SVM captcha model not found: {DEFAULT_MODEL_PATH}")

        metadata = _get_svm_meta()
        image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
        hog_params = dict(metadata.get("hog_params", DEFAULT_HOG_PARAMS))
        threshold = metadata.get("threshold")
        target_width = int(metadata.get("target_width", DEFAULT_CAPTCHA_WIDTH))
        target_height = int(metadata.get("target_height", DEFAULT_CAPTCHA_HEIGHT))

        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        cropped_bgr = _crop_to_foreground(image_bgr, threshold=threshold)
        aligned_bgr = _fit_to_canvas(
            cropped_bgr,
            target_width=target_width,
            target_height=target_height,
        )

        vectors: list[np.ndarray] = []
        for left, right in _slot_bounds(aligned_bgr.shape[1]):
            slot = aligned_bgr[:, left:right]
            vectors.append(
                _extract_hog_features(
                    slot,
                    size=image_size,
                    hog_params=hog_params,
                    threshold=threshold,
                )
            )

        result = "".join(_predict_svm(model, np.vstack(vectors)))
        return result[:4]

    def _recognize_with_ddddocr(self, image: Image.Image) -> str:
        result = _get_ddddocr_instance().classification(image)
        return result[:4]

    def recognize(self, captcha_control) -> str:
        try:
            image = _grab_control_image(captcha_control)
        except Exception as exc:
            self.logger.error("captcha_capture_failed", error=str(exc))
            return ""

        engine = project_config_instance.captcha_engine
        if engine == "svm":
            try:
                result = self._recognize_with_svm(image)
                if result:
                    return result
            except Exception as exc:
                self.logger.warning("captcha_svm_failed", error=str(exc))

        try:
            return self._recognize_with_ddddocr(image)
        except Exception as exc:
            self.logger.error("captcha_ocr_failed", error=str(exc))
            return ""


@functools.lru_cache(maxsize=1)
def get_captcha_ocr_server() -> CaptchaOCR:
    return CaptchaOCR()
