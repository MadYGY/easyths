"""
Dataset and Augmentation for Captcha Recognition.

Dataset format: {captcha_text}_{uuid}.png
Labels are extracted from filenames automatically.
"""

import os
import random

import numpy as np
import torch

from .crnn import sort_charset, DEFAULT_CHARSET


class EnhancedAugmentation:
    """Data augmentation for captcha images."""

    def __init__(
            self,
            brightness_prob: float = 0.5,
            brightness_range: tuple = (0.8, 1.2),
            contrast_prob: float = 0.5,
            contrast_range: tuple = (0.8, 1.2),
            rotation_prob: float = 0.3,
            rotation_range: tuple = (-5, 5),
            noise_prob: float = 0.3,
            noise_std: float = 0.02,
            blur_prob: float = 0.2,
            blur_kernel_range: tuple = (3, 5),
            cutout_prob: float = 0.1,
            cutout_size: int = 10
    ):
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if random.random() < self.brightness_prob:
            factor = random.uniform(*self.brightness_range)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if random.random() < self.contrast_prob:
            factor = random.uniform(*self.contrast_range)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        if random.random() < self.rotation_prob:
            angle = random.uniform(*self.rotation_range)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std * 255, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < self.blur_prob:
            k = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
            img = cv2.GaussianBlur(img, (k, k), 0)

        if random.random() < self.cutout_prob:
            h, w = img.shape[:2]
            y = random.randint(0, max(0, h - self.cutout_size))
            x = random.randint(0, max(0, w - self.cutout_size))
            img[y:y + self.cutout_size, x:x + self.cutout_size] = 0

        return img


class CaptchaDataset(torch.utils.data.Dataset):
    """
    Dataset for captcha recognition.

    Filename format: {captcha_text}_{uuid}.png

    Args:
        data_dir: Directory containing captcha images
        character: Character set (will be sorted internally)
        img_h, img_w: Image dimensions
        nc: Input channels (1=grayscale, 3=RGB)
        augment: Enable basic augmentation
        augmentation_config: EnhancedAugmentation config dict
    """

    def __init__(
            self,
            data_dir: str,
            character: str = DEFAULT_CHARSET,
            img_h: int = 64,
            img_w: int = 256,
            nc: int = 1,
            augment: bool = True,
            augmentation_config: dict = None
    ):
        self.data_dir = data_dir
        self.img_h = img_h
        self.img_w = img_w
        self.nc = nc
        self.augment = augment
        self.training = True

        # Sort character set (same order as model)
        self.character = sort_charset(character)
        self.char_to_idx = {c: i for i, c in enumerate(self.character)}
        self.blank_idx = len(self.character)

        # Augmentation
        if augmentation_config and augmentation_config.get('enabled', False):
            self.augmentor = EnhancedAugmentation(**{
                k: v for k, v in augmentation_config.items() if k != 'enabled'
            })
        else:
            self.augmentor = None

        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory not found: {self.data_dir}")
            return samples

        for filename in os.listdir(self.data_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            name = os.path.splitext(filename)[0]
            pos = name.rfind('_')
            label = name[:pos] if pos > 0 else name

            if all(c in self.char_to_idx for c in label):
                samples.append((os.path.join(self.data_dir, filename), label))
            else:
                print(f"Warning: Skipping file with invalid characters: {filename}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2

        img_path, label = self.samples[idx]

        if self.nc == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        img = cv2.resize(img, (self.img_w, self.img_h))

        # Augmentation
        if self.training and self.augmentor:
            img = self.augmentor(img)
        elif self.training and self.augment:
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                mean = img.mean()
                img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        if self.nc == 1:
            img = np.expand_dims(img, axis=0)       # (1, H, W)
        else:
            img = np.transpose(img, (2, 0, 1))      # (C, H, W)

        img = torch.from_numpy(img).float()
        label_tensor = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)

        return img, label_tensor, len(label)
