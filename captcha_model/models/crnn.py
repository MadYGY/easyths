"""
CRNN Model for Captcha Recognition.

Architecture: CaptchaCNN (modern ResNet-style CNN) + BiLSTM + CTC Head

Author: noimank (康康)
Email: noimank@163.com
"""

import torch
import torch.nn as nn

# Default character set (sorted for consistency)
DEFAULT_CHARSET = "".join(sorted("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def sort_charset(character: str) -> str:
    """Sort character set for consistent char-to-index mapping."""
    return "".join(sorted(character))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Conv-BN-GELU block with optional depthwise separable convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, depthwise_separable: bool = False):
        super().__init__()
        if depthwise_separable and in_ch == out_ch:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with SE attention. Two conv layers + skip connection."""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + x)


class CaptchaCNN(nn.Module):
    """Modern CNN backbone with residual blocks and SE attention.

    Architecture:
        Stage 1: Conv(nc->32) + Pool(2,2)       -> H/2, W/2
        Stage 2: Conv(32->64) + Pool(2,2)        -> H/4, W/4
        Stage 3: ResBlock(64) + Pool((2,2),(2,1)) -> H/8, W~
        Stage 4: Conv(64->128) + ResBlock(128) + Pool((2,2),(2,1)) -> H/16, W~
        Stage 5: Conv(128->128, 2x1, no pad)      -> H final collapse
    """

    def __init__(self, nc: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: stem
            ConvBlock(nc, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Stage 2: expand channels
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Stage 3: residual learning with SE attention
            ResBlock(64, use_se=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            # Stage 4: higher capacity + residual
            ConvBlock(64, 128, 3, 1, 1),
            ResBlock(128, use_se=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            # Stage 5: final feature compression
            ConvBlock(128, 128, (2, 1), 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CRNN(nn.Module):
    """
    CRNN model for captcha recognition.

    num_classes is derived automatically: len(character) + 1 (blank).
    character is always sorted internally to guarantee consistent mapping.

    Args:
        character: Character set string (will be sorted internally)
        img_h: Input image height
        img_w: Input image width
        nc: Number of input channels (1=grayscale, 3=RGB)
        hidden_size: LSTM hidden size
    """

    def __init__(
            self,
            character: str = DEFAULT_CHARSET,
            img_h: int = 64,
            img_w: int = 256,
            nc: int = 1,
            hidden_size: int = 128,
    ):
        super().__init__()

        # Sort character set for consistent mapping
        self.character = sort_charset(character)
        self.num_classes = len(self.character) + 1  # +1 for CTC blank
        self.img_h = img_h
        self.img_w = img_w
        self.nc = nc
        self.hidden_size = hidden_size

        # Character mapping
        self.char_to_idx = {c: i for i, c in enumerate(self.character)}
        self.idx_to_char = {i: c for i, c in enumerate(self.character)}
        self.blank_idx = self.num_classes - 1

        # CNN backbone
        self.cnn = CaptchaCNN(nc=nc)

        # Probe CNN output shape
        with torch.no_grad():
            probe = self.cnn(torch.zeros(1, nc, img_h, img_w))
            _, c, h, w = probe.shape
            self.lstm_input_size = c * h
            self.seq_len = w

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )

        # Classifier
        self.fc = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes, T)
        """
        feat = self.cnn(x)                          # (B, C, H, W)
        b, c, h, w = feat.shape
        feat = feat.permute(3, 0, 1, 2)             # (W, B, C, H)
        feat = feat.reshape(w, b, c * h)             # (W, B, C*H)
        feat, _ = self.lstm(feat)                    # (W, B, hidden*2)
        logits = self.fc(feat)                       # (W, B, num_classes)
        return logits.permute(1, 2, 0)               # (B, num_classes, W)

    def decode(self, logits: torch.Tensor) -> list[str]:
        """CTC greedy decode."""
        preds = logits.argmax(dim=1).cpu().numpy()   # (B, T)
        results = []
        for pred in preds:
            chars, prev = [], -1
            for idx in pred:
                idx = int(idx)
                if idx != prev and idx != self.blank_idx and idx < len(self.character):
                    chars.append(self.character[idx])
                prev = idx
            results.append("".join(chars))
        return results

    def get_seq_len(self) -> int:
        """Output sequence length (time steps)."""
        return self.seq_len


if __name__ == "__main__":
    model = CRNN(character=DEFAULT_CHARSET, nc=1)
    x = torch.randn(2, 1, 64, 256)
    logits = model(x)
    print(f"Input: {x.shape}  Output: {logits.shape}  SeqLen: {model.get_seq_len()}")
    print(f"Character (sorted): {model.character}")
    print(f"num_classes: {model.num_classes}")
    print(f"Decoded: {model.decode(logits)}")
    total = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    print(f"Parameters: {total:,} (CNN: {cnn_params:,})")
