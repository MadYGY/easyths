# 验证码识别模型

基于 ResNet34 + CTC 的验证码识别模型，专门用于识别同花顺交易软件的验证码。

## 目录结构

```
captcha_model/
├── config.yaml              # 训练配置文件
├── requirements.txt         # Python 依赖
├── data_generate.py         # 数据集生成脚本（仿同花顺风格）
├── data_generate2.py        # 数据集生成脚本（备选方案）
├── data_generate3.py        # 数据集生成脚本（备选方案）
├── train.py                 # 模型训练脚本
├── eval.py                  # 模型评估脚本
├── export_onnx.py           # 导出 ONNX 模型
├── infer_onnx.py            # ONNX 推理脚本
├── eval_ddddocr.py          # ddddocr 评估对比脚本
├── infer_read.py            # PyTorch 推理脚本
├── utils.py                 # 工具函数
├── fonts/                   # 字体文件目录
│   ├── arial.ttf
│   ├── Roboto-VariableFont_wdth,wght.ttf
│   └── ...
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── resnet_ocr.py        # ResNetOCR 模型
│   ├── attention_modules.py # 注意力模块
│   └── loss.py              # 损失函数和数据集
├── data/                    # 数据集目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── outputs/                 # 训练输出
│   ├── best_model.pt        # 最佳模型
│   ├── last_model.pt        # 最终模型
│   └── train_log.csv        # 训练日志
└── onnx_model/              # ONNX 模型输出
    └── captcha_ocr.onnx     # 导出的 ONNX 模型
```

## 环境准备

### 安装依赖

```bash
# 使用 uv 安装依赖
cd captcha_model
uv pip install -r requirements.txt
```

### 依赖列表

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- onnx >= 1.14.0
- onnxruntime >= 1.15.0
- Pillow >= 10.0.0
- PyYAML >= 6.0
- numpy >= 1.24.0

## 数据集准备

### 1. 准备字体文件

将字体文件放入 `fonts/` 目录。推荐使用多种字体以增加数据多样性：

```bash
# 查看已有字体
ls fonts/
```

### 2. 生成数据集

使用 `data_generate.py` 生成仿同花顺风格的验证码数据集：

```bash
# 生成训练集（10000 张）
python data_generate.py --num_samples 10000 --output_dir data/train

# 生成验证集（2000 张）
python data_generate.py --num_samples 2000 --output_dir data/val

# 生成测试集（1000 张）
python data_generate.py --num_samples 1000 --output_dir data/test
```

### 3. 数据集格式

生成的图片命名格式：`{label}_{uuid}.png`

```
data/train/
├── aB3x_abc123.png
├── Xy9k_def456.png
└── ...
```

### 4. 数据生成参数

```bash
python data_generate.py \
    --num_samples 10000 \       # 样本数量
    --output_dir data/train \   # 输出目录
    --charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" \  # 字符集
    --length 4 \                # 验证码长度
    --font_dir fonts            # 字体目录
```

### 5. 同花顺验证码特征

本项目的数据生成器精确模拟了同花顺验证码的特征：

- 尺寸：84x38 像素
- 背景：纯白 RGB(255, 255, 255)
- 顶部边线：淡蓝色 RGB(219, 233, 242)
- 文字颜色：天蓝色 RGB(0, 160, 233)
- 字符数：4 位
- 无旋转、无干扰线、无噪点

## 模型训练

### 1. 配置文件

编辑 `config.yaml` 配置训练参数：

```yaml
Global:
  use_gpu: true
  character: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  img_h: 64
  img_w: 256
  output_dir: "outputs"
  resume: true

Architecture:
  name: ResNetOCR
  backbone:
    name: ResNet34
    pretrained: true
    freeze_backbone: false

Optimizer:
  name: AdamW
  lr: 0.0001
  weight_decay: 0.00001

Training:
  batch_size: 64
  num_workers: 4
  save_interval: 20
  early_stopping:
    patience: 20
    min_delta: 0.001

Dataset:
  train_dir: data/train
  val_dir: data/val
```

### 2. 开始训练

```bash
# 使用默认配置
python train.py

# 使用自定义配置
python train.py --config custom_config.yaml
```

### 3. 训练输出

```
outputs/
├── best_model.pt      # 最佳模型（根据验证集序列准确率）
├── last_model.pt      # 最终模型
├── epoch_20.pt        # 周期性检查点
├── epoch_40.pt
└── train_log.csv      # 训练日志
```

### 4. 训练日志

```csv
epoch,train_loss,val_loss,char_acc,seq_acc,lr,time
1,2.3456,1.8765,0.4521,0.2341,0.000020,12.5
2,1.6543,1.2345,0.6234,0.4567,0.000040,11.8
...
```

### 5. 恢复训练

设置 `resume: true` 可从最佳模型恢复训练：

```yaml
Global:
  resume: true
```

## 模型评估

```bash
# 使用默认配置评估
python eval.py

# 指定模型和测试目录
python eval.py --model outputs/best_model.pt --test_dir data/test

# 指定设备
python eval.py --device cuda --batch_size 64
```

### 评估指标

- **字符准确率 (Char Acc)**：单个字符的识别准确率
- **序列准确率 (Seq Acc)**：整个验证码完全正确的比例
- **平均延迟 (Latency)**：单张图片推理时间

## 导出 ONNX 模型

### 1. 导出模型

```bash
# 使用默认配置导出
python export_onnx.py

# 指定模型路径和输出目录
python export_onnx.py --model outputs/best_model.pt --output onnx_model

# 指定 ONNX opset 版本
python export_onnx.py --opset 18
```

### 2. 导出参数

```bash
python export_onnx.py \
    --config config.yaml \        # 配置文件
    --model outputs/best_model.pt \  # PyTorch 模型路径
    --output onnx_model \         # 输出目录
    --name captcha_ocr.onnx \     # 输出文件名
    --opset 18                    # ONNX opset 版本
```

### 3. 模型元数据

导出的 ONNX 模型会自动嵌入以下元数据：

- `character`：字符集
- `img_h`：图片高度
- `img_w`：图片宽度

这意味着推理时不需要额外的配置文件，所有参数都从模型元数据中读取。

## ONNX 推理

### 1. 单张图片推理

```bash
python infer_onnx.py --model onnx_model/captcha_ocr.onnx --image captcha.png
```

### 2. 批量推理

```bash
# 推理整个目录
python infer_onnx.py --model onnx_model/captcha_ocr.onnx --dir data/test
```

### 3. 性能基准测试

```bash
python infer_onnx.py --model onnx_model/captcha_ocr.onnx --benchmark
```

### 4. 使用 GPU 推理

```bash
python infer_onnx.py --model onnx_model/captcha_ocr.onnx --providers CUDAExecutionProvider
```

## 部署到 EasyTHS

### 1. 复制模型文件

将导出的 ONNX 模型复制到 EasyTHS 项目的资源目录：

```bash
# 复制到默认位置
cp onnx_model/captcha_ocr.onnx ../easyths/assets/onnx_model/
cp onnx_model/captcha_ocr.onnx.data ../easyths/assets/onnx_model/  # 如果存在
```

### 2. 配置模型路径（可选）

在 `config.toml` 中指定自定义模型路径：

```toml
[app]
onnx_model_dir = "path/to/your/onnx_model"
```

### 3. 使用验证码识别

EasyTHS 中的 `CaptchaOCR` 类会自动加载 ONNX 模型：

```python
from easyths.utils import get_captcha_ocr_server

# 获取 OCR 服务实例
ocr = get_captcha_ocr_server()

# 识别验证码控件
result = ocr.recognize(captcha_control)
print(f"验证码: {result}")
```

### 4. 模型加载逻辑

EasyTHS 的模型加载优先级：

1. 如果 `config.toml` 中配置了 `onnx_model_dir`，使用配置的路径
2. 否则使用默认路径 `easyths/assets/onnx_model/captcha_ocr.onnx`

## 模型架构

### ResNetOCR

```
Input (B, 3, 64, 256)
    ↓
ResNet34 Backbone (pretrained)
    ├── conv1 + bn1 + relu + maxpool
    ├── layer1 (frozen)
    ├── layer2 (frozen)
    └── layer3 + ECA attention (trainable)
    ↓
Spatial Attention Pooling
    ↓
Sequence Encoder (B, 256, 32)
    ↓
CTC Head (2 conv layers + residual)
    ↓
Output (B, 63, 32)
    ↓
CTC Greedy Decoding
    ↓
Predicted Text
```

### 关键特性

- **预训练 ResNet34**：使用 ImageNet 预训练权重加速收敛
- **ECA 注意力**：增强通道注意力，提升特征表达
- **空间注意力池化**：自适应序列编码
- **CTC 损失**：无需字符级标注，支持变长输出
- **早停机制**：防止过拟合

## 常见问题

### Q: 训练时 loss 不下降？

1. 检查数据集是否正确生成
2. 尝试降低学习率
3. 增加训练数据量
4. 检查字符集配置是否正确

### Q: 识别准确率低？

1. 确保训练数据与真实验证码风格一致
2. 增加字体多样性
3. 使用真实样本进行微调
4. 调整数据增强参数

### Q: ONNX 导出失败？

1. 确保 PyTorch 版本 >= 2.0
2. 尝试降低 opset 版本（如 14 或 15）
3. 检查模型是否有不支持的操作

### Q: 推理速度慢？

1. 使用 GPU 推理（CUDAExecutionProvider）
2. 确保模型已正确导出
3. 检查图片预处理是否高效

## 参考资料

- [CTC Loss 解释](https://distill.pub/2017/ctc/)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)
- [ONNX Runtime](https://onnxruntime.ai/)

## 作者

- 作者：noimank
- 邮箱：noimank@163.com
