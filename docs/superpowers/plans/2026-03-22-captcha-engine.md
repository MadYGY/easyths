# Captcha Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 easyths 增加配置驱动的验证码识别引擎切换能力，默认保留数字验证码的 `ddddocr`，复杂验证码改走 SVM。

**Architecture:** 在配置层新增 `captcha_type` 字段并归一化为内部引擎标识，OCR 层按该标识路由到 `ddddocr` 或 SVM。业务操作层继续复用现有 `recognize(control)` 接口，不感知引擎差异。

**Tech Stack:** Python 3.12, toml, pillow, numpy, ddddocr, joblib, opencv-python, pytest

---

### Task 1: 配置字段与归一化

**Files:**
- Modify: `easyths/utils/config.py`
- Modify: `easyths/assets/config_example.toml`
- Test: `test/test_captcha_config.py`

- [ ] **Step 1: 写失败测试，覆盖默认值和复杂验证码值**

```python
def test_captcha_type_defaults_to_ddddocr():
    ...

def test_captcha_type_complex_uses_svm():
    ...
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_captcha_config.py -q`
Expected: FAIL，提示缺少 `captcha_type` 行为或断言不成立

- [ ] **Step 3: 实现最小配置解析**

```python
captcha_type = "数字验证码"

@property
def captcha_engine(self) -> str:
    return "svm" if self.captcha_type == "复杂验证码" else "ddddocr"
```

- [ ] **Step 4: 更新示例配置**

在 `[trading]` 中新增 `captcha_type` 注释和默认值说明

- [ ] **Step 5: 重新运行测试**

Run: `python -m pytest test/test_captcha_config.py -q`
Expected: PASS

### Task 2: OCR 路由测试先行

**Files:**
- Modify: `easyths/utils/captcha_ocr.py`
- Test: `test/test_captcha_ocr.py`

- [ ] **Step 1: 写失败测试，覆盖 ddddocr 路径和 svm 路径选择**

```python
def test_recognize_uses_ddddocr_when_engine_is_default():
    ...

def test_recognize_uses_svm_when_engine_is_complex():
    ...
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_captcha_ocr.py -q`
Expected: FAIL，提示当前实现没有按配置分流

- [ ] **Step 3: 在 OCR 层实现引擎路由**

```python
if project_config_instance.captcha_engine == "svm":
    return self._recognize_with_svm(image)
return self._recognize_with_ddddocr(image)
```

- [ ] **Step 4: 重新运行测试**

Run: `python -m pytest test/test_captcha_ocr.py -q`
Expected: PASS

### Task 3: 最小化合入 SVM 实现

**Files:**
- Modify: `easyths/utils/captcha_ocr.py`
- Create: `easyths/assets/captcha_svm.joblib`

- [ ] **Step 1: 合入最小所需的模型加载和图像预处理代码**

包括：
- 模型加载
- 前景裁剪
- 画布对齐
- 四槽切分
- HOG 特征提取
- SVM 预测

- [ ] **Step 2: 确认不引入调试导出和成功样本保存逻辑**

检查实现中不包含：
- 调试目录写入
- 验证码成功样本持久化
- 字符样本导出

- [ ] **Step 3: 运行 OCR 测试**

Run: `python -m pytest test/test_captcha_ocr.py -q`
Expected: PASS

### Task 4: 依赖与包资源

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 补充 SVM 所需依赖**

在 `server` 依赖组加入：

```toml
"joblib>=1.4.0",
"opencv-python>=4.10.0",
```

- [ ] **Step 2: 确认模型资源会被打包**

Run: `python - <<'PY'\nfrom pathlib import Path\nprint(Path('easyths/assets/captcha_svm.joblib').exists())\nPY`
Expected: `True`

### Task 5: 验证

**Files:**
- Test: `test/test_captcha_config.py`
- Test: `test/test_captcha_ocr.py`

- [ ] **Step 1: 运行新增测试**

Run: `python -m pytest test/test_captcha_config.py test/test_captcha_ocr.py -q`
Expected: PASS

- [ ] **Step 2: 运行全量测试并记录既有失败**

Run: `python -m pytest -q`
Expected: 新增测试通过；已有基线失败仍可能存在，主要是依赖本地同花顺客户端和本地 API 服务的用例
