from PIL import Image

import easyths.utils.captcha_ocr as captcha_ocr


def test_recognize_uses_ddddocr_when_engine_is_default(monkeypatch) -> None:
    calls = {"svm": 0, "ddddocr": 0}

    monkeypatch.setattr(captcha_ocr.project_config_instance, "captcha_type", "数字验证码")
    monkeypatch.setattr(
        captcha_ocr,
        "_grab_control_image",
        lambda control: Image.new("RGB", (12, 12), "white"),
    )

    def fake_svm(self, image):
        calls["svm"] += 1
        return "ABCD"

    class FakeDdddOcr:
        def classification(self, image):
            calls["ddddocr"] += 1
            return "1234"

    monkeypatch.setattr(captcha_ocr.CaptchaOCR, "_recognize_with_svm", fake_svm)
    monkeypatch.setattr(captcha_ocr, "_get_ddddocr_instance", lambda: FakeDdddOcr())

    result = captcha_ocr.CaptchaOCR().recognize(object())

    assert result == "1234"
    assert calls == {"svm": 0, "ddddocr": 1}


def test_recognize_uses_svm_when_engine_is_complex(monkeypatch) -> None:
    calls = {"svm": 0, "ddddocr": 0}

    monkeypatch.setattr(captcha_ocr.project_config_instance, "captcha_type", "复杂验证码")
    monkeypatch.setattr(
        captcha_ocr,
        "_grab_control_image",
        lambda control: Image.new("RGB", (12, 12), "white"),
    )

    def fake_svm(self, image):
        calls["svm"] += 1
        return "Ab9Z"

    class FakeDdddOcr:
        def classification(self, image):
            calls["ddddocr"] += 1
            return "1234"

    monkeypatch.setattr(captcha_ocr.CaptchaOCR, "_recognize_with_svm", fake_svm)
    monkeypatch.setattr(captcha_ocr, "_get_ddddocr_instance", lambda: FakeDdddOcr())

    result = captcha_ocr.CaptchaOCR().recognize(object())

    assert result == "Ab9Z"
    assert calls == {"svm": 1, "ddddocr": 0}
