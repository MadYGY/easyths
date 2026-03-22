from pathlib import Path

from easyths.utils.config import ProjectConfig


def _write_config(tmp_path: Path, trading_lines: list[str]) -> Path:
    config_path = tmp_path / "config.toml"
    content = "\n".join(["[trading]", *trading_lines, ""])
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_captcha_type_defaults_to_ddddocr(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, ['app_path = "C:/ths/xiadan.exe"'])

    config = ProjectConfig()
    config.update_from_toml_file(str(config_path))

    assert config.captcha_type == "数字验证码"
    assert config.captcha_engine == "ddddocr"


def test_captcha_type_complex_uses_svm(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        [
            'app_path = "C:/ths/xiadan.exe"',
            'captcha_type = "复杂验证码"',
        ],
    )

    config = ProjectConfig()
    config.update_from_toml_file(str(config_path))

    assert config.captcha_type == "复杂验证码"
    assert config.captcha_engine == "svm"
