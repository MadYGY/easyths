import os
from pathlib import Path

import toml


class ProjectConfig:
    # App 配置
    app_name = os.getenv("APP_NAME", "同花顺交易自动化程序")
    app_version = os.getenv("APP_VERSION", "1.0.0")

    # Trading 配置
    trading_app_path = os.getenv(
        "TRADING_APP_PATH",
        "C:/同花顺远航版/transaction/xiadan.exe",
    )
    captcha_type = os.getenv("CAPTCHA_TYPE", "数字验证码")

    # Queue 配置
    queue_max_size = int(os.getenv("QUEUE_MAX_SIZE", 1000))
    queue_priority_levels = int(os.getenv("QUEUE_PRIORITY_LEVELS", 5))
    queue_batch_size = int(os.getenv("QUEUE_BATCH_SIZE", 10))

    # API 配置
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 7648))
    api_rate_limit = int(os.getenv("API_RATE_LIMIT", 10))
    api_cors_origins = os.getenv("API_CORS_ORIGINS", "*")
    api_key = os.getenv("API_KEY", None)
    api_ip_whitelist = os.getenv("API_IP_WHITELIST", None)
    api_mcp_server_type = os.getenv("API_MCP_SERVER_TYPE", "streamable-http")

    # Logging 配置
    logging_level = os.getenv("LOGGING_LEVEL", "INFO")
    logging_file = (
        str(Path("~/easyths/log.txt").expanduser())
        if os.getenv("LOGGING_FILE", "") == ""
        else os.getenv("LOGGING_FILE")
    )

    def update_from_toml_file(
        self,
        toml_file_path: str,
        exe_path: str | None = None,
    ) -> None:
        """从 TOML 配置文件更新配置。"""
        config = toml.load(toml_file_path)

        if "app" in config:
            app_config = config["app"]
            if "name" in app_config:
                self.app_name = app_config["name"]
            if "version" in app_config:
                self.app_version = app_config["version"]

        if "trading" in config:
            trading_config = config["trading"]
            if "app_path" in trading_config:
                self.trading_app_path = trading_config["app_path"]
            if "captcha_type" in trading_config:
                self.captcha_type = trading_config["captcha_type"] or "数字验证码"

        if "queue" in config:
            queue_config = config["queue"]
            if "max_size" in queue_config:
                self.queue_max_size = queue_config["max_size"]
            if "priority_levels" in queue_config:
                self.queue_priority_levels = queue_config["priority_levels"]
            if "batch_size" in queue_config:
                self.queue_batch_size = queue_config["batch_size"]

        if "api" in config:
            api_config = config["api"]
            if "host" in api_config:
                self.api_host = api_config["host"]
            if "port" in api_config:
                self.api_port = api_config["port"]
            if "rate_limit" in api_config:
                self.api_rate_limit = api_config["rate_limit"]
            if "cors_origins" in api_config:
                self.api_cors_origins = api_config["cors_origins"]
            if "key" in api_config:
                self.api_key = api_config["key"] or None
            if "ip_whitelist" in api_config:
                self.api_ip_whitelist = api_config["ip_whitelist"] or None
            if "mcp_server_type" in api_config:
                valid_types = ["http", "streamable-http", "sse"]
                mcp_type = api_config["mcp_server_type"]
                if mcp_type in valid_types:
                    self.api_mcp_server_type = mcp_type
                else:
                    raise ValueError(
                        f"无效的 mcp_server_type: {mcp_type}，可选值: {valid_types}"
                    )

        if "logging" in config:
            logging_config = config["logging"]
            if "level" in logging_config:
                self.logging_level = logging_config["level"]
            if "file" in logging_config:
                self.logging_file = (
                    str(Path("~/easyths/log.txt").expanduser())
                    if logging_config["file"] == ""
                    else logging_config["file"]
                )

        if exe_path:
            self.trading_app_path = exe_path

    @property
    def captcha_engine(self) -> str:
        normalized = str(self.captcha_type or "").strip().lower()
        if normalized in {"复杂验证码", "svm"}:
            return "svm"
        return "ddddocr"

    @property
    def api_ip_whitelist_list(self) -> list[str] | None:
        """获取 IP 白名单列表。"""
        if not self.api_ip_whitelist:
            return None
        return [ip.strip() for ip in self.api_ip_whitelist.split(",") if ip.strip()]

    @property
    def api_cors_origins_list(self) -> list[str]:
        """获取 CORS 允许的源列表。"""
        if not self.api_cors_origins:
            return ["*"]
        if self.api_cors_origins == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.api_cors_origins.split(",")
            if origin.strip()
        ]


project_config_instance = ProjectConfig()
