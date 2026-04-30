"""LLM 与 Embedding 客户端封装。

设计说明：
    阿里云 DashScope 与 sglang 都兼容 OpenAI Chat / Embedding 接口，因此本模块
    统一用 `openai` SDK 作为底层调用，通过 `base_url` 与 `api_key` 区分 provider。

加载顺序（参考美团 WOWService 的 provider 切换思路，做了精简版）：
    1. 加载 `.env` 到进程环境变量
    2. 读取 `config/config.yaml`
    3. 解析 `provider`：环境变量 `provider_env` > yaml `provider` 字段
    4. 取出对应 provider 的 `name / base_url / api_key_env`
    5. 从环境变量中拿真实 api_key
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# 路径与配置加载
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """加载 yaml 配置，并自动注入 .env 中的环境变量。"""
    if DEFAULT_ENV_PATH.exists():
        load_dotenv(DEFAULT_ENV_PATH, override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Provider 解析
# ---------------------------------------------------------------------------

@dataclass
class ProviderResolved:
    """解析后的 provider 信息（供客户端直接使用）。"""

    provider: str          # provider 标识（如 aliyun / sglang）
    model_name: str        # 模型名（如 qwen-turbo / text-embedding-v3）
    base_url: str
    api_key: str
    extra: Dict[str, Any]  # 其他参数（如 max_tokens / request_timeout_sec）


def _resolve_provider(section: Dict[str, Any]) -> ProviderResolved:
    """从 yaml 的某个 section（model 或 embedding）解析当前激活的 provider。"""
    provider_env = section.get("provider_env")
    default_provider = section.get("provider")
    provider = (os.getenv(provider_env) if provider_env else None) or default_provider

    if not provider:
        raise ValueError("配置中未指定 provider，且未设置对应环境变量")

    providers = section.get("providers") or {}
    if provider not in providers:
        raise ValueError(
            f"未在 providers 中找到 {provider}，可选: {list(providers.keys())}"
        )

    cfg = providers[provider]
    api_key_env = cfg.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key:
        # sglang 本地服务可能不需要 key，这里给一个占位避免 SDK 报错
        api_key = "EMPTY"

    extra = {k: v for k, v in section.items() if k not in ("provider", "provider_env", "providers")}
    extra.update({k: v for k, v in cfg.items() if k not in ("name", "api_key_env", "base_url")})

    return ProviderResolved(
        provider=provider,
        model_name=cfg["name"],
        base_url=cfg["base_url"],
        api_key=api_key,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# LLM 客户端
# ---------------------------------------------------------------------------

class LLMClient:
    """对话补全客户端。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config()
        self._resolved = _resolve_provider(cfg["model"])
        self._client = OpenAI(
            api_key=self._resolved.api_key,
            base_url=self._resolved.base_url,
            timeout=self._resolved.extra.get("request_timeout_sec", 90),
            max_retries=self._resolved.extra.get("max_retries", 1),
        )

    @property
    def provider(self) -> str:
        return self._resolved.provider

    @property
    def model_name(self) -> str:
        return self._resolved.model_name

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """发起一次 chat 调用并返回首条回复的文本。"""
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._resolved.extra.get("temperature", 0),
            "max_tokens": max_tokens or self._resolved.extra.get("max_tokens", 2048),
        }
        params.update(kwargs)

        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Embedding 客户端
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """文本向量化客户端。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config()
        self._resolved = _resolve_provider(cfg["embedding"])
        self._client = OpenAI(
            api_key=self._resolved.api_key,
            base_url=self._resolved.base_url,
            timeout=self._resolved.extra.get("request_timeout_sec", 60),
        )

    @property
    def provider(self) -> str:
        return self._resolved.provider

    @property
    def model_name(self) -> str:
        return self._resolved.model_name

    def embed(self, text: str) -> List[float]:
        """对单条文本生成向量。"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。"""
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# 工厂函数（推荐外部入口）
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    return LLMClient()


def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()
