"""
AI Provider slot system: unified chat interface with pluggable providers.
Usage:
    from . import build_chat_provider, create_provider
    # 通过环境变量
    provider = build_chat_provider(prefix='LLM')
    # 或直接传参构建
    provider = create_provider('oai_http', api_key, base_url, model)

Env keys (by prefix):
    {PREFIX}_PROVIDER: one of [oai_http, openai, openai_sdk, volcengine, gemini, dashscope]
    {PREFIX}_API_KEY
    {PREFIX}_BASE_URL
    {PREFIX}_MODEL
"""
import os
from .base import ChatProvider


def create_provider(provider_name: str, api_key: str, base_url: str | None, model: str) -> ChatProvider | None:
    name = (provider_name or 'oai_http').lower()
    try:
        if name in ("oai_http",):
            from .providers.oai_http import OAIHTTPProvider
            if not base_url:
                base_url = "https://api.openai.com/v1"
            return OAIHTTPProvider(api_key=api_key, base_url=base_url, model=model)
        if name in ("openai", "openai_sdk"):
            from .providers.openai_sdk import OpenAISDKProvider
            return OpenAISDKProvider(api_key=api_key, base_url=base_url, model=model)
        if name == "volcengine":
            from .providers.ark import ArkProvider
            return ArkProvider(api_key=api_key, base_url=base_url, model=model)
        if name == "gemini":
            from .providers.gemini import GeminiProvider
            return GeminiProvider(api_key=api_key, base_url=base_url, model=model)
        if name == "dashscope":
            from .providers.dashscope import DashscopeProvider
            return DashscopeProvider(api_key=api_key, base_url=base_url, model=model)
    except Exception as e:
        print(f"构建 Provider 失败: {name}: {e}")
        return None
    print(f"未知 Provider: {name}")
    return None


def build_chat_provider(prefix: str = 'LLM') -> ChatProvider | None:
    provider_name = os.getenv(f"{prefix}_PROVIDER", "oai_http").lower()
    api_key = os.getenv(f"{prefix}_API_KEY", "")
    base_url = os.getenv(f"{prefix}_BASE_URL", "")
    model = os.getenv(f"{prefix}_MODEL", "")
    return create_provider(provider_name, api_key, base_url, model)
