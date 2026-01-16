from typing import List, Dict, Any, Optional
from ..base import ChatProvider


class OpenAISDKProvider(ChatProvider):
    """OpenAI 官方 SDK provider。"""

    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        super().__init__(api_key, base_url, model)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("未安装 openai，请先安装") from e
        # base_url 可为 None，OpenAI SDK 会使用默认 https://api.openai.com/v1
        self.client = OpenAI(api_key=api_key, base_url=(self.base_url or None))

    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # 兼容文本和多模态
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            payload["messages"] = [{"role": "user", "content": content}]
        else:
            payload["messages"] = messages or [{"role": "user", "content": prompt}]

        try:
            resp = self.client.chat.completions.create(**payload)
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI SDK 调用失败: {e}"

