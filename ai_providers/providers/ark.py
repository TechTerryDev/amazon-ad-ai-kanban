from typing import List, Dict, Any, Optional
from ..base import ChatProvider


class ArkProvider(ChatProvider):
    """火山引擎 Ark SDK provider（OpenAI 兼容格式）。"""

    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        super().__init__(api_key, base_url, model)
        try:
            from volcenginesdkarkruntime import Ark  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "未安装 volcenginesdkarkruntime（建议：pip install 'volcengine-python-sdk[ark]'；或 pip install volcenginesdkarkruntime）"
            ) from e
        self.client = Ark(api_key=api_key)

    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            payload["messages"] = [{"role": "user", "content": content}]

        try:
            resp = self.client.chat.completions.create(timeout=int(timeout or 180), **payload)
        except TypeError:
            resp = self.client.chat.completions.create(**payload)
        if hasattr(resp, 'choices') and resp.choices:
            return resp.choices[0].message.content
        return str(resp)
