from typing import List, Dict, Any, Optional
from ..base import ChatProvider


class DashscopeProvider(ChatProvider):
    """阿里 DashScope provider（MultiModalConversation）。"""

    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        super().__init__(api_key, base_url, model)
        try:
            import dashscope  # type: ignore
        except Exception as e:
            raise RuntimeError("未安装 dashscope，请先安装") from e
        import os
        os.environ['DASHSCOPE_API_KEY'] = api_key
        self.dashscope = dashscope

    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        from dashscope import MultiModalConversation  # type: ignore
        msgs = messages or [{"role": "user", "content": [{"text": prompt}]}]
        resp = MultiModalConversation.call(
            model=self.model,
            messages=msgs,
            result_format='message',
            max_tokens=max_tokens or 1024,
        )
        try:
            return resp["output"]["choices"][0]["message"]["content"][0]["text"]
        except Exception:
            return str(resp)

