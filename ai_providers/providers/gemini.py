import json
from typing import List, Dict, Any, Optional
from ..base import ChatProvider


class GeminiProvider(ChatProvider):
    """Google Gemini provider。优先使用官方 SDK，失败则回退到 HTTP v1beta generateContent。"""

    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        super().__init__(api_key, base_url, model)
        self._sdk = None
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            self._sdk = genai
        except Exception:
            self._sdk = None

    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        if self._sdk is not None:
            try:
                genai = self._sdk
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(prompt)
                if hasattr(resp, 'text'):
                    return resp.text
                # SDK 结构变化时的兼容
                return str(resp)
            except Exception as e:
                return f"Gemini SDK 调用失败: {e}"

        # HTTP 回退
        import requests
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base.rstrip('/')}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}: {resp.text[:500]}"
            data = resp.json()
            # 解析 text
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(data, ensure_ascii=False)[:800]
        except Exception as e:
            return f"Gemini HTTP 调用失败: {e}"

