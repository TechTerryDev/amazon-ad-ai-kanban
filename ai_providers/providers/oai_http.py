import json
import urllib.error
import urllib.request
from typing import List, Dict, Any, Optional
from ..base import ChatProvider


class OAIHTTPProvider(ChatProvider):
    """OpenAI 兼容 HTTP provider (/chat/completions)。"""

    def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> tuple[int, str]:
        """
        只用标准库发送 HTTP JSON 请求，避免额外依赖（requests）。
        返回: (status_code, response_text)
        """
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=int(timeout or 180)) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                text = resp.read().decode("utf-8", errors="replace")
                return status, text
        except urllib.error.HTTPError as e:
            try:
                return int(e.code or 0), (e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e))
            except Exception:
                return int(e.code or 0), str(e)
        except Exception as e:
            return 0, f"请求失败: {e}"

    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        api_url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # 简单 multimodal 支持（将本地图片路径/URL 作为 image_url 传入）
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            payload["messages"] = [{"role": "user", "content": content}]

        status, text = self._post_json(api_url, headers=headers, payload=payload, timeout=timeout)
        if status != 200:
            return f"HTTP {status}: {text[:500]}"
        try:
            data = json.loads(text)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"响应解析失败: {e} | {text[:500]}"
