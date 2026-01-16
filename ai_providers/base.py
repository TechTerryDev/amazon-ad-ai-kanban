from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ChatProvider(ABC):
    """统一聊天生成接口。"""

    def __init__(self, api_key: str, base_url: Optional[str], model: str):
        self.api_key = api_key
        self.base_url = (base_url or '').rstrip('/') if base_url else ''
        self.model = model

    @abstractmethod
    def generate(self,
                 prompt: str,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 images: Optional[List[str]] = None,
                 timeout: int = 180) -> str:
        """返回生成的文本内容字符串。"""
        raise NotImplementedError

