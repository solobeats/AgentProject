from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

# 全局会话存储字典
# Key: session_id (即微信的 from_user)
# Value: BaseChatMessageHistory 对象
_session_storage = {}

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    一个基于内存字典的聊天消息历史记录存储类。
    用于在单个应用实例的生命周期内暂存对话历史。
    """
    def __init__(self):
        self.messages = []

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        self._messages = value

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    根据 session_id 获取用户的对话历史。
    如果不存在，则创建一个新的。
    """
    if session_id not in _session_storage:
        _session_storage[session_id] = InMemoryChatMessageHistory()
    return _session_storage[session_id]
