"""LLM 与 Embedding 联通性测试。

直接运行：
    python tests/test_llm.py
预期输出：
    [LLM]    provider=aliyun model=qwen-turbo
    >>> 你好...（模型回复）
    [Embed]  provider=aliyun model=text-embedding-v3 dim=1024
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import get_embedding_client, get_llm_client


def test_llm_chat() -> None:
    client = get_llm_client()
    print(f"[LLM]    provider={client.provider} model={client.model_name}")

    messages = [
        {"role": "system", "content": "你是一位资深吉他老师，回答简洁。"},
        {"role": "user", "content": "C 大调音阶在第一把位包含哪些音？请用一句话回答。"},
    ]
    reply = client.chat(messages)
    print(">>>", reply)
    assert reply.strip(), "LLM 返回为空"


def test_embedding() -> None:
    client = get_embedding_client()
    vector = client.embed("Am 和弦是吉他的基础和弦之一。")
    print(f"[Embed]  provider={client.provider} model={client.model_name} dim={len(vector)}")
    assert len(vector) > 0, "Embedding 返回向量为空"


def main() -> None:
    print("=" * 60)
    print("FretSense - LLM 联通性测试")
    print("=" * 60)
    test_llm_chat()
    print()
    test_embedding()
    print()
    print("✅ 全部测试通过")


if __name__ == "__main__":
    main()
