"""guitar-agent 项目入口。

当前阶段：MVP-0
    本入口仅完成「LLM 直连 + 系统提示词扮演吉他老师」的最小闭环。
    Multi-Agent 编排、认知诊断、分层记忆、多模态感知尚未接入。

演进路线（详见 PROJECT_GOAL.md 第九节）：
    Phase 1: 在 _build_response 中替换为 Multi-Agent 编排（Diagnose / Planner / Tutor / Critic）
    Phase 2: 在启动时加载学习者画像，注入到 system prompt（学生层 Context）
    Phase 3: 增加 :load <pdf_path> 命令，触发 Perceptual Module 解析谱子
"""

from __future__ import annotations

import sys
from typing import Dict, List

from src.llm_client import LLMClient, get_llm_client


# ---------------------------------------------------------------------------
# 系统提示词（MVP-0 临时版，Phase 1 之后由 Tutor Agent 接管）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是 guitar-agent 项目的吉他教学助教，名叫 Fret（取自吉他「品柱」一词）。

角色设定：
1. 你具备扎实的乐理与吉他演奏知识，能根据学生水平调整讲解深度。
2. 回答简洁、有条理，复杂概念用比喻或类比，避免长篇大论。
3. 学生提问模糊时，先主动反问以了解他的水平、学习目标或当前练习内容，再给建议。
4. 鼓励学生表达练习困惑，而不是只回答标准问题。

诚实约束（非常重要）：
- 你目前未接入完整的认知诊断系统，无法精确建模学生水平。
- 当学生信息不足时，明确告知"我目前对你的水平掌握有限"，再给出基于经验的建议。
- 不要假装记得用户的历史，跨会话记忆当前还没接入。
"""

EXIT_COMMANDS = {":q", ":quit", "exit", "quit"}
HELP_COMMANDS = {":h", ":help"}


# ---------------------------------------------------------------------------
# REPL 主循环
# ---------------------------------------------------------------------------

def _print_banner(client: LLMClient) -> None:
    bar = "=" * 60
    print(bar)
    print("  guitar-agent  -  自适应吉他教学 Agent")
    print(f"  Stage: MVP-0   |   LLM: {client.provider}/{client.model_name}")
    print(bar)
    print("欢迎使用 guitar-agent，我是你的吉他老师助教 Fret。")
    print(f"输入问题开始对话；输入 {' / '.join(sorted(EXIT_COMMANDS))} 退出，:h 查看帮助。\n")


def _print_help() -> None:
    print()
    print("可用命令：")
    print("  :h / :help     查看帮助")
    print("  :q / :quit     退出")
    print("  :reset         清空对话历史，重新开始一轮")
    print()


def _build_response(client: LLMClient, history: List[Dict[str, str]]) -> str:
    """生成助教回复。

    MVP-0 直接调 LLM。Phase 1 替换为 Multi-Agent 编排入口。
    """
    return client.chat(history)


def repl() -> int:
    try:
        client = get_llm_client()
    except Exception as exc:  # noqa: BLE001
        print(f"[初始化失败] {exc}")
        print("请检查 config/config.yaml 与 .env 中的 API Key 是否配置正确。")
        return 1

    _print_banner(client)

    history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见，继续保持练习。")
            return 0

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in EXIT_COMMANDS:
            print("再见，继续保持练习。")
            return 0

        if cmd in HELP_COMMANDS:
            _print_help()
            continue

        if cmd == ":reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[对话历史已清空]\n")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            reply = _build_response(client, history)
        except Exception as exc:  # noqa: BLE001
            print(f"[模型调用失败] {exc}\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"\nFret: {reply}\n")


if __name__ == "__main__":
    sys.exit(repl())
