"""
TESrACT LLM routing integration tests.

Exercises task classification, direct routing, and the full agent graph
(including Groq tool calling).

Usage:
    python test_routing.py
    python main.py test
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import TypeAlias

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

import llm_router
from llm_router import (
    ROUTING_MODE,
    build_groq_llm,
    choose_route,
    classify_task,
    colab_configured,
    colab_is_healthy,
    route_and_invoke,
)
import actions

ClassificationCase: TypeAlias = tuple[str, str, str] | tuple[str, str, str, int]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _result(label: str, preferred: str, actual: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"       Preferred route : {preferred.upper()}")
    print(f"       Backend used    : {actual.upper()}")
    if detail:
        print(f"       Detail          : {detail}")
    return ok


# ---------------------------------------------------------------------------
# 1. Classification (offline — no API keys required)
# ---------------------------------------------------------------------------

def test_classification() -> bool:
    _header("1. Task classification (offline)")

    cases: list[ClassificationCase] = [
        ("What time is it?", "groq", "light greeting / time query"),
        ("Hello, how are you?", "groq", "short casual message"),
        ("What is the capital of France?", "groq", "short factual query"),
        ("Who was the first president of the United States?", "groq", "simple who-question"),
        (
            "Analyze in detail the pros and cons of microservices architecture step by step",
            "colab",
            "heavy keyword + long analysis request",
        ),
        (
            "Write a detailed 500-word essay on the causes of World War I",
            "colab",
            "long-form writing with word-count hint",
        ),
        (
            "Provide a comprehensive analysis of climate change impacts on agriculture",
            "colab",
            "deep analysis request",
        ),
        ("use colab for a deep analysis of this API design", "colab", "explicit Colab request"),
        ("Search the web for the latest news on AI", "groq", "tool-friendly web search"),
        ("What is the capital of France?", "groq", "factual query even with long thread", 15),
    ]

    all_ok = True
    for case in cases:
        if len(case) == 4:
            text, expected, reason, history_len = case
        else:
            text, expected, reason = case
            history_len = 0
        got = classify_task(text, history_len=history_len)
        ok = got == expected
        all_ok = _result(f"Classify: {reason}", expected, got, ok) and all_ok
    return all_ok


# ---------------------------------------------------------------------------
# 2. Direct routing invoke (requires GROQ_API_KEY; Colab optional)
# ---------------------------------------------------------------------------

def _run_direct_route(label: str, user_text: str, expect_preferred: str) -> bool:
    if not os.getenv("GROQ_API_KEY"):
        print(f"\n[SKIP] {label} — GROQ_API_KEY not set")
        return True

    sys_msg = SystemMessage(content="You are TESrACT. Reply in one short sentence, Sir.")
    history: list[BaseMessage] = [HumanMessage(content=user_text)]
    preferred = choose_route(user_text, history_len=len(history))

    llm = build_groq_llm()
    llm_tools = llm.bind_tools(actions.tools)

    try:
        response, actual = route_and_invoke(
            llm_with_tools=llm_tools,
            sys_msg=sys_msg,
            history=history,
            control_allowed=False,
            user_text=user_text,
            history_len=len(history),
        )
        snippet = str(response.content or "")[:80].replace("\n", " ")
        ok = preferred == expect_preferred
        # Heavy tasks may fall back to Groq when Colab is offline — that's expected.
        if expect_preferred == "colab" and actual == "groq" and not colab_is_healthy(log_failure=False):
            ok = True
            detail = f"Colab offline — Groq fallback OK | reply: {snippet!r}"
        elif expect_preferred == "colab" and actual == "colab":
            detail = f"Colab served request | reply: {snippet!r}"
        else:
            detail = f"reply: {snippet!r}"
        return _result(label, preferred, actual, ok, detail)
    except Exception as exc:
        print(f"\n[FAIL] {label}: {exc}")
        traceback.print_exc()
        return False


def test_direct_routing() -> bool:
    _header("2. Direct routing invoke (live API)")

    ok_light = _run_direct_route(
        "Light task → Groq",
        "What time is it right now?",
        "groq",
    )
    ok_heavy = _run_direct_route(
        "Heavy task → Colab (or Groq fallback)",
        "Provide a comprehensive step-by-step analysis of REST vs GraphQL API design",
        "colab",
    )
    return ok_light and ok_heavy


# ---------------------------------------------------------------------------
# 3. Full agent graph + Groq tool calling
# ---------------------------------------------------------------------------

def test_agent_tool_calling() -> bool:
    _header("3. Full agent graph — Groq tool calling")

    if not os.getenv("GROQ_API_KEY"):
        print("\n[SKIP] Agent tool test — GROQ_API_KEY not set")
        return True

    # Import here so FastAPI / hardware init only runs when needed.
    from main import TESRACT_SYSTEM_PROMPT, compiled_app

    thread_id = "routing_test_tools"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    prompt = "What is the current time? You must use the get_current_time tool to answer."

    try:
        compiled_app.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "control_allowed": True,
            },
            config=config,
        )
        state = compiled_app.get_state(config)
        state_values = state.values if state else None
        messages = (state_values or {}).get("messages", [])

        tool_called = any(isinstance(m, ToolMessage) for m in messages)
        tool_names = [
            str(getattr(m, "name", "") or "")
            for m in messages
            if isinstance(m, ToolMessage)
        ]
        backend = (llm_router.last_route_used or "unknown").upper()
        final_reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final_reply = str(msg.content)[:100]
                break

        ok = tool_called and backend == "GROQ"
        detail = (
            f"tools invoked: {tool_names or 'none'} | "
            f"final reply: {final_reply!r}"
        )
        return _result(
            "Agent uses Groq + get_current_time tool",
            "groq",
            backend.lower(),
            ok,
            detail,
        )
    except Exception as exc:
        print(f"\n[FAIL] Agent tool test: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_tests() -> int:
    print("TESrACT LLM Routing Test Suite")
    print(f"  LLM_ROUTING_MODE : {ROUTING_MODE}")
    print(f"  GROQ_API_KEY     : {'set' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
    print(f"  COLAB_LLM_URL    : {'set' if colab_configured() else 'not set'}")
    if colab_configured():
        print(f"  Colab health     : {'online' if colab_is_healthy() else 'offline'}")

    results = [
        test_classification(),
        test_direct_routing(),
        test_agent_tool_calling(),
    ]

    _header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"\n  {passed}/{total} test sections passed")

    if passed == total:
        print("\n  All routing tests passed ✓")
        return 0
    print("\n  Some tests failed ✗")
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())