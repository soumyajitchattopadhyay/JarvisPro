"""
TESrACT LLM routing integration tests.

Exercises task classification, direct routing, and the full agent graph
(including local Ollama + Groq tool calling fallback).

Usage:
    python test_routing.py
    python main.py test
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Union
from typing_extensions import TypeAlias

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
    ollama_is_healthy,
    local_status,
    hybrid_status,
    route_and_invoke,
)
import actions

ClassificationCase: TypeAlias = Union[tuple[str, str, str], tuple[str, str, str, int]]


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


def test_classification() -> bool:
    _header("1. Task classification (offline)")

    cases: list[ClassificationCase] = [
        ("What time is it?", "local", "light greeting / time query"),
        ("Hello, how are you?", "local", "short casual message"),
        ("What is the capital of France?", "local", "short factual query"),
        ("Who was the first president of the United States?", "local", "simple who-question"),
        (
            "Analyze in detail the pros and cons of microservices architecture step by step",
            "local",
            "heavy keyword + long analysis request",
        ),
        (
            "Write a detailed 500-word essay on the causes of World War I",
            "local",
            "long-form writing with word-count hint",
        ),
        (
            "Provide a comprehensive analysis of climate change impacts on agriculture",
            "local",
            "deep analysis request",
        ),
        ("use colab for a deep analysis of this API design", "colab", "explicit Colab request"),
        ("Search the web for the latest news on AI", "local", "tool-friendly web search"),
        ("Generate an image of a cyberpunk robot", "local", "image generation tool task"),
        ("Read the main.py file and summarize it", "local", "file read tool task"),
        ("Recall what we discussed about AI last time", "local", "memory recall tool task"),
        ("Remember that my preferred language is Python", "local", "memory save tool task"),
        ("What is the capital of France?", "local", "factual query even with long thread", 15),
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


def _run_direct_route(label: str, user_text: str, expect_preferred: str) -> bool:
    has_local = ollama_is_healthy(log_failure=False)
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    if not has_local and not has_groq:
        print(f"\n[SKIP] {label} — Ollama offline and GROQ_API_KEY not set")
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
            tools=actions.tools,
        )
        snippet = str(response.content or "")[:80].replace("\n", " ")
        ok = preferred == expect_preferred
        has_remote = hybrid_status().get("remote_mac_healthy", False)
        if expect_preferred == "local" and actual in ("groq", "colab") and not has_local and not has_remote:
            ok = True
            detail = f"Ollama offline — {actual.upper()} fallback OK | reply: {snippet!r}"
        elif expect_preferred == "local" and actual == "local":
            detail = f"Local Ollama served | reply: {snippet!r}"
        elif expect_preferred == "local" and actual == "remote_mac":
            detail = f"Remote Mac tunnel served | reply: {snippet!r}"
        elif expect_preferred == "local" and actual in ("groq", "colab"):
            ok = True
            detail = f"Local preferred but cloud served ({actual}) | reply: {snippet!r}"
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
        "Light task → Local (Ollama light)",
        "What time is it right now?",
        "local",
    )
    ok_heavy = _run_direct_route(
        "Heavy task → Local heavy (or cloud fallback)",
        "Provide a comprehensive step-by-step analysis of REST vs GraphQL API design",
        "local",
    )
    return ok_light and ok_heavy


def test_force_tools_and_synthesize() -> bool:
    _header("3a-ii. Force tools + synthesize (offline)")

    from main import force_tools_node, synthesize_node, extract_final_reply

    state = {
        "messages": [
            HumanMessage(content="What time is it right now?"),
            AIMessage(content="The current time is [Insert Current Time]."),
        ],
        "control_allowed": False,
    }
    forced = force_tools_node(state)
    msgs = state["messages"] + forced.get("messages", [])
    state2 = {**state, "messages": msgs}
    synth = synthesize_node(state2)
    final_msgs = msgs + synth.get("messages", [])
    reply = extract_final_reply(final_msgs)

    ok = (
        forced.get("messages")
        and "get_current_time" in str(forced["messages"][0].name)
        and "STATUS: SUCCESS" in str(forced["messages"][0].content)
        and "[" not in reply
        and "Insert" not in reply
    )
    detail = f"forced={bool(forced.get('messages'))} | reply={reply[:100]!r}"
    return _result("Force tool + synthesize time query", "offline", "offline", ok, detail)


def test_tool_reply_helpers() -> bool:
    _header("3a-i. Tool reply helpers (offline)")

    from langchain_core.messages import ToolMessage
    from main import (
        extract_final_reply,
        _pending_tool_synthesis,
        _has_placeholders,
    )

    time_result = (
        "TOOL: get_current_time\nSTATUS: SUCCESS\nRESULT:\n"
        "It is Wednesday, July 08 2026, 03:45 PM."
    )
    messages = [
        HumanMessage(content="What time is it?"),
        AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "1"}]),
        ToolMessage(content=time_result, name="get_current_time", tool_call_id="1"),
    ]

    ok_pending = _pending_tool_synthesis(messages)
    ok_placeholder = _has_placeholders("The time is [Insert Current Time]")
    fallback = extract_final_reply(messages)
    ok_fallback = "03:45 PM" in fallback and "[" not in fallback

    bad_messages = messages + [AIMessage(content="It is [Insert Current Time]")]
    recovered = extract_final_reply(bad_messages)
    ok_recover = "03:45 PM" in recovered and "[" not in recovered

    all_ok = ok_pending and ok_placeholder and ok_fallback and ok_recover
    detail = (
        f"pending_synthesis={ok_pending} | placeholder_detect={ok_placeholder} | "
        f"fallback={fallback!r} | recovered={recovered!r}"
    )
    return _result("Tool reply extraction", "offline", "offline", all_ok, detail)


def test_agent_tool_calling() -> bool:
    _header("3b. Full agent graph — tool calling (local or Groq)")

    if not ollama_is_healthy(log_failure=False) and not os.getenv("GROQ_API_KEY"):
        print("\n[SKIP] Agent tool test — Ollama offline and GROQ_API_KEY not set")
        return True

    from main import TESRACT_SYSTEM_PROMPT, compiled_app, extract_final_reply

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
        final_reply = extract_final_reply(messages)

        ok = (
            tool_called
            and backend in ("LOCAL", "REMOTE_MAC", "GROQ")
            and "[" not in final_reply
            and "Insert" not in final_reply
        )
        detail = (
            f"tools invoked: {tool_names or 'none'} | "
            f"backend: {backend} | "
            f"final reply: {final_reply[:120]!r}"
        )
        return _result(
            "Agent uses local/Groq + get_current_time tool",
            "local",
            backend.lower(),
            ok,
            detail,
        )
    except Exception as exc:
        print(f"\n[FAIL] Agent tool test: {exc}")
        traceback.print_exc()
        return False


def run_all_tests() -> int:
    print("TESrACT LLM Routing Test Suite")
    print(f"  LLM_ROUTING_MODE : {ROUTING_MODE}")
    _local = local_status()
    print(f"  Ollama           : {'online' if _local['ollama_healthy'] else 'offline'}")
    print(f"  Light model      : {_local['light_model']}")
    print(f"  Heavy model      : {_local['heavy_model']}")
    print(f"  GROQ_API_KEY     : {'set' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
    print(f"  COLAB_LLM_URL    : {'set' if colab_configured() else 'not set'}")
    if colab_configured():
        print(f"  Colab health     : {'online' if colab_is_healthy() else 'offline'}")
    _hybrid = hybrid_status()
    print(f"  Hybrid routing   : {'enabled' if _hybrid['enabled'] else 'disabled'}")
    if _hybrid["enabled"] and _hybrid.get("local_instance_url"):
        print(
            f"  Remote Mac       : "
            f"{'online' if _hybrid['remote_mac_healthy'] else 'offline'} "
            f"({_hybrid['local_instance_url'][:48]})"
        )

    results = [
        test_classification(),
        test_direct_routing(),
        test_force_tools_and_synthesize(),
        test_tool_reply_helpers(),
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