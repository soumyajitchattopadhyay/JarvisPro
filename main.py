import os
from typing import Annotated, TypedDict, List, Literal, Union, Dict, cast
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

import actions 
from speak import speak 
from listen import listen_for_command

load_dotenv()

# --- BRAIN PROTOCOLS ---
PROTOCOLS = {
    "CORE": "You are TESrACT. Execute first, talk second. Use the 'execute_python_code' tool for ALL math, logic, and file ops.",
    "CODE_STYLE": "All files created must include Type Hints (PEP 484) and Google-style docstrings.",
    "ERROR_RECOVERY": "If a tool fails, read the traceback, search for the fix, and retry WITHOUT asking the user.",
    "COMMUNICATION": "Keep verbal updates under 15 words.",
    "PERMISSION": "If CURRENT_CONTROL is RESTRICTED, you are FORBIDDEN from using system tools (execute_python_code, open_url_in_browser). You must first ask the user: 'I need to perform a system action. Allow Control?'",
    "ACTIVATION": "Only once the user says 'Allow Control' will CURRENT_CONTROL become ACTIVE. Do not simulate or assume permission."
}


class JProState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: NotRequired[str]
    control_allowed: bool # Added to state for persistence

# --- NODES ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(actions.tools)

def call_brain(state: JProState):
    is_allowed = state.get("control_allowed", False)
    status = "ACTIVE" if is_allowed else "RESTRICTED"
    
    # EXPLICITLY filter tools or warn LLM based on status
    sys_msg = SystemMessage(content=f"SYSTEM PROTOCOLS:\n{PROTOCOLS}\nCURRENT_CONTROL: {status}")
    
    # We pass the flag to tools via tool_calls args
    response = llm_with_tools.invoke([sys_msg] + state["messages"][-10:])
    
    if response.tool_calls:
        for t in response.tool_calls:
            t['args']['control_allowed'] = is_allowed

    return {"messages": [response]}

def reviewer_node(state: JProState):
    last_msg = state["messages"][-1]
    content_str = str(last_msg.content).lower() if last_msg.content else ""
    
    if "allow control" in content_str: return {"control_allowed": True}
    if "stop control" in content_str: return {"control_allowed": False}
    if "error" in content_str:
        return {"messages": [SystemMessage(content="Action failed. Pivot strategy.")]}
    return state

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(JProState)
workflow.add_node("agent", call_brain)
workflow.add_node("tools", ToolNode(actions.tools))
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "agent")

def route_after_agent(state: JProState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "reviewer")
workflow.add_edge("reviewer", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- EXECUTION LOOP (FIXED INDENTATION) ---
def main_loop():
    speak("TESrACT system online.", wait_for_speech=True)
    config: RunnableConfig = {"configurable": {"thread_id": "jarvis_session_001"}}
    
    while True:
        command = listen_for_command()
        if not command: continue
        
        if any(word in command.lower() for word in ["shutdown", "exit"]):
            speak("Powering down.")
            break

        # --- INDENTED INSIDE WHILE LOOP ---
        current_state = app.get_state(config)
        is_allowed = current_state.values.get("control_allowed", False)

        # Fix: Cast dict to JProState to resolve type assignment error
        inputs = cast(JProState, {
            "messages": [HumanMessage(content=command)], 
            "control_allowed": is_allowed
        })
        
        try:
            final_msg = None
            for event in app.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event and event["messages"]:
                    final_msg = event["messages"][-1]
            
            if final_msg and isinstance(final_msg, AIMessage) and not final_msg.tool_calls:
                speak(str(final_msg.content).replace("`", ""), wait_for_speech=True)
                
        except Exception as e:
            print(f"Graph Error: {e}")
            speak("Internal logic error encountered.")

if __name__ == "__main__":
    main_loop()