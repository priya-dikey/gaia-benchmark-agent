import os
import json
import re
import math
import requests
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState

# Good options (all support tool/function calling via HF Inference):
#   "Qwen/Qwen2.5-72B-Instruct"          ← default, strong & fast
#   "meta-llama/Llama-3.3-70B-Instruct"
#   "mistralai/Mistral-7B-Instruct-v0.3"

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
MAX_TOKENS = 1024
MAX_ITERATIONS = 8  # safety cap on agentic rounds

_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
 
with open(_PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()

def web_search(query: str) -> str:
    """DuckDuckGo Instant Answer API — no key required."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        snippets = [
            t["Text"] for t in data.get("RelatedTopics", [])[:3]
            if isinstance(t, dict) and t.get("Text")
        ]
        return "\n".join(snippets) if snippets else f"No results found for: {query}"
    except Exception as e:
        return f"Web search error: {e}"


def calculator(expression: str) -> str:
    """Safe math eval with Python's math module."""
    try:
        cleaned = expression.replace("^", "**").replace(",", "")
        safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_ns["__builtins__"] = {}
        return str(eval(cleaned, safe_ns))
    except Exception as e:
        return f"Calculation error: {e}"


def wikipedia(query: str) -> str:
    """Wikipedia REST API summary."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query)
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("extract", "No summary found.")
        return f"Wikipedia: page not found for '{query}'"
    except Exception as e:
        return f"Wikipedia error: {e}"


TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "wikipedia": wikipedia,
}

# OpenAI-compatible tool schemas (HF Inference uses this format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, log, ceil, floor, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. 'ceil(1002 * 0.04)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia",
            "description": "Look up a Wikipedia article summary for a person, place, or concept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Wikipedia article title or search term",
                    }
                },
                "required": ["query"],
            },
        },
    },
]



def call_hf(messages: list[dict], use_tools: bool = True) -> dict:
    """
    Call the HF Inference OpenAI-compatible chat completions endpoint.
    Requires HF_TOKEN with Inference permission.
    """
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }
    if use_tools:
        body["tools"] = TOOL_SCHEMAS
        body["tool_choice"] = "auto"

    response = requests.post(HF_API_URL, headers=headers, json=body, timeout=90)
    response.raise_for_status()
    return response.json()


def run_agent(question: str) -> str:
    """
    Agentic loop: call HF model → execute tool calls → feed results back → repeat
    until the model stops requesting tools and gives a final answer.
    """
    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        response = call_hf(conversation)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        # Add assistant message to history
        conversation.append(message)

        assistant_text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        # No tool calls → final answer
        if finish_reason == "stop" or not tool_calls:
            return extract_final_answer(assistant_text)

        # Execute each tool call and collect results
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                fn_args = {}

            tool_fn = TOOLS.get(fn_name)
            if tool_fn:
                try:
                    result = tool_fn(**fn_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            print(f"  [Tool] {fn_name}({fn_args}) → {str(result)[:120]}")

            # Append tool result as a "tool" role message
            conversation.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": str(result),
            })

    # Max iterations reached — ask for final answer without tools
    conversation.append({"role": "user", "content": "Please give your final answer now."})
    final = call_hf(conversation, use_tools=False)
    final_text = final["choices"][0]["message"].get("content", "")
    return extract_final_answer(final_text)


def extract_final_answer(text: str) -> str:
    """Extract text after 'FINAL ANSWER:' if present, else return the last line."""
    if not text:
        return ""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def build_graph():
    def hf_agent_node(state: MessagesState):
        user_query = state["messages"][-1].content
        print(f"[Agent] Processing: {user_query[:80]}...")
        answer = run_agent(user_query)
        print(f"[Agent] Answer: {answer[:120]}")
        return {"messages": state["messages"] + [AIMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("hf_agent", hf_agent_node)
    builder.set_entry_point("hf_agent")
    builder.set_finish_point("hf_agent")
    return builder.compile()


graph = build_graph()