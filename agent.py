import os
import re
import json
import math
import base64
import tempfile
import mimetypes
import requests
from typing import Any

import arxiv
from tavily import TavilyClient

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState

TEXT_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

HF_API_URL   = "https://router.huggingface.co/v1/chat/completions"
GAIA_API_URL = "https://agents-course-unit4-scoring.hf.space"

MAX_TOKENS     = 2048
MAX_ITERATIONS = 10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
with open(_PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()


def _hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")


def call_hf(messages: list[dict], model: str, tools: list | None = None) -> dict:
    """POST to HF router; retries without tools on 400."""
    headers = {
        "Authorization": f"Bearer {_hf_token()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    resp = requests.post(HF_API_URL, headers=headers, json=body, timeout=120)

    if not resp.ok:
        print(f"  [HF {resp.status_code}] {resp.text[:400]}")
        if resp.status_code == 400 and tools:
            print("  [Retrying without tools]")
            body.pop("tools", None)
            body.pop("tool_choice", None)
            resp = requests.post(HF_API_URL, headers=headers, json=body, timeout=120)

    resp.raise_for_status()
    return resp.json()


def tavily_search(query: str, search_depth: str = "advanced") -> str:
    """
    Real-time web search via Tavily. Returns rich snippets with source URLs.
    search_depth: "basic" (faster) or "advanced" (more thorough, costs 2 credits).
    Falls back to Wikipedia search if TAVILY_API_KEY is not set.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("  [Tavily] No TAVILY_API_KEY — falling back to Wikipedia search")
        return wikipedia(query)

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=5,
            include_answer=True,        # LLM-generated answer from results
            include_raw_content=False,
        )

        parts = []

        # Direct answer if available
        if response.get("answer"):
            parts.append(f"Answer: {response['answer']}")

        # Top results with title, url, snippet
        for r in response.get("results", []):
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = r.get("content", "")[:400]
            parts.append(f"[{title}] ({url})\n{content}")

        return "\n\n".join(parts) if parts else f"No results found for: {query}"

    except Exception as e:
        return f"Tavily search error: {e}"


def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers. Returns title, authors, published date,
    and abstract for each result. Useful for science/math/CS questions.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        parts = []
        for result in client.results(search):
            authors = ", ".join(a.name for a in result.authors[:3])
            if len(result.authors) > 3:
                authors += " et al."
            parts.append(
                f"Title: {result.title}\n"
                f"Authors: {authors}\n"
                f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                f"ArXiv ID: {result.entry_id.split('/')[-1]}\n"
                f"Abstract: {result.summary[:500]}"
            )

        return "\n\n---\n\n".join(parts) if parts else f"No arXiv papers found for: {query}"

    except Exception as e:
        return f"arXiv search error: {e}"


def calculator(expression: str) -> str:
    """
    Safe math eval. Supports +,-,*,/,**,sqrt,log,log10,ceil,floor,pi,e,factorial, etc.
    Always use this for arithmetic — never compute mentally.
    """
    try:
        cleaned = expression.replace("^", "**").replace(",", "")
        safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_ns["__builtins__"] = {}
        result = eval(cleaned, safe_ns)
        # Return int if whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def wikipedia(query: str) -> str:
    """Fetch a Wikipedia article summary (up to 2000 chars)."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query)
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            extract = r.json().get("extract", "")
            return extract[:2000] if extract else "No summary found."
        # Fall back to Wikipedia search API
        r2 = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 3},
            timeout=10,
        )
        items = r2.json().get("query", {}).get("search", [])
        if items:
            snippets = [
                f"{i['title']}: {re.sub(r'<[^>]+>', '', i.get('snippet',''))}"
                for i in items
            ]
            return "\n".join(snippets)
        return f"Wikipedia: nothing found for '{query}'"
    except Exception as e:
        return f"Wikipedia error: {e}"


TOOLS = {
    "tavily_search": tavily_search,
    "arxiv_search":  arxiv_search,
    "calculator":    calculator,
    "wikipedia":     wikipedia,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Real-time web search. Use for any factual question, current events, "
                "people, places, history, products, or anything that needs up-to-date information. "
                "Preferred over wikipedia for most questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific, well-formed search query"
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Use 'advanced' for complex questions, 'basic' for simple lookups"
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search academic papers on arXiv. Use for questions about scientific research, "
                "papers, authors, published findings in physics, math, CS, biology, etc. "
                "Free, no API key required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — can include author names, paper titles, topics"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of papers to return (default 3, max 5)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate any math expression. Supports: +,-,*,/,**,sqrt(),log(),log10(),"
                "ceil(),floor(),pi,e,factorial(),sin(),cos(),tan(). "
                "Always use this instead of computing mentally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression e.g. 'ceil(1002 * 0.04)' or 'sqrt(144)'"
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
            "description": (
                "Fetch a Wikipedia article summary. Good for well-known people, places, "
                "concepts, or events. Use tavily_search if you need more detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Wikipedia article title or topic"
                    }
                },
                "required": ["query"],
            },
        },
    },
]



def _parse_xml_tool_calls(text: str) -> list[dict]:
    pattern = re.compile(
        r"<function=(\w+)"
        r"(?:\{(.+?)\})?"
        r">(.*?)</function>",
        re.DOTALL,
    )
    calls = []
    for i, m in enumerate(pattern.finditer(text)):
        name     = m.group(1)
        args_str = (m.group(2) or m.group(3) or "{}").strip()
        if args_str and not args_str.startswith("{"):
            args_str = "{" + args_str + "}"
        try:
            json.loads(args_str)
        except json.JSONDecodeError:
            args_str = "{}"
        calls.append({
            "id": f"xml_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return calls


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().splitlines()[0].strip()
        answer = answer.strip('"\'')
        # Remove trailing period/comma unless it's a decimal number
        if answer and answer[-1] in ".," and not re.match(r"^\d+\.\d+$", answer):
            answer = answer[:-1]
        return answer
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _fetch_task_file(task_id: str) -> bytes | None:
    try:
        r = requests.get(
            f"{GAIA_API_URL}/files/{task_id}",
            headers={"Authorization": f"Bearer {_hf_token()}"},
            timeout=30,
        )
        return r.content if r.status_code == 200 else None
    except Exception as e:
        print(f"  [File fetch error] {e}")
        return None


def _to_data_url(data: bytes, filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        if data[:4] == b"\x89PNG":    mime = "image/png"
        elif data[:2] == b"\xff\xd8": mime = "image/jpeg"
        elif data[:4] == b"GIF8":     mime = "image/gif"
        else:                          mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def describe_image(image_data: bytes, filename: str, question: str) -> str:
    data_url = _to_data_url(image_data, filename)
    resp = call_hf(
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": (
                    f"Analyse this image carefully to help answer a GAIA benchmark question.\n"
                    f"Question: {question}\n"
                    f"Describe all visible text, numbers, labels, objects, and details. "
                    f"Be thorough — small details may be the answer."
                )},
            ],
        }],
        model=VISION_MODEL,
        tools=None,
    )
    return resp["choices"][0]["message"].get("content", "")


def extract_video_frames(video_data: bytes, filename: str, n_frames: int = 6) -> list[bytes]:
    try:
        import cv2
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total > 0:
            for i in range(n_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n_frames))
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame)
                    frames.append(buf.tobytes())
        cap.release()
        os.unlink(tmp_path)
        return frames
    except ImportError:
        print("  [Video] pip install opencv-python-headless")
        return []
    except Exception as e:
        print(f"  [Video] {e}")
        return []


def describe_video(video_data: bytes, filename: str, question: str) -> str:
    frames = extract_video_frames(video_data, filename)
    if not frames:
        return "Could not extract video frames — install opencv-python-headless."
    descs = [f"Frame {i+1}: {describe_image(fb, f'frame_{i+1}.jpg', question)}"
             for i, fb in enumerate(frames)]
    combined = "\n\n".join(descs)
    resp = call_hf(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content":
          f"Video frame descriptions:\n\n{combined}\n\nQuestion: {question}\n\nFINAL ANSWER: <answer>"}],
        model=TEXT_MODEL,
        tools=None,
    )
    return resp["choices"][0]["message"].get("content", combined)


def handle_attachment(task_id: str, filename: str, question: str) -> str | None:
    if not task_id or not filename:
        return None
    ext  = os.path.splitext(filename.lower())[1]
    data = _fetch_task_file(task_id)
    if data is None:
        return f"[Could not download: {filename}]"
    print(f"  [Attachment] {filename} ({len(data)//1024}KB, ext={ext})")

    if ext in IMAGE_EXTS:
        return f"[Image '{filename}']:\n{describe_image(data, filename, question)}"
    elif ext in VIDEO_EXTS:
        return f"[Video '{filename}']:\n{describe_video(data, filename, question)}"
    elif ext in AUDIO_EXTS:
        return f"[Audio '{filename}' — transcription not supported yet]"
    elif ext == ".pdf":
        try:
            from io import BytesIO
            from pdfminer.high_level import extract_text
            return f"[PDF '{filename}']:\n{extract_text(BytesIO(data))[:4000]}"
        except ImportError:
            return f"[PDF '{filename}' — install pdfminer.six]"
        except Exception as e:
            return f"[PDF error: {e}]"
    elif ext in (".txt", ".csv", ".md", ".json", ".xml", ".html", ".py"):
        try:
            return f"[File '{filename}']:\n{data.decode('utf-8', errors='replace')[:4000]}"
        except Exception as e:
            return f"[Decode error: {e}]"
    else:
        return f"[Unsupported file type: {filename}]"



def run_agent(question: str, task_id: str = "", file_name: str = "") -> str:
    # Step 1: handle attachment
    attachment_ctx = ""
    if task_id and file_name:
        attachment_ctx = handle_attachment(task_id, file_name, question) or ""

    # Step 2: build initial conversation
    user_content = f"{attachment_ctx}\n\n{question}".strip() if attachment_ctx else question
    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Step 3: agentic loop
    for iteration in range(MAX_ITERATIONS):
        response = call_hf(conversation, model=TEXT_MODEL, tools=TOOL_SCHEMAS)

        choices = response.get("choices")
        if not choices:
            return extract_final_answer(str(response))

        choice        = choices[0]
        message       = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        if isinstance(message, str):
            return extract_final_answer(message)

        assistant_text = message.get("content") or ""
        tool_calls     = message.get("tool_calls") or []

        # XML fallback
        if not tool_calls and assistant_text:
            tool_calls = _parse_xml_tool_calls(assistant_text)
            if tool_calls:
                print(f"  [XML fallback] {len(tool_calls)} call(s)")
                assistant_text = re.sub(
                    r"<function=.*?</function>", "", assistant_text, flags=re.DOTALL
                ).strip()

        # Append assistant turn to history
        conversation.append({
            "role":    "assistant",
            "content": assistant_text,
            **({"tool_calls": message.get("tool_calls") or [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ]} if tool_calls else {}),
        })

        if finish_reason == "stop" or not tool_calls:
            return extract_final_answer(assistant_text)

        # Execute each tool call
        for tc in tool_calls:
            fn      = tc.get("function", {})
            fn_name = fn.get("name", "")
            try:
                fn_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                fn_args = {}

            tool_fn = TOOLS.get(fn_name)
            try:
                result = tool_fn(**fn_args) if tool_fn else f"Unknown tool: {fn_name}"
            except Exception as e:
                result = f"Tool error: {e}"

            print(f"  [Tool] {fn_name}({fn_args}) → {str(result)[:150]}")

            conversation.append({
                "role":         "tool",
                "tool_call_id": tc.get("id", ""),
                "name":         fn_name,
                "content":      str(result),
            })

    # Force final answer after hitting iteration cap
    conversation.append({
        "role":    "user",
        "content": "Based on everything above, provide only: FINAL ANSWER: <answer>",
    })
    final = call_hf(conversation, model=TEXT_MODEL, tools=None)
    return extract_final_answer(final["choices"][0]["message"].get("content", ""))


def build_graph():
    def hf_agent_node(state: MessagesState):
        last      = state["messages"][-1]
        meta      = getattr(last, "additional_kwargs", {}) or {}
        task_id   = meta.get("task_id", "")
        file_name = meta.get("file_name", "")

        print(f"[Agent] {last.content[:100]}...")
        if file_name:
            print(f"[Agent] Attachment: {file_name} (task={task_id})")

        answer = run_agent(last.content, task_id=task_id, file_name=file_name)
        print(f"[Agent] → {answer[:120]}")
        return {"messages": state["messages"] + [AIMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("hf_agent", hf_agent_node)
    builder.set_entry_point("hf_agent")
    builder.set_finish_point("hf_agent")
    return builder.compile()


graph = build_graph()