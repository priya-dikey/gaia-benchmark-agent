"""
GAIA Benchmark Agent — powered by Hugging Face Inference API
Supports: text questions, image attachments, video (frame extraction), PDFs, text files

Models:
  TEXT  : meta-llama/Llama-3.3-70B-Instruct  (tool-calling)
  VISION: meta-llama/Llama-3.2-11B-Vision-Instruct  (image understanding)

Key: HF_TOKEN  (or HUGGINGFACEHUB_API_TOKEN)
"""

import os
import re
import json
import math
import base64
import tempfile
import mimetypes
import requests
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState


# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

# Text + tool-calling model (no vision)
# Qwen2.5-72B has more reliable JSON-format tool calling than Llama-3.3-70B
# Alternative: "meta-llama/Llama-3.3-70B-Instruct" (may emit XML-style calls)
TEXT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Vision model — handles images and video frames
# Alternatives: "Qwen/Qwen2.5-VL-7B-Instruct", "google/gemma-3-27b-it"
VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

HF_API_URL    = "https://router.huggingface.co/v1/chat/completions"
GAIA_API_URL  = "https://agents-course-unit4-scoring.hf.space"

MAX_TOKENS     = 1024
MAX_ITERATIONS = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")
with open(_PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()


# ─────────────────────────────────────────────────────────────────────────────
# HF API helper
# ─────────────────────────────────────────────────────────────────────────────

def _hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")


def call_hf(messages: list[dict], model: str, tools: list | None = None) -> dict:
    """
    POST to the HF router chat-completions endpoint.
    Retries without tools on 400 so we always get an answer.
    """
    headers = {
        "Authorization": f"Bearer {_hf_token()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    response = requests.post(HF_API_URL, headers=headers, json=body, timeout=120)

    if not response.ok:
        print(f"  [HF {response.status_code}] {response.text[:400]}")
        if response.status_code == 400 and tools:
            print("  [Retrying without tools]")
            body.pop("tools", None)
            body.pop("tool_choice", None)
            response = requests.post(HF_API_URL, headers=headers, json=body, timeout=120)

    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_task_file(task_id: str) -> bytes | None:
    """Download the attached file for a GAIA task from the scoring API."""
    try:
        url = f"{GAIA_API_URL}/files/{task_id}"
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {_hf_token()}"},
            timeout=30,
        )
        if r.status_code == 200:
            return r.content
        print(f"  [File fetch] {r.status_code} for task {task_id}")
        return None
    except Exception as e:
        print(f"  [File fetch error] {e}")
        return None


def _to_data_url(data: bytes, filename: str) -> str:
    """Encode raw bytes as a base64 data URL, guessing MIME from filename + magic bytes."""
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        # Magic-byte sniffing
        if data[:4] == b"\x89PNG":
            mime = "image/png"
        elif data[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif data[:4] == b"GIF8":
            mime = "image/gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "application/octet-stream"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe_image(image_data: bytes, filename: str, question: str) -> str:
    """Send an image to the vision model and ask the question about it."""
    data_url = _to_data_url(image_data, filename)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {
                    "type": "text",
                    "text": (
                        f"You are analysing this image to answer a GAIA benchmark question.\n"
                        f"Question: {question}\n"
                        f"Describe all relevant visual details and answer the question directly."
                    ),
                },
            ],
        }
    ]
    print(f"  [Vision] {filename} → {VISION_MODEL}")
    resp = call_hf(messages, model=VISION_MODEL, tools=None)
    return resp["choices"][0]["message"].get("content", "")


def extract_video_frames(video_data: bytes, filename: str, n_frames: int = 4) -> list[bytes]:
    """
    Extract N evenly-spaced frames from a video as JPEG bytes.
    Requires opencv-python. Returns [] if unavailable.
    """
    try:
        import cv2

        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total > 0:
            indices = [int(i * total / n_frames) for i in range(n_frames)]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame)
                    frames.append(buf.tobytes())

        cap.release()
        os.unlink(tmp_path)
        return frames

    except ImportError:
        print("  [Video] opencv-python not installed — pip install opencv-python")
        return []
    except Exception as e:
        print(f"  [Video] Frame extraction error: {e}")
        return []


def describe_video(video_data: bytes, filename: str, question: str) -> str:
    """
    Extract frames, describe each with the vision model, then synthesise an answer
    using the text model.
    """
    print(f"  [Video] Extracting frames from {filename} ({len(video_data)//1024}KB)")
    frames = extract_video_frames(video_data, filename, n_frames=4)

    if not frames:
        return (
            "Could not extract frames from the video file. "
            "Install opencv-python (pip install opencv-python) to enable video analysis."
        )

    frame_descriptions = []
    for i, frame_bytes in enumerate(frames):
        desc = describe_image(frame_bytes, f"frame_{i+1}.jpg", question)
        frame_descriptions.append(f"Frame {i+1}: {desc}")
        print(f"  [Video] Frame {i+1} described.")

    combined = "\n\n".join(frame_descriptions)

    # Synthesise across frames using the text model
    synthesis = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"I extracted {len(frames)} frames from a video and described each:\n\n"
                f"{combined}\n\n"
                f"Based on these descriptions, answer the following question:\n{question}\n\n"
                f"End with: FINAL ANSWER: <your answer>"
            ),
        },
    ]
    resp = call_hf(synthesis, model=TEXT_MODEL, tools=None)
    return resp["choices"][0]["message"].get("content", combined)


def handle_attachment(task_id: str, filename: str, question: str) -> str | None:
    """
    Download the task file and route to the correct handler.
    Returns a context string to prepend to the agent prompt, or None.
    """
    if not task_id or not filename:
        return None

    ext = os.path.splitext(filename.lower())[1]
    print(f"  [Attachment] Fetching {filename} (ext={ext})")

    file_data = _fetch_task_file(task_id)
    if file_data is None:
        return f"[Could not download attachment: {filename}]"

    # ── Images ────────────────────────────────────────────────────────────────
    if ext in IMAGE_EXTS:
        desc = describe_image(file_data, filename, question)
        return f"[Image analysis of '{filename}']:\n{desc}"

    # ── Videos ────────────────────────────────────────────────────────────────
    elif ext in VIDEO_EXTS:
        desc = describe_video(file_data, filename, question)
        return f"[Video analysis of '{filename}']:\n{desc}"

    # ── Audio ─────────────────────────────────────────────────────────────────
    elif ext in AUDIO_EXTS:
        # Whisper transcription would go here; returning a note for now
        return (
            f"[Audio file '{filename}' detected. "
            "Full audio transcription requires whisper integration. "
            "Consider answering from context if possible.]"
        )

    # ── PDFs ──────────────────────────────────────────────────────────────────
    elif ext == ".pdf":
        try:
            from io import BytesIO
            from pdfminer.high_level import extract_text
            text = extract_text(BytesIO(file_data))
            return f"[PDF content of '{filename}']:\n{text[:4000]}"
        except ImportError:
            return f"[PDF '{filename}' — install pdfminer.six for text extraction]"
        except Exception as e:
            return f"[PDF extraction error for '{filename}': {e}]"

    # ── Plain text / structured text ──────────────────────────────────────────
    elif ext in (".txt", ".csv", ".md", ".json", ".xml", ".html", ".py", ".js"):
        try:
            text = file_data.decode("utf-8", errors="replace")
            return f"[Content of '{filename}']:\n{text[:4000]}"
        except Exception as e:
            return f"[Text decode error for '{filename}': {e}]"

    else:
        return f"[Unsupported attachment type: '{filename}' — cannot process {ext} files]"


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """DuckDuckGo Instant Answer API — no key required."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
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


TOOLS = {"web_search": web_search, "calculator": calculator, "wikipedia": wikipedia}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +,-,*,/,**,sqrt,log,ceil,floor.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
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
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]



# ─────────────────────────────────────────────────────────────────────────────
# XML tool-call fallback parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Some models (e.g. Llama-3.3) emit tool calls in an old XML format:
        <function=tool_name{"arg": "value"}></function>
    or:
        <function=tool_name>{"arg": "value"}</function>

    This parser detects and converts them into the standard OpenAI tool_call dicts
    so the agentic loop can handle them normally.
    """
    # Pattern: <function=NAME{JSON}></function>  or  <function=NAME>JSON</function>
    pattern = re.compile(
        r"<function=([\w_]+)"   # tool name
        r"(?:\{(.+?)\})?"       # optional inline JSON args  {…}
        r">(?:(.*?))?</function>",# optional body JSON args
        re.DOTALL,
    )
    calls = []
    for i, m in enumerate(pattern.finditer(text)):
        name = m.group(1)
        args_str = m.group(2) or m.group(3) or "{}"
        args_str = args_str.strip()
        # Wrap bare key:value if not valid JSON
        if args_str and not args_str.startswith("{"):
            args_str = "{" + args_str + "}"
        try:
            json.loads(args_str)   # validate
        except json.JSONDecodeError:
            args_str = "{}"
        calls.append({
            "id": f"xml_call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return calls

# ─────────────────────────────────────────────────────────────────────────────
# Core agentic loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(question: str, task_id: str = "", file_name: str = "") -> str:
    """
    Full agentic loop with optional multimodal pre-processing.

    If task_id + file_name are provided, the attachment is fetched and analysed
    first (image→vision model, video→frame extraction+vision, pdf→text extraction),
    and the result is prepended to the user message so the text model has full context.
    """
    # Step 1: handle attachment
    attachment_context = ""
    if task_id and file_name:
        attachment_context = handle_attachment(task_id, file_name, question) or ""
        if attachment_context:
            print(f"  [Context] {attachment_context[:200]}...")

    # Step 2: build initial conversation
    user_content = f"{attachment_context}\n\n{question}" if attachment_context else question

    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Step 3: agentic tool-use loop
    for _ in range(MAX_ITERATIONS):
        response = call_hf(conversation, model=TEXT_MODEL, tools=TOOL_SCHEMAS)

        choices = response.get("choices")
        if not choices:
            print(f"  [Unexpected response] {str(response)[:300]}")
            return extract_final_answer(str(response))

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        if isinstance(message, str):
            return extract_final_answer(message)

        conversation.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
        })

        assistant_text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        # Fallback: some models emit XML-style tool calls in the text content
        if not tool_calls and assistant_text:
            tool_calls = _parse_xml_tool_calls(assistant_text)
            if tool_calls:
                print(f"  [XML fallback] Parsed {len(tool_calls)} tool call(s) from text")
                # Strip the XML from the visible text
                assistant_text = re.sub(r"<function=.*?</function>", "", assistant_text, flags=re.DOTALL).strip()

        if finish_reason == "stop" or not tool_calls:
            return extract_final_answer(assistant_text)

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            try:
                fn_args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, KeyError):
                fn_args = {}

            tool_fn = TOOLS.get(fn_name)
            try:
                result = tool_fn(**fn_args) if tool_fn else f"Unknown tool: {fn_name}"
            except Exception as e:
                result = f"Tool error: {e}"

            print(f"  [Tool] {fn_name}({fn_args}) → {str(result)[:120]}")

            conversation.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": fn_name,
                "content": str(result),
            })

    # Max iterations — force a final answer
    conversation.append({"role": "user", "content": "Please give your final answer now."})
    final = call_hf(conversation, model=TEXT_MODEL, tools=None)
    return extract_final_answer(final["choices"][0]["message"].get("content", ""))


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()

def build_graph():
    def hf_agent_node(state: MessagesState):
        last = state["messages"][-1]
        meta = getattr(last, "additional_kwargs", {}) or {}
        task_id  = meta.get("task_id", "")
        file_name = meta.get("file_name", "")

        user_query = last.content
        print(f"[Agent] Processing: {user_query[:80]}...")
        if file_name:
            print(f"[Agent] Attachment: {file_name} (task={task_id})")

        answer = run_agent(user_query, task_id=task_id, file_name=file_name)
        print(f"[Agent] Answer: {answer[:120]}")
        return {"messages": state["messages"] + [AIMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("hf_agent", hf_agent_node)
    builder.set_entry_point("hf_agent")
    builder.set_finish_point("hf_agent")
    return builder.compile()


graph = build_graph()