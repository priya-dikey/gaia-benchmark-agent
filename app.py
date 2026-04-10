import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from agent import build_graph 

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


class BasicAgent:
    """LangGraph agent powered by HF Inference with web_search, calculator, wikipedia tools."""

    def __init__(self):
        print("BasicAgent (HF Inference) initialized.")
        if not os.environ.get("HF_TOKEN"):
            raise EnvironmentError(
                "HF_TOKEN environment variable not set. "
                "Add it as a HF Space secret or in your .env file."
            )
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 80 chars): {question[:80]}...")
        messages = [HumanMessage(content=question)]
        result = self.graph.invoke({"messages": messages})

        if not isinstance(result, dict):
            return "Graph returned an unexpected result format."

        if "messages" in result:
            return result["messages"][-1].content
        return f"Graph returned: {result} (missing 'messages')"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetch all GAIA questions, run the agent, submit answers, display results."""
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # 3. Run Agent on each question
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Submit
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except Exception:
            error_detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {error_detail}", pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# ─── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Benchmark Agent — Powered by Hugging Face Inference")
    gr.Markdown(
        """
**Instructions:**
1. Set `HF_TOKEN` as a HF Space secret (Settings → Variables and secrets → New secret).
   Your token needs **Inference** permission (a free HF account token works).
2. Log in to your Hugging Face account using the button below.
3. Click **Run Evaluation & Submit All Answers**.

The agent uses **`Qwen/Qwen2.5-72B-Instruct`** via the HF Inference API with three tools:
`web_search`, `calculator`, and `wikipedia`.
It runs an agentic loop — the model can call tools multiple times before giving a final answer.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
    else:
        print("ℹ️  SPACE_HOST not found (running locally?).")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
    else:
        print("ℹ️  SPACE_ID not found (running locally?).")

    print("-" * (60 + len(" App Starting ")) + "\n")
    demo.launch(debug=True, share=False)