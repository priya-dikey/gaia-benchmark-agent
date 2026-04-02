import json
import os
import csv
import json
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState

INPUT_CSV = "data_clean.csv"

def load_docs(csv_path):
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row["content"]

            try:
                metadata = json.loads(row.get("metadata", "{}"))
            except json.JSONDecodeError:
                metadata = {}

            docs.append(Document(page_content=content, metadata=metadata))
    return docs


docs = load_docs(INPUT_CSV)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db"
)
vector_store.persist()
print("Векторная база создана и сохранена в 'chroma_db'")


def find_answer(query, k=1) -> str:
    """
    Searches for an answer in the vector database based on the user's query.
    Returns a string with the final answer or the last text of the document.
    :param query: User query
    :param k: number of possible answers
    :return: User's answer
    """
    results = vector_store.similarity_search(query, k=k)
    if not results:
        return "Ответ не найден"

    content = results[0].page_content

    if "Final answer :" in content:
        return content.split("Final answer :", 1)[1].strip()
    elif "Answer:" in content:
        return content.split("Answer:", 1)[1].strip()
    else:
        return content.strip().splitlines()[-1]


def build_graph():
    def retriever_node(state: MessagesState):
        user_query = state["messages"][-1].content
        answer_text = find_answer(user_query)
        return {"messages": state["messages"] + [AIMessage(content=answer_text)]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever_node)
    builder.set_entry_point("retriever")
    builder.set_finish_point("retriever")
    return builder.compile()

graph = build_graph()
