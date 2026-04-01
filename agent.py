import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import YandexGPT
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.tools import PythonREPLTool 
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import faiss
import numpy as np



load_dotenv()

python_repl = PythonREPLTool()

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}
    

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
class LocalRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self.model.encode(documents)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]


# Load your data (replace with CSV later if needed)
documents = [
    "How to use LangChain for retrieval?",
    "What is FAISS similarity search?",
    "How do embeddings work in NLP?",
    "What is Retrieval Augmented Generation?",
]

local_retriever = LocalRetriever(documents)


@tool
def retriever_tool(query: str) -> str:
    """Retrieve similar questions from local vector store."""
    results = local_retriever.search(query)
    return "\n\n".join(results)

tools = [
    wiki_search,
    web_search,
    arvix_search,
    PythonREPLTool(),
    retriever_tool,
]

def build_graph():
    llm = HuggingFaceEndpoint( 
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text2text-generation", 
        model_kwargs={ "temperature": 0, "max_length": 512 }
        , 
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN") 
    )

    #llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = local_retriever.search(state["messages"][0].content)

        print("Similar questions:")
        print(similar_question)

        if len(similar_question) > 0:
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference:\n\n{similar_question[0]}",
            )
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}

        return {"messages": [sys_msg] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge("tools", "assistant")

    return builder.compile()

# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    graph = build_graph(provider="groq")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
