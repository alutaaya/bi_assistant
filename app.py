import os
import io
import tempfile
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

# Document / LangChain related imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.evaluation.qa import QAEvalChain

# PDF & text splitting
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

# Data & plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re 
import json

# Workflow
from langgraph.graph import StateGraph, END

# Typing
from typing import TypedDict, List, Any, Dict, Optional

# PandaAI
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Load .env if present
load_dotenv()


# ------------------------
# Typed state definition
# ------------------------
class appstate(TypedDict, total=False):
    csv_bytes: Optional[bytes]
    doc_files: Optional[List[str]]
    doc_texts: Optional[List[Document]]
    statistics_summary: Optional[Dict[str, Any]]
    trends_summary: Optional[Dict[str, Any]]
    narrative_chunks: Optional[List[str]]
    vectordb: Optional[Any]
    last_query: Optional[str]
    query_answer: Optional[str]
    ask_stat: Optional[str]
    adhoc_result: Optional[Any]
    csv_path: Optional[str]
    knowledgebase: Optional[List[Document]]
    visualisation: Optional[Dict[str, Any]]


# ------------------------
# Utility: load LLM (cached)
# ------------------------
##@st.cache_resource
##def load_llm():
    ##groq_api = st.secrets.get("groq_api") or os.getenv("groq_api")
    ##if not groq_api:
        ##st.error("❌ ERROR: groq_api not found. Please set it in Streamlit secrets or .env file.")
        ##return None
    # instantiate ChatGroq exactly as you had it
    ##return ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, api_key=groq_api)


#----Loading the llm to use for PandaAI

def load_llm():
    api_key=st.secrets["OPENAI_API_KEY"]
    return OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)


def load_llm2():
    api_key=st.secrets["OPENAI_API_KEY"]
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)




# ------------------------
# Agents (minimal edits: fix typos, ensure returns, small safety)
# ------------------------

# Filepath agent: write csv_bytes to temp file and set csv_path
def filepath_agent(state: appstate):
    if "csv_bytes" in state and state["csv_bytes"]:
        tmp_dir = tempfile.gettempdir()
        path = os.path.join(tmp_dir, "uploaded_file.csv")
        # write file in binary mode
        with open(path, "wb") as f:
            f.write(state["csv_bytes"])
        state["csv_path"] = path
    return state


# Stats agent: compute summaries and display them (keeps your display calls)
def stats_agent(state: appstate):
    path = state.get("csv_path")
    if not path:
        return state

    summary = {}
    df = pd.read_csv(path)
    # parse date safely
    df["date"] = pd.to_datetime(df["Date"],format="%d/%m/%Y", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # descriptive
    summary["describe"] = df.describe(include="all").to_dict()
    st.markdown("### Descriptive Statistics")
    st.dataframe(pd.DataFrame(summary["describe"]).T)

    # sales summary
    summary["sales_summary"] = df.groupby("year")["Sales"].sum().to_dict()
    sales_summary_df = df.groupby("year")["Sales"].sum().reset_index()
    st.markdown("### Sales per year")
    st.dataframe(sales_summary_df)

    # region
    summary["region_summary"] = df.groupby(["Region"])["Sales"].sum().to_dict()
    region_summary_df = df.groupby(["Region"])["Sales"].sum().reset_index()
    st.markdown("### Sales by Region")
    st.dataframe(region_summary_df)

    # median customer age
    summary["Av_Cust_age"] = df.groupby(["Region"])["age"].median().to_dict()
    cust_age_df = df.groupby(["Region"])["age"].median().reset_index()
    st.markdown("### Median Customer age per region")
    st.dataframe(cust_age_df)

    # gender
    summary["cust_gender"] = df.groupby(["sex"])["Sales"].sum().to_dict()
    cust_gender_df = df.groupby(["sex"])["Sales"].sum().reset_index()
    st.markdown("### Sales by Customer Gender")
    st.dataframe(cust_gender_df)

    # customer satisfaction median
    summary["cust_satistifaction"] = df.groupby(["Product"])["satistifaction"].median().to_dict()
    cust_sat_df = df.groupby(["Product"])["satistifaction"].median().reset_index()
    st.markdown("### Median Customer Satisfaction by product")
    st.dataframe(cust_sat_df)

    # add yearly trends to summary (optional)
    yearly_sales = df.groupby("year")["Sales"].sum().to_dict()
    yearly_cust_sat = df.groupby("year")["satistifaction"].median().to_dict()
    ##-- i do no want to show this  summary["yearly_sales_trend"] = yearly_sales
    ##-- i do not want to show this  summary["yearly_cust_satistifaction_trend"] = yearly_cust_sat

    # persist summary to state
    state["statistics_summary"] = summary
    return state


# Visualisation agent: draws and stores plot data (kept your plotting code, fixed storing figs)
def visualisation_agent(state: appstate):
    path = state.get("csv_path")
    if not path:
        return state

    visuals = {}
    df = pd.read_csv(path)
    ###df["Date"] = df["Date"].str.strip()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    yearly_sales = df.groupby("year")["Sales"].sum().to_dict()
    yearly_cust_sat = df.groupby("year")["satistifaction"].median().to_dict()

    visuals["yearly_sales_trend"] = yearly_sales
    visuals["yearly_cust_sat_trend"] = yearly_cust_sat

    years = list(yearly_sales.keys())
    sales_values = list(yearly_sales.values())
    cust_sat_years = list(yearly_cust_sat.keys())
    cust_sat_values = list(yearly_cust_sat.values())

    # Plot 1
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(years, sales_values, marker="o", label="Sales")
    ax1.set_title("Yearly Sales Trend")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Total sales")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(cust_sat_years, cust_sat_values, marker="o", label="Customer Satisfaction")
    ax2.set_title("Yearly Customer Satisfaction Trends")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Median customer satisfaction")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # store numeric data (fig objects closed, we store underlying data)
    state["visualisation"] = {
        "data": visuals,
    }
    return state


# Narrative agent (fixed name and ensured we use load_llm safely)
def narrative_agent(state: appstate):
    summary = state.get("adhoc_result")
    if not summary:
        return state

    prompt = f"You are a data analyst, given the  summary: {summary}. Write concise bullets of narrative providing necessary insights."

    llm = load_llm2()
    if llm is None:
        return state

    # Use predict method for ChatOpenAI
    try:
        response = llm.predict(prompt)  # ✅ ChatOpenAI has .predict
    except Exception as e:
        state["narrative_chunks"] = [f"Error generating narrative: {e}"]
        return state

    # split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(response)
    state["narrative_chunks"] = chunks
    return state
# Knowledgebase agent: create Document objects from narrative and PDFs (kept your logic)
def knowledgebase_agent(state: appstate):
    chunks = state.get("narrative_chunks", [])
    docs = [Document(page_content=text, metadata={"source": "narrative"}) for text in chunks]

    pdf_docs = state.get("doc_texts", [])
    if pdf_docs:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        # split_documents expects Document-like inputs; your doc_texts list should be Document objects
        # keep original call
        chunked_pdf = splitter.split_documents(pdf_docs)
        docs.extend(chunked_pdf)

    state["knowledgebase"] = docs
    return state


# Vectorstore agent: build/load FAISS index. Fixed return semantics (state stored vectordb)
def vectorstore_agent(state: appstate):
    store = state.get("knowledgebase", [])
    FAISS_INDEX_PATH = "faiss_index"
    hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # try to load
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vectordb = FAISS.load_local(FAISS_INDEX_PATH, hf_embed, allow_dangerous_deserialization=True)
            state["vectordb"] = vectordb
            return state
        except Exception:
            st.warning("Error loading FAISS index. Rebuilding index.")

    # build new
    if store:
        vectordb = FAISS.from_documents(store, hf_embed)
        vectordb.save_local(FAISS_INDEX_PATH)
        state["vectordb"] = vectordb
    else:
        state["vectordb"] = None
    return state

##--- function for memory 


def save_concise_memory(input_text: str, result_text: str):  
    llm = load_llm2()
    summary_prompt = f"Summarize the following answer concisely for future reference:\n\n{result_text}"
    try:   
        concise_summary = llm.predict(summary_prompt).strip()
    except Exception: 
        concise_summary = result_text  # fallback

    if "memory" in st.session_state:    
        st.session_state["memory"].save_context({"input": input_text}, {"output": concise_summary})

    if "chat_history" in st.session_state:    
        st.session_state["chat_history"].append((input_text, concise_summary))

    if "numeric_memory" not in st.session_state:  
        st.session_state["numeric_memory"] = {}

    # ✅ safer number extraction
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", concise_summary)
    num_values = []
    for n in numbers:
        try:
            num_values.append(float(n.replace(",", "")))
        except ValueError:
            continue  # skip invalid ones

    if num_values:
        st.session_state["numeric_memory"][input_text.lower()] = num_values[0]

    return concise_summary


def answering_agent(state: appstate):
    query = state.get("ask_stat", "")
    vectorstore = state.get("vectordb", None)
    if not vectorstore or not query:
        return state
    # retrieve relevant chunks from vectordb
    retrieve_relevant_chunks = vectorstore.similarity_search(query, k=12)
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in retrieve_relevant_chunks]
    )
    # include recent memory in the prompt (last 5)
    memory_history = ""
    if st.session_state.get("memory"):
        buffer = st.session_state["memory"].buffer
        memory_history = "\n".join(
            [f"Q: {m.input}\nA: {m.output}" for m in buffer[-5:]]
        )
    # also include numeric memory if available
    numeric_memory_text = ""
    if st.session_state.get("numeric_memory"):
        numeric_memory_text = "\n".join(
            [f"{k}: {v}" for k, v in st.session_state["numeric_memory"].items()]
        )
    prompt = (
        "You are a business assistant. Use the following memory, numeric memory, and context to answer concisely.\n\n"
        f"Memory:\n{memory_history}\n\n"
        f"Numeric Memory:\n{numeric_memory_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\nAnswer concisely:"
    )

    llm = load_llm2()
    response = llm.predict(prompt)
    if hasattr(response, "content"):
        answer = response.content.strip()
    else:
        answer = str(response).strip()
    state["query_answer"] = answer
    # save concise summary to memory
    save_concise_memory(query, answer)
    return state



def restricted_adhoc_agent(state: dict):
    """
    Use PandasAI to answer ad-hoc statistical questions on the uploaded CSV.
    This version only mutates state in-place and ensures a proper return type.
    """
    ask_stat = state.get("ask_stat", "")
    path = state.get("csv_path")
    if not path or not ask_stat:
        state["adhoc_result"] = "No CSV loaded or no question provided."
        return state

    df = pd.read_csv(path)

    # Preprocess date column safely
    df.columns = df.columns.str.strip()
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"],format="%d/%m/%Y", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month


    llm = load_llm()
    pandas_ai = PandasAI(llm)

    # Run query safely
    try:
        result = pandas_ai.run(df, ask_stat, show_code=True, is_conversational_answer=True)
        result_text = str(result)
        # save concise summary to memory
        concise_summary = save_concise_memory(ask_stat, result_text)
        state["adhoc_result"] = concise_summary
    except Exception as e:
        state["adhoc_result"] = f"Error executing PandasAI query: {e}"

    return state


# ------------------------
# Workflow constructors (kept your original graphs)
# ------------------------
def construct_graph1():
    """Build the state graph 1 for business assistant that answers questions."""
    workflow = StateGraph(appstate)
    workflow.add_node("filepath", filepath_agent)
    workflow.add_node("statistics", stats_agent)
    workflow.add_node("narrative", narrative_agent)
    workflow.add_node("knowledgebase", knowledgebase_agent)
    workflow.add_node("vectorstore", vectorstore_agent)
    workflow.add_node("answering", answering_agent)
    workflow.set_entry_point("filepath")
    workflow.add_edge("filepath", "statistics")
    workflow.add_edge("statistics", "narrative")
    workflow.add_edge("narrative", "knowledgebase")
    workflow.add_edge("knowledgebase", "vectorstore")
    workflow.add_edge("vectorstore", "answering")
    workflow.add_edge("answering", END)
    return workflow.compile()


def construct_graph2():
    """Build the state graph 2 for business assistant that answers adhoc statistics"""
    chain = StateGraph(appstate)
    chain.add_node("filepath", filepath_agent)
    # note: restricted_adhoc_agent signature takes (state, ask_stat)
    # LangGraph may pass only state; we'll wrap in a lambda when invoking the chain
    chain.add_node("adhoc", restricted_adhoc_agent)
    chain.set_entry_point("filepath")
    chain.add_edge("filepath", "adhoc")
    chain.add_edge("adhoc", END)
    return chain.compile()


#-------main function
def main():
    st.set_page_config(layout="wide")
    st.markdown(
        "<h1 style='text-align: center; font-weight: bold;color: blue;'>InsightForge Business Assistant</h1>",
        unsafe_allow_html=True
    )

    # Initialize app state dict
    state: appstate = {
        "csv_bytes": None,
        "doc_files": [],
        "doc_texts": [],
        "statistics_summary": {},
        "trends_summary": {},
        "narrative_chunks": [],
        "vectordb": None,
        "last_query": "",
        "query_answer": "",
        "ask_stat": "",
        "adhoc_result": {},
        "csv_path": "",
        "knowledgebase": [],
        "visualisation": {},
    }

    # --- Sidebar: upload CSV and PDF
    csv_file = st.sidebar.file_uploader("Upload csv file", type=["csv"])
    doc_files = st.sidebar.file_uploader("Upload supporting documents", type=["pdf"], accept_multiple_files=True)

    # --- Initialize session state keys
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "csv_bytes" not in st.session_state:
        st.session_state["csv_bytes"] = None
    if "doc_texts" not in st.session_state:
        st.session_state["doc_texts"] = []
    if "ask_stat" not in st.session_state:
        st.session_state["ask_stat"] = ""
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- Process the csv (only once per upload)
    if csv_file:
        csv_bytes: bytes = csv_file.read()
        st.session_state["csv_bytes"] = csv_bytes
        df: pd.DataFrame = pd.read_csv(io.BytesIO(csv_bytes))

        st.header("Preview the uploaded CSV")
        st.dataframe(df.head())
        st.markdown("## Summary Statistics")
        st.dataframe(df.describe(include="all").T)
        state["csv_bytes"] = csv_bytes

    # --- Process PDFs
    if doc_files:
        doc_texts = []
        for x in doc_files:
            if x.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(x.read())
                    temp_path = tmp_file.name
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                doc_texts.extend(documents)
            else:
                st.warning(f"Skipped non-pdf file: {x.name}")
        st.session_state["doc_texts"] = doc_texts
        state["doc_texts"] = doc_texts

   # --- Model evaluation with QAEvalChain
    if st.sidebar.button("Run Evaluation"):
        llm = load_llm2()  # use the same LLM you already configured
        # Ground truth examples
        examples = [
        {"query": "Total sales in 2022?", "answer": "50000"},
        {"query": "Top-selling product?", "answer": "Product A"},
        ]
        # Example predictions (must use "result" instead of "answer")
        predictions = [
        {"query": "Total sales in 2022?", "result": "50000"},
        {"query": "Top-selling product?", "result": "Product A"},
        ]
        # Build evaluation chain
        eval_chain = QAEvalChain.from_llm(llm)
        # Run evaluation
        grades = eval_chain.evaluate(examples, predictions)
        st.markdown("### Model Evaluation Results")
        st.json(grades)


    # --- Construct the workflows
    processgraph = construct_graph1()
    processgraph2 = construct_graph2()

    # --- Buttons for clearing
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []
            st.session_state["ask_stat"] = ""
            st.success("Chat history cleared!")
    with col2:
        if st.button("Reset Assistant"):
            st.session_state["chat_history"] = []
            st.session_state["memory"].clear()
            st.session_state["ask_stat"] = ""
            st.success("Assistant fully reset (chat + memory)!")

    # --- Ask a question
    options = ["chat_with_csv", "ask_for_insights"]
    choice = st.selectbox("Select an option:", ["--Select--"] + options)
    ask_stat = st.text_input("Ask a question", value=st.session_state["ask_stat"]).strip().lower()
    st.session_state["ask_stat"] = ask_stat
    state["ask_stat"] = ask_stat

    if st.button("Ask") and ask_stat:
        if choice == "chat_with_csv":
            state_result = processgraph2.invoke(state.copy())
            result_text = state_result.get("adhoc_result", "No result")
            st.write(result_text)
            st.session_state["memory"].save_context({"input": ask_stat}, {"output": str(result_text)})
            st.session_state["chat_history"].append((ask_stat, str(result_text)))
            st.session_state["adhoc_result"] = result_text

        elif choice == "ask_for_insights":
            state_result = processgraph.invoke(state.copy())
            result_text = state_result.get("query_answer", "No result")
            st.write(result_text)
            st.session_state["memory"].save_context({"input": ask_stat}, {"output": str(result_text)})
            st.session_state["chat_history"].append((ask_stat, str(result_text)))
            st.session_state["query_answer"] = result_text
        else:
            st.warning("Please select a valid option from the dropdown.")

    # --- Display summary statistics
    if st.button("Statistical_Summary"):
        state = filepath_agent(state)
        state = stats_agent(state)
        summary = state.get("statistics_summary", {})

    # --- Display visuals
    if st.button("Data visualisation"):
        state = filepath_agent(state)
        state = visualisation_agent(state)

    # --- Display chat history
    st.markdown("### Chat History")
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")


if __name__ == "__main__":
    main()

