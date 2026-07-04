from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
from langchain_core.tools import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma

load_dotenv(override=True)

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY", "")

st.set_page_config(page_title="Meditation AI Guide", page_icon="🧘", layout="centered")
st.title("🧘 Meditation AI Assistant")
st.caption("Ask questions about meditation techniques from your PDF or search the web.")

@st.cache_resource
def init_agent():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="meditation_pdf_search",
        description="Search this tool first for any questions regarding meditation techniques, posture, and breathing exercises."
    )
    tavily_tool = TavilySearchResults(max_results=3)
    tools = [retriever_tool, tavily_tool]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    memory = MemorySaver()
    
    return create_react_agent(llm, tools, checkpointer=memory)

agent_executor = init_agent()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help you explore meditation practices. What's on your mind today?"}
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Type your message here..."):
    
    with st.chat_message("user"):
        st.write(user_input)
        
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            initial_state = {"messages": [("user", user_input)]}
            response = agent_executor.invoke(initial_state, config=config)
            
            final_reply = response["messages"][-1].content
            
            st.write(final_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": final_reply})