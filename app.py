from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma

load_dotenv(override=True)

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY", "")

st.set_page_config(page_title="Code of Conduct Assistant", page_icon="💰", layout="centered")
st.title("Code of Conduct Assistant")
st.caption("Ask questions about Code of Conduct of Apple and Google.")

@st.cache_resource
def init_agent():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./vector_store", embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="code_of_conduct_search_for_apple_and_google",
        description="Search this tool first for any questions regarding Code of Conduct of Apple and Google."
    )
    tools = [retriever_tool]
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    memory = MemorySaver()
    
    return create_react_agent(llm, tools, checkpointer=memory)

agent_executor = init_agent()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help you with Code of Conduct of Apple and Google."}
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
            
            final_reply = response["messages"][-1].content[0]["text"]
            
            st.write(final_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": final_reply})