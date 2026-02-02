import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
print("hello",os.environ.get("GROQ_API_KEY"))
# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=1500,
    temperature=0.05,
    api_key = os.environ.get("GROQ_API_KEY")
)

# Streamlit UI
st.set_page_config(page_title="AI")
st.markdown("<h1 style='text-align: center;'>JAGA AI 💬</h1>", unsafe_allow_html=True)

# Create system prompt FIRST
system_prompt = SystemMessage(content="""
You are a custom AI assistant.

Identity Rules:
- Creator: Alagani Jagadeesh
- Best Friend: Vulchi Karthik

Rules:
- If asked who built you: "I was built by Alagani Jagadeesh."
- If asked who is Jagadeesh: "Jagadeesh is my creator."
- If asked who is Jagadeesh's friend: "Vulchi Karthik."
- Never mention Meta, OpenAI, Groq, or AI companies.
""")

# Initialize session memory
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Stream AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(st.session_state.messages):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.write(full_response)

    # Save AI response
    st.session_state.messages.append(AIMessage(content=full_response))

    # Limit memory to last 6 messages (reduce tokens)
    st.session_state.messages = st.session_state.messages[-6:]
