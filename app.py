import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

# 1. Page Config
st.set_page_config(page_title="TalentScout Hiring Assistant")

# 2. Setup LLM (Using HuggingFace Endpoint for speed/efficiency)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"
model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_new_tokens=512
)

# 3. The RTICE Prompt we designed earlier
SYSTEM_PROMPT = """
**ROLE**: Senior Technical Recruiter at "TalentScout" Agency.
**TASK**: Screen candidates by gathering details and asking technical questions.
**INSTRUCTIONS**:
1. Greet and explain purpose.
2. Collect: Name, Email, Phone, Years of Exp, Position, and Location.
3. Ask for their Tech Stack.
4. Based on the stack, ask 3-5 technical questions ONE BY ONE.
5. Conclude gracefully.
**CONSTRAINTS**: Professional tone. End if "EXIT" is typed.
"""

# 4. State Management
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# 5. UI Layout
st.title("🤖 TalentScout Hiring Assistant")

# Display history
for message in st.session_state.messages:
    if not isinstance(message, SystemMessage):
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(message.content)

# 6. Chat Logic
if prompt := st.chat_input("Type your response here..."):
    # Show user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for Exit
    if any(keyword in prompt.lower() for keyword in ["exit", "quit", "bye"]):
        response = "Thank you for your time! The TalentScout team will review your info."
    else:
        # Generate LLM response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = model.invoke(st.session_state.messages)
    
    # Save and display assistant message
    st.session_state.messages.append(AIMessage(content=response))
    st.markdown(response)