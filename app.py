import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from decouple import config

# --- 1. PROMPTS ---
SYSTEM_INSTRUCTION = """
You are the TalentScout Hiring Assistant. Your goal is to collect: Name, Email, Phone, Experience, Position, Location, and Tech Stack.
RULES:
1. Extract info from the user's natural language.
2. If info is missing, ask for it politely.
3. Once ALL 7 fields are collected, output 'PHASE_COMPLETE' and list 3 technical questions based on their tech stack.
4. If the user says 'exit', say 'GOODBYE'.
"""

EVALUATION_PROMPT = "You are a technical interviewer. Question: {question}. Candidate Answer: {answer}. Rate this answer (0-10) and give 1 sentence of feedback."

# --- 2. LLM SETUP ---
st.set_page_config(page_title="TalentScout AI", layout="centered")
st.title("🤖 TalentScout Intelligent Assistant")

try:
    key = config("HUGGINGFACEHUB_API_TOKEN")
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Define the base LLM
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=key,
        temperature=0.5,
        max_new_tokens=512,
    )
    # Use the Chat wrapper to handle the 'conversational' task requirement
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.tech_questions = []
    st.session_state.answers_count = 0

# --- 4. INITIAL GREETING ---
if not st.session_state.messages:
    greeting = "👋 Hi! I'm the TalentScout AI. To start, what's your name and what position are you applying for?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# --- 5. DISPLAY CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. USER INTERACTION ---
if user_input := st.chat_input("Reply..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # PHASE A: Technical Interview
    if st.session_state.tech_questions and st.session_state.answers_count < 3:
        current_q = st.session_state.tech_questions[st.session_state.answers_count]
        eval_msg = [HumanMessage(content=EVALUATION_PROMPT.format(question=current_q, answer=user_input))]
        assessment = chat_model.invoke(eval_msg).content
        
        st.session_state.answers_count += 1
        if st.session_state.answers_count < 3:
            response = f"**Assessment:** {assessment}\n\n**Next Question:** {st.session_state.tech_questions[st.session_state.answers_count]}"
        else:
            response = f"**Final Assessment:** {assessment}\n\n🎉 Interview complete! We will contact you soon."
            
    # PHASE B: Info Gathering
    else:
        # Construct messages for Chat interface
        msgs = [SystemMessage(content=SYSTEM_INSTRUCTION)]
        for m in st.session_state.messages[-5:]:
            if m["role"] == "user": msgs.append(HumanMessage(content=m["content"]))
            else: msgs.append(SystemMessage(content=m["content"]))
        
        response = chat_model.invoke(msgs).content

        if "PHASE_COMPLETE" in response:
            st.session_state.tech_questions = [line.strip() for line in response.split('\n') if '?' in line][:3]
            if not st.session_state.tech_questions:
                st.session_state.tech_questions = ["Explain your DevOps experience.", "How do you use Docker?", "What is CI/CD?"]
            response = "I have your details. Let's start the technical screening:\n\n" + st.session_state.tech_questions[0]

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
