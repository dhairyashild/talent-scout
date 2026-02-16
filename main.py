import streamlit as st
import os
from decouple import config
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. API Configuration
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config("HUGGINGFACEHUB_API_TOKEN")

# 2. Model Initialization
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.7,
    max_new_tokens=512,
)
model = ChatHuggingFace(llm=llm)

st.title("TALENT-SCOUT HIRING CHATBOT")

# 3. System Prompt (Your logic remains the same)
hiring_system_prompt = """
You are TalentScout Hiring Assistant. EXCELLENT PROFESSIONAL HIRING BOT.

***STAY ON CURRENT STEP UNTIL VALID INPUT - THEN MOVE TO NEXT***
PHASE 1: INFORMATION GATHERING (Steps 1-7)
1. FULL NAME → 2. EMAIL → 3. PHONE → 4. YEARS EXPERIENCE → 5. DESIRED POSITION → 6. LOCATION → 7. TECH STACK

PHASE 2: TECHNICAL ASSESSMENT (Step 8)
8. Ask 3 technical question based on their EXACT tech stack , whether answers given from student may be correct or wrong just take it do not validate the answer only when student gives answer of 3 questions asked based on techstack given .in other time llm can check the input validation given according to above.

VALIDATION RULES (Reject invalid → "Please provide VALID [field]"):
- NAME: "John Doe" (real name, space required)
- EMAIL: name@domain.com format  
- PHONE: 10 digits or +91xxxxxxxxxx
- EXPERIENCE: Number of years (0 for freshers, 1, 2, 5, etc.) or "X years" format
- POSITION: Any technology role (DevOps, Developer, Engineer, Data Scientist, etc.)
- LOCATION: "City, State" or "City, Country" or "city"
- TECH STACK: "Python, Django, AWS" or "MLflow, Kubeflow"

TECH QUESTIONS (After Step 7 - response):
**Tech Stack Questions**: Generate 3-5 relevant technical questions based on declared stack.whether answers given from student may be correct or wrong just take it do not validate the answer only when student gives answer of 3 questions asked based on techstack given .in other time llm can check the input validation given according to above.whatever questions u will ask should ask one by one dont ask 3 questions simultaneously and wait for user entering the anser

CRITICAL BEHAVIOR:
✅ "Step 1/8 ✓ Great! Next:" (valid answer)
❌ "Please enter VALID [field]. Step X/8" (invalid)
🔚 if aspirant enter anytime "bye/exit" →then chatbot should exit the conversation "Thank you! Profile saved. HR will contact you in 48hrs."

RESPONSE FORMAT:
- Professional, brief (2-3 sentences max)
- Track progress: "Step X/8"
- Reference history: "Since you mentioned Python..."
- NEVER ask 2 questions at once
- End after 3 tech question answer: "Interview complete!"

EXAMPLE FLOW:
Bot: "Step 1/8: Please share your FULL NAME"
User: "John Doe" → Bot: "Step 1/8 ✓ Great! Step 2/8: What's your EMAIL?"
...
User: "Python,Django" → Bot: "Step 7/8 ✓ Perfect! Based on tech stack generate 3 question dont just stick to python or django aspirant whatever enters the tech stack according to that generate questions 
User: "Answer..." → Bot: "Excellent! Interview complete. Thank you!"

CRITICAL: Read ALL chat history before responding. Stay on track.
"""

# 4. Initialize Session State with GREETING
if "msg" not in st.session_state:
    greeting = """👋 **Welcome to TalentScout Hiring Assistant!**

I'm here to help screen you for technology positions. This will take just 5-8 minutes.

**To begin, please share your FULL NAME:**"""
    st.session_state.msg = [{"role": "assistant", "content": greeting}]

# 5. Display existing chat history
for chat in st.session_state.msg:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# 6. Define Prompt with History Placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", hiring_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # This is the "Memory"
    ("human", "{user_input}")
])

# 7. Handle User Input
start = st.chat_input("Enter your response...")

if start:
    # Append User Input to History
    st.session_state.msg.append({"role": "user", "content": start})
    with st.chat_message("user", avatar="user"):
        st.write(start)

    # Generate Response
    with st.chat_message("assistant", avatar="assistant"):
        # We pass the existing history list to the prompt
        ready_prompt = prompt.format_messages(
            user_input=start,
            chat_history=st.session_state.msg
        )
        
        output = model.invoke(ready_prompt)
        response_text = output.content
        
        st.write(response_text)
        
        # Append AI Response to History
        st.session_state.msg.append({"role": "assistant", "content": response_text})
