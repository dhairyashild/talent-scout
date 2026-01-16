import os
os.environ["LANGCHAIN_VERBOSE"] = "false"

import streamlit as st
from decouple import config

# LangChain imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Initialize Streamlit
st.set_page_config(page_title="TalentScout AI", layout="centered")
st.title("🤖 TalentScout AI - PG-AGI")

# ========== LANGCHAIN SETUP ==========
class TalentScoutLangChain:
    def __init__(self):
        # Initialize LLM
        token = config("HUGGINGFACEHUB_API_TOKEN")
        endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=token,
            temperature=0.1,
            max_new_tokens=50
        )
        self.llm = ChatHuggingFace(llm=endpoint)
        
        # Define validation chains for each field
        self.validation_chains = self._create_validation_chains()
        
        # Define tech question chain
        self.tech_chain = self._create_tech_chain()
    
    def _create_validation_chains(self):
        """Create specific validation chains for each field"""
        chains = {}
        
        # Name validation: must have only letters
        name_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid full name containing ONLY letters, spaces, hyphens, or apostrophes?
            Rules: NO numbers allowed. Must be at least 2 characters.
            
            Examples:
            - "John Smith" → VALID
            - "John123" → INVALID
            - "5" → INVALID
            - "Mary O'Connor" → VALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["full name"] = RunnableSequence(name_prompt, self.llm, StrOutputParser())
        
        # Email validation: must have @ and domain
        email_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid email address with @ symbol and proper domain?
            Rules: Must have @, must have domain with dot, no spaces.
            
            Examples:
            - "john@example.com" → VALID
            - "bh" → INVALID
            - "john.com" → INVALID
            - "name@domain.co.uk" → VALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["email"] = RunnableSequence(email_prompt, self.llm, StrOutputParser())
        
        # Phone validation: must be 10 digits only
        phone_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid 10-digit phone number containing ONLY digits 0-9?
            Rules: Exactly 10 digits, no letters, no symbols.
            
            Examples:
            - "1234567890" → VALID
            - "54" → INVALID
            - "123-456-7890" → INVALID
            - "123456789012" → INVALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["phone"] = RunnableSequence(phone_prompt, self.llm, StrOutputParser())
        
        # Location validation: reasonable location name
        location_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid location name?
            Rules: Must be at least 2 characters, can contain letters, numbers, spaces, commas.
            
            Examples:
            - "Pune" → VALID
            - "New York" → VALID
            - "123 Main St" → VALID
            - "A" → INVALID
            - "@#$" → INVALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["location"] = RunnableSequence(location_prompt, self.llm, StrOutputParser())
        
        # Experience validation: numbers only
        experience_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input valid experience in numbers only?
            Rules: Must contain only digits 0-9, can include "years" but must have numbers.
            
            Examples:
            - "4" → VALID
            - "2 years" → VALID
            - "five" → INVALID
            - "4+" → VALID
            - "1-3" → VALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["experience"] = RunnableSequence(experience_prompt, self.llm, StrOutputParser())
        
        # Position validation: letters only, no numbers
        position_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid job position containing ONLY letters, no numbers?
            Rules: Must have only letters, spaces, slashes, hyphens. NO numbers.
            
            Examples:
            - "AI/ML Intern" → VALID
            - "4" → INVALID
            - "Data Scientist" → VALID
            - "Developer 2" → INVALID
            - "AI Intern 2024" → INVALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["position"] = RunnableSequence(position_prompt, self.llm, StrOutputParser())
        
        # Tech stack validation: comma-separated list
        tech_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Is this input a valid tech stack list?
            Rules: Comma separated list of technologies, at least 1 item.
            
            Examples:
            - "Python, Django" → VALID
            - "nh,ki" → INVALID (not real tech names)
            - "React, Node.js, MongoDB" → VALID
            - "" → INVALID
            - "Python" → VALID
            
            Input: {input}
            Answer (VALID or INVALID only):"""
        )
        chains["tech stack"] = RunnableSequence(tech_prompt, self.llm, StrOutputParser())
        
        return chains
    
    def _create_tech_chain(self) -> RunnableSequence:
        """Create LangChain tech question chain"""
        tech_prompt = PromptTemplate(
            input_variables=["tech_stack", "topic"],
            template="""Ask one technical question about {topic} for an AI/ML intern.
            Tech Stack: {tech_stack}
            Question:"""
        )
        
        return RunnableSequence(
            tech_prompt,
            self.llm,
            StrOutputParser()
        )
    
    def validate(self, user_input: str, field: str) -> bool:
        """Run validation chain"""
        try:
            if field not in self.validation_chains:
                return False
                
            result = self.validation_chains[field].invoke({"input": user_input})
            response = result.strip().upper()
            
            # Debug: show what LLM returned
            st.sidebar.write(f"Validation for {field}: '{user_input}' → '{response}'")
            
            return "VALID" in response and "INVALID" not in response
        except Exception as e:
            st.sidebar.error(f"Validation error for {field}: {e}")
            return False
    
    def generate_tech_question(self, tech_stack: str, topic: str) -> str:
        """Run tech question chain"""
        try:
            return self.tech_chain.invoke({
                "tech_stack": tech_stack,
                "topic": topic
            })
        except Exception as e:
            return f"Explain {topic}."

# Initialize LangChain system
try:
    langchain_system = TalentScoutLangChain()
    st.success("✅ LangChain system initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize LangChain: {e}")
    st.stop()

# ========== APPLICATION STATE ==========
class AppState:
    def __init__(self):
        self.questions = [
            {"field": "full name", "text": "What is your full name? (letters only, no numbers)"},
            {"field": "email", "text": "What is your email? (must have @ and domain)"},
            {"field": "phone", "text": "What is your phone number? (10 digits only)"},
            {"field": "location", "text": "What is your current location?"},
            {"field": "experience", "text": "Years of experience? (numbers only)"},
            {"field": "position", "text": "What position? (letters only, no numbers)"},
            {"field": "tech stack", "text": "List your tech stack: (comma separated real technologies)"}
        ]
        
        self.tech_topics = ["Python", "Machine Learning", "Algorithms"]
        
        # Initialize session state
        if "current_step" not in st.session_state:
            st.session_state.current_step = 0
            st.session_state.answers = {}
            st.session_state.chat = []
            st.session_state.tech_q_count = 0
            st.session_state.retry_count = 0
            
            # Start with first question
            self.add_to_chat("assistant", self.questions[0]["text"])
    
    def add_to_chat(self, role: str, message: str):
        """Add message to chat history"""
        st.session_state.chat.append({"role": role, "content": message})
    
    def get_current_question(self):
        """Get current question based on step"""
        if st.session_state.current_step < len(self.questions):
            return self.questions[st.session_state.current_step]
        return None
    
    def handle_validation(self, user_input: str):
        """Handle input validation using LangChain"""
        current_q = self.get_current_question()
        
        if not current_q:
            return
        
        # Validate using LangChain
        is_valid = langchain_system.validate(user_input, current_q["field"])
        
        if is_valid:
            # Store answer
            field_key = current_q["field"].replace(" ", "_")
            st.session_state.answers[field_key] = user_input
            st.session_state.current_step += 1
            st.session_state.retry_count = 0
            
            # Check if we need tech questions
            if st.session_state.current_step >= len(self.questions):
                self.start_tech_questions()
            else:
                # Ask next question
                next_q = self.questions[st.session_state.current_step]
                self.add_to_chat("assistant", next_q["text"])
        else:
            # Invalid input
            st.session_state.retry_count += 1
            
            if st.session_state.retry_count >= 3:
                # After 3 retries, force move
                field_key = current_q["field"].replace(" ", "_")
                st.session_state.answers[field_key] = f"[Invalid: {user_input}]"
                st.session_state.current_step += 1
                st.session_state.retry_count = 0
                
                if st.session_state.current_step >= len(self.questions):
                    self.start_tech_questions()
                else:
                    next_q = self.questions[st.session_state.current_step]
                    self.add_to_chat("assistant", f"⚠️ Skipping. {next_q['text']}")
            else:
                # Ask again
                self.add_to_chat("assistant", f"❌ Invalid {current_q['field']}. Try again ({st.session_state.retry_count}/3):")
    
    def start_tech_questions(self):
        """Start technical questioning phase"""
        self.add_to_chat("assistant", "✅ Basic info collected! Now technical questions...")
        self.ask_next_tech_question()
    
    def ask_next_tech_question(self):
        """Ask next technical question using LangChain"""
        if st.session_state.tech_q_count < len(self.tech_topics):
            topic = self.tech_topics[st.session_state.tech_q_count]
            tech_stack = st.session_state.answers.get("tech_stack", "")
            
            # Generate tech question using LangChain
            question = langchain_system.generate_tech_question(tech_stack, topic)
            
            self.add_to_chat("assistant", 
                           f"**Tech Q{st.session_state.tech_q_count + 1}:** {question}")
        else:
            self.add_to_chat("assistant", "🎉 Screening complete! Thank you.")

# Initialize app state
app_state = AppState()

# ========== STREAMLIT UI ==========
# Sidebar
with st.sidebar:
    st.header("📊 Progress & Debug")
    
    # Calculate progress
    total_steps = len(app_state.questions) + len(app_state.tech_topics)
    current_progress = st.session_state.current_step + st.session_state.tech_q_count
    
    if current_progress > 0:
        progress = current_progress / total_steps
        st.progress(min(progress, 1.0))
    
    # Show current phase
    if st.session_state.current_step < len(app_state.questions):
        st.info(f"🔹 Info Collection: {st.session_state.current_step + 1}/{len(app_state.questions)}")
        if st.session_state.retry_count > 0:
            st.warning(f"Retries: {st.session_state.retry_count}/3")
    elif st.session_state.tech_q_count < len(app_state.tech_topics):
        st.info(f"🔸 Tech Questions: {st.session_state.tech_q_count + 1}/{len(app_state.tech_topics)}")
    else:
        st.success("✅ Complete")
    
    # Show collected info
    if st.session_state.answers:
        st.divider()
        st.subheader("📋 Collected Info")
        for key, value in st.session_state.answers.items():
            if value:
                st.write(f"**{key.title()}:** {value}")

# Display chat
for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_input := st.chat_input("Type your answer..."):
    # Add user message
    app_state.add_to_chat("user", user_input)
    
    # Check current phase
    if st.session_state.current_step < len(app_state.questions):
        # In info collection phase - validate
        app_state.handle_validation(user_input)
    elif st.session_state.tech_q_count < len(app_state.tech_topics):
        # In tech questions phase - just accept answer
        st.session_state.tech_q_count += 1
        if st.session_state.tech_q_count < len(app_state.tech_topics):
            app_state.ask_next_tech_question()
        else:
            app_state.add_to_chat("assistant", "🎉 All questions answered! Screening complete.")
    
    st.rerun()

# Footer
st.divider()
st.caption("PG-AGI TalentScout AI | Strict LLM Validation")
