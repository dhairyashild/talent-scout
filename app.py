# === COMPLETE CHAT SECTION (Copy this) ===
# Initialize LangChain messages
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Show chat history
for message in st.session_state.messages[1:]:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input + LLM call
if prompt := st.chat_input("Enter your response:"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if "exit" in prompt.lower():
        response = "Thank you! Team will contact you soon."
    else:
        with st.chat_message("assistant"):
            response = model.invoke(st.session_state.messages).content
            st.markdown(response)
    
    st.session_state.messages.append(AIMessage(content=response))
