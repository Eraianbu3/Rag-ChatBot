import streamlit as st
import os
import nest_asyncio
from main import BossWallahChatbot

nest_asyncio.apply()

st.set_page_config(
    page_title="ChatBot AI Support",
    page_icon="🎓",
    layout="wide"
)

os.environ['GOOGLE_API_KEY'] = ''

def load_chatbot():
    return BossWallahChatbot()

def get_chatbot_response(query, chatbot, selected_language):
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "response": "",
        "language": selected_language.lower(),
        "relevance_score": 0.0,
        "has_relevant_info": False
    }
    
    result = chatbot.app.invoke(initial_state)
    return result

def main():
    st.title("ChatBot AI Support Agent")
    st.write("Ask me anything about courses!")
    
    with st.sidebar:
        st.header("Settings")
        language_options = ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam"]
        selected_language = st.selectbox(
            "Response Language",
            options=language_options,
            index=0
        )
    
    try:
        chatbot = load_chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.stop()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What would you like to know about our courses?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = get_chatbot_response(prompt, chatbot, selected_language)
                    response = result.get('response', 'No response generated')
                    
                    st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sample Questions")
    sample_questions = [
        "Tell me about honey bee farming course",
        "I want to learn how to start a poultry farm",
        "Do you have any courses in Tamil?",
        "I am a recent high school graduate, are there any opportunities for me?",
        "What courses are available for entrepreneurs?"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(question, key=f"sample_{question[:20]}"):
            st.session_state.messages.append({"role": "user", "content": question})
            
            try:
                result = get_chatbot_response(question, chatbot, selected_language)
                response = result.get('response', 'No response generated')
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()

if __name__ == "__main__":
    main()