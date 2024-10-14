import requests
import json
import re
import streamlit as st

# TODO: RAG of the different tools, its architecture, pros and cons, tasks its good at, what its trained on

def display_chatbot():
    st.subheader("Chatbot Assistant")
    st.write("Describe your business requirement, and the assistant will recommend the best tool and model to use.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
        
        st.session_state.chat_history.append({"role": "system", "content": "You are a helpful assistant specialized in computer vision models."})

    user_input = st.text_input("You:", key="input")
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        
        conversation = ''
        for message in st.session_state.chat_history:
            if message['role'] == 'system':
                conversation += f"{message['content']}\n\n"
            elif message['role'] == 'user':
                conversation += f"User: {message['content']}\n"
            else:
                conversation += f"Assistant: {message['content']}\n"

        conversation += "Assistant:"

        # Send the conversation to Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                headers={'Content-Type': 'application/json'},
                data=json.dumps({
                    'model': 'llama3.1', 
                    'prompt': conversation,
                })
            )

            if response.status_code == 200:
                response_text = '[' + re.sub(r'\}\s*\{', '}, {', response.text) + ']'
                try:
                    response_data= json.loads(response_text)
                    responses=[obj['response'] for obj in response_data]
                    assistant_reply = " ".join(responses).strip()
        
                except json.JSONDecodeError:
                    assistant_reply = "Error in processing the response. Check the formatting."
                # Append reply to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            else:
                assistant_reply = "Sorry, I couldn't process your request."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        except requests.exceptions.RequestException as e:
            st.error("Error connecting to Ollama API. Please ensure it's running.")
            st.stop()

    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message['role'] == "assistant":
            st.markdown(f"**Assistant:** {message['content']}")
        # elif message['role'] == "system":
            
        #     pass