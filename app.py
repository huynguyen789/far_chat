import os
import streamlit as st
import google.generativeai as genai
from streamlit.components.v1 import html

# Function to load prompts from files
def load_prompt(filename):
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(prompt_path, "r") as file:
        return file.read().strip()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Load FAR document
@st.cache_resource
def load_far_document():
    with open('/Users/huyknguyen/Desktop/redhorse/code_projects/far_chat/docs/FAR_28-39.rtf', 'r') as file:
        return file.read()

far_text = load_far_document()

# Create the model
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    # "top_k": 64,
    "max_output_tokens": 8192,
}

system_instruction = load_prompt("system_instruction.txt")
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    }
]

model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=system_instruction,
            safety_settings=safety_settings
        )

def chat_with_far(query):
    # Prepare the full context for the model
    conversation_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
    full_context = load_prompt("chat_content.txt").format(
        far_text=far_text,
        conversation_history=conversation_string,
        query=query
    )
    
    response = model.generate_content(full_context, stream=True)
    
    full_response = ""
    for chunk in response:
        content = chunk.text
        full_response += content
        yield content
    
    # Add the query and response to the conversation history
    st.session_state.conversation_history.append({"role": "human", "content": query})
    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
    
    # Limit the conversation history to the last 10 exchanges
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history = st.session_state.conversation_history[-20:]
    
    return full_response

def summarize_conversation():
    conversation_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
    summary_prompt = load_prompt("summary_prompt.txt").format(conversation=conversation_string)

    summary_response = model.generate_content(summary_prompt)
    return summary_response.text

# JavaScript code to scroll to the bottom
scroll_script = """
<script>
    function scrollToBottom() {
        var chatContainer = parent.document.querySelector('section.main');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    scrollToBottom();
</script>
"""






# Streamlit UI

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'introduced' not in st.session_state:
    st.session_state.introduced = False

st.title("Federal Acquisition Regulation (FAR) Chat Assistant")

# Add a sidebar for debugging and the clear button
with st.sidebar:
    st.header("Debugging Information")
    st.subheader("Full Conversation History")
    st.json(st.session_state.conversation_history)
    
    # Move the clear button to the sidebar
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.summary = ""
        st.session_state.introduced = False
        st.rerun()

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    if st.session_state.summary:
        with st.expander("Conversation Summary", expanded=False):
            st.markdown(st.session_state.summary)
    
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Introduction message
if not st.session_state.introduced:
    with chat_container:
        with st.chat_message("assistant"):
            intro_message = "Hello! I'm your Federal Acquisition Regulation (FAR) Chat Assistant. I'm here to help you navigate and understand the Federal Acquisition Regulation. Feel free to ask me any questions about FAR, and I'll do my best to provide accurate and helpful information. How can I assist you today?"
            st.markdown(intro_message)
            st.session_state.conversation_history.append({"role": "assistant", "content": intro_message})
    st.session_state.introduced = True

# User input
if prompt := st.chat_input("Ask a question about FAR:"):
    # Display user message in chat message container
    with chat_container:
        with st.chat_message("human"):
            st.markdown(prompt)

    # Generate and display assistant response
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in chat_with_far(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response)

            # Scroll down after each response chunk
            st.markdown(scroll_script, unsafe_allow_html=True)
    # Check if we need to summarize (every 20 messages)
    if len(st.session_state.conversation_history) % 20 == 0:
        st.session_state.summary = summarize_conversation()
        st.experimental_rerun()  # Rerun