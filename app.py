import os
import streamlit as st
import google.generativeai as genai
from streamlit.components.v1 import html
import time

#Functions
def load_prompt(filename):
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(prompt_path, "r") as file:
        return file.read().strip()

def summarize_conversation():
    conversation_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
    summary_prompt = load_prompt("summary_prompt.txt").format(conversation=conversation_string)

    summary_response = model.generate_content(summary_prompt)
    return summary_response.text

# Handle user feedback
def set_user_feedback(feedback):
    st.session_state.user_feedback = feedback  # Replace previous feedback with new feedback

# Callback function to handle feedback submission
def submit_feedback():
    if st.session_state.feedback_input:
        set_user_feedback(st.session_state.feedback_input)
        st.session_state.feedback_input = ""  # Clear the input
        st.success("Thank you for your feedback! It has been incorporated into the current session.")
    else:
        st.warning("Please enter some feedback before submitting.")

# New function to clear feedback
def clear_feedback():
    st.session_state.user_feedback = ""
    st.success("Feedback has been cleared.")

#Calculate pricing
def calculate_price(input_tokens, output_tokens):
    input_price = 0.35 if input_tokens <= 128000 else 0.70
    output_price = 1.05 if input_tokens <= 128000 else 2.10
    
    return (input_tokens / 1000000) * input_price + (output_tokens / 1000000) * output_price


def display_pricing():
    st.header("Chat Pricing and Performance")
    
    if 'query_info' not in st.session_state:
        st.session_state.query_info = []
    
    total_price = sum(info['price'] for info in st.session_state.query_info)
    
    st.subheader("Query Information")
    for index, info in enumerate(st.session_state.query_info, start=1):
        st.text(f"Query {index}: ${info['price']:.6f}.\nTime: {info['time']:.2f}s")
    
    st.markdown("---")
    st.markdown(f"**Total: ${total_price:.6f}**")

def update_pricing(prompt_tokens, candidates_tokens):
    if 'query_info' not in st.session_state:
        st.session_state.query_info = []
    
    current_price = calculate_price(prompt_tokens, candidates_tokens)
    query_time = time.time() - st.session_state.query_start_time
    st.session_state.query_info.append({
        'price': current_price,
        'time': query_time
    })
    
    return current_price

# Function to update token counts
def update_token_counts(prompt_tokens, candidates_tokens):
    if 'token_counts' not in st.session_state:
        st.session_state.token_counts = []
    
    st.session_state.token_counts.append({
        'promptTokenCount': prompt_tokens,
        'candidatesTokenCount': candidates_tokens
    })
    
def chat_with_far(query):
    # Prepare the full context for the model
    conversation_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
    
    full_context = load_prompt("chat_content.txt").format(
        far_text=far_text,
        conversation_history=conversation_string,
        query=query,
        user_feedback=st.session_state.user_feedback
    )
    
    try:
        st.session_state.query_start_time = time.time()
        response = model.generate_content(full_context, stream=True)
        
        full_response = ""
        for chunk in response:
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text
                    full_response += content
                    yield content

                # Check finish reason after each chunk
                if candidate.finish_reason == "SAFETY":
                    safety_message = "\n\nNote: The response was filtered due to safety concerns.\nSafety ratings:\n"
                    for rating in candidate.safety_ratings:
                        safety_message += f"- Category: {rating.category}, Probability: {rating.probability}\n"
                    yield safety_message
                    break  # Stop streaming if we hit a safety filter
                
        # After processing all chunks, yield the query info
        if hasattr(response, 'usage_metadata'):
            prompt_tokens = response.usage_metadata.prompt_token_count
            candidates_tokens = response.usage_metadata.candidates_token_count
            current_price = update_pricing(prompt_tokens, candidates_tokens)
            query_time = time.time() - st.session_state.query_start_time
            
            # Yield a special message to signal query info
            yield f"QUERY_INFO:{prompt_tokens},{candidates_tokens},{current_price},{query_time}"



            # If no candidates or content, yield an empty string to maintain the stream
            if not chunk.candidates or not candidate.content or not candidate.content.parts:
                yield ""

        # Add the query and response to the conversation history
        st.session_state.conversation_history.append({"role": "human", "content": query})
        st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
        

    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.error(error_message)
        yield "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
 
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
#End functions
#############################################################################################################



#Set up
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 0,
    # "top_p": 0.95,
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

# Load FAR document
@st.cache_resource
def load_far_document():
    with open('./docs/far29-38.rtf', 'r') as file:
        return file.read()

far_text = load_far_document()
#End set up
#############################################################################################################





# Streamlit UI

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'introduced' not in st.session_state:
    st.session_state.introduced = False
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = ""  # Initialize as an empty string
if 'token_counts' not in st.session_state:
    st.session_state.token_counts = []
if 'query_prices' not in st.session_state:
    st.session_state.query_prices = []
if 'user_feedback_input' not in st.session_state:
    st.session_state.user_feedback_input = ""

st.title("Federal Acquisition Regulation (FAR) Chat Assistant")

#Side bar
with st.sidebar:
    # # Debugging information box
    # st.header("Debugging Information")
    # st.subheader("Full Conversation History")
    # st.json(st.session_state.conversation_history)
    
    # Move the clear button to the sidebar
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.summary = ""
        st.session_state.introduced = False
        st.rerun()
    
    # Feedback section 
    st.header("Feedback")
    st.markdown("""
    Provide feedback on how you'd like the assistant to behave. Examples:
    - "Please provide more concise/comprehensive answers"
    - "Quoted exactly the FAR text"
    - "Use simpler language for easier understanding"
    """)
    
    # Use a form to group the input and button
    with st.form(key='feedback_form'):
        user_feedback_input = st.text_area("Enter your feedback:", key="feedback_input", height=100)
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("Submit Feedback", on_click=submit_feedback)
        with col2:
            clear_button = st.form_submit_button("Clear Feedback", on_click=clear_feedback)
    
    # Display the current feedback
    st.subheader("Current Feedback:")
    st.text_area(
        label="Current feedback",
        value=st.session_state.user_feedback,
        height=100,
        disabled=True,
        key="current_feedback_display"
    )
    
    
    
# Main chat interface
# Use session state to maintain the feedback input value
user_feedback_input = st.session_state.user_feedback_input

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


if prompt := st.chat_input("Ask a question about FAR:"):
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in chat_with_far(prompt):
            if chunk.startswith("QUERY_INFO:"):
                # Extract query info and update session state
                prompt_tokens, candidate_tokens, current_price, query_time = map(float, chunk.split(":")[1].split(","))
                st.session_state.query_info.append({
                    'price': current_price,
                    'time': query_time
                })
            else:
                full_response += chunk
                message_placeholder.markdown(full_response)
        
        with st.sidebar:
            st.empty()  # Clear the previous content
            display_pricing()  # Display updated pricing and query info
        # Scroll down after each response chunk
        st.markdown(scroll_script, unsafe_allow_html=True)

