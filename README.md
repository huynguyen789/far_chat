# Federal Acquisition Regulation (FAR) Chat Assistant

## Description

This Streamlit application serves as an interactive chat assistant for the Federal Acquisition Regulation (FAR). It uses Google's Generative AI to provide accurate and helpful information about FAR, allowing users to ask questions and receive detailed responses.

## Features

- Interactive chat interface
- Real-time responses using Google's Generative AI
- Conversation history tracking
- Automatic conversation summarization
- Debugging information in the sidebar
- Ability to clear conversation history

## Requirements

- Python 3.7+
- Streamlit
- Google Generative AI library
- Access to Google AI API (API key required)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/far-chat-assistant.git
   cd far-chat-assistant
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google AI API key:
   - Create a `secrets.toml` file in the `.streamlit` directory
   - Add your API key to the file:
     ```
     GOOGLE_API_KEY = "your-api-key-here"
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Start chatting with the FAR assistant by typing your questions in the input box.

## File Structure

- `app.py`: Main application file
- `prompts/`: Directory containing prompt templates
  - `system_instruction.txt`: System instructions for the AI model
  - `chat_content.txt`: Template for chat content
  - `summary_prompt.txt`: Template for conversation summarization
- `docs/`: Directory containing the FAR document
  - `FAR_28-39.rtf`: FAR document (sections 28-39)

## How It Works

1. The application loads the FAR document and initializes the Google Generative AI model.
2. Users input questions about FAR in the chat interface.
3. The app processes the query, combining it with the conversation history and FAR document.
4. The AI model generates a response, which is displayed in real-time.
5. The conversation history is updated and summarized periodically.
6. Debugging information and a clear conversation option are available in the sidebar.

## Customization

- Adjust the `generation_config` in the code to modify the AI's response characteristics.
- Update the `safety_settings` to change content filtering preferences.
- Modify the prompt templates in the `prompts/` directory to alter the AI's behavior or context.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Disclaimer

This application is for informational purposes only and should not be considered as legal advice. Always consult with a qualified legal professional for specific FAR-related inquiries.