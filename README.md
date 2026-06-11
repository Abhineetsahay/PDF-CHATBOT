# PDF Chatbot

A simple Streamlit app for chatting with the contents of an uploaded PDF.

## Features

- Upload a PDF file
- Split the document into chunks
- Create embeddings with Hugging Face
- Store chunks in Chroma
- Ask questions using Groq-powered chat responses
- Keep chat history during the session

## Requirements

- Python 3.10+
- A Groq API key in your environment

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set your environment variables in a `.env` file if needed.

## Run

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Upload a PDF.
3. Ask questions about the document.
4. Use the sidebar button to clear the chat.

## Notes

- The app uses the uploaded PDF only for the current session.
- Large PDFs may take a moment to process.