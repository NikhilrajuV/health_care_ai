# HealthCare AI

A web-based healthcare assistant built with Python and Streamlit.

## Features

- AI-powered health question answering
- Medical PDF upload and document management
- RAG using Google AI embeddings, LangChain and FAISS
- Image upload and AI image analysis
- Context-aware chat history
- Conversation export
- Medical source references when available
- Safety disclaimers for educational use

## Requirements

- Python 3.12
- Groq API key
- Google AI API key for embeddings and RAG indexing

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file using `.env.example` as a template.
5. Add your API keys locally.
6. Start the app:

   `streamlit run app.py`

## Medical Disclaimer

This application is for educational and informational purposes only. It does not provide confirmed medical diagnoses and is not a substitute for professional medical care.
