## Introduction:

This project implements a LLM based Chatbot using Transformers and Langchain. It is specifically used for assisting the users planning to apply to Karunya.

## Key Features

- Developed a chat interface using Streamlit.
- Scraped the relevant data from Karunya's website and stored it in a MySQL database.
- Included support for vernacular languages.
- Utilised a RAG system for data withdrawal from a vector database using `langchain`
- Added functionality for the chatbot to remember previous chats.
- Implemented a NER (Named Entity Recognition) based user data collection for Vernacular Language Chat.
- Implemented a personalized style based on the field of interest of the user.
- Implemented TTS with vernacular support

## Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/kevin-291/llm-chatbot.git
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running Scripts:

```bash
streamlit run app.py
```
- Do ensure that `streamlit` is installed in your system before running the script.

### 4. Further Development:

- Incorporating `vLLM` for a high-throughput and memory-efficient inference and serving engine for LLMs.
- Incorporating STT with support for vernacular languages.

