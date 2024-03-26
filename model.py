from language_utils import translate_text, detect_language, text_to_speech
from indic_trans import find_lang, english_to_indic, indic_to_english
from IndicNER import get_predictions, tokenizer, model

import streamlit as st
from PyPDF2 import PdfReader
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline
)
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
#api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings()
    global vector_store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_entities(text):
    predicted_labels = get_predictions(sentence=text, tokenizer=tokenizer, model=model)
    entities = []
    for index in range(len(text.split(' '))):
        token = text.split(' ')[index]
        label = predicted_labels[index]
        if label != 'O':
            entities.append((token, label))
    return entities

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model = AutoModelForCausalLM.from_pretrained(model,
                                                # quantization_config=gptq.QuantizationConfig(
                                                # quantization_format="gptq",
                                                # tensor_name_to_quantization_configs={
                                                #     "lmhead.weight": gptq.DefaultQuantizationConfig(
                                                #     quantization_bit=8,
                                                #     distributed_tensors=True,
                                                #     quantization_scheme="tensor",
                                                #                                     ),
                                                # "lm_head.weight": gptq.DefaultQuantizationConfig(
                                                # quantization_bit=8,
                                                # distributed_tensors=True,
                                                # quantization_scheme="tensor",
                                                #                                     ),
                                                #                                 },
                                                #                             ),
                                             load_in_8bit=True,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    pipe = pipeline(
        task="text-generation", 
        model=model, 

        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16, 
        temperature = 0,
        device_map="auto")
    
    tiny_llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    retriever = vector_store.as_retriever()
    chain = ConversationalRetrievalChain(tiny_llm, retriever=retriever, memory=memory, prompt=prompt)
    return chain

def user_input(user_question):
    # embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    # response = chain({"context": docs, "question": user_question}, return_only_outputs=True)
    # st.write("Reply: ", response["output_text"])
    lang = detect_language(user_question)
    indic_lan = find_lang(lang)

    if lang != 'en':
        translated_question = indic_to_english(user_question, indic_lan)
        result = chain({"query": translated_question, "context": docs}, return_only_outputs=True)
        translated_response = english_to_indic(result["result"], indic_lan)
        st.write("Reply: ", translated_response)
        entities = get_entities(user_question)
        print("Detected Entities:", entities)

        audio_data = text_to_speech(translated_response, lang)
    else:
        result = chain({"query": user_question, "context": docs}, return_only_outputs=True)
        st.write("Reply: ", result["result"])
        audio_data = text_to_speech(result["result"], 'en')

    return audio_data