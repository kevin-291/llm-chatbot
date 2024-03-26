import streamlit as st
from model import user_input, get_pdf_text, get_text_chunks, get_vector_store
from language_utils import translate_text, detect_language
from indic_trans import find_lang, english_to_indic, indic_to_english
import base64
import tempfile

def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    question_lang = detect_language(user_question)
    indic_lan = find_lang(question_lang)

    if user_question:
        if question_lang != 'en':  # Ensure API key and user question are provided
            translated_question = indic_to_english(user_question, indic_lan)
            # user_input(user_question)
            audio_data = user_input(translated_question)
        else:
            audio_data = user_input(user_question)

        if st.button("Play Audio"):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(audio_data.read())
            tmp.close()
            audio_file = open(tmp.name, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") is True:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

