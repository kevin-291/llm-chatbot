from langdetect import detect
from googletrans import Translator
from gtts import gTTS
import os
import io

translator = Translator()

def detect_language(text):
    """
    Detects the language of the given text.
    """
    try:
        lang = detect(text)
        if lang == 'en':
            return lang
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "en"  # Assume English if detection fails

def translate_text(text, target_lang="en"):
    """
    Translates the given text to the specified target language.
    If target_lang is not provided, it defaults to English.
    """
    try:
        src_lang = detect_language(text)
        if src_lang == target_lang:
            return text
        translated_text = translator.translate(text, src=src_lang, dest=target_lang).text
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text
    
def text_to_speech(text, lang):
    """
    Converts the given text to speech in the specified language using Google Text-to-Speech.
    Returns the audio data as a bytes-like object.
    """
    tts = gTTS(text=text, lang=lang)
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

# Demo working    
# text = 'हैलो, क्या हाल हैं'
# lang = detect_language(text)
# print(lang)
# print(translate_text(text))