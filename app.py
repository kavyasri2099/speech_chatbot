import streamlit as st
from transformers import pipeline
import pyttsx3
import soundfile as sf
import torch
import numpy as np

# Initialize models
stt_model = pipeline("automatic-speech-recognition", model="openai/whisper-large")
nlp_model = pipeline("text-generation", model="EleutherAI/gpt-j-6B")
tts_engine = pyttsx3.init()

# Set up the Streamlit page
st.set_page_config(page_title="🎙️ Voice Bot", layout="wide")
st.title("🎙️ Speech Bot")
st.sidebar.title("`Speak with LLMs` \n`in any language`")

def record_voice():
    st.write("Click the button below and speak.")
    audio_file = st.file_uploader("Upload audio file", type=["wav"])
    if audio_file:
        audio, sr = sf.read(audio_file)
        return audio, sr
    return None, None

def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    st.sidebar.header("Configuration")
    language = st.sidebar.selectbox("Language", ["en"])
    model = st.sidebar.selectbox("Model", ["GPT-J"])

    audio, sr = record_voice()

    if audio is not None:
        # Convert speech to text
        st.write("Processing your voice...")
        audio = torch.tensor(audio).unsqueeze(0).float()
        stt_result = stt_model(audio)["text"]

        st.write(f"You said: {stt_result}")

        # Generate a response
        response = nlp_model(stt_result, max_length=50)[0]['generated_text']
        st.write(f"Response: {response}")

        # Convert text to speech
        text_to_speech(response)

if __name__ == "__main__":
    main()
