import streamlit as st
import pyttsx3
import whisper
import numpy as np
import sounddevice as sd

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize Text-to-Speech
engine = pyttsx3.init()

def recognize_speech():
    st.write("Listening...")
    # Record audio
    duration = 5  # seconds
    sample_rate = 16000
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    recording = np.squeeze(recording)

    # Transcribe audio using Whisper
    result = whisper_model.transcribe(recording, language="en")
    text = result["text"]
    st.write(f"Recognized: {text}")
    return text

def generate_response(prompt):
    # Simple echo response for demonstration
    return f"You said: {prompt}"

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

st.title("Audio Chatbot with Whisper")

if st.button("Start"):
    user_text = recognize_speech()
    if user_text:
        response_text = generate_response(user_text)
        st.write(f"Response: {response_text}")
        speak_text(response_text)
