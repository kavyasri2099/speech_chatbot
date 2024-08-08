import os
import base64
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init

# Initialize Float feature
float_init()

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # Use GPT-2 for demonstration
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_file_path = 'temp_audio.mp3'
    tts.save(audio_file_path)
    return audio_file_path

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]

initialize_session_state()

st.title("Speech-to-Speech Conversational Bot 🤖")

footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.wav"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            if st.session_state.messages[-1]["role"] != "user":
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
            os.remove(webm_file_path)

# Check if the last message is from the user before generating a response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages[-1]["content"])
        with st.spinner("Generating audio response..."):    
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

footer_container.float("bottom: 0rem;")
