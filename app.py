import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import speech_recognition as sr
from gtts import gTTS
import os
import transformers
import numpy as np

# Initialize ChatterBot
bot = ChatBot("Candice")
trainer = ListTrainer(bot)
trainer.train(['What is your name?', 'My name is Candice'])
trainer.train(['Who are you?', 'I am a bot'])
trainer.train(['Who created you?', 'Tony Stark', 'Buddha', 'You?'])

corpus_trainer = ChatterBotCorpusTrainer(bot)
corpus_trainer.train("chatterbot.corpus.english")

# Initialize AI voice bot
class VoiceChatBot():
    def __init__(self, name):
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            st.write("Listening...")
            audio = recognizer.listen(mic)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Could not understand audio")
            return ""
        except sr.RequestError:
            st.write("Error with the request")
            return ""

    @staticmethod
    def text_to_speech(text):
        st.write(f"AI Response: {text}")
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("response.mp3")
        os.system("afplay response.mp3" if os.name == "posix" else "start response.mp3")
        os.remove("response.mp3")

    def wake_up(self, text):
        return self.name.lower() in text.lower()

    @staticmethod
    def action_time():
        return datetime.datetime.now().strftime('%H:%M')

voice_ai = VoiceChatBot(name="AI Buddha")
nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")

# Streamlit app layout
st.title("AI Voice Chatbot")

# Voice input button
if st.button("Speak"):
    user_input = voice_ai.speech_to_text()

    if voice_ai.wake_up(user_input):
        response = "Hello, I am Maya the AI. What can I do for you?"

    elif "time" in user_input:
        response = voice_ai.action_time()

    elif any(phrase in user_input for phrase in ["thank", "thanks"]):
        response = np.random.choice([
            "You're welcome!", "Anytime!", "No problem!", "Cool!", "I'm here if you need me!", "Peace out!"
        ])
    else:
        chat = nlp(transformers.Conversation(user_input), pad_token_id=50256)
        response = str(chat)
        response = response[response.find("bot >> ") + 6:].strip()

    voice_ai.text_to_speech(response)

# Text input box
user_text = st.text_input("You: ")

if user_text:
    bot_response = bot.get_response(user_text)
    st.write(f"Bot: {bot_response}")
