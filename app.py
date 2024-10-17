import os
import subprocess
import pkg_resources
import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
import threading

# Function to install required packages if they are not already installed
def install_if_needed(package, version):
    try:
        pkg = pkg_resources.get_distribution(package)
    except pkg_resources.DistributionNotFound:
        subprocess.check_call(["pip", "install", f"{package}=={version}"])

# Install necessary packages
install_if_needed("langchain", "0.2.2")
install_if_needed("langchain-openai", "0.1.8")
install_if_needed("unstructured", "0.14.4")
install_if_needed("chromadb", "0.5.0")

# Set your OpenAI API key
openai_api_key = "openai key"  

# Initialize speech recognizer and TTS engine
recognizer = sr.Recognizer()

# Function to convert text to speech using threading
def text_to_speech(text):
    def speak():
        tts_engine = pyttsx3.init()  # Initialize TTS engine inside the thread
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 1)
        tts_engine.say(text)
        tts_engine.runAndWait()

    # Start the TTS in a separate thread
    tts_thread = threading.Thread(target=speak)
    tts_thread.start()

# Set up Streamlit page configuration
st.set_page_config(page_title="AutoMate", layout="wide")

# CSS to style the title and options
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: navy;
    }
    .options {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: navy;
    }
    .answer {
        font-size: 24px;
        font-weight: bold;
        color: orange;  /* Couleur orange */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the centered title
st.markdown('<h1 class="title">Drive with Confidence, Powered by Intelligence</h1>', unsafe_allow_html=True)

# Load the background image
with open("C:/Users/hamdi/Desktop/ragù/d.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode()

# Create the style with the base64-encoded image
page_bg_img = f'''
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{base64_image}");
    background-size: cover;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Option for user to select input type (audio or text)
input_option = st.radio("", ('Speak out loud', 'Type it out'))

# Store query and answer in session state to prevent infinite loops
if 'rag_answer' not in st.session_state:
    st.session_state['rag_answer'] = None
if 'query' not in st.session_state:
    st.session_state['query'] = ""

# Function to capture audio input
def capture_audio():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.write(f"Recognized: {query}")
            return query
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.write("Could not request results from the service.")
    return ""

# Load the HTML file as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="C:/Users/hamdi/Desktop/ragù/mg-zs-warning-messages.html")
car_docs = loader.load()

# Initialize RecursiveCharacterTextSplitter to make chunks of HTML text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(car_docs)

# Initialize Chroma vectorstore with documents as splits
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Setup vectorstore as retriever
retriever = vectorstore.as_retriever()

# Define RAG prompt
prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
)

# Initialize chat-based LLM
model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

# Setup the chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

# Handle input/output
if input_option == 'Type it out':
    user_input = st.text_input("Ask a question", key="typed_query")
    if user_input:  # Check if there's new input
        st.session_state['query'] = user_input
        st.session_state['rag_answer'] = None  # Reset answer for new query

else:
    if st.button("Record Audio"):
        st.session_state['query'] = capture_audio()
        st.session_state['rag_answer'] = None  # Reset answer for new audio input

# Check for specific dangerous situation and respond accordingly
if "my car is burning" in st.session_state['query'].lower():
    st.session_state['rag_answer'] = "Jump! There is no solution now, just save your life."
    text_to_speech(st.session_state['rag_answer'])  # Call text-to-speech for this specific answer

# Only run if there's a query and no answer has been generated yet
elif st.session_state['query'] and not st.session_state['rag_answer']:
    st.session_state['rag_answer'] = rag_chain.invoke(st.session_state['query']).content
    text_to_speech(st.session_state['rag_answer'])  # Call text-to-speech after generating the answer

# Display the answer if it exists
if st.session_state['rag_answer']:
    st.markdown(f'<div class="answer">{st.session_state["rag_answer"]}</div>', unsafe_allow_html=True)

# Reset button (optional) to allow new queries and reset the app state
if st.button("Reset"):
    st.session_state['query'] = ""
    st.session_state['rag_answer'] = None  # Reset the answer as well
