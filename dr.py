import streamlit as st
from agents import Agent, Runner , OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
import os
import asyncio
from dotenv import load_dotenv

st.set_page_config(page_title="*Doctor Agent*",  page_icon="🩺", layout="wide")
st.title("Doctor Agent 🩺")
st.write("### This agent provides medical advice and treatment plans.📃 ###")

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key, #get key from environment variable 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # google api key is used 
)

model = OpenAIChatCompletionsModel( #model of ai assistant
    model="gemini-2.0-flash", #its a free model 
    openai_client=external_client # using openai framework 
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

Doctor = Agent(
        name ="Doctor",
        instructions ="You are a doctor. Provide medical advice and treatment plans based on the symptoms described by the patient.",
)

user_input = st.text_area("Enter your symptoms or medical questions 🩺:", height=200 )
result = None
if st.button("Get Medical Advice 👩🏼‍⚕️"):
    async def run_medical_advice():
        response = await Runner.run(
            Doctor,
            input = user_input,
            run_config = config
        )

        return response.final_output
    result = asyncio.run(run_medical_advice())
if result is not None:
    st.write("### Medical Advice Result👇🏼💉: ###")
    st.write(result)