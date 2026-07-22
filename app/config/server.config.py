from dotenv import load_dotenv
load_dotenv()

import os

PORT = os.getenv("PORT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_TEMPERATURE = os.getenv("GROQ_TEMPERATURE")
GROQ_MAX_TOKENS = os.getenv("GROQ_MAX_TOKENS")

def Server_Credentials()->dict:
    return {
        "PORT":PORT,
        "GROQ_API_KEY":GROQ_API_KEY,
        "GROQ_MODEL":GROQ_MODEL,
        "GROQ_TEMPERATURE":GROQ_TEMPERATURE,
        "GROQ_MAX_TOKENS":GROQ_MAX_TOKENS
    }
