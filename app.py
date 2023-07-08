import os
from dotenv import load_dotenv

from typing import Union

from supabase.client import Client, create_client

from vectorstore import CustomSupabaseVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings()


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_PRIVATE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

vectorstore = CustomSupabaseVectorStore(
    client=supabase, table_name="documents", embedding=embeddings
)


@app.get("/")
def home():
    return "sup"


@app.get("/search")
def search(query: str):
    docs = vectorstore.similarity_search(query)

    return docs[0].page_content
