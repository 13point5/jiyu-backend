import os
from dotenv import load_dotenv

from typing import Union

from supabase.client import Client, create_client

from vectorstore import CustomSupabaseVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

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

blockId = "k3u46gu4bg"
# blockId = "l136hn5j24n"

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-0613"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"filter": {"blockId": blockId}}),
)


@app.get("/")
def home():
    return "sup"


@app.get("/search")
def search(query: str):
    return qa.run(query)
