import json
import os
from dotenv import load_dotenv

from typing import Union
from pydantic import BaseModel, Field

from supabase.client import Client, create_client

from vectorstore import CustomSupabaseVectorStore
import utils

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

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
query = "What is the summary of <@block:{}>?".format(blockId)

query = "What is the difference between the key ideas in <@block:{}> and <@block:{}>?".format(
    "pdf_cognitivism", "l136hn5j24n"
)

# blockId = "l136hn5j24n"

chat_llm = ChatOpenAI(model="gpt-3.5-turbo-0613")


@app.get("/")
def home():
    return "sup"


class BlockTool(BaseModel):
    question: str = Field()


def block_tool_func(retriever, source_docs):
    def actual_tool_func(question: str):
        res = RetrievalQA.from_chain_type(
            llm=chat_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )(question)

        source_docs.append(res["source_documents"])
        return res["result"]

    return actual_tool_func


@app.get("/search")
def search():
    mention_ids = utils.extract_mention_ids(query)
    print("Query: {}".format(query))

    source_docs = []

    tools = [
        Tool(
            name="get_info_about_block_{}".format(mention_id),
            func=block_tool_func(
                retriever=vectorstore.as_retriever(
                    search_kwargs={"filter": {"blockId": mention_id}}
                ),
                source_docs=source_docs,
            ),
            args_schema=BlockTool,
            description="useful for finding information about block {}".format(
                mention_id
            ),
        )
        for mention_id in mention_ids
    ]

    mrkl = initialize_agent(
        tools,
        llm=chat_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        return_intermediate_steps=True,
    )

    res = mrkl({"input": query})
    print(source_docs)
    return res["output"]
