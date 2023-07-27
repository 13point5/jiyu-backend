import json
import os
from dotenv import load_dotenv

from typing import Union
from pydantic import BaseModel, Field

from supabase.client import Client, create_client

from vectorstore import CustomSupabaseVectorStore
import utils

from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema.messages import (
    SystemMessage,
)

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

# block1 = "k3u46gu4bg"
# block2 = "l136hn5j24n"
# block3 = "pdf_cognitivism"
# query = "What is the summary of <@block:{}>?".format(blockId)
# query = "What is the difference between the key ideas in <@block:{}> and <@block:{}>?".format(
#     "pdf_cognitivism", "l136hn5j24n"
# )
# query = "How are the key ideas in <@block:{}> and <@block:{}> related?".format(
#     block2, block3
# )


gpt_3 = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
gpt_4 = ChatOpenAI(model="gpt-4")

AGENT_SYSTEM_MESSAGE = """
You are a helpful AI assistant. 
A block is a reference to a document that contains information about it. 
When you respond to queries that mention blocks in the format <@block:BLOCK_ID>, you need to ensure that your response references the blocks in the same format.
"""


@app.get("/")
def home():
    return "sup"


class BlockTool(BaseModel):
    question: str = Field()


def block_tool_func(retriever, source_docs):
    def actual_tool_func(question: str):
        res = RetrievalQA.from_chain_type(
            llm=gpt_3,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )(question)

        source_docs.append(res["source_documents"])
        return res["result"]

    return actual_tool_func


class SearchResponse(BaseModel):
    output: str
    source_docs: list


@app.get("/qa")
def search(query: str) -> SearchResponse:
    mention_ids = utils.extract_mention_ids(query)

    if len(mention_ids) == 0:
        response = gpt_3.predict(query)
        return SearchResponse(output=response, source_docs=[])

    source_docs = []

    tools = [
        Tool(
            name="answer_question_about_block_{}".format(mention_id),
            func=block_tool_func(
                retriever=vectorstore.as_retriever(
                    search_kwargs={"filter": {"blockId": mention_id}}
                ),
                source_docs=source_docs,
            ),
            args_schema=BlockTool,
            description="useful for answering questions about block {}".format(
                mention_id
            ),
        )
        for mention_id in mention_ids
    ]

    agent = initialize_agent(
        tools,
        llm=gpt_4,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs={"system_message": SystemMessage(content=AGENT_SYSTEM_MESSAGE)},
        max_iterations=len(mention_ids) + 1,
        # verbose=True,
        # return_intermediate_steps=True,
    )

    res = agent({"input": query})

    return SearchResponse(output=res["output"], source_docs=source_docs)
