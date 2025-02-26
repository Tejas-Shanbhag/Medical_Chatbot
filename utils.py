import numpy as numpy
import pandas as pd

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore,Pinecone
from langchain_openai import OpenAI,ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  


def get_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def get_docstore(collection_name,embedding_name,api_key):
    docstore = PineconeVectorStore(index_name=collection_name,embedding=embedding_name,pinecone_api_key=api_key)
    return docstore

def load_gpt_model(openai_api_key):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.1,max_tokens=500,api_key=openai_api_key)
    return llm

def load_groq_model(groq_api_key):
    llm = ChatGroq(model="gemma2-9b-it",temperature=0.1,max_tokens=500,api_key=groq_api_key)
    return llm

def get_prompt_template():
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer or any question is out of context just say that you don't know."
                "Use five sentences maximum and keep the answer concise."
                "\n\n"
                "Context:{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("human","{input}")
        ]
    )

    return prompt
