import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import  HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain

import os
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

def Vector_stores(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return db

def get_conversation_chain(db, query , k=3):
    docs = db.similarity_search(query, k=k)
    docs_text = "".join([doc.page_content for doc in docs])
    local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    propmt = PromptTemplate(
        input_variables= ['question', 'context'],
        template = """
    Answer the question as clearly and completely as possible using the provided context.
    If the answer requires multiple points, list them as bullet points.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )
    chain = LLMChain(llm=llm, prompt=propmt, verbose=True)
    response = chain.run({
        'question': query,
        'context': docs_text
    })
    response = response.replace('\n', ' ')
    return response

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    text_chunks = text_spliter.split_text(text)
    return text_chunks

