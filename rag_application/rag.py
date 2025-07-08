import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import  HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

def Vector_stores(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return db

# Use this version of the function if you want to use the local model
# 
# def get_conversation_chain(db, query , k=3):
    # docs = db.similarity_search(query, k=k)
    # docs_text = "".join([doc.page_content for doc in docs])
    # local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    # llm = HuggingFacePipeline(pipeline=local_pipeline)
    # propmt = PromptTemplate(
    #     input_variables= ['question', 'context'],
    #     template = """
    # Answer the question as clearly and completely as possible using the provided context.
    # If the answer requires multiple points, list them as bullet points.

    # Context:
    # {context}

    # Question:
    # {question}

    # Answer:
    # """
    # )
    # chain = LLMChain(llm=llm, prompt=propmt, verbose=True)
    # response = chain.run({
    #     'question': query,
    #     'context': docs_text
    # })
    # response = response.replace('\n', ' ')
    # return response

# Use this version of the function if you want to use the google model(cloud model)
def get_conversation_chain(db, query , k=3):


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, api_key=os.getenv("GOOGLE_API_KEY"))  # Store the API key in the .env file
    docs = db.similarity_search(query, k=k)
    docs_text = "".join([doc.page_content for doc in docs])
    prompt = PromptTemplate(
        input_variables= ['question', 'context'],
        template = """
        "You are a helpful and honest assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not contained within the context, just say you don't know. Do not make up an answer.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
        """
    )
  
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = chain.run({
        'question': query,
        'context': docs_text
    })
    return response, f"\n Source is {docs_text.strip()}"


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

