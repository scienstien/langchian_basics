import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

import os
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

def Vector_stores(text):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text, embedding=embeddings)
    return vector_store

def get_conversation_chain(Vector):
    # Use a local model pipeline instead of HuggingFaceHub
    local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question.

        Context: {context}
        Question: {question}
        Answer:"""
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory
    )

    retriever = Vector.as_retriever()

    qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=chain,
        return_source_documents=True
    )

    return qa

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    text = text_spliter.split_text(raw_text)
    return text

def main():
    load_dotenv()
    print("HF Token Loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    st.set_page_config(page_title="Spider Task 2", page_icon=":books:")

    st.header("Spider Task 2 :books:")
    st.text_input("Ask questions from the pdfs")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    with st.sidebar:
        st.subheader("Your Header")
        pdf_docs = st.file_uploader('Upload your pdf here', accept_multiple_files=True)
        if st.button("Thinkingt"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                Vector = Vector_stores(text_chunks)
                st.session_state.conversation = get_conversation_chain(Vector)
                st.success("Successfully loaded")

    st.session_state.conversation

if __name__ == '__main__':
    main()
