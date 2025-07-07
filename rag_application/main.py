import streamlit as st
import rag_model
import textwrap


st.title("Rag Model")
with st.sidebar :
    with st.form("my_form"):
        pdf_docs = st.file_uploader('Upload your pdf here', accept_multiple_files=True)
        query = st.text_input("Ask questions from the pdfs")
        submit_button = st.form_submit_button(label="Submit")

if pdf_docs and query and submit_button:
    with st.spinner("Processing..."):
        raw_text = rag_model.get_pdf_text(pdf_docs)
        text_chunks = rag_model.get_text_chunks(raw_text)
        Vector = rag_model.Vector_stores(text_chunks)
        response = rag_model.get_conversation_chain(Vector, query)

        if response:
            st.success("Successfully loaded")
            wrapped_response = textwrap.fill(response, width=80)
            st.text_area("Response", value=wrapped_response, height=300)
        else:
            st.error("No response generated.")
