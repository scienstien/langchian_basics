import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import time

# Try different import strategies for different LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader


def validate_huggingface_token(token):
    """Validate HuggingFace API token"""
    try:
        # Test with HuggingFace whoami API
        headers = {"Authorization": f"Bearer {token}"}
        test_url = "https://huggingface.co/api/whoami"
        
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            st.success(f"✅ Token valid for user: {user_info.get('name', 'Unknown')}")
            return True
        elif response.status_code == 401:
            st.error("❌ Token is invalid or unauthorized")
            return False
        else:
            st.warning(f"Token validation returned status: {response.status_code}")
            return False
            
    except Exception as e:
        st.warning(f"Could not validate token: {str(e)}")
        return False


def get_available_models():
    """Get list of available models from HuggingFace"""
    try:
        # These are known working models that are publicly available
        reliable_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "google/flan-t5-small",
            "google/flan-t5-base",
            "facebook/blenderbot-400M-distill",
            "huggingface/CodeBERTa-small-v1",
            "distilgpt2",
            "gpt2"
        ]
        return reliable_models
    except Exception as e:
        st.error(f"Error getting model list: {str(e)}")
        return []


def test_model_availability(model_name, token):
    """Test if a specific model is available and working"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        test_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # First, check if model exists
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            return False, "Model not found"
        elif response.status_code == 403:
            return False, "Access forbidden"
        elif response.status_code == 401:
            return False, "Unauthorized"
        
        # Test with a simple inference
        payload = {"inputs": "Hello, how are you?"}
        response = requests.post(test_url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            return True, "Available"
        elif response.status_code == 503:
            return False, "Model loading"
        else:
            return False, f"Status {response.status_code}"
            
    except Exception as e:
        return False, str(e)


class SimpleHuggingFaceLLM:
    """Simple HuggingFace API client with improved error handling"""
    
    def __init__(self, repo_id="microsoft/DialoGPT-medium", api_token=None):
        self.repo_id = repo_id
        self.api_token = api_token
        self.api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        
        # Different models need different parameter configurations
        self.model_configs = {
            "microsoft/DialoGPT-medium": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            },
            "microsoft/DialoGPT-small": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            },
            "google/flan-t5-small": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True
            },
            "google/flan-t5-base": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True
            },
            "facebook/blenderbot-400M-distill": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True
            },
            "gpt2": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            },
            "distilgpt2": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256
            }
        }
    
    def __call__(self, prompt):
        """Make API call to HuggingFace with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get model-specific configuration
                config = self.model_configs.get(self.repo_id, {
                    "max_length": 100,
                    "temperature": 0.7,
                    "do_sample": True
                })
                
                # Prepare payload
                payload = {
                    "inputs": prompt,
                    "parameters": config,
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                }
                
                # Make API call
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            generated_text = result[0].get('generated_text', '')
                        else:
                            generated_text = str(result[0])
                    elif isinstance(result, dict):
                        generated_text = result.get('generated_text', '')
                    else:
                        generated_text = str(result)
                    
                    # Clean up the response
                    if generated_text:
                        # Remove the original prompt if it's repeated
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        
                        # Clean up common artifacts
                        generated_text = generated_text.replace('<|endoftext|>', '').strip()
                        
                        if generated_text:
                            return generated_text
                    
                    return "I couldn't generate a proper response. Please try rephrasing your question."
                
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        st.info(f"Model is loading, waiting... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(5 * (attempt + 1))  # Longer wait for model loading
                        continue
                    return "Model is currently loading. Please try again in a moment."
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        st.warning(f"Rate limit hit, waiting... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(10)  # Rate limit - wait longer
                        continue
                    return "Rate limit exceeded. Please try again later."
                elif response.status_code == 401:
                    return "Authentication failed. Please check your API token."
                elif response.status_code == 403:
                    return "Access forbidden. Please check your token permissions."
                elif response.status_code == 404:
                    return "Model not found. The model may have been moved or is no longer available."
                else:
                    error_text = response.text if response.text else "Unknown error"
                    return f"API Error: {response.status_code} - {error_text}"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                return "Request timed out. Please try again."
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Error occurred, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                    continue
                return f"Error: {str(e)}"
        
        return "Failed to get response after multiple attempts."


def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks with better error handling"""
    try:
        if not text_chunks:
            st.error("No text chunks to process")
            return None
            
        # Filter out empty chunks
        valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]
        
        if not valid_chunks:
            st.error("No valid text chunks found")
            return None
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.from_texts(texts=valid_chunks, embedding=embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def create_llm():
    """Create HuggingFace LLM with better model selection and token validation"""
    try:
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            st.error("No API token found!")
            return None
        
        # First, validate the token
        if not validate_huggingface_token(api_token):
            st.error("❌ Invalid HuggingFace API token. Please check your token.")
            return None
        
        # Get available models
        models = get_available_models()
        
        # Test models and find the first working one
        st.info("🔍 Testing available models...")
        
        for model in models:
            with st.spinner(f"Testing {model}..."):
                is_available, status = test_model_availability(model, api_token)
                
                if is_available:
                    st.success(f"✅ {model} is available")
                    # Try to create and test the LLM
                    try:
                        llm = SimpleHuggingFaceLLM(repo_id=model, api_token=api_token)
                        test_response = llm("Hello")
                        
                        if (test_response and 
                            "Error" not in test_response and 
                            "403" not in test_response and
                            "401" not in test_response and
                            "404" not in test_response and
                            "loading" not in test_response.lower() and 
                            "failed" not in test_response.lower()):
                            st.success(f"✅ Successfully connected to model: {model}")
                            st.info(f"Test response: {test_response[:100]}...")
                            return llm
                        else:
                            st.warning(f"⚠️ Model {model} gave poor response: {test_response}")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Model {model} failed during testing: {str(e)}")
                        continue
                else:
                    st.warning(f"❌ Model {model} gave poor response: {status}")
        
        st.error("❌ Could not connect to any HuggingFace model")
        st.error("❌ Failed to create LLM.")
        
        # Provide helpful troubleshooting information
        st.info("💡 Troubleshooting tips:")
        st.info("1. Check your HuggingFace token permissions (needs 'read' access)")
        st.info("2. Try generating a new token from https://huggingface.co/settings/tokens")
        st.info("3. Wait a few minutes and try again (models might be loading)")
        st.info("4. Check HuggingFace status page for service issues")
        
        return None
        
    except Exception as e:
        st.error(f"Error creating LLM: {str(e)}")
        return None


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with better error handling"""
    text = ''
    try:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                pdf_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1} of {pdf.name}")
                        continue
                
                if pdf_text.strip():
                    text += pdf_text
                    st.info(f"✅ Extracted text from {pdf.name}")
                else:
                    st.warning(f"⚠️ No text found in {pdf.name}")
                    
            except Exception as e:
                st.error(f"Error reading {pdf.name}: {str(e)}")
                continue
                
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDFs: {str(e)}")
        return ""


def get_text_chunks(raw_text):
    """Split text into chunks for processing with validation"""
    try:
        if not raw_text or not raw_text.strip():
            st.error("No text to split into chunks")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(raw_text)
        
        # Filter out very short chunks
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        if not valid_chunks:
            st.error("No valid text chunks could be created")
            return []
            
        return valid_chunks
        
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []


def simple_qa(question, vector_store, llm):
    """Improved QA function with better prompt engineering"""
    try:
        if not question.strip():
            return {"answer": "Please ask a valid question.", "source_documents": []}
        
        # Get relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "source_documents": []
            }
        
        # Combine relevant text (limit context to avoid token limits)
        context_parts = []
        total_length = 0
        max_context_length = 800  # Reduced for better model compatibility
        
        for doc in docs:
            if total_length + len(doc.page_content) < max_context_length:
                context_parts.append(doc.page_content)
                total_length += len(doc.page_content)
            else:
                # Add partial content if it fits
                remaining_space = max_context_length - total_length
                if remaining_space > 100:  # Only add if meaningful amount of space left
                    context_parts.append(doc.page_content[:remaining_space])
                break
        
        context = "\n\n".join(context_parts)
        
        # Create a better prompt for different model types
        if "flan-t5" in llm.repo_id:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Based on this context, answer the question:\n\n{context}\n\nQ: {question}\nA:"
        
        # Get answer from LLM
        answer = llm(prompt)
        
        # Clean up the answer
        if answer:
            answer = answer.strip()
            # Remove any repetition of the question
            if answer.lower().startswith(question.lower()):
                answer = answer[len(question):].strip()
            # Remove common artifacts
            answer = answer.replace("A:", "").replace("Answer:", "").strip()
        
        return {
            "answer": answer if answer else "I couldn't generate a proper answer.",
            "source_documents": docs
        }
        
    except Exception as e:
        st.error(f"Error in QA: {str(e)}")
        return {"answer": f"Error processing question: {str(e)}", "source_documents": []}


def handle_user_input(user_question):
    """Handle user question and display response"""
    if not user_question.strip():
        st.warning("Please enter a question.")
        return
        
    if st.session_state.vector_store is not None and st.session_state.llm is not None:
        with st.spinner("Thinking..."):
            try:
                result = simple_qa(user_question, st.session_state.vector_store, st.session_state.llm)
                
                if result:
                    # Display the answer
                    st.write("**Answer:**")
                    st.write(result["answer"])
                    
                    # Display source documents if available
                    if result["source_documents"]:
                        with st.expander("📄 View Sources"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.write(f"**Source {i+1}:**")
                                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.write(preview)
                                st.write("---")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": result["answer"]
                    })
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    else:
        st.warning("Please upload and process PDF files first!")


def display_chat_history():
    """Display chat history"""
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")


def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API token is available
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        st.error("⚠️ Please set your HUGGINGFACEHUB_API_TOKEN in your .env file")
        st.info("Create a .env file with: HUGGINGFACEHUB_API_TOKEN=your_token_here")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        st.stop()
    
    st.set_page_config(
        page_title="PDF Q&A Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.header("📚 PDF Question & Answer Assistant")
    st.markdown("Upload PDF files and ask questions about their content!")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input
        user_question = st.text_input("💬 Ask a question about your PDFs:", key="user_input")
        if user_question:
            handle_user_input(user_question)
        
        # Display chat history
        if st.session_state.chat_history:
            display_chat_history()
    
    with col2:
        # PDF upload section
        st.subheader("📄 Upload Documents")
        pdf_docs = st.file_uploader(
            'Choose PDF files',
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to analyze"
        )
        
        if pdf_docs:
            st.write(f"Selected {len(pdf_docs)} file(s):")
            for pdf in pdf_docs:
                st.write(f"- {pdf.name}")
        
        if st.button("🔄 Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Get PDF text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text could be extracted from the PDFs. Please check your files.")
                            st.stop()
                        
                        st.info(f"📄 Extracted {len(raw_text)} characters from PDF(s)")
                        
                        # Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                            st.error("❌ Could not create text chunks from the PDFs.")
                            st.stop()
                        
                        st.info(f"📝 Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        with st.spinner("Creating vector store..."):
                            vector_store = create_vector_store(text_chunks)
                        
                        if vector_store is None:
                            st.error("❌ Failed to create vector store.")
                            st.stop()
                        
                        st.info("✅ Vector store created successfully")
                        
                        # Create LLM
                        with st.spinner("Connecting to HuggingFace..."):
                            llm = create_llm()
                        
                        if llm is None:
                            st.error("❌ Failed to create LLM.")
                            st.stop()
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.llm = llm
                        st.session_state.processed_files = [pdf.name for pdf in pdf_docs]
                        
                        st.success(f"✅ Successfully processed {len(pdf_docs)} PDF(s) with {len(text_chunks)} text chunks!")
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred while processing: {str(e)}")
                        st.error("Please check your PDF files and try again.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        # Display current status
        st.subheader("📊 Status")
        if st.session_state.vector_store is not None and st.session_state.llm is not None:
            st.success("✅ Documents processed and ready!")
            if st.session_state.processed_files:
                st.write("**Processed files:**")
                for file in st.session_state.processed_files:
                    st.write(f"- {file}")
            
            # Show current model
            st.info(f"🤖 Using model: {st.session_state.llm.repo_id}")
            st.info(f"🔑 Using API token: {api_token[:10]}...****")
        else:
            st.info("📄 Upload PDFs and click 'Process Documents' to start")
        
        # Test connection button
        if st.button("🔧 Test API Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                # First validate token
                if not validate_huggingface_token(api_token):
                    st.error("❌ Token validation failed!")
                    st.info("Please check your HuggingFace token and permissions")
                    return
                
                # Test model availability
                models = get_available_models()
                st.info(f"Testing {len(models)} models...")
                
                working_models = []
                for model in models[:3]:  # Test first 3 models
                    is_available, status = test_model_availability(model, api_token)
                    if is_available:
                        working_models.append(model)
                        st.success(f"✅ {model}: {status}")
                    else:
                        st.warning(f"❌ {model}: {status}")
                
                if working_models:
                    st.success(f"✅ Found {len(working_models)} working models!")
                else:
                    st.error("❌ No working models found")
                    st.info("💡 Try again later or check HuggingFace status")
        
        # Clear session button
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.llm = None
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.success("Session cleared!")
            st.rerun()


if __name__ == '__main__':
    main()


import os, requests
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
res = requests.get("https://huggingface.co/api/whoami", headers={
    "Authorization": f"Bearer {token}"
})

print("Status code:", res.status_code)
print("Response:", res.text)
