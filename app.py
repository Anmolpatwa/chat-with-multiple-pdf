import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding_model_name = "models/embedding-001"
chat_model_name = "models/gemini-1.5-pro-latest"  


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say "The answer is not available in the context." Do not make up an answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model=chat_model_name,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a question about the uploaded PDF:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()
