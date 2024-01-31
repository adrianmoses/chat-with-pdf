import tempfile

from dotenv import load_dotenv
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_community.docstore.document import Document
import chromadb

# Load environment variables
load_dotenv()


def ingest_documents(documents):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10)
    chunks = text_splitter.split_documents(documents)
    client = chromadb.Client()

    if not client.list_collections():
        client.create_collection("consent_collection")
    else:
        print("Collection already exists")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
    )

    vectordb.persist()
    return vectordb


def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def main():
    st.title("Chat with your PDF 💬")

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    documents = []

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        for i, page in enumerate(pdf_reader.pages):
            doc = Document(page_content=page.extract_text(), metadata={'page': i})
            documents.append(doc)

        # Create the knowledge base vectordb
        knowledgeBase = ingest_documents(documents)

        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledgeBase.similarity_search(query)

            chain = create_agent_chain()

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)


            st.write(response)



if __name__ == "__main__":
    main()
