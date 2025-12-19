from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_data(file_path):
    loader=PyPDFLoader(file_path)
    data=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=10)
    splitted_data=splitter.split_documents(data)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return splitted_data,embedding