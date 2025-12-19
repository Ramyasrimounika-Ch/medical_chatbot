from src.helper import read_data
from langchain_community.vectorstores import FAISS
import pickle

file_path="vector.pkl"
def load_data():
    splitted_data,embedding=read_data("medical_data.pdf")
    db=FAISS.from_documents(splitted_data,embedding)
    with open(file_path,"wb") as f:
        pickle.dump(db,f)