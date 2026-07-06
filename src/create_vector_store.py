import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

data_dir=os.path.join(os.path.dirname(__file__),"../data")



data={}
with open(os.path.join(data_dir,'articles.json'),'r') as f:
    data=json.load(f)
    
    
documents=[]
for key,value in zip(data.keys(),data.values()):
    article_no=key.split()[1]
    documents.append(Document(metadata={"Article":article_no},page_content=f"{key} \n: {value}"))


data={}
with open(os.path.join(data_dir,'penal_code_sections.json'),'r') as f:
    data=json.load(f)


for key,value in data.items():
    section_no=key.split()[2]
    documents.append(Document(metadata={"Section":section_no},page_content=f"{key} \n{value}"))
    

vector_store=Chroma(embedding_function=embeddings,collection_name="constitution_and_ipc",persist_directory=os.path.join(data_dir,"constitution_and_ipc.chroma"))
vector_store.add_documents(documents)