import json
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
import os

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "workflow", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    wf_config = yaml.safe_load(f)

model_name = wf_config["embeddings"]["model_name"]
embeddings = BedrockEmbeddings(model_id=model_name, region_name=wf_config["models"]["region"])

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

vector_store=Chroma(
    embedding_function=embeddings,
    collection_name=wf_config["vector_store"]["collection_name"],
    persist_directory=os.path.join(data_dir,"constitution_and_ipc.chroma")
)
vector_store.add_documents(documents)