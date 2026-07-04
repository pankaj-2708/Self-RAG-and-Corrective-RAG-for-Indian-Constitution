import os
import json
import ast
import warnings
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.run_config import RunConfig

warnings.filterwarnings("ignore")
os.environ['LANGSMITH_TRACING_V2'] = "false"

load_dotenv()

if not os.environ.get('OLLAMA_API_KEY'):
    raise ValueError("No ollama api key")
else:
    OLLAMA_API_KEY = os.environ['OLLAMA_API_KEY']

# Resolve paths relative to the file location to make it run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
sampled_df_path = os.path.join(DATA_DIR, "sampled_df.csv")
kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")

sampled_df = pd.read_csv(sampled_df_path)

generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)

generator_llm = ChatOllama(
    model="nemotron-3-ultra:cloud",
    base_url="https://ollama.com",
    temperature=0,
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)

documents = []
for i in range(sampled_df.shape[0]):
    documents.append(
        Document(
            page_content=sampled_df['document'].values[i],
            metadata=json.loads(json.dumps(ast.literal_eval(sampled_df['metadata'].values[i])))
        )
    )

my_run_config = RunConfig(
    max_workers=4,      
    timeout=180,
    max_retries=10,
    max_wait=60,
    log_tenacity=True,
)

if os.path.isfile(kg_path):
    kg = KnowledgeGraph().load(kg_path)
else:
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        ))


trans = default_transforms(documents=documents, llm=generator_llm, embedding_model=generator_embeddings)

print("Applying transformations to Knowledge Graph...")
apply_transforms(kg, trans, run_config=my_run_config)

kg.save(kg_path)
print(f"Knowledge graph saved successfully to {kg_path}")
