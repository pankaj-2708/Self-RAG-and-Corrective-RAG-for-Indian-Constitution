from phoenix.otel import register
tracer_provider = register(
  project_name="constitution",
  auto_instrument=True 
)

import os
import json
import yaml
import ast
import sys
import warnings
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_aws import ChatBedrockConverse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.run_config import RunConfig

warnings.filterwarnings("ignore")
os.environ['LANGSMITH_TRACING_V2'] = "false"

load_dotenv()

if not os.environ.get('AWS_BEARER_TOKEN_BEDROCK'):
    raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable is not set")

# Resolve paths relative to the file location to make it run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))
sampled_df_path = os.path.join(DATA_DIR, "sampled_df.csv")
kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")


with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)["knowledge_graph"]

create_knowledge_graph = params["create_knowledge_graph"]

if create_knowledge_graph:
    pass
else:
    # assuming knowledge_graph.json already exsits just rewritng it for dvc
    kg = KnowledgeGraph().load(os.path.join(DATA_DIR,"knowledge_graph2.json"))
    print(f"Loaded knowledge graph from {kg_path}")
    kg.save(kg_path)
    sys.exit()
    

generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)

generator_llm = ChatBedrockConverse(
    model="deepseek.v3.2",
    api_key=os.environ['AWS_BEARER_TOKEN_BEDROCK'],
    region_name="us-east-1",
    temperature=0,
)
generator_llm = LangchainLLMWrapper(generator_llm)

sampled_df = pd.read_csv(sampled_df_path)

documents = []
for i in range(sampled_df.shape[0]):
    documents.append(
        Document(
            page_content=sampled_df['document'].values[i],
            metadata=json.loads(json.dumps(ast.literal_eval(sampled_df['metadata'].values[i])))
        )
    )

my_run_config = RunConfig(
    max_workers=10,      
    timeout=180,
    max_retries=10,
    max_wait=60,
    log_tenacity=True,
)

# if os.path.isfile(kg_path):
#     kg = KnowledgeGraph().load(kg_path)
# else:
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
