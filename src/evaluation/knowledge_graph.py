import os
import sys
import yaml

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))
kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)["knowledge_graph"]

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    eval_config = yaml.safe_load(f)

create_knowledge_graph = params.get("create_knowledge_graph", True)

if not create_knowledge_graph:
    # assuming knowledge_graph.json already exists, just rewriting it for dvc
    if os.path.exists(kg_path):
        from ragas.testset.graph import KnowledgeGraph
        kg = KnowledgeGraph().load(kg_path)
        print(f"create_knowledge_graph is set to False in params.yaml. Loaded knowledge graph from {kg_path}")
        kg.save(kg_path)
    else:
        print(f"create_knowledge_graph is set to False in params.yaml and {kg_path} does not exist. Exiting.")
    sys.exit(0)

from phoenix.otel import register
tracer_provider = register(
  project_name="constitution",
  auto_instrument=True 
)

import json
import ast
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

sampled_df_path = os.path.join(DATA_DIR, "sampled_df.csv")
    

generator_embeddings = HuggingFaceEmbeddings(model_name=eval_config["embeddings"]["model_name"])
generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)

generator_llm = ChatBedrockConverse(
    model=eval_config["models"]["llm_model_id"],
    api_key=os.environ['AWS_BEARER_TOKEN_BEDROCK'],
    region_name=eval_config["models"]["region"],
    temperature=eval_config["models"]["llm_temperature"],
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

rc = eval_config["knowledge_graph"]["run_config"]
my_run_config = RunConfig(
    max_workers=rc["max_workers"],      
    timeout=rc["timeout"],
    max_retries=rc["max_retries"],
    max_wait=rc["max_wait"],
    log_tenacity=rc["log_tenacity"],
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
