import os
import sys
import yaml

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)["test_set_generation"]

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    eval_config = yaml.safe_load(f)

generate_test_set = params.get("generate_test_set", True)
test_size = params.get("test_size", 50)

if not generate_test_set:
    print("generate_test_set is set to False in params.yaml. Exiting test set generation stage.")
    sys.exit(0)

from phoenix.otel import register
tracer_provider = register(
  project_name="constitution",
  auto_instrument=True 
)

import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper

warnings.filterwarnings("ignore")

load_dotenv()

kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")
test_set_path = os.path.join(DATA_DIR, "test_set.csv")

generator_embeddings = HuggingFaceEmbeddings(model_name=eval_config["embeddings"]["model_name"])
generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)

generator_llm = ChatBedrockConverse(
    model=eval_config["models"]["llm_model_id"],
    api_key=os.environ['AWS_BEARER_TOKEN_BEDROCK'],
    region_name=eval_config["models"]["region"],
    temperature=eval_config["models"]["llm_temperature"],
  )

generator_llm = LangchainLLMWrapper(generator_llm)

kg = KnowledgeGraph().load(kg_path)

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)

query_distribution = default_query_distribution(generator_llm)

print(f"Generating test set (size={test_size})...")
dataset = generator.generate(testset_size=test_size, query_distribution=query_distribution)

# Convert to pandas and save to CSV
df_dataset = dataset.to_pandas()
df_dataset.to_csv(test_set_path, index=False)
print(f"Test set generated and saved to {test_set_path}")
