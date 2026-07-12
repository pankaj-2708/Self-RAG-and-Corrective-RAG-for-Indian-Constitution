from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

if not os.environ.get('AWS_BEARER_TOKEN_BEDROCK'):
    raise ValueError("No AWS bearer token for Bedrock")
else:
    AWS_BEARER_TOKEN_BEDROCK = os.environ['AWS_BEARER_TOKEN_BEDROCK']

# Models
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Shared Bedrock kwargs — DeepSeek V3
_bedrock_kwargs = dict(
    model="deepseek.v3.2",
    api_key=AWS_BEARER_TOKEN_BEDROCK,
    region_name="us-east-1",
)

# Shared Bedrock kwargs — DeepSeek R1 (reasoning model)
# R1 controls reasoning internally; temperature must be 1 on Bedrock
_r1_kwargs = dict(
    model="us.deepseek.r1-v1:0",
    api_key=AWS_BEARER_TOKEN_BEDROCK,
    region_name="us-east-1",
)

# --- Task-specific models (previously all main_model) ---

# For routing the query: retrieval / web_search / None
# R1's chain-of-thought reasoning handles complex tiebreaker rules
retrieval_decider_model = ChatBedrockConverse(**_r1_kwargs, temperature=1)

# For binary relevance check on each retrieved chunk (runs N times per query)
# V3 is sufficient and much cheaper for this simple entailment task
decision_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.0)

# For generating retriever & web-search queries
# Moderate temp → some variety in query formulation
query_gen_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.4)

# For open-ended answer generation: direct_generation
# Higher temp → natural, fluent responses
generation_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.7)

# For answer_from_context: R1's chain-of-thought reasoning produces
# more grounded, well-reasoned answers from retrieved context
# R1 controls reasoning internally; temperature must be 1 on Bedrock
context_answer_model = ChatBedrockConverse(**_r1_kwargs, temperature=1)

# For groundedness verification: check_answer_grounded
# Zero temp → strict, factual evaluation
grounding_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.0)


# Rewrites answers to better address the user's query
# R1's chain-of-thought reasoning produces better targeted rewrites
answer_rewrite_model = ChatBedrockConverse(**_r1_kwargs, temperature=1)

# Judges whether an answer is relevant to the user's query
# R1's reasoning helps catch subtle relevance issues
judge_model = ChatBedrockConverse(**_r1_kwargs, temperature=1)

# Revises/improves an answer that failed grounding
# R1's chain-of-thought reasoning produces more accurate revisions
critic_model = ChatBedrockConverse(**_r1_kwargs, temperature=1)

# vector_store
vector_store = Chroma(
    collection_name="constitution_and_ipc",
    persist_directory="C:\\Users\\panka\\genai_project\\constitution_rag\\data\\constitution_and_ipc.chroma",
    embedding_function=embeddings,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# search tools
tavily_tool = TavilySearch(max_results=2)
