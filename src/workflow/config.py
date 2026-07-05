from langchain_community.vectorstores import FAISS
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

# Shared Bedrock kwargs
_bedrock_kwargs = dict(
    model="deepseek.v3.2",
    api_key=AWS_BEARER_TOKEN_BEDROCK,
    region_name="us-east-1",
)

# --- Task-specific models (previously all main_model) ---

# For binary decisions: retrieval_decider, is_relevant
# Low temp → deterministic, consistent classification
decision_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.0)

# For generating retriever & web-search queries
# Moderate temp → some variety in query formulation
query_gen_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.4)

# For open-ended answer generation: direct_generation, answer_from_context
# Higher temp → natural, fluent responses
generation_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.7)

# For groundedness verification: check_answer_grounded
# Zero temp → strict, factual evaluation
grounding_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.0)


# Rewrites queries to improve retrieval quality
query_rewrite_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.3)

# Judges whether an answer is useful to the user
judge_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.0)

# Revises/improves an answer that failed grounding
critic_model = ChatBedrockConverse(**_bedrock_kwargs, temperature=0.5)

# vector_store
vector_store = FAISS.load_local(
    "C:\\Users\\panka\\genai_project\\constitution_rag\\data\\constitution_and_ipc.faiss",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# search tools
tavily_tool = TavilySearch(max_results=3)
