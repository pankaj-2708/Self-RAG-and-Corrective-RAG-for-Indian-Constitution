from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

# ── Load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

_models = _cfg["models"]
_emb = _cfg["embeddings"]
_vs = _cfg["vector_store"]
_ret = _cfg["retriever"]
_ws = _cfg["web_search"]

# ── AWS auth ───────────────────────────────────────────────────────────────────
if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    raise ValueError("No AWS bearer token for Bedrock")
AWS_BEARER_TOKEN_BEDROCK = os.environ["AWS_BEARER_TOKEN_BEDROCK"]

# ── Embeddings ─────────────────────────────────────────────────────────────────
embeddings = BedrockEmbeddings(
    model_id=_emb["model_name"], region_name=_models["region"]
)

# ── Shared Bedrock kwargs ──────────────────────────────────────────────────────
_bedrock_kwargs = dict(
    model=_models["v3_model_id"],
    api_key=AWS_BEARER_TOKEN_BEDROCK,
    region_name=_models["region"],
)

# DeepSeek R1 (reasoning model) — temperature must be 1 on Bedrock
_r1_kwargs = dict(
    model=_models["r1_model_id"],
    api_key=AWS_BEARER_TOKEN_BEDROCK,
    region_name=_models["region"],
)

# ── Task-specific models ───────────────────────────────────────────────────────

# Routing: retrieval / web_search / None  — R1 chain-of-thought for tiebreaker rules
retrieval_decider_model = ChatBedrockConverse(
    **_r1_kwargs, temperature=_models["r1_temperature"]
)

# Binary relevance check per chunk — V3 sufficient and cheaper for simple entailment
decision_model = ChatBedrockConverse(
    **_bedrock_kwargs, temperature=_models["decision_temperature"]
)

# Sub-query & web-query generation — moderate temp for variety
query_gen_model = ChatBedrockConverse(
    **_bedrock_kwargs, temperature=_models["query_gen_temperature"]
)

# Open-ended direct generation — higher temp for fluency
generation_model = ChatBedrockConverse(
    **_bedrock_kwargs, temperature=_models["generation_temperature"]
)

# Answer from context — R1 chain-of-thought for grounded, well-reasoned answers
context_answer_model = ChatBedrockConverse(
    **_r1_kwargs, temperature=_models["r1_temperature"]
)

# Groundedness verification — zero temp for strict factual evaluation
grounding_model = ChatBedrockConverse(
    **_bedrock_kwargs, temperature=_models["grounding_temperature"]
)

# Answer rewriter — R1 for better targeted rewrites
answer_rewrite_model = ChatBedrockConverse(
    **_r1_kwargs, temperature=_models["r1_temperature"]
)

# Relevance judge — R1 to catch subtle relevance issues
judge_model = ChatBedrockConverse(**_r1_kwargs, temperature=_models["r1_temperature"])

# Grounding critic — R1 for accurate revisions of ungrounded answers
critic_model = ChatBedrockConverse(**_r1_kwargs, temperature=_models["r1_temperature"])

# ── Vector store ───────────────────────────────────────────────────────────────
_vs_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", _vs["persist_directory"])
)
vector_store = Chroma(
    collection_name=_vs["collection_name"],
    persist_directory=_vs_path,
    embedding_function=embeddings,
)
retriever = vector_store.as_retriever(
    search_type=_ret["search_type"],
    search_kwargs={"k": _ret["k"]},
)
max_retriever_queries = _ret.get("max_queries", 3)

# ── Web search ─────────────────────────────────────────────────────────────────
tavily_tool = TavilySearch(max_results=_ws["max_results"])
