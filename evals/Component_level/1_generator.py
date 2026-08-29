import os
import yaml
import pandas as pd
# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)['standalone_generator']
    
TEST_SET_PATH = os.path.abspath(os.path.join(DATA_DIR, params['test_set']))
OUTPUT_PATH = os.path.abspath(os.path.join(DATA_DIR, params['output']))


if not params['enabled']:
    print("Standalone generator evaluation is disabled. Set 'enabled' to True in params.yaml to enable it.")
    # create an empty csv file
    pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
    exit(0)
    
import sys
import mlflow
import time
from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import AmazonBedrockModel
from dotenv import load_dotenv


load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/pankaj-2708/Self-RAG-and-Corrective-RAG-for-Indian-Constitution.mlflow")
mlflow.set_experiment("constitution-rag")

if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_TRACING_V2") == "false":
    try:
        from phoenix.otel import register
        tracer_provider = register(
            project_name="constitution",
            auto_instrument=True 
        )
    except ImportError:
        pass




vector_store_path = os.path.abspath(os.path.join(DATA_DIR, params['vector_store_name']))
embeddings = BedrockEmbeddings(model_id=params["embeddings"]["model_name"], region_name=params["models"]["region"])

vector_store = Chroma(
    collection_name=params['vector_store_collection_name'],
    persist_directory=vector_store_path,
    embedding_function=embeddings,
)
retriever = vector_store.as_retriever(
    search_type=params["search_type"],
    search_kwargs={"k": params["k"]},
)

generator_llm = ChatBedrockConverse(
    model=params["llm_model_id"],
    region_name=params["region"],
    temperature=params["llm_temperature"]
)

judge_model = AmazonBedrockModel(
    model=params["llm_model_id"],
    region=params["region"],
    generation_kwargs={"temperature": params["llm_temperature"]} 
)

test_set = pd.read_csv(TEST_SET_PATH)

test_cases = []
latencies = []  # wall-clock seconds per row for retriever + generator pipeline
input_tokens = []
output_tokens = []
total_tokens = []

for i in range(len(test_set)):
    query = test_set.iloc[i]['user_input']
    ground_truth = test_set.iloc[i]['reference']
    t0 = time.perf_counter()
    retrieved_docs = retriever.invoke(query)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    
    context_str = "\n\n".join(retrieved_docs_text)
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
    
    response = generator_llm.invoke(prompt)
    latencies.append(time.perf_counter() - t0)
    
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    input_tokens.append(in_tok)
    output_tokens.append(out_tok)
    total_tokens.append(in_tok + out_tok)

    if isinstance(response.content, list):
        actual_output = " ".join([b["text"] for b in response.content if isinstance(b, dict) and b.get("type") == "text"]).strip()
    else:
        actual_output = str(response.content)
    
    test_cases.append(
        LLMTestCase(
            input=query,
            expected_output=ground_truth,
            retrieval_context=retrieved_docs_text,
            actual_output=actual_output
        )
    )

metrics = [
    FaithfulnessMetric(threshold=params['FAITHFULNESS_THRESHOLD'], model=judge_model, include_reason=True),
    AnswerRelevancyMetric(threshold=params['ANSWER_RELEVANCY_THRESHOLD'], model=judge_model, include_reason=True),
]

evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "FaithfulnessMetric_llm": params['llm_model_id'],
        "FaithfulnessMetric_threshold": params["FAITHFULNESS_THRESHOLD"],
        "AnswerRelevancyMetric_llm": params['llm_model_id'],
        "AnswerRelevancyMetric_threshold": params["ANSWER_RELEVANCY_THRESHOLD"],
        "Name": params['name'],
        "k": params["k"],
        "dataset_size": len(test_set),
    }
)

parsed_results = []
for test_result in evaluation_results.test_results:
    row = {
        "input": test_result.input,
        "actual_output": test_result.actual_output,
        "expected_output": test_result.expected_output,
        "retrieved_context": test_result.retrieval_context,
        "success": test_result.success
    }
    
    for metric in test_result.metrics_data:
        row[f"{metric.name} Score"] = metric.score
        row[f"{metric.name} Reason"] = metric.reason
        
    parsed_results.append(row)

df = pd.DataFrame(parsed_results)
df["latency_seconds"] = latencies   # align by position (same order as test_cases)
df["input_tokens"] = input_tokens
df["output_tokens"] = output_tokens
df["total_tokens"] = total_tokens

faithfulness = df['Faithfulness Score'].mean() if 'Faithfulness Score' in df.columns else None
answer_relevancy = df['Answer Relevancy Score'].mean() if 'Answer Relevancy Score' in df.columns else None

print(f"Faithfulness: {faithfulness}")
print(f"Answer Relevancy: {answer_relevancy}")

# ── Latency statistics via pandas quantile ────────────────────────────────────
latency_series = pd.Series(latencies)
latency_stats = {
    "avg_latency": float(latency_series.mean()),
    "p50_latency": float(latency_series.quantile(0.50)),
    "p90_latency": float(latency_series.quantile(0.90)),
    "p95_latency": float(latency_series.quantile(0.95)),
    "p99_latency": float(latency_series.quantile(0.99)),
}
print(f"Avg latency: {latency_stats['avg_latency']:.2f}s | "
      f"p50: {latency_stats['p50_latency']:.2f}s | "
      f"p90: {latency_stats['p90_latency']:.2f}s | "
      f"p95: {latency_stats['p95_latency']:.2f}s | "
      f"p99: {latency_stats['p99_latency']:.2f}s")

# ── Token statistics ──────────────────────────────────────────────────────────
token_stats = {
    "total_input_tokens": int(df["input_tokens"].sum()),
    "total_output_tokens": int(df["output_tokens"].sum()),
    "total_tokens": int(df["total_tokens"].sum()),
    "avg_input_tokens": float(df["input_tokens"].mean()),
    "avg_output_tokens": float(df["output_tokens"].mean()),
    "avg_total_tokens": float(df["total_tokens"].mean()),
}
print(f"Total tokens: {token_stats['total_tokens']} (Input: {token_stats['total_input_tokens']}, Output: {token_stats['total_output_tokens']}) | Avg per row: {token_stats['avg_total_tokens']:.1f}")

# ── Concatenate summary rows to output DataFrame & save to single file ────────
summary_rows = [
    {"input": f"[LATENCY_STAT] {k}", "latency_seconds": v}
    for k, v in latency_stats.items()
] + [
    {"input": f"[TOKEN_STAT] {k}", "total_tokens": v}
    for k, v in token_stats.items()
]
combined_df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
combined_df.to_csv(OUTPUT_PATH, index=False)

with mlflow.start_run(run_name=params['name']) as parent_run:
    # ── Log params once using log_params ──────────────────────────────────────
    eval_params = {
        "FaithfulnessMetric_llm": params['llm_model_id'],
        "FaithfulnessMetric_threshold": params["FAITHFULNESS_THRESHOLD"],
        "AnswerRelevancyMetric_llm": params['llm_model_id'],
        "AnswerRelevancyMetric_threshold": params["ANSWER_RELEVANCY_THRESHOLD"],
        "Name": params['name'],
        "k": params["k"],
        "dataset_size": len(test_set),
    }
    mlflow.log_params(eval_params)

    # ── Log metrics once using log_metrics ────────────────────────────────────
    all_metrics = {}
    if faithfulness is not None:
        all_metrics["faithfulness"] = float(faithfulness)
    if answer_relevancy is not None:
        all_metrics["answer_relevancy"] = float(answer_relevancy)
    all_metrics.update(latency_stats)
    all_metrics.update(token_stats)
    mlflow.log_metrics(all_metrics)

    # ── Artifact: concatenated output and latency CSV file ───────────────────
    mlflow.log_artifact(OUTPUT_PATH, "eval_results")
