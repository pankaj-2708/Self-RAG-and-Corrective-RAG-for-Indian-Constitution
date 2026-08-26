import os
import sys
import yaml
import mlflow
import pandas as pd
import time
from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import AmazonBedrockModel
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_TRACING_V2")=="false":
    try:
        from phoenix.otel import register
        tracer_provider = register(
            project_name="constitution",
            auto_instrument=True 
        )
    except ImportError:
        pass

mlflow.set_tracking_uri("https://dagshub.com/pankaj-2708/Self-RAG-and-Corrective-RAG-for-Indian-Constitution.mlflow")
mlflow.set_experiment("constitution-rag")

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)['standalone_retriever']

TEST_SET_PATH = os.path.abspath(os.path.join(DATA_DIR, params['test_set']))
OUTPUT_PATH = os.path.abspath(os.path.join(DATA_DIR, params['output']))


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

judge_model = AmazonBedrockModel(
    model=params["llm_model_id"],
    region=params["region"],
    generation_kwargs={"temperature": params["llm_temperature"]} 
)

test_set=pd.read_csv(TEST_SET_PATH)

test_cases=[]
latencies = []  # wall-clock seconds per row for retriever.invoke()
for i in range(len(test_set)):
    query=test_set.iloc[i]['user_input']
    ground_truth=test_set.iloc[i]['reference']
    t0 = time.perf_counter()
    retrieved_docs=retriever.invoke(query)
    latencies.append(time.perf_counter() - t0)
    retrieved_docs=[doc.page_content for doc in retrieved_docs]
    
    test_cases.append(
        LLMTestCase(
            input=query,
            expected_output=ground_truth,
            retrieval_context=retrieved_docs,
            actual_output="not required here"
        )
    )

metrics = [
    ContextualRecallMetric(threshold=params['RECALL_THRESHOLD'], model=judge_model, include_reason=True),
    ContextualPrecisionMetric(threshold=params['PRECISION_THRESHOLD'], model=judge_model, include_reason=True),
]

evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "ContextualRecallMetric_llm": params['llm_model_id'],
        "ContextualRecallMetric_threshold": params["RECALL_THRESHOLD"],
        "ContextualPrecisionMetric_llm": params['llm_model_id'],
        "ContextualPrecisionMetric_threshold": params["PRECISION_THRESHOLD"],
        "Name":params['name'],
        "k":params["k"],
        "dataset_size":len(test_set),
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

precision=df['Contextual Precision Score'].mean()
recall=df['Contextual Recall Score'].mean()

print(f"Precision: {precision}")
print(f"Recall: {recall}")

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

# ── Concatenate latency summary rows to output DataFrame & save to single file ──
latency_summary_rows = [
    {"input": f"[LATENCY_STAT] {k}", "latency_seconds": v}
    for k, v in latency_stats.items()
]
combined_df = pd.concat([df, pd.DataFrame(latency_summary_rows)], ignore_index=True)
combined_df.to_csv(OUTPUT_PATH, index=False)

with mlflow.start_run() as parent_run:
    # ── Log params once using log_params ──────────────────────────────────────
    eval_params = {
        "ContextualRecallMetric_llm": params['llm_model_id'],
        "ContextualRecallMetric_threshold": params["RECALL_THRESHOLD"],
        "ContextualPrecisionMetric_llm": params['llm_model_id'],
        "ContextualPrecisionMetric_threshold": params["PRECISION_THRESHOLD"],
        "Name": params['name'],
        "k": params["k"],
        "dataset_size": len(test_set),
    }
    mlflow.log_params(eval_params)

    # ── Log metrics once using log_metrics ────────────────────────────────────
    all_metrics = {
        "precision": float(precision),
        "recall": float(recall),
        **latency_stats,
    }
    mlflow.log_metrics(all_metrics)

    # ── Artifact: concatenated output and latency CSV file ───────────────────
    mlflow.log_artifact(OUTPUT_PATH, "eval_results")