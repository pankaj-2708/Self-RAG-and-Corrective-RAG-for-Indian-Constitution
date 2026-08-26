import os
import sys
import yaml
import mlflow
import pandas as pd
import asyncio
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/pankaj-2708/Self-RAG-and-Corrective-RAG-for-Indian-Constitution.mlflow")

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from deepeval.models import AmazonBedrockModel
from src.workflow import get_workflow

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

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)['pipeline_eval']

TEST_SET_PATH = os.path.abspath(os.path.join(DATA_DIR, params['test_set']))
OUTPUT_PATH = os.path.abspath(os.path.join(DATA_DIR, params['output']))

judge_model = AmazonBedrockModel(
    model=params["llm_model_id"],
    region=params["region"],
    generation_kwargs={"temperature": params["llm_temperature"]} 
)

test_set = pd.read_csv(TEST_SET_PATH)

async def generate_test_cases():
    test_cases = []
    latencies = []  # wall-clock seconds per row for workflow.ainvoke()
    async with get_workflow() as (workflow, ck_ptr):
        for i in range(len(test_set)):
            query = test_set.iloc[i]['user_input']
            ground_truth = test_set.iloc[i]['reference']
            thread_id = str(uuid.uuid4())
            initial_state = {
                "user_query": query,
                "k": params.get("k", 3),
                "max_retry_for_groundness_checking": 1,
                "max_retry_for_answer_relevant_checking": 1,
            }
            t0 = time.perf_counter()
            res = await workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            latencies.append(time.perf_counter() - t0)
            actual_output = res.get("generated_response", "")
            raw_contexts = res.get("relevant_contexts") or res.get("retrieved_contexts") or []
            rel_contexts = [c for c in raw_contexts if c != "-1"]
            
            test_cases.append(
                LLMTestCase(
                    input=query,
                    expected_output=ground_truth,
                    retrieval_context=rel_contexts,
                    actual_output=actual_output
                )
            )
    return test_cases, latencies

test_cases, latencies = asyncio.run(generate_test_cases())

metrics = [
    ContextualRecallMetric(threshold=params['RECALL_THRESHOLD'], model=judge_model, include_reason=True),
    ContextualPrecisionMetric(threshold=params['PRECISION_THRESHOLD'], model=judge_model, include_reason=True),
    FaithfulnessMetric(threshold=params['FAITHFULNESS_THRESHOLD'], model=judge_model, include_reason=True),
    AnswerRelevancyMetric(threshold=params['ANSWER_RELEVANCY_THRESHOLD'], model=judge_model, include_reason=True),
    ContextualRelevancyMetric(threshold=params['CONTEXTUAL_RELEVANCY_THRESHOLD'], model=judge_model, include_reason=True),
]

evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "ContextualRecallMetric_llm": params['llm_model_id'],
        "ContextualRecallMetric_threshold": params["RECALL_THRESHOLD"],
        "ContextualPrecisionMetric_llm": params['llm_model_id'],
        "ContextualPrecisionMetric_threshold": params["PRECISION_THRESHOLD"],
        "FaithfulnessMetric_llm": params['llm_model_id'],
        "FaithfulnessMetric_threshold": params["FAITHFULNESS_THRESHOLD"],
        "AnswerRelevancyMetric_llm": params['llm_model_id'],
        "AnswerRelevancyMetric_threshold": params["ANSWER_RELEVANCY_THRESHOLD"],
        "ContextualRelevancyMetric_llm": params['llm_model_id'],
        "ContextualRelevancyMetric_threshold": params["CONTEXTUAL_RELEVANCY_THRESHOLD"],
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

recall = df['Contextual Recall Score'].mean() if 'Contextual Recall Score' in df.columns else None
precision = df['Contextual Precision Score'].mean() if 'Contextual Precision Score' in df.columns else None
faithfulness = df['Faithfulness Score'].mean() if 'Faithfulness Score' in df.columns else None
answer_relevancy = df['Answer Relevancy Score'].mean() if 'Answer Relevancy Score' in df.columns else None
contextual_relevancy = df['Contextual Relevancy Score'].mean() if 'Contextual Relevancy Score' in df.columns else None

metrics_dict = {}
if recall is not None:
    metrics_dict["recall"] = float(recall)
    print(f"Recall: {recall}")
if precision is not None:
    metrics_dict["precision"] = float(precision)
    print(f"Precision: {precision}")
if faithfulness is not None:
    metrics_dict["faithfulness"] = float(faithfulness)
    print(f"Faithfulness: {faithfulness}")
if answer_relevancy is not None:
    metrics_dict["answer_relevancy"] = float(answer_relevancy)
    print(f"Answer Relevancy: {answer_relevancy}")
if contextual_relevancy is not None:
    metrics_dict["contextual_relevancy"] = float(contextual_relevancy)
    print(f"Contextual Relevancy: {contextual_relevancy}")

# ── Latency statistics via pandas quantile ────────────────────────────────────
latency_series = pd.Series(latencies)
latency_stats = {
    "avg_latency":  float(latency_series.mean()),
    "p50_latency":  float(latency_series.quantile(0.50)),
    "p90_latency":  float(latency_series.quantile(0.90)),
    "p95_latency":  float(latency_series.quantile(0.95)),
    "p99_latency":  float(latency_series.quantile(0.99)),
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
    # ── Flatten params into a dict and log once using log_params ────────────
    eval_params = {}
    def flatten_params(d, prefix=""):
        for k, v in d.items():
            param_key = f"{prefix}{k}" if prefix else str(k)
            if isinstance(v, dict):
                flatten_params(v, prefix=f"{param_key}.")
            else:
                eval_params[param_key] = v

    flatten_params(params)
    eval_params["dataset_size"] = len(test_set)
    mlflow.log_params(eval_params)

    # ── Log metrics once using log_metrics ────────────────────────────────────
    all_metrics = {**metrics_dict, **latency_stats}
    mlflow.log_metrics(all_metrics)

    # ── Artifact: concatenated output and latency CSV file ───────────────────
    mlflow.log_artifact(OUTPUT_PATH, "eval_results")