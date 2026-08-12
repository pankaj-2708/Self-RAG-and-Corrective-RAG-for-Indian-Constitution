import os
import sys
import yaml

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)["evaluate"]

evaluate_rag = params.get("evaluate_rag", True)

if not evaluate_rag:
    print("evaluate_rag is set to False in params.yaml. Exiting evaluation stage.")
    sys.exit(0)

from phoenix.otel import register
tracer_provider = register(
  project_name="constitution",
  auto_instrument=True 
)

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import ast
import json
import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    NonLLMContextPrecisionWithReference,
    context_recall,
)
from langchain_aws import ChatBedrockConverse
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))
from workflow import graph

# Load environment variables from .env
load_dotenv()

import glob
import re

test_set_path = os.path.join(DATA_DIR, "test_set.csv")

RESULTS_DIR = os.path.join(DATA_DIR, "evaluation_progress_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

progress_path = os.path.join(RESULTS_DIR, "evaluation_progress2.csv")
history_path = os.path.join(RESULTS_DIR, "evaluation_history2.csv")
all_results_path = os.path.join(RESULTS_DIR, "all_results2.csv")

# Determine next version
existing_files = glob.glob(os.path.join(RESULTS_DIR, "results_v*.csv"))
versions = []
for f in existing_files:
    match = re.search(r'results_v(\d+)\.csv$', os.path.basename(f))
    if match:
        versions.append(int(match.group(1)))

next_version = max(versions) + 1 if versions else 1
results_path = os.path.join(RESULTS_DIR, f"results_v{next_version}.csv")

# Initialize evaluator LLM and wrap it for Ragas 0.4.x
evaluater_llm = ChatBedrockConverse(
    model="deepseek.v3.2",
    api_key=os.environ['AWS_BEARER_TOKEN_BEDROCK'],
    region_name="us-east-1",
    temperature=0,
)
evaluater_llm = LangchainLLMWrapper(evaluater_llm)

# Initialize embedding model and wrap it for Ragas evaluation
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
evaluater_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

# Compile the LangGraph application (without checkpointer to avoid SQLite database overhead)
app = graph.compile()

# Load evaluation data via pandas
df = pd.read_csv(test_set_path)

async def run_langgraph_rag(user_query, app, pbar=None):
    async def _run():
        state_updates = {
            "relevant_contexts": [],
            "generated_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "is_answer_relevant": None,
            "relevance_explanation": "",
            "is_grounded": None,
            "evidence": "",
            "retrieval_method": None,
            "web_searched": False
        }
        async for chunk in app.astream({
            "user_query": user_query,
            "k": 2,
            "max_retry_for_groundness_checking": 1,
            "max_retry_for_answer_relevant_checking": 1,
            "input_tokens": 0,
            "output_tokens": 0
        }, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if pbar is not None:
                    pbar.set_postfix(node=node_name)
                
                if node_state and "generated_response" in node_state:
                    state_updates["generated_response"] = node_state["generated_response"]
                
                if node_state and "relevant_contexts" in node_state:
                    rc = node_state["relevant_contexts"]
                    if isinstance(rc, list):
                        state_updates["relevant_contexts"].extend(rc)
                    else:
                        state_updates["relevant_contexts"].append(rc)
                
                # Accumulate tokens from each node
                if node_state and "input_tokens" in node_state:
                    state_updates["input_tokens"] += node_state["input_tokens"]
                if node_state and "output_tokens" in node_state:
                    state_updates["output_tokens"] += node_state["output_tokens"]
                
                # Capture retrieval method from retrieval_decider_node
                if node_state and "retrieval_required" in node_state:
                    state_updates["retrieval_method"] = node_state["retrieval_required"]
                
                # Capture grounding result (last value wins if node runs multiple times)
                if node_state and "is_grounded" in node_state:
                    state_updates["is_grounded"] = node_state["is_grounded"]
                if node_state and "evidence" in node_state:
                    state_updates["evidence"] = node_state["evidence"]
                
                # Capture answer relevance result (last value wins if node runs multiple times)
                if node_state and "is_answer_relevant" in node_state:
                    state_updates["is_answer_relevant"] = node_state["is_answer_relevant"]
                if node_state and "relevance_explanation" in node_state:
                    state_updates["relevance_explanation"] = node_state["relevance_explanation"]
                
                # Capture web_searched flag
                if node_state and "web_searched" in node_state:
                    state_updates["web_searched"] = node_state["web_searched"]

        return state_updates

    try:
        res = await _run()
    except (ValidationError, OutputParserException) as e:
        print(f"\n[Warning] Encountered validation/parsing error: {e}. Retrying once...")
        res = await _run()
    
    # Extract text content of context messages (due to add_messages annotation in the state schema)
    contexts_raw = res.get('relevant_contexts', [])
    contexts = [c.content if hasattr(c, 'content') else str(c) for c in contexts_raw]
    
    return (
        res.get('generated_response', ''),
        contexts,
        {
            "input_tokens": res.get("input_tokens", 0),
            "output_tokens": res.get("output_tokens", 0),
            "is_answer_relevant": res.get("is_answer_relevant"),
            "relevance_explanation": res.get("relevance_explanation", ""),
            "is_grounded": res.get("is_grounded"),
            "evidence": res.get("evidence", ""),
            "retrieval_method": res.get("retrieval_method"),
            "web_searched": res.get("web_searched", False),
        }
    )

# Check for existing progress file
evaluation_data = []
completed_questions = set()

if os.path.exists(progress_path):
    if params['mode']=="dvc":
        choice=params['continue_previous_run']
    else:
        choice = input("Previous run progress found. Do you want to continue the previous run? (yes/no): ").strip().lower()
    if choice in ['y', 'yes']:
        print(f"Loading progress from {progress_path}...")
        progress_df = pd.read_csv(progress_path)
        for _, row in progress_df.iterrows():
            ctx_val = list(set(ast.literal_eval(row["contexts"])))
            evaluation_data.append({
                "question": row["question"],
                "answer": row["answer"],
                "contexts": ctx_val,
                "ground_truth": row["ground_truth"],
                "reference_contexts": ast.literal_eval(row["reference_contexts"]),
                "synthesizer": row.get("synthesizer", ""),
                "input_tokens": row.get("input_tokens", 0),
                "output_tokens": row.get("output_tokens", 0),
                "is_answer_relevant": row.get("is_answer_relevant"),
                "relevance_explanation": row.get("relevance_explanation", ""),
                "is_grounded": row.get("is_grounded", "not_fully_supported"),
                "evidence": row.get("evidence", ""),
                "retrieval_method": row.get("retrieval_method"),
                "web_searched": row.get("web_searched", False),
            })
            completed_questions.add(row["question"])
        print(f"Resuming run. {len(completed_questions)} items already completed.")
    else:
        print("Starting fresh run. Previous progress file will be deleted/overwritten.")
        try:
            os.remove(progress_path)
        except OSError:
            pass

# Filter out already completed queries
df_to_run = df[~df["user_input"].isin(completed_questions)]

print("Running LangGraph workflow across test set...")
pbar = tqdm(df_to_run.iterrows(), total=len(df), initial=len(completed_questions), desc="Evaluating queries")
for _, row in pbar:
    question = row["user_input"]
    ground_truth = row["reference"] 
    
    tqdm.write(f"Processing query: {question}")
    import asyncio
    answer, contexts, metadata = asyncio.run(run_langgraph_rag(question, app, pbar))
    
    evaluation_data.append({
        "question": question,
        "answer": answer,         
        "contexts": contexts,     
        "reference_contexts":ast.literal_eval(row['reference_contexts']),
        "ground_truth": ground_truth,
        "synthesizer": row["synthesizer_name"],
        "input_tokens": metadata["input_tokens"],
        "output_tokens": metadata["output_tokens"],
        # If evidence is "max_retries_exhausted", the grounding check was skipped
        # because retries ran out — the answer is not truly grounded
        "is_answer_relevant": metadata["is_answer_relevant"] if metadata.get("relevance_explanation") != "max_retries_exhausted" else False,
        "relevance_explanation": metadata["relevance_explanation"],
        "is_grounded": metadata.get("is_grounded", "not_fully_supported") if metadata.get("evidence") != "max_retries_exhausted" else "not_fully_supported",
        "evidence": metadata["evidence"] if metadata["evidence"] else "",
        "retrieval_method": metadata["retrieval_method"],
        "web_searched": metadata["web_searched"],
    })
    completed_questions.add(question)
    
    # Save current progress after each query by serializing contexts to JSON
    progress_rows = []
    for item in evaluation_data:
        progress_rows.append({
            **item,
            "contexts": json.dumps(item["contexts"])
        })
    pd.DataFrame(progress_rows).to_csv(progress_path, index=False)

# Normalize data types to prevent PyArrow mixed-type errors
for item in evaluation_data:
    # Ensure is_grounded is always a string Literal ("fully_supported" / "not_fully_supported")
    ig = item.get("is_grounded")
    if isinstance(ig, bool):
        item["is_grounded"] = "fully_supported" if ig else "not_fully_supported"
    elif ig is None:
        item["is_grounded"] = "not_fully_supported"
    elif isinstance(ig, str) and ig not in ("fully_supported", "not_fully_supported"):
        item["is_grounded"] = "fully_supported" if ig.lower() in ("true", "fully_supported") else "not_fully_supported"

    # Ensure string fields are never NaN/None (replace with empty string)
    for str_field in ("evidence", "relevance_explanation", "retrieval_method", "synthesizer"):
        val = item.get(str_field)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            item[str_field] = ""

    # Ensure token counts are int
    for int_field in ("input_tokens", "output_tokens"):
        val = item.get(int_field)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            item[int_field] = 0
        else:
            item[int_field] = int(val)

eval_dataset = Dataset.from_list(evaluation_data)

print("Running Ragas evaluation...")
ragas_results = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        NonLLMContextPrecisionWithReference(),
        context_recall,
    ],
    llm=evaluater_llm,
    embeddings=evaluater_embeddings
)
print(ragas_results)

# Convert results to a DataFrame and save
ragas_results_df = ragas_results.to_pandas()

# RAGAS discards all non-standard columns during evaluate().
# Merge our custom metadata columns back by index (row order is preserved).
custom_columns = [
    "input_tokens", "output_tokens", "is_answer_relevant",
    "relevance_explanation", "is_grounded", "evidence",
    "retrieval_method", "web_searched", "synthesizer",
]
custom_data = pd.DataFrame(evaluation_data)[custom_columns]
# Reset index on both sides to guarantee alignment
custom_data.index = ragas_results_df.index
for col in custom_columns:
    ragas_results_df[col] = custom_data[col]

ragas_results_df.to_csv(results_path, index=False)

# Update all_results.csv with overall metrics
overall_metrics = {"id": f"v{next_version}"}
# Check if ragas_results supports .items() directly or if we should iterate over it
try:
    for metric_name, score in ragas_results.items():
        overall_metrics[metric_name] = score
except AttributeError:
    # Fallback to computing average from dataframe
    metrics_cols = [c for c in ragas_results_df.columns if pd.api.types.is_numeric_dtype(ragas_results_df[c])]
    for c in metrics_cols:
        overall_metrics[c] = ragas_results_df[c].mean()

metrics_df = pd.DataFrame([overall_metrics])
if os.path.exists(all_results_path):
    all_results_df = pd.read_csv(all_results_path)
    all_results_df = pd.concat([all_results_df, metrics_df], ignore_index=True)
else:
    all_results_df = metrics_df

all_results_df.to_csv(all_results_path, index=False)


# Append to history file with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ragas_results_df["run_id"] = timestamp

if os.path.exists(history_path):
    history_df = pd.read_csv(history_path)
    history_df = pd.concat([history_df, ragas_results_df], ignore_index=True)
else:
    history_df = ragas_results_df
    
history_df.to_csv(history_path, index=False)
print(f"Results successfully saved and history updated in {history_path}")

# Clean up progress file on successful completion of the full run
if os.path.exists(progress_path):
    os.remove(progress_path)