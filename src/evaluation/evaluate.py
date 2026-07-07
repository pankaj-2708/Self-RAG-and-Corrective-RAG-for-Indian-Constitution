import os
import sys
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
    context_precision,
    context_recall,
)
from langchain_aws import ChatBedrockConverse
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',)))
from workflow import graph

# Load environment variables from .env
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
test_set_path = os.path.join(DATA_DIR, "test_set.csv")
results_path = os.path.join(DATA_DIR, "results.csv")
progress_path = os.path.join(DATA_DIR, "evaluation_progress.csv")
history_path = os.path.join(DATA_DIR, "evaluation_history.csv")

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

def run_langgraph_rag(user_query, app, pbar=None):
    def _run():
        state_updates = {
            "relevant_contexts": [],
            "generated_response": ""
        }
        for chunk in app.stream({
            "user_query": user_query,
            "k": 2,
            "max_retry_for_revise_answer": 2,
            "max_retry_for_rewrite_query": 1
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
        return state_updates

    try:
        res = _run()
    except (ValidationError, OutputParserException) as e:
        print(f"\n[Warning] Encountered validation/parsing error: {e}. Retrying once...")
        res = _run()
    
    # Extract text content of context messages (due to add_messages annotation in the state schema)
    contexts_raw = res.get('relevant_contexts', [])
    contexts = [c.content if hasattr(c, 'content') else str(c) for c in contexts_raw]
    
    return res.get('generated_response', ''), contexts

# Check for existing progress file
evaluation_data = []
completed_questions = set()

if os.path.exists(progress_path):
    choice = input("Previous run progress found. Do you want to continue the previous run? (yes/no): ").strip().lower()
    if choice in ['y', 'yes']:
        print(f"Loading progress from {progress_path}...")
        progress_df = pd.read_csv(progress_path)
        for _, row in progress_df.iterrows():
            ctx_val = row["contexts"]
            if isinstance(ctx_val, str):
                try:
                    contexts_list = json.loads(ctx_val)
                except Exception:
                    try:
                        contexts_list = ast.literal_eval(ctx_val)
                    except Exception:
                        contexts_list = [ctx_val]
            else:
                contexts_list = list(ctx_val) if pd.notna(ctx_val) else []
            
            evaluation_data.append({
                "question": row["question"],
                "answer": row["answer"],
                "contexts": contexts_list,
                "ground_truth": row["ground_truth"],
                "synthesizer": row.get("synthesizer", "")
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
    answer, contexts = run_langgraph_rag(question, app, pbar)
    
    evaluation_data.append({
        "question": question,
        "answer": answer,         
        "contexts": contexts,     
        "ground_truth": ground_truth,
        "synthesizer": row["synthesizer_name"]
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

eval_dataset = Dataset.from_list(evaluation_data)

print("Running Ragas evaluation...")
ragas_results = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=evaluater_llm,
    embeddings=evaluater_embeddings
)
print(ragas_results)

# Convert results to a DataFrame and save
ragas_results_df = ragas_results.to_pandas()
ragas_results_df.to_csv(results_path, index=False)

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