import os
import sys
import yaml
import mlflow
import pandas as pd
from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import AmazonBedrockModel

if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGSMITH_TRACING_V2" == "false"):
    try:
        from phoenix.otel import register
        tracer_provider = register(
            project_name="constitution",
            auto_instrument=True 
        )
    except ImportError:
        pass

# Resolve paths and check execution parameter before loading heavy modules or telemetry
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
PARAMS_PATH = os.path.abspath(os.path.join(DATA_DIR, "../params.yaml"))

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)['standalone_generator']

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
for i in range(len(test_set)):
    query = test_set.iloc[i]['user_input']
    ground_truth = test_set.iloc[i]['reference']
    retrieved_docs = retriever.invoke(query)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    
    context_str = "\n\n".join(retrieved_docs_text)
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
    
    response = generator_llm.invoke(prompt)
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
df.to_csv(OUTPUT_PATH, index=False)

faithfulness = df['Faithfulness Score'].mean() if 'Faithfulness Score' in df.columns else None
answer_relevancy = df['Answer Relevancy Score'].mean() if 'Answer Relevancy Score' in df.columns else None

print(f"Faithfulness: {faithfulness}")
print(f"Answer Relevancy: {answer_relevancy}")


with mlflow.start_run():
    mlflow.log_param("FaithfulnessMetric_llm", params['llm_model_id'])
    mlflow.log_param("FaithfulnessMetric_threshold", params["FAITHFULNESS_THRESHOLD"])
    mlflow.log_param("AnswerRelevancyMetric_llm", params['llm_model_id'])
    mlflow.log_param("AnswerRelevancyMetric_threshold", params["ANSWER_RELEVANCY_THRESHOLD"])
    mlflow.log_param("Name", params['name'])
    mlflow.log_param("k", params["k"])
    mlflow.log_param("dataset_size", len(test_set))
    mlflow.log_metrics({"faithfulness": faithfulness, "answer_relevancy": answer_relevancy})
    mlflow.log_artifact(OUTPUT_PATH, "eval_results")
