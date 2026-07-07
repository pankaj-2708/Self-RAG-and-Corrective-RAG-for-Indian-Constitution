# Self-RAG & Corrective RAG System for the Indian Constitution & IPC

An advanced, agentic Retrieval-Augmented Generation (RAG) system utilizing **LangGraph** to implement Self-RAG and Corrective RAG (CRAG) patterns. It provides highly reliable question-answering over the **Constitution of India** and the **Indian Penal Code (IPC)**.

The system dynamically decides between internal database retrieval, web search fallback, and direct generation, incorporating validation checks for context relevance, answer grounding, and utility to rewrite queries and self-correct when necessary.

---

## 🗺️ Workflow Architecture

Below is the complete state machine representing the LangGraph workflow:

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrieval_decider_node(retrieval_decider_node)
	generate_retriever_query_node(generate_retriever_query_node)
	retrieve_node(retrieve_node)
	direct_generation_node(direct_generation_node)
	is_relevant_node(is_relevant_node)
	answer_from_context_node(answer_from_context_node)
	check_answer_grounded_node(check_answer_grounded_node)
	revise_answer_node(revise_answer_node)
	is_answer_useful_node(is_answer_useful_node)
	rewrite_answer_node(rewrite_answer_node)
	generate_web_search_query_node(generate_web_search_query_node)
	web_search_node(web_search_node)
	aggregate_retrieval(aggregate_retrieval)
	aggregate_relevance(aggregate_relevance)
	__end__([<p>__end__</p>]):::last
	__start__ --> retrieval_decider_node;
	aggregate_relevance -. &nbsp;True&nbsp; .-> answer_from_context_node;
	aggregate_relevance -. &nbsp;False&nbsp; .-> generate_web_search_query_node;
	aggregate_retrieval -.-> is_relevant_node;
	answer_from_context_node --> check_answer_grounded_node;
	check_answer_grounded_node -. &nbsp;True&nbsp; .-> is_answer_useful_node;
	check_answer_grounded_node -. &nbsp;False&nbsp; .-> revise_answer_node;
	generate_retriever_query_node -.-> retrieve_node;
	generate_web_search_query_node --> web_search_node;
	is_answer_useful_node -. &nbsp;True&nbsp; .-> __end__;
	is_answer_useful_node -. &nbsp;False&nbsp; .-> rewrite_answer_node;
	is_relevant_node --> aggregate_relevance;
	retrieval_decider_node -. &nbsp;None&nbsp; .-> direct_generation_node;
	retrieval_decider_node -. &nbsp;retrieval&nbsp; .-> generate_retriever_query_node;
	retrieval_decider_node -. &nbsp;web_search&nbsp; .-> generate_web_search_query_node;
	retrieve_node --> aggregate_retrieval;
	revise_answer_node --> check_answer_grounded_node;
	rewrite_answer_node --> is_answer_useful_node;
	web_search_node --> answer_from_context_node;
	direct_generation_node --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

---

## 🚀 Key Features

1. **Intelligent Query Routing (`retrieval_decider_node`)**: Evaluates if the query is a general knowledge question (`None`), requires constitutional/IPC lookup (`retrieval`), or concerns recent happenings/un-indexed details (`web_search`).
2. **Parallel Multi-Query Retrieval (`generate_retriever_query_node`, `fanout_retrieve_node` & `retrieve_node`)**: Generates optimized search queries and executes parallel vector searches for comprehensive context fetching.
3. **Parallel Relevance Verification (`fanout_relevant_node` & `is_relevant_node`)**: Distributes retrieved documents in parallel (Map step) to score and filter irrelevant content, and combines them (Reduce step) to clean up noise.
4. **Web Search Fallback (`web_search_node`)**: Integrates the Tavily Search API as a backup when local retrieval fails to find relevant context.
5. **Hallucination & Grounding Check (`check_answer_grounded_node`)**: Grades the synthesized answer against the retrieved evidence. If the answer is ungrounded, it invokes `revise_answer_node` to regenerate using a critic model.
6. **Answer Refinement (`rewrite_answer_node`)**: If the final response is judged non-useful (in `is_answer_useful_node`), it rewrites the generated answer directly based on context and evaluates its utility again.
7. **Thread-Based Memory (`SqliteSaver`)**: Persists chat history across sessions using SQLite state checkpointers.

---

## 📁 Repository Structure

*   **`src/workflow/`**: The core graph package.
    *   `__init__.py`: Combines state, nodes, and conditional edges into the compiled `StateGraph`.
    *   `state.py`: Defines the `schema` (TypedDict) that maintains the shared workflow variables.
    *   `nodes.py`: Houses logic for all graph states, invoking the underlying LLMs, Tavily tools, and vector stores.
    *   `edges.py`: Logic for conditional routing (decider route, relevance, grounding, and utility).
    *   `config.py`: Loads credentials and defines model components (using Ollama and HuggingFace).
    *   `prompts.py`: Holds system instructions for evaluation, generation, revision, and rewriting.
    *   `schemas.py`: Implements Pydantic parser schemas for structured JSON output.
*   **`src/create_vector_store.py`**: Ingestion script that parses raw article and section files and saves them as a Chroma database.
*   **`src/cli.py`**: Interactive terminal shell utilizing `rich` for pretty printing and real-time step streaming.
*   **`data/`**: Directory containing raw JSON datasets and Chroma database files.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.12 or higher.
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip` for managing dependencies.

### 2. Install Dependencies
Run the following command in the project root:
```bash
uv sync
```
*Or using traditional pip:*
```bash
pip install -r pyproject.toml
```

### 3. Environment Variables
Create a `.env` file in the root directory (or update the existing one) with your API keys:
```env
LANGSMITH_TRACING_V2="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="constitution"
TAVILY_API_KEY="your-tavily-api-key"
OLLAMA_API_KEY="your-ollama-api-key"
```

### 4. Create the Vector Store
To ingest the Indian Constitution and Penal Code data into the local Chroma vector store:
```bash
python src/create_vector_store.py
```
*(This parses raw inputs from `data/articles.json` and `data/penal_code_sections.json` and compiles them under `data/constitution_and_ipc.chroma`).*

---

## 💬 Usage

Launch the interactive console with the CLI script:
```bash
python src/cli.py
```

### Special Commands inside CLI:
- `/new`: Resets the chat history and spawns a new conversation thread.
- `exit`: Shuts down the interactive loop.
