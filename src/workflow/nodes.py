from langchain_core.messages import HumanMessage, SystemMessage
from workflow.state import schema
from workflow.config import (
    decision_model, retrieval_decider_model, query_gen_model, generation_model,
    context_answer_model, grounding_model,
    critic_model, judge_model, answer_rewrite_model,
    retriever, tavily_tool, vector_store
)
from workflow.schemas import (
    parser_for_retrieval_decider_node,
    parser_for_is_relevant_node,
    parser_for_answer_from_context_node,
    parser_for_schema_for_check_answer_grounded_node,
    parser_for_revise_answer_node,
    parser_for_revise_answer_node,
    parser_for_is_answer_relevant_node,
    parser_for_rewrite_answer_node,
    parser_for_retriever_query_node,
    parser_for_web_search_query_node
)
from workflow.prompts import (
    sys_prompt_for_retrieval_decider_node,
    sys_prompt_for_is_relevant_node,
    sys_prompt_for_answer_from_context_node,
    sys_prompt_for_check_answer_grounded_node,
    sys_prompt_for_revise_answer_node,
    sys_prompt_for_is_answer_relevant_node,
    sys_prompt_for_rewrite_answer_node,
    sys_prompt_for_retriever_query_node,
    sys_prompt_for_web_search_query_node
)
from langgraph.types import Send

def _extract_r1_text(content) -> str:
    """Extract the final text output from a DeepSeek R1 response via Bedrock.
    
    LangChain's ChatBedrockConverse returns response.content as a list of dicts
    for R1, e.g.:
      [{'type': 'text', 'text': '...'},
       {'type': 'reasoning_content', 'reasoning_content': {'text': '...'}}]
    This helper extracts only the 'text' block, ignoring the reasoning block.
    Falls back gracefully if content is already a plain string (other models).
    """
    if isinstance(content, list):
        parts = [block["text"] for block in content if block.get("type") == "text"]
        return " ".join(parts).strip()
    return content  # plain string — other models

def retrieval_decider_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retrieval_decider_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    response = retrieval_decider_model.invoke(inp)
    usage = response.usage_metadata or {}
    # R1 via Bedrock returns content as a list of dicts — extract only the 'text' block
    content = _extract_r1_text(response.content)
    res = parser_for_retrieval_decider_node.invoke(
        content
    ).retrieval_required
    return {
        "retrieval_required": res,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def generate_retriever_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retriever_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    response = query_gen_model.invoke(inp)
    usage = response.usage_metadata or {}
    res = parser_for_retriever_query_node.invoke(
        response.content
    )
    # Explicit check to enforce at most 3 queries
    queries = res.retriever_queries[:3]
    
    queries_dicts = [
        {
            "query": item.query,
            "doc_type": item.doc_type,
            "number": item.number
        }
        for item in queries
    ]
    return {
        "retriever_queries": queries_dicts,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def fanout_retrieve_node(state: schema):
    queries = state.get("retriever_queries")
    if not queries:
        queries = [{"query": state["user_query"], "doc_type": "None", "number": None}]
    
    lst = []
    for q_item in queries:
        lst.append(Send("retrieve_node", {"query_item": q_item, "user_query": state['user_query']}))
    
    return lst

def retrieve_node(inp):
    global vector_store
    
    q_item = inp.get("query_item")
    if not q_item:
        q_item = {"query": inp["user_query"], "doc_type": "None", "number": None}
        
    all_contexts = []
    seen = set()
    k_val = inp.get("k") or 3
    
    query_str = q_item.get("query")
    doc_type = q_item.get("doc_type")
    number = q_item.get("number")
    
    filter_dict = None
    if doc_type == "Constitution" and number:
        filter_dict = {"Article": str(number).strip()}
    elif doc_type == "IPC" and number:
        filter_dict = {"Section": str(number).strip()}
        
    if filter_dict:
        retrieved_contexts = vector_store.similarity_search(query_str, k=k_val, filter=filter_dict)
    else:
        retrieved_contexts = vector_store.similarity_search(query_str, k=k_val)
        
    for doc in retrieved_contexts:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_contexts.append(doc.page_content)
    return {"retrieved_contexts": all_contexts}

def aggregate_retrieval(state: schema):
    """Sync barrier after all fanned-out retrieve_nodes complete.
    No state mutation — just ensures all retrieve_nodes finish before
    relevance evaluation begins."""

    return {}
    

def direct_generation_node(state: schema):
    response = generation_model.invoke(state["user_query"])
    usage = response.usage_metadata or {}
    return {
        "generated_response": response.content,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def fanout_relevant_node(state:schema):
    contexts = state.get("retrieved_contexts", [])
    
    # Deduplicate contexts since operator.add reducer may accumulate duplicates
    seen = set()
    unique_contexts = []
    for ctx in contexts:
        if ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)
    
    lst = []
    for context in unique_contexts:
        lst.append(Send("is_relevant_node",{"context":context,"user_query":state['user_query']}))
    
    return lst

def is_relevant_node(inp):
    sys_prompt = SystemMessage(content=sys_prompt_for_is_relevant_node)
    hmn_prompt = f"Query - {inp['user_query']}" + f"\n Context - \n {inp['context']}"

    response = decision_model.invoke(
        [sys_prompt, HumanMessage(content=hmn_prompt)]
    )
    usage = response.usage_metadata or {}
    res = parser_for_is_relevant_node.invoke(response.content)
    
    out = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }
    if res.is_relevant_context :
        out["relevant_contexts"] = [inp['context']]
    return out

def aggregate_relevance(state):
    # Deduplication is now handled by the custom state reducer.
    return {}

def answer_from_context_node(state: schema):
    contexts = state["relevant_contexts"]
    sys_prompt = SystemMessage(content=sys_prompt_for_answer_from_context_node)

    context = ""
    for i in contexts:
        context += i
        context += "\n"

    hmn_prompt = HumanMessage(
        content=f"Query - {state['user_query']} \n\n Contexts - \n {context}"
    )
    inp = [sys_prompt,hmn_prompt]

    response = context_answer_model.invoke(inp)
    usage = response.usage_metadata or {}
    # R1 via Bedrock returns content as a list of dicts — extract only the 'text' block
    content = _extract_r1_text(response.content)
    res = parser_for_answer_from_context_node.invoke(
        content
    ).response
    return {
        "generated_response": res,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def check_answer_grounded_node(state: schema):

    if state['max_retry_for_groundness_checking'] <= 0:
        return {
            # because max_retry_for_groundness_checking =0 so even if answer is not grounded we are not going to modify it so there is no sense of checking here
            "is_grounded": "fully_supported",
            "evidence": "max_retries_exhausted",
            "input_tokens": 0,
            "output_tokens": 0
        }
    contexts = state["relevant_contexts"]
    sys_prompt = SystemMessage(content=sys_prompt_for_check_answer_grounded_node)
    context = ""
    for i in contexts:
        context += i
        context += "\n"

    human_pr = HumanMessage(
        content=f"Answer - {state['generated_response']} \n Contexts - {context}"
    )

    response = grounding_model.invoke([sys_prompt, human_pr])
    usage = response.usage_metadata or {}
    res = parser_for_schema_for_check_answer_grounded_node.invoke(
        response.content
    )

    return {
        "is_grounded": res.is_grounded,
        "evidence": res.evidence,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def revise_answer_node(state: schema):
    if state['max_retry_for_answer_relevant_checking'] <= 0:
        return {
            # because max_retry_for_answer_relevant_checking =0 so even if answer is not relevant we are not going to modify it so there is no sense of checking here
            "generated_response": state["generated_response"],
            "input_tokens": 0,
            "output_tokens": 0
        }
    contexts = state["relevant_contexts"]
    context = ""
    for i in contexts:
        context += i
        context += "\n"
    generated_response = state["generated_response"]
    user_query = state["user_query"]
    evidence = state["evidence"]

    sys_prompt = SystemMessage(content=sys_prompt_for_revise_answer_node)

    human_pr = HumanMessage(
        content=f"""Query - {user_query} \n\n Generated Response - {generated_response} \n\n Contexts - {context} \n\n Evidence - {evidence}"""
    )

    response = critic_model.invoke([sys_prompt, human_pr])
    usage = response.usage_metadata or {}
    # R1 via Bedrock returns content as a list of dicts — extract only the 'text' block
    content = _extract_r1_text(response.content)
    revised_answer = parser_for_revise_answer_node.invoke(
        content
    )

    return {
        "generated_response": revised_answer.revised_response,
        "max_retry_for_groundness_checking": state["max_retry_for_groundness_checking"] - 1,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def is_answer_relevant_node(state: schema):
    if state['max_retry_for_answer_relevant_checking'] <= 0:
        return {
            # because max_retry_for_answer_relevant_checking =0 so even if answer is not relevant we are not going to modify it so there is no sense of checking here
            "is_answer_relevant": True,
            "relevance_explanation": "max_retries_exhausted",
            "input_tokens": 0,
            "output_tokens": 0
        }
    user_query = state["user_query"]
    generated_response = state["generated_response"]

    sys_prompt = SystemMessage(content=sys_prompt_for_is_answer_relevant_node)
    human_pr = HumanMessage(
        content=f"Query - {user_query} \n\n Generated Response - {generated_response}"
    )

    response = judge_model.invoke([sys_prompt, human_pr])
    usage = response.usage_metadata or {}
    # R1 via Bedrock returns content as a list of dicts — extract only the 'text' block
    content = _extract_r1_text(response.content)
    res = parser_for_is_answer_relevant_node.invoke(content)
    return {
        "is_answer_relevant": res.is_relevant,
        "relevance_explanation": res.explanation,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def rewrite_answer_node(state: schema):
    contexts = state["relevant_contexts"]
    context = ""
    for i in contexts:
        context += i
        context += "\n"

    generated_response = state["generated_response"]
    user_query = state["user_query"]
    relevance_explanation = state.get("relevance_explanation", "")

    sys_prompt = SystemMessage(content=sys_prompt_for_rewrite_answer_node)
    human_pr = HumanMessage(
        content=f"""Query - {user_query} \n\n Previous Answer - {generated_response} \n\n Contexts - {context} \n\n Relevance Explanation - {relevance_explanation}"""
    )

    response = answer_rewrite_model.invoke([sys_prompt, human_pr])
    usage = response.usage_metadata or {}
    # R1 via Bedrock returns content as a list of dicts — extract only the 'text' block
    content = _extract_r1_text(response.content)
    res = parser_for_rewrite_answer_node.invoke(content)
    return {
        "generated_response": res.rewritten_response,
        "max_retry_for_answer_relevant_checking": state["max_retry_for_answer_relevant_checking"] - 1,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def generate_web_search_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_web_search_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    response = query_gen_model.invoke(inp)
    usage = response.usage_metadata or {}
    res = parser_for_web_search_query_node.invoke(
        response.content
    ).web_search_queries[:3]
    return {
        "web_search_queries": res,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0)
    }

def web_search_node(state: schema):
    queries = state.get("web_search_queries") or [state["user_query"]]
    if not queries:
        queries = [state["user_query"]]
    
    res = []
    seen_urls = set()
    for query in queries:
        try:
            x = tavily_tool.invoke(query)
            for r in x.get("results", []):
                if r['url'] not in seen_urls:
                    seen_urls.add(r['url'])
                    p = f"Source - {r['url']} \n title - {r['title']} \n {r['content']}"
                    res.append(p)
        except Exception as e:
            # Print/log the exception and keep going with other queries
            print(f"Error querying Tavily for '{query}': {e}")
    return {"relevant_contexts": res, "web_searched": True}
