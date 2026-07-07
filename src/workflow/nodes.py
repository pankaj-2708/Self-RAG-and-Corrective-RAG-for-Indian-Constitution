from langchain_core.messages import HumanMessage, SystemMessage
from workflow.state import schema
from workflow.config import (
    decision_model, query_gen_model, generation_model, grounding_model,
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
    parser_for_is_answer_useful_node,
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
    sys_prompt_for_is_answer_useful_node,
    sys_prompt_for_rewrite_answer_node,
    sys_prompt_for_retriever_query_node,
    sys_prompt_for_web_search_query_node
)
from langgraph.types import Send

def retrieval_decider_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retrieval_decider_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_retrieval_decider_node.invoke(
        decision_model.invoke(inp).content
    ).retrieval_required
    return {"retrieval_required": res}

def generate_retriever_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retriever_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_retriever_query_node.invoke(
        query_gen_model.invoke(inp).content
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
    return {"retriever_queries": queries_dicts}

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

    # removing duplicates and capping maximum no of relevent contexts
    state['relevant_contexts']=list(set(state['relevant_contexts']))
    state['relevant_contexts']=state['relevant_contexts'][:5]
    return state

def direct_generation_node(state: schema):
    res = generation_model.invoke(state["user_query"]).content
    return {"generated_response": res}

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

    res = parser_for_is_relevant_node.invoke(
            decision_model.invoke(
                [sys_prompt, HumanMessage(content=hmn_prompt)]
            ).content
        )
    
    if res.is_relevant_context :
        return {"relevant_contexts": [inp['context']] }
    else:
        return {}

def aggregate_relevance(state):
    # plain pass-through node, just a sync point
    return {}

def answer_from_context_node(state: schema):
    contexts = [x.content for x  in state["relevant_contexts"]]
    sys_prompt = SystemMessage(content=sys_prompt_for_answer_from_context_node)

    context = ""
    for i in contexts:
        context += i
        context += "\n"

    hmn_prompt = HumanMessage(
        content=f"Query - {state['user_query']} \n\n Contexts - \n {context}"
    )
    inp = [sys_prompt,hmn_prompt]

    res = parser_for_answer_from_context_node.invoke(
        generation_model.invoke(inp).content
    ).response
    return {"generated_response": res}

def check_answer_grounded_node(state: schema):
    contexts = [x.content for x  in state["relevant_contexts"]]
    sys_prompt = SystemMessage(content=sys_prompt_for_check_answer_grounded_node)
    context = ""
    for i in contexts:
        context += i
        context += "\n"

    human_pr = HumanMessage(
        content=f"Answer - {state['generated_response']} \n Contexts - {context}"
    )

    res = parser_for_schema_for_check_answer_grounded_node.invoke(
        grounding_model.invoke([sys_prompt, human_pr]).content
    )

    return {"is_grounded": res.is_grounded, "evidence": res.evidence}

def revise_answer_node(state: schema):
    contexts =[x.content for x  in state["relevant_contexts"]]
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

    revised_answer = parser_for_revise_answer_node.invoke(
        critic_model.invoke([sys_prompt, human_pr]).content
    )

    return {
        "generated_response": revised_answer.revised_response,
        "max_retry_for_revise_answer": state["max_retry_for_revise_answer"] - 1,
    }

def is_answer_useful_node(state: schema):
    user_query = state["user_query"]
    generated_response = state["generated_response"]

    sys_prompt = SystemMessage(content=sys_prompt_for_is_answer_useful_node)
    human_pr = HumanMessage(
        content=f"Query - {user_query} \n\n Generated Response - {generated_response}"
    )

    res = parser_for_is_answer_useful_node.invoke(
        judge_model.invoke([sys_prompt, human_pr]).content
    )
    return {"is_answer_useful": res.is_useful}

def rewrite_answer_node(state: schema):
    contexts = [x.content for x in state["relevant_contexts"]]
    context = ""
    for i in contexts:
        context += i
        context += "\n"

    generated_response = state["generated_response"]
    user_query = state["user_query"]

    sys_prompt = SystemMessage(content=sys_prompt_for_rewrite_answer_node)
    human_pr = HumanMessage(
        content=f"""Query - {user_query} \n\n Previous Answer - {generated_response} \n\n Contexts - {context}"""
    )

    res = parser_for_rewrite_answer_node.invoke(
        answer_rewrite_model.invoke([sys_prompt, human_pr]).content
    )
    return {
        "generated_response": res.rewritten_response,
        "max_retry_for_answer_relevancy": state["max_retry_for_answer_relevancy"] - 1,
    }

def generate_web_search_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_web_search_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_web_search_query_node.invoke(
        query_gen_model.invoke(inp).content
    ).web_search_queries[:3]
    return {"web_search_queries": res}

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
