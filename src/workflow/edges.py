from workflow.state import schema

def retrieval_decider_condition(state: schema):
    return state["retrieval_required"]

def is_relevant_condition(state: schema):
    return len(state["relevant_contexts"]) > 0

def is_grounded_condition(state: schema):
    return state["is_grounded"]=="fully_supported" or state["max_retry_for_groundness_checking"] <= 0

def is_answer_relevant_condition(state: schema):
    return state["is_answer_relevant"] or state["max_retry_for_answer_relevant_checking"] <= 0

def memory_summary_condition(state: schema):
    counter = state.get("turns_until_summary_update")
    if counter is not None and counter <= 0:
        return "summarize"
    return "end"

