import uvicorn
from starlette.applications import Starlette
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
import sys
import os
import json

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)
from src.workflow import get_workflow
from utils import _extract_node_details


skill = AgentSkill(
    id='constitution_rag_workflow',
    name='Indian Constitution & IPC Self-RAG',
    description=(
        'A Self-RAG (Retrieval-Augmented Generation) pipeline that answers questions about '
        'the Indian Constitution and the Indian Penal Code (IPC). '
        'The workflow: (1) decides whether retrieval, web search, or direct generation is needed; '
        '(2) generates optimised retriever queries and fetches relevant passages; '
        '(3) filters passages for relevance, falling back to web search when necessary; '
        '(4) generates a grounded answer and verifies it against the source contexts; '
        '(5) revises the answer if it is not fully supported, then checks answer relevance; '
        '(6) rewrites the answer if it does not address the question; '
        '(7) maintains multi-turn conversational memory with automatic summarisation. '
        'Supports follow-up questions within the same session.'
    ),
    input_modes=['text/plain'],
    output_modes=['text/plain'],
    tags=['legal', 'india', 'constitution', 'ipc', 'rag', 'self-rag', 'multi-turn'],
    examples=[
        'What are the Fundamental Rights guaranteed by the Indian Constitution?',
        'What does Article 21 say?',
        'Explain Section 302 of the IPC.',
        'What is the punishment for theft under the IPC?',
        'How does the amendment process work under Article 368?',
        'What did you just explain? (follow-up)',
    ],
)

public_agent_card = AgentCard(
    name='Indian Constitution & IPC Self-RAG Agent',
    description=(
        'An AI agent powered by a Self-RAG workflow for querying the Indian Constitution '
        'and the Indian Penal Code (IPC). '
        'Capabilities include: multi-query vector retrieval, context relevance filtering, '
        'web search fallback, grounding verification with iterative answer revision, '
        'answer relevance checking with rewriting, and persistent multi-turn memory '
        'with automatic conversation summarisation.'
    ),
    version='0.1.0',
    default_input_modes=['text/plain'],
    default_output_modes=['text/plain'],
    capabilities=AgentCapabilities(streaming=True, extended_agent_card=False),
    supported_interfaces=[
        AgentInterface(
            protocol_binding='JSONRPC',
            url='http://127.0.0.1:9999',
            protocol_version='1.0'
        )
    ],
    skills=[skill]
)

class WorkflowExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(
                event_queue=event_queue, task_id=task.id, context_id=task.context_id
            )
        try:

            await task_updater.update_status(
                state=TaskState.TASK_STATE_WORKING,
                message=new_text_message('Processing request...'),
            )
            user_query = get_message_text(context.message)
            if not user_query:
                await task_updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message('No text input is provided!'),
                )
                return

            thread_id=task.context_id

            async with get_workflow() as (workflow, ck_ptr):
                existing = await ck_ptr.aget_tuple(
                    {"configurable": {"thread_id": thread_id}}
                )
                if existing:
                    # if conv already exists
                    initial_state = {
                        "user_query": user_query,
                    }
                else:
                    initial_state = {
                        "user_query": user_query,
                        "k": 2,
                        "max_retriever_queries": 3,
                        "max_retry_for_groundness_checking": 1,
                        "max_retry_for_answer_relevant_checking": 1,
                        "max_turns_before_summarisation": 2,
                        "messages_to_include": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }

                async for chunk in workflow.astream(
                    initial_state,
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="updates",
                ):
                    node_name = list(chunk.keys())[0]
                    node_data = chunk[node_name]
                    details = _extract_node_details(node_name, node_data)
                    await task_updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=new_text_message(f"{json.dumps({ 'node': node_name, 'details': details })}")
                    )
                response = await workflow.aget_state(
                    config={"configurable": {"thread_id": thread_id}}
                )
                ai_response = response.values["generated_response"]
                

                await task_updater.update_status(
                    TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message(
                        f"{json.dumps({ 'response': ai_response})}"
                    )
                )
        except Exception as e:
            await task_updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=new_text_message(f"Error executing workflow: {str(e)}"),
            )

    async def cancel(self,context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if task is None:
            return

        running = self._running_tasks.get(task.id)
        if running is not None and not running.done():
            running.cancel()
        else:
            task_updater = TaskUpdater(
                event_queue=event_queue, task_id=task.id, context_id=task.context_id
            )
            await task_updater.update_status(
                state=TaskState.TASK_STATE_CANCELED,
                message=new_text_message('Task was not running or already finished.'),
            )

request_handler = DefaultRequestHandler(
    agent_executor=WorkflowExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=public_agent_card
)

routes = []
routes.extend(create_agent_card_routes(public_agent_card))  
routes.extend(create_jsonrpc_routes(request_handler, '/'))  

app = Starlette(routes=routes)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9999)
        
