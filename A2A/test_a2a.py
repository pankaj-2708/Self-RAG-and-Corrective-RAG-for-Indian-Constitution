import asyncio
import httpx
from a2a.client import ClientConfig, create_client ,A2ACardResolver
from a2a.helpers import new_text_message, get_stream_response_text
from a2a.types import Role, SendMessageRequest, AgentCard

# Rename your wrapper function so it doesn't conflict
async def main(): 
    server_url = "http://127.0.0.1:9999/"
    
    async with httpx.AsyncClient() as http_client:
        resolver=A2ACardResolver(httpx_client=http_client,base_url=server_url)
        public_agent_card = await resolver.get_agent_card()

    custom_timeout = httpx.Timeout(120.0)
    my_http_client = httpx.AsyncClient(timeout=custom_timeout)
    client_config = ClientConfig(streaming=True,httpx_client=my_http_client)
    
    streaming_client = await create_client(agent=public_agent_card, client_config=client_config)

    raw_text = "What does IPC Section 134 say about abetting assault in the Navvy?"
    message = new_text_message(raw_text, role=Role.ROLE_USER)
    
    async for chunk in streaming_client.send_message(SendMessageRequest(message=message)):
        # text_content = get_stream_response_text(chunk)
        # if text_content:
        #     print(text_content, end="", flush=True)
        print(chunk)

    await streaming_client.close()

if __name__ == "__main__":
    asyncio.run(main())