# base image
FROM python:3.12-slim

# setting a working directory
WORKDIR /app

#paths are related to the directory from where i run docker build (in my case it is root)
COPY ./Backend/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./Backend ./Backend
COPY ./data/constitution_and_ipc.chroma ./data/constitution_and_ipc.chroma
COPY ./src/workflow ./src/workflow

EXPOSE 8000

CMD ["python", "Backend/main.py"]

# Docker build command -> docker build -f ./Backend/Dockerfile -t pankajmaulekhi/backend . (from root)
# docker run -e LANGSMITH_API_KEY="<your-key>" -e AWS_BEARER_TOKEN_BEDROCK="<your-key>" -e LANGSMITH_ENDPOINT="https://api.smith.langchain.com" -e TAVILY_API_KEY="<your-key>" -e LANGSMITH_PROJECT="constitution-deployed" -e LANGSMITH_TRACING_V2="true" -p 8000:8000 backend