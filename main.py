import logging

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from financial_graph import AgentResponse, QueryInput, run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Advisor API")

@app.post("/chat", response_model=AgentResponse)
async def chat(request: QueryInput) -> AgentResponse:
    try:
        return await run_in_threadpool(run_query, request)
    except Exception as e:
        logger.exception("Error en el chat")
        raise
