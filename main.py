import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from financial_graph import AgentResponse, QueryInput, run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Advisor API")


@app.post("/chat", response_model=AgentResponse)
async def chat(request: QueryInput) -> AgentResponse:
    request_id = str(uuid.uuid4())

    try:
        logger.info("/chat request started | request_id=%s", request_id)
        result = await run_in_threadpool(run_query, request)
        logger.info("/chat request completed | request_id=%s", request_id)
        return result
    except ValueError as exc:
        logger.warning(
            "Validation/runtime error in /chat | request_id=%s | error=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        error_text = str(exc)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            logger.warning(
                "Upstream quota exceeded in /chat | request_id=%s | error=%s",
                request_id,
                error_text,
            )
            raise HTTPException(
                status_code=429,
                detail="Model quota exceeded. Retry later or lower request volume.",
            ) from exc

        logger.exception(
            "Unhandled error in /chat | request_id=%s",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing chat request.",
        ) from exc
