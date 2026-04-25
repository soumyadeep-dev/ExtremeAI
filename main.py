import os
import sys
import asyncio
import json

os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from researcher import run_deep_research
from coder_agent import detect_language, generate_code, SUPPORTED_LANGUAGES
from memory import save_to_memory

app = FastAPI(title="ExtremeAI Backend")


class ResearchRequest(BaseModel):
    topic: str
    language_override: str | None = None  # Passed from UI dropdown; None = auto-detect


class ResearchResponse(BaseModel):
    status: str
    topic: str
    report: str
    language: str
    generated_code: str
    queries: list[str]
    sources: list[str]
    raw_data: list[str]


@app.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    # Validate override if provided
    override = request.language_override
    if override and override not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language override: {override}")

    try:
        # language detection + research run in parallel
        # detect_language respects override — skips OpenAI call entirely if set
        language, research_data = await asyncio.gather(
            detect_language(request.topic, language_override=override),
            run_deep_research(request.topic)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {str(e)}")

    try:
        generated_code = await generate_code(request.topic, research_data["report"], language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

    metadata = {"topic": request.topic, "language": language}
    background_tasks.add_task(save_to_memory, research_data["report"], metadata)

    return {
        "status": "success",
        "topic": request.topic,
        "report": research_data["report"],
        "language": language,
        "generated_code": generated_code,
        "queries": research_data["queries"],
        "sources": research_data["sources"],
        "raw_data": research_data["raw_data"]
    }


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    override = request.language_override
    if override and override not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language override: {override}")

    async def event_stream():
        try:
            language, research_data = await asyncio.gather(
                detect_language(request.topic, language_override=override),
                run_deep_research(request.topic)
            )

            yield f"data: {json.dumps({'type': 'meta', 'language': language, 'queries': research_data['queries'], 'sources': research_data['sources']})}\n\n"

            from langchain_ollama import ChatOllama
            from langchain_core.prompts import PromptTemplate
            from coder_agent import LANGUAGE_CONFIGS

            heavy_llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
            config = LANGUAGE_CONFIGS.get(language, {"role": "Senior Developer", "rules": "Write clean code."})

            prompt = PromptTemplate.from_template(
                "You are a {role}. Provide the implementation for: {topic}\n\n"
                "RESEARCH:\n{research}\n\n"
                "CRITICAL RULE: Return ONLY the raw code for {language}. "
                "Rules: {rules}\n\nRETURN ONLY THE RAW CODE. NO MARKDOWN. NO EXPLANATION."
            )
            chain = prompt | heavy_llm

            full_code = ""
            async for chunk in chain.astream({
                "role": config["role"],
                "language": language,
                "topic": request.topic,
                "research": research_data["report"][:4000],
                "rules": config["rules"]
            }):
                token = chunk.content
                full_code += token
                yield f"data: {json.dumps({'type': 'code_chunk', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'report', 'report': research_data['report']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, loop="asyncio", reload=False)