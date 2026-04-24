import os
import sys
import asyncio

# 1. Prevent LLM unloading to ensure instant switching between 1B and 8B models
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

# 2. Critical Windows Async Patch
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from researcher import run_deep_research
from coder_agent import detect_language, generate_code
from memory import save_to_memory

app = FastAPI(title="ExtremeAI Backend")

# --- DATA MODELS ---
class ResearchRequest(BaseModel):
    topic: str

class ResearchResponse(BaseModel):
    status: str
    topic: str
    report: str
    language: str
    generated_code: str
    queries: list[str]
    sources: list[str]
    raw_data: list[str]

# --- ENDPOINTS ---
@app.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    The Orchestrator: Routes to 1B model, runs parallel 8B research, 
    and triggers background memory persistence.
    """
    # 1. Intent Detection & Routing (Uses Fast 1B Model)
    language = await detect_language(request.topic)
    
    # 2. Parallel Deep Research (Uses Parallel Scrapers + 8B Model)
    research_data = await run_deep_research(request.topic)
    
    # 3. specialist Code Generation (Uses 8B Model + Zero-Wrapper Prompt)
    generated_code = await generate_code(request.topic, research_data["report"], language)
    
    # 4. Background Persistence (FAISS Archive)
    # We pass this as a background task so the UI shows results immediately
    metadata = {"topic": request.topic, "language": language}
    background_tasks.add_task(save_to_memory, research_data["report"], metadata)
    
    # 5. Return complete payload to UI
    return {
        "status": "success", 
        "topic": request.topic,
        "report": research_data["report"],
        "language": language,
        "generated_code": generated_code,
        "queries": research_data["queries"],
        "sources": research_data["sources"],
        "raw_data": research_data["raw_data"] # Added to support the 'Raw Data' tab in UI
    }

if __name__ == "__main__":
    # Ensure loop="asyncio" is set for Windows stability
    uvicorn.run("main:app", host="127.0.0.1", port=8000, loop="asyncio", reload=False)