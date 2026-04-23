import sys
import asyncio

# Windows async loop patch
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
from researcher import run_deep_research
from memory import save_to_memory
from terraform_agent import generate_terraform

app = FastAPI(title="Local Deep Research API")

class ResearchRequest(BaseModel):
    topic: str

# Updated response model to match your UI
class ResearchResponse(BaseModel):
    topic: str
    report: str
    status: str
    terraform_code: str
    queries: list[str] = []
    sources: list[str] = []
    raw_data: list[str] = []

@app.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    try:
        # 1. Run the Researcher (It now returns a full dictionary!)
        research_data = await run_deep_research(request.topic)
        
        # 2. Execute the Terraform Agent using just the report string from the dictionary
        print("Generating Terraform configuration...")
        tf_code = await generate_terraform(request.topic, research_data["report"])
        
        # 3. Save to local FAISS memory
        metadata = {"topic": request.topic}
        background_tasks.add_task(save_to_memory, research_data["report"], metadata)
        
        # 4. Send EVERYTHING to the UI
        return ResearchResponse(
            topic=request.topic,
            report=research_data["report"],
            status="success",
            terraform_code=tf_code,
            queries=research_data["queries"],     # <-- Now passing the real data
            sources=research_data["sources"],     # <-- Now passing the real data
            raw_data=research_data["raw_data"]    # <-- Now passing the real data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # loop="asyncio" forces Uvicorn to respect our Windows Proactor patch!
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, loop="asyncio")