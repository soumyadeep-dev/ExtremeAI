from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from researcher import run_deep_research
from memory import save_to_memory

app = FastAPI(title="Local Deep Research API")

class ResearchRequest(BaseModel):
    topic: str

class ResearchResponse(BaseModel):
    topic: str
    report: str
    status: str

@app.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    try:
        # Run the multi-agent AI pipeline
        final_report = await run_deep_research(request.topic)
        
        # Save to FAISS without blocking the API response
        metadata = {"topic": request.topic}
        background_tasks.add_task(save_to_memory, final_report, metadata)
        
        return ResearchResponse(
            topic=request.topic,
            report=final_report,
            status="success - saved to local FAISS"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))