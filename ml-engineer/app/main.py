from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .utils import search_videos, summarize_transcript

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

class SummarizeRequest(BaseModel):
    video_id: str
    title: str

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}


@app.post("/search")
def search(request: SearchRequest):
    try:
        results = search_videos(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    try:
        summary = summarize_transcript(request.video_id, request.title)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
