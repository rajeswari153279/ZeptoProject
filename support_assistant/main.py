# ============================================================
# MODULE 3 - PART C: FASTAPI APP (POST /ask)
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from b_langgraph_app import app_graph

app = FastAPI(title="Zepto Support Assistant")

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = app_graph.invoke({
        "question": request.query,
        "intent": "", "answer": "", "sources": [], "confidence": 0.0
    })
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )

@app.get("/")
def root():
    return {"status": "Support assistant is running. POST to /ask with {'query': '...'}"}