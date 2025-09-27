from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from main import BossWallahChatbot

app = FastAPI(title="Chatbot API", version="1.0.0")

chatbot = None

class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "english"

class QueryResponse(BaseModel):
    response: str
    language: str
    has_relevant_info: bool
    relevance_score: float

@app.on_event("startup")
async def startup_event():
    global chatbot
    print("Initializing Chatbot...")
    chatbot = BossWallahChatbot()
    print("Chatbot initialized successfully!")

@app.get("/")
async def root():
    return {"message": "Chatbot API is running"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        if not chatbot:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        initial_state = {
            "query": request.question,
            "retrieved_docs": [],
            "response": "",
            "language": request.language,
            "relevance_score": 0.0,
            "has_relevant_info": False
        }
        
        result = chatbot.app.invoke(initial_state)
        
        return QueryResponse(
            response=result.get('response', 'No response generated'),
            language=result.get('language', 'english'),
            has_relevant_info=result.get('has_relevant_info', False),
            relevance_score=result.get('relevance_score', 0.0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chatbot_initialized": chatbot is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)