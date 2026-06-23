"""
Yojana AI — Backend API Server
================================
Run: uvicorn server:app --reload --port 8000

Endpoints:
  POST /find-schemes   — Main endpoint for finding schemes
  GET  /health         — Health check
  GET  /stats          — DB stats
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import httpx  # For calling AMD/Claude API

# Import our RAG pipeline
from rag_pipeline import search_schemes, format_for_prompt

app = FastAPI(title="Yojana AI API", version="1.0.0")

# Allow frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
# Before July 6: uses Claude (OpenRouter)
# After July 6:  swap AMD_MODE = True → uses AMD endpoint
AMD_MODE = os.getenv("AMD_MODE", "false").lower() == "true"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
AMD_API_KEY = os.getenv("AMD_API_KEY", "")
AMD_BASE_URL = os.getenv("AMD_BASE_URL", "https://api.amd.com/v1")  # update on July 6

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ml": "Malayalam",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati"
}

# ── Request/Response Models ───────────────────────────────────────────────────
class SchemeRequest(BaseModel):
    age: Optional[str] = None
    income: Optional[str] = None
    state: Optional[str] = None
    gender: Optional[str] = None
    categories: Optional[List[str]] = []
    free_text: Optional[str] = None
    language: Optional[str] = "en"

class SchemeResult(BaseModel):
    name: str
    ministry: str
    match: str
    description: str
    benefit: str
    tags: List[str]
    applyUrl: str

class SchemeResponse(BaseModel):
    schemes: List[SchemeResult]
    summary: str
    source: str  # "rag" or "ai_only"

# ── Build user profile string ─────────────────────────────────────────────────
def build_profile_string(req: SchemeRequest) -> str:
    parts = []
    if req.age: parts.append(f"Age: {req.age}")
    if req.gender: parts.append(f"Gender: {req.gender}")
    if req.state: parts.append(f"State: {req.state}")
    if req.income: parts.append(f"Annual income: ₹{req.income}")
    if req.categories: parts.append(f"Categories: {', '.join(req.categories)}")
    if req.free_text: parts.append(req.free_text)
    return '. '.join(parts)

# ── Call AI (AMD or Claude) ───────────────────────────────────────────────────
async def call_ai(prompt: str, system: str) -> str:
    """Single function — swaps between AMD and Claude based on AMD_MODE env var"""
    
    if AMD_MODE:
        # ── AMD endpoint (July 6 onwards) ──
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{AMD_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {AMD_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/Llama-3.1-8B-Instruct",  # or Qwen on AMD
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3
                }
            )
            data = response.json()
            if "choices" not in data:
                raise Exception(f"OpenRouter error: {data.get('error', data)}")
            return data["choices"][0]["message"]["content"]

    else:
        # ── Claude via OpenRouter (before July 6) ──
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 3000
                }
            )
            data = response.json()
            if "choices" not in data:
                raise Exception(f"OpenRouter error: {data.get('error', data)}")
            return data["choices"][0]["message"]["content"]

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/find-schemes", response_model=SchemeResponse)
async def find_schemes(req: SchemeRequest):
    
    profile = build_profile_string(req)
    if not profile.strip():
        raise HTTPException(status_code=400, detail="Please provide at least some profile info")
    
    lang = req.language or "en"
    lang_name = LANG_NAMES.get(lang, "English")
    
    # ── Step 1: RAG Search ──
    rag_context = ""
    source = "ai_only"
    
    try:
        schemes_from_db = search_schemes(
            user_profile=profile,
            n_results=8,
            state_filter=req.state
        )
        if schemes_from_db:
            rag_context = format_for_prompt(schemes_from_db)
            source = "rag"
    except Exception as e:
        # DB not ready yet — fallback to AI only
        rag_context = ""
        source = "ai_only"
    
    # ── Step 2: Build prompt ──
    system_prompt = f"""You are an expert on Indian government welfare schemes — both Central and State level.
Your job is to match users with schemes they qualify for and explain them clearly in {lang_name}.
Always respond with accurate, helpful information. 
Respond ONLY in valid JSON — no markdown, no backticks, no explanation outside JSON. CRITICAL: when writing non-English text (especially Malayalam, Tamil, or other Indic scripts), ensure all string values are properly JSON-escaped — escape any double quotes, backslashes, and newlines inside text fields. Double-check the JSON is syntactically valid before responding."""

    user_prompt = f"""User Profile:
{profile}

{f'Relevant schemes from our database:{chr(10)}{rag_context}' if rag_context else 'Use your knowledge of Indian government schemes.'}

Instructions:
1. Based on the profile and {'the database schemes above' if rag_context else 'your knowledge'}, select the 4-6 BEST matching schemes
2. Prioritize schemes the user is most likely to qualify for
3. Write descriptions and benefits in {lang_name}
4. Return ONLY this JSON structure, nothing else:

{{
  "schemes": [
    {{
      "name": "Scheme name",
      "ministry": "Ministry/Department",
      "match": "High or Medium",
      "description": "2-3 sentences in {lang_name} about what this scheme does",
      "benefit": "Specific benefit amount or type in {lang_name}",
      "tags": ["relevant", "category", "tags"],
      "applyUrl": "https://actual-url.gov.in"
    }}
  ],
  "summary": "One line in {lang_name} explaining why these schemes were selected for this user"
}}"""

    # ── Step 3: Call AI ──
    try:
        raw = await call_ai(user_prompt, system_prompt)
        clean = raw.replace("```json", "").replace("```", "").strip()

        # Aggressively extract JSON — find first { to last }
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise HTTPException(status_code=500, detail=f"No JSON in response: {clean[:200]}")
        clean = clean[start:end]

        parsed = json.loads(clean)
        
        return SchemeResponse(
            schemes=[SchemeResult(**s) for s in parsed["schemes"]],
            summary=parsed.get("summary", ""),
            source=source
        )
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse failed: {str(e)} | Raw: {clean[:300]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Utility endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "AMD" if AMD_MODE else "Claude/OpenRouter",
        "rag": os.path.exists("./yojana_db")
    }

@app.get("/stats")
def stats():
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./yojana_db")
        collection = client.get_collection("schemes")
        return {"total_schemes": collection.count(), "status": "ready"}
    except:
        return {"total_schemes": 0, "status": "db_not_built"}
