from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
import asyncio
from datetime import datetime
import uuid

from backend.models.hybrid_model import HybridFakeNewsDetector
from backend.detectors.linguistic_analyzer import LinguisticAnalyzer
from backend.detectors.source_verifier import SourceVerifier

app = FastAPI(title="Fake News Detection API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = HybridFakeNewsDetector()
linguistic_analyzer = LinguisticAnalyzer()
source_verifier = SourceVerifier()

# Load models
detector.load_models()

# Request/Response models
class NewsAnalysisRequest(BaseModel):
    text: str
    url: Optional[HttpUrl] = None
    source: Optional[str] = None
    context: Optional[Dict] = None

class NewsAnalysisResponse(BaseModel):
    id: str
    timestamp: datetime
    prediction: Dict
    linguistic_analysis: Dict
    source_verification: Optional[Dict]
    fact_checks: List[Dict]
    summary: Dict
    recommendations: List[str]

@app.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze news article for fake news detection"""
    
    # Validate input
    if not request.text and not request.url:
        raise HTTPException(400, "Either text or URL must be provided")
    
    # Extract text from URL if provided
    text = request.text
    if request.url:
        text = await extract_text_from_url(str(request.url))
    
    # Run analyses in parallel
    prediction_task = asyncio.create_task(analyze_prediction(text))
    linguistic_task = asyncio.create_task(analyze_linguistics(text))
    source_task = asyncio.create_task(analyze_source(request.source, request.url))
    fact_check_task = asyncio.create_task(check_facts(text))
    
    # Gather results
    prediction, linguistic, source, fact_checks = await asyncio.gather(
        prediction_task, linguistic_task, source_task, fact_check_task
    )
    
    # Generate summary and recommendations
    summary = generate_summary(prediction, linguistic, source)
    recommendations = generate_recommendations(prediction, linguistic, source)
    
    # Store in database (background task)
    background_tasks.add_task(store_analysis, {
        'id': str(uuid.uuid4()),
        'text': text,
        'prediction': prediction,
        'timestamp': datetime.now()
    })
    
    return NewsAnalysisResponse(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        prediction=prediction,
        linguistic_analysis=linguistic,
        source_verification=source,
        fact_checks=fact_checks,
        summary=summary,
        recommendations=recommendations
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": detector.bert_model is not None,
        "timestamp": datetime.now()
    }

@app.post("/batch-analyze")
async def batch_analyze(requests: List[NewsAnalysisRequest]):
    """Batch analysis of multiple news articles"""
    tasks = [analyze_news(req, BackgroundTasks()) for req in requests]
    results = await asyncio.gather(*tasks)
    return {"results": results}

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "total_analyses": await get_total_analyses(),
        "average_confidence": await get_average_confidence(),
        "fake_ratio": await get_fake_ratio(),
        "top_sources": await get_top_sources()
    }

async def analyze_prediction(text: str) -> Dict:
    """Get model predictions"""
    return detector.predict_hybrid(text)

async def analyze_linguistics(text: str) -> Dict:
    """Get linguistic analysis"""
    return linguistic_analyzer.analyze(text)

async def analyze_source(source: Optional[str], url: Optional[str]) -> Optional[Dict]:
    """Verify news source"""
    if source:
        return source_verifier.verify_source(source)
    elif url:
        return source_verifier.verify_source(str(url))
    return None

async def check_facts(text: str) -> List[Dict]:
    """Check facts against external APIs"""
    return source_verifier.check_fact_check_apis(text)

async def extract_text_from_url(url: str) -> str:
    """Extract article text from URL"""
    # Implement web scraping logic here
    # Use libraries like newspaper3k, readability-lxml
    return "Extracted text from URL"

def generate_summary(prediction: Dict, linguistic: Dict, source: Optional[Dict]) -> Dict:
    """Generate comprehensive summary"""
    return {
        'verdict': prediction['verdict'],
        'confidence': prediction['confidence'],
        'key_risk_factors': get_key_risk_factors(prediction, linguistic),
        'linguistic_risk': linguistic.get('overall_linguistic_risk', 0),
        'source_trust': source.get('trust_score', 0) if source else None
    }

def generate_recommendations(prediction: Dict, linguistic: Dict, source: Optional[Dict]) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if prediction['is_fake']:
        recommendations.append("⚠️ This content shows strong indicators of being fake news")
        recommendations.append("🔍 Verify this information with trusted sources before sharing")
    
    if linguistic.get('exaggeration_score', 0) > 0.7:
        recommendations.append("📢 Content contains exaggerated claims - fact-check thoroughly")
    
    if linguistic.get('sentiment_analysis', {}).get('sentiment_risk', 0) > 0.7:
        recommendations.append("🎭 Highly emotional language detected - maintain critical thinking")
    
    if source and source.get('trust_score', 0) < 0.4:
        recommendations.append(f"⚠️ Source {source.get('domain')} has low trust score")
    
    if not recommendations:
        recommendations.append("✓ Content appears legitimate, but always verify important claims")
    
    return recommendations

async def store_analysis(analysis: Dict):
    """Store analysis in database"""
    # Implement database storage
    pass

async def get_total_analyses() -> int:
    """Get total number of analyses"""
    return 0  # Implement database query

async def get_average_confidence() -> float:
    """Get average confidence score"""
    return 0.0

async def get_fake_ratio() -> float:
    """Get ratio of fake news detected"""
    return 0.0

async def get_top_sources() -> List[Dict]:
    """Get most analyzed sources"""
    return []