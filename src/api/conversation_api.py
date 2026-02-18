"""
XYZ Analytics API - Conversation Processing Endpoints

This module provides the core API endpoints for the Conversation Analytics Platform.
It handles conversation ingestion, processing, and retrieval operations.

Note: This is a simplified version for assessment purposes. The actual implementation
connects to Azure SQL shards, Elasticsearch, and Redis in production.
"""

from fastapi import APIRouter, HTTPException, Header, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import re
import asyncio

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


# ============================================================================
# Models
# ============================================================================

class ConversationType(str, Enum):
    CALL = "call"
    CHAT = "chat"
    EMAIL = "email"


class SentimentScore(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ConversationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Speaker(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., pattern="^(agent|customer|system)$")
    name: Optional[str] = Field(None, max_length=200)


class Utterance(BaseModel):
    speaker_id: str
    text: str = Field(..., min_length=1, max_length=10000)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @validator('end_time')
    def end_time_after_start(cls, v, values):
        if 'start_time' in values and v < values['start_time']:
            raise ValueError('end_time must be greater than or equal to start_time')
        return v


class ConversationCreate(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=255)
    conversation_type: ConversationType
    speakers: List[Speaker] = Field(..., min_items=1, max_items=10)
    utterances: List[Utterance] = Field(..., min_items=1, max_items=5000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    recorded_at: datetime
    duration_seconds: float = Field(..., gt=0, le=86400)  # Max 24 hours
    language: str = Field(default="en", pattern="^[a-z]{2}$")

    @validator('speakers')
    def validate_unique_speaker_ids(cls, v):
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError('Speaker IDs must be unique')
        return v

    @validator('utterances')
    def validate_speaker_references(cls, v, values):
        if 'speakers' in values:
            valid_ids = {s.id for s in values['speakers']}
            for utt in v:
                if utt.speaker_id not in valid_ids:
                    raise ValueError(f'Invalid speaker_id: {utt.speaker_id}')
        return v


class ConversationResponse(BaseModel):
    id: str
    external_id: str
    customer_id: str
    conversation_type: ConversationType
    status: ConversationStatus
    speakers: List[Speaker]
    utterance_count: int
    duration_seconds: float
    recorded_at: datetime
    created_at: datetime
    processed_at: Optional[datetime] = None


class ConversationDetail(ConversationResponse):
    utterances: List[Utterance]
    metadata: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseModel):
    conversation_id: str
    overall_sentiment: SentimentScore
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)
    topics: List[str]
    key_phrases: List[str]
    qa_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    compliance_flags: List[str]
    processing_time_ms: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    conversation_types: Optional[List[ConversationType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sentiment: Optional[SentimentScore] = None
    min_qa_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    max_qa_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @validator('date_to')
    def date_range_valid(cls, v, values):
        if v and 'date_from' in values and values['date_from']:
            if v < values['date_from']:
                raise ValueError('date_to must be after date_from')
        return v


class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: List[ConversationResponse]


class BulkOperationRequest(BaseModel):
    conversation_ids: List[str] = Field(..., min_items=1, max_items=100)
    operation: str = Field(..., pattern="^(reprocess|delete|archive)$")


class BulkOperationResponse(BaseModel):
    operation_id: str
    status: str
    processed: int
    failed: int
    errors: List[Dict[str, str]]


# ============================================================================
# Mock Data Store (In-memory for assessment purposes)
# ============================================================================

class MockDataStore:
    """
    Simulates the data layer. In production, this would connect to:
    - Azure SQL (sharded) for structured data
    - Elasticsearch for search operations
    - Redis for caching
    """
    
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.analyses: Dict[str, Dict] = {}
        self.api_keys: Dict[str, Dict] = {
            # Pre-seeded API keys for testing
            "CUS_test_customer_123": {
                "customer_id": "cust_001",
                "type": "customer",
                "status": "active",
                "rate_limit": 100
            },
            "CUS_premium_customer": {
                "customer_id": "cust_002", 
                "type": "customer",
                "status": "active",
                "rate_limit": 500
            },
            "CUS_suspended_key": {
                "customer_id": "cust_003",
                "type": "customer",
                "status": "suspended",
                "rate_limit": 100
            },
            "ADM_admin_key_456": {
                "customer_id": None,
                "type": "admin",
                "status": "active",
                "rate_limit": 1000
            }
        }
        self._seed_test_data()
    
    def _seed_test_data(self):
        """Seed some initial test conversations"""
        test_conversations = [
            {
                "id": "conv_001",
                "external_id": "ext_001",
                "customer_id": "cust_001",
                "conversation_type": "call",
                "status": "completed",
                "speakers": [
                    {"id": "spk_1", "role": "agent", "name": "John Smith"},
                    {"id": "spk_2", "role": "customer", "name": "Jane Doe"}
                ],
                "utterances": [
                    {"speaker_id": "spk_1", "text": "Hello, thank you for calling support.", "start_time": 0.0, "end_time": 2.5, "confidence": 0.95},
                    {"speaker_id": "spk_2", "text": "Hi, I have an issue with my account.", "start_time": 2.6, "end_time": 5.0, "confidence": 0.92}
                ],
                "metadata": {"source": "zendesk", "ticket_id": "TKT-12345"},
                "duration_seconds": 300.5,
                "recorded_at": datetime(2024, 1, 15, 10, 30),
                "created_at": datetime(2024, 1, 15, 11, 0),
                "processed_at": datetime(2024, 1, 15, 11, 5),
                "language": "en"
            },
            {
                "id": "conv_002",
                "external_id": "ext_002",
                "customer_id": "cust_001",
                "conversation_type": "chat",
                "status": "completed",
                "speakers": [
                    {"id": "spk_3", "role": "agent", "name": "Alice Johnson"},
                    {"id": "spk_4", "role": "customer", "name": None}
                ],
                "utterances": [
                    {"speaker_id": "spk_3", "text": "Welcome to live chat!", "start_time": 0.0, "end_time": 1.0, "confidence": 1.0},
                    {"speaker_id": "spk_4", "text": "I need help canceling my subscription.", "start_time": 1.5, "end_time": 4.0, "confidence": 0.98}
                ],
                "metadata": {"channel": "web", "page_url": "/pricing"},
                "duration_seconds": 180.0,
                "recorded_at": datetime(2024, 1, 16, 14, 0),
                "created_at": datetime(2024, 1, 16, 14, 5),
                "processed_at": datetime(2024, 1, 16, 14, 8),
                "language": "en"
            }
        ]
        
        for conv in test_conversations:
            self.conversations[conv["id"]] = conv
            # Add corresponding analysis
            self.analyses[conv["id"]] = {
                "conversation_id": conv["id"],
                "overall_sentiment": "neutral",
                "sentiment_confidence": 0.85,
                "topics": ["account", "support"],
                "key_phrases": ["account issue", "need help"],
                "qa_score": 78.5,
                "compliance_flags": [],
                "processing_time_ms": 1250
            }


# Global data store instance
data_store = MockDataStore()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> Dict:
    """
    Validates the API key and returns the associated context.
    
    BUG: There's an intentional bug here for candidates to discover.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    
    # Check if key exists
    key_data = data_store.api_keys.get(x_api_key)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # BUG: Status check has an issue - can you spot it?
    if key_data.get("status") != "active":
        raise HTTPException(status_code=403, detail="API key is not active")
    
    return {
        "customer_id": key_data["customer_id"],
        "key_type": key_data["type"],
        "rate_limit": key_data["rate_limit"]
    }


async def require_customer_key(auth: Dict = Depends(validate_api_key)) -> str:
    """Ensures the API key is a customer key and returns customer_id"""
    if auth["key_type"] != "customer":
        raise HTTPException(
            status_code=403, 
            detail="This endpoint requires a customer API key"
        )
    return auth["customer_id"]


async def require_admin_key(auth: Dict = Depends(validate_api_key)) -> Dict:
    """Ensures the API key is an admin key"""
    if auth["key_type"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires an admin API key"
        )
    return auth


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    conversation: ConversationCreate,
    background_tasks: BackgroundTasks,
    customer_id: str = Depends(require_customer_key)
):
    """
    Ingest a new conversation for processing.
    
    The conversation will be queued for AI-powered analysis including:
    - Sentiment analysis
    - Topic extraction
    - Quality assurance scoring
    - Compliance checking
    """
    # Generate unique ID
    conv_id = f"conv_{hashlib.md5(f'{customer_id}_{conversation.external_id}'.encode()).hexdigest()[:12]}"
    
    # Check for duplicate external_id
    for existing in data_store.conversations.values():
        if existing["external_id"] == conversation.external_id and existing["customer_id"] == customer_id:
            raise HTTPException(
                status_code=409,
                detail=f"Conversation with external_id '{conversation.external_id}' already exists"
            )
    
    now = datetime.utcnow()
    
    conv_data = {
        "id": conv_id,
        "external_id": conversation.external_id,
        "customer_id": customer_id,
        "conversation_type": conversation.conversation_type.value,
        "status": "pending",
        "speakers": [s.dict() for s in conversation.speakers],
        "utterances": [u.dict() for u in conversation.utterances],
        "metadata": conversation.metadata,
        "duration_seconds": conversation.duration_seconds,
        "recorded_at": conversation.recorded_at,
        "created_at": now,
        "processed_at": None,
        "language": conversation.language
    }
    
    data_store.conversations[conv_id] = conv_data
    
    # Queue background processing
    background_tasks.add_task(process_conversation, conv_id)
    
    return ConversationResponse(
        id=conv_id,
        external_id=conversation.external_id,
        customer_id=customer_id,
        conversation_type=conversation.conversation_type,
        status=ConversationStatus.PENDING,
        speakers=conversation.speakers,
        utterance_count=len(conversation.utterances),
        duration_seconds=conversation.duration_seconds,
        recorded_at=conversation.recorded_at,
        created_at=now
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    customer_id: str = Depends(require_customer_key)
):
    """
    Retrieve a specific conversation by ID.
    
    Returns the full conversation including all utterances and analysis results
    if processing is complete.
    """
    conv = data_store.conversations.get(conversation_id)
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Tenant isolation check
    if conv["customer_id"] != customer_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    analysis = data_store.analyses.get(conversation_id)
    
    return ConversationDetail(
        id=conv["id"],
        external_id=conv["external_id"],
        customer_id=conv["customer_id"],
        conversation_type=ConversationType(conv["conversation_type"]),
        status=ConversationStatus(conv["status"]),
        speakers=[Speaker(**s) for s in conv["speakers"]],
        utterance_count=len(conv["utterances"]),
        utterances=[Utterance(**u) for u in conv["utterances"]],
        duration_seconds=conv["duration_seconds"],
        recorded_at=conv["recorded_at"],
        created_at=conv["created_at"],
        processed_at=conv.get("processed_at"),
        metadata=conv.get("metadata", {}),
        analysis=analysis
    )


@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(
    customer_id: str = Depends(require_customer_key),
    conversation_type: Optional[ConversationType] = None,
    status: Optional[ConversationStatus] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    List all conversations for the authenticated customer.
    
    Supports filtering by type, status, and date range.
    """
    results = []
    
    for conv in data_store.conversations.values():
        # Tenant isolation
        if conv["customer_id"] != customer_id:
            continue
        
        # Apply filters
        if conversation_type and conv["conversation_type"] != conversation_type.value:
            continue
        if status and conv["status"] != status.value:
            continue
        if date_from and conv["recorded_at"] < date_from:
            continue
        if date_to and conv["recorded_at"] > date_to:
            continue
        
        results.append(ConversationResponse(
            id=conv["id"],
            external_id=conv["external_id"],
            customer_id=conv["customer_id"],
            conversation_type=ConversationType(conv["conversation_type"]),
            status=ConversationStatus(conv["status"]),
            speakers=[Speaker(**s) for s in conv["speakers"]],
            utterance_count=len(conv["utterances"]),
            duration_seconds=conv["duration_seconds"],
            recorded_at=conv["recorded_at"],
            created_at=conv["created_at"],
            processed_at=conv.get("processed_at")
        ))
    
    # Sort by created_at descending
    results.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    return results[offset:offset + limit]


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    customer_id: str = Depends(require_customer_key)
):
    """
    Delete a conversation and all associated data.
    
    This is a hard delete - the data cannot be recovered.
    """
    conv = data_store.conversations.get(conversation_id)
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv["customer_id"] != customer_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del data_store.conversations[conversation_id]
    if conversation_id in data_store.analyses:
        del data_store.analyses[conversation_id]
    
    return None


@router.post("/search", response_model=SearchResponse)
async def search_conversations(
    search_request: SearchRequest,
    customer_id: str = Depends(require_customer_key)
):
    """
    Perform a full-text search across conversations.
    
    In production, this uses Elasticsearch with semantic search capabilities.
    For this mock implementation, we use simple substring matching.
    """
    results = []
    query_lower = search_request.query.lower()
    
    for conv in data_store.conversations.values():
        # Tenant isolation
        if conv["customer_id"] != customer_id:
            continue
        
        # Type filter
        if search_request.conversation_types:
            if conv["conversation_type"] not in [t.value for t in search_request.conversation_types]:
                continue
        
        # Date filters
        if search_request.date_from and conv["recorded_at"] < search_request.date_from:
            continue
        if search_request.date_to and conv["recorded_at"] > search_request.date_to:
            continue
        
        # Sentiment filter
        analysis = data_store.analyses.get(conv["id"])
        if search_request.sentiment and analysis:
            if analysis["overall_sentiment"] != search_request.sentiment.value:
                continue
        
        # QA score filters
        if analysis:
            if search_request.min_qa_score and analysis["qa_score"] < search_request.min_qa_score:
                continue
            if search_request.max_qa_score and analysis["qa_score"] > search_request.max_qa_score:
                continue
        
        # Text search across utterances
        found = False
        for utt in conv["utterances"]:
            if query_lower in utt["text"].lower():
                found = True
                break
        
        if not found:
            continue
        
        results.append(ConversationResponse(
            id=conv["id"],
            external_id=conv["external_id"],
            customer_id=conv["customer_id"],
            conversation_type=ConversationType(conv["conversation_type"]),
            status=ConversationStatus(conv["status"]),
            speakers=[Speaker(**s) for s in conv["speakers"]],
            utterance_count=len(conv["utterances"]),
            duration_seconds=conv["duration_seconds"],
            recorded_at=conv["recorded_at"],
            created_at=conv["created_at"],
            processed_at=conv.get("processed_at")
        ))
    
    # Sort and paginate
    results.sort(key=lambda x: x.created_at, reverse=True)
    start = (search_request.page - 1) * search_request.page_size
    end = start + search_request.page_size
    
    return SearchResponse(
        total=len(results),
        page=search_request.page,
        page_size=search_request.page_size,
        results=results[start:end]
    )


@router.get("/{conversation_id}/analysis", response_model=AnalysisResult)
async def get_analysis(
    conversation_id: str,
    customer_id: str = Depends(require_customer_key)
):
    """
    Retrieve the AI analysis results for a conversation.
    """
    conv = data_store.conversations.get(conversation_id)
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv["customer_id"] != customer_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not available. Conversation status: {conv['status']}"
        )
    
    analysis = data_store.analyses.get(conversation_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisResult(**analysis)


@router.post("/{conversation_id}/reprocess", response_model=ConversationResponse)
async def reprocess_conversation(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    customer_id: str = Depends(require_customer_key)
):
    """
    Queue a conversation for reprocessing.
    
    Useful when analysis models have been updated or if the initial
    processing encountered issues.
    """
    conv = data_store.conversations.get(conversation_id)
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv["customer_id"] != customer_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Update status
    conv["status"] = "pending"
    conv["processed_at"] = None
    
    # Clear existing analysis
    if conversation_id in data_store.analyses:
        del data_store.analyses[conversation_id]
    
    # Queue reprocessing
    background_tasks.add_task(process_conversation, conversation_id)
    
    return ConversationResponse(
        id=conv["id"],
        external_id=conv["external_id"],
        customer_id=conv["customer_id"],
        conversation_type=ConversationType(conv["conversation_type"]),
        status=ConversationStatus.PENDING,
        speakers=[Speaker(**s) for s in conv["speakers"]],
        utterance_count=len(conv["utterances"]),
        duration_seconds=conv["duration_seconds"],
        recorded_at=conv["recorded_at"],
        created_at=conv["created_at"],
        processed_at=None
    )


@router.post("/bulk", response_model=BulkOperationResponse)
async def bulk_operation(
    request: BulkOperationRequest,
    background_tasks: BackgroundTasks,
    customer_id: str = Depends(require_customer_key)
):
    """
    Perform bulk operations on multiple conversations.
    
    Supported operations:
    - reprocess: Queue conversations for reanalysis
    - delete: Remove conversations and associated data
    - archive: Mark conversations as archived (not implemented)
    """
    operation_id = f"bulk_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"
    processed = 0
    failed = 0
    errors = []
    
    for conv_id in request.conversation_ids:
        conv = data_store.conversations.get(conv_id)
        
        if not conv:
            failed += 1
            errors.append({"conversation_id": conv_id, "error": "Not found"})
            continue
        
        if conv["customer_id"] != customer_id:
            failed += 1
            errors.append({"conversation_id": conv_id, "error": "Not found"})
            continue
        
        try:
            if request.operation == "delete":
                del data_store.conversations[conv_id]
                if conv_id in data_store.analyses:
                    del data_store.analyses[conv_id]
            elif request.operation == "reprocess":
                conv["status"] = "pending"
                background_tasks.add_task(process_conversation, conv_id)
            elif request.operation == "archive":
                # BUG: Archive not implemented but doesn't raise error
                pass
            
            processed += 1
        except Exception as e:
            failed += 1
            errors.append({"conversation_id": conv_id, "error": str(e)})
    
    return BulkOperationResponse(
        operation_id=operation_id,
        status="completed" if failed == 0 else "partial",
        processed=processed,
        failed=failed,
        errors=errors
    )


# ============================================================================
# Admin Endpoints
# ============================================================================

@router.get("/admin/stats", tags=["admin"])
async def get_system_stats(auth: Dict = Depends(require_admin_key)):
    """
    Get system-wide statistics. Admin only.
    """
    total_conversations = len(data_store.conversations)
    by_status = {}
    by_type = {}
    
    for conv in data_store.conversations.values():
        status = conv["status"]
        by_status[status] = by_status.get(status, 0) + 1
        
        conv_type = conv["conversation_type"]
        by_type[conv_type] = by_type.get(conv_type, 0) + 1
    
    return {
        "total_conversations": total_conversations,
        "by_status": by_status,
        "by_type": by_type,
        "total_customers": len(set(c["customer_id"] for c in data_store.conversations.values()))
    }


@router.get("/admin/customer/{customer_id}/conversations", tags=["admin"])
async def admin_get_customer_conversations(
    customer_id: str,
    auth: Dict = Depends(require_admin_key),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Admin endpoint to view any customer's conversations.
    """
    results = []
    for conv in data_store.conversations.values():
        if conv["customer_id"] == customer_id:
            results.append(conv)
    
    return results[:limit]


# ============================================================================
# Background Processing (Simulated)
# ============================================================================

async def process_conversation(conversation_id: str):
    """
    Simulates the background processing pipeline.
    
    In production, this would:
    1. Update status to 'processing'
    2. Generate embeddings via Azure OpenAI
    3. Run sentiment analysis
    4. Extract topics and key phrases
    5. Calculate QA scores
    6. Check compliance rules
    7. Persist to SQL and Elasticsearch
    8. Update status to 'completed'
    """
    # Simulate processing delay
    await asyncio.sleep(0.5)
    
    conv = data_store.conversations.get(conversation_id)
    if not conv:
        return
    
    conv["status"] = "processing"
    await asyncio.sleep(1.0)  # Simulate AI processing
    
    # Generate mock analysis results
    analysis = {
        "conversation_id": conversation_id,
        "overall_sentiment": "neutral",
        "sentiment_confidence": 0.82,
        "topics": ["support", "inquiry"],
        "key_phrases": ["help needed", "account"],
        "qa_score": 75.0 + (hash(conversation_id) % 25),  # Random-ish score
        "compliance_flags": [],
        "processing_time_ms": 1500
    }
    
    data_store.analyses[conversation_id] = analysis
    conv["status"] = "completed"
    conv["processed_at"] = datetime.utcnow()
