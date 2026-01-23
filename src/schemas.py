from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, TypedDict
from datetime import datetime

class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str= Field(description="Identificador del Documento")
    content:str= Field(description="El contenido del texto")
    metadata: Dict[str, Any]= Field(default_factory=dict, description="Metadata adicional")
    relevance_score: float= Field(default=0.0, description="valoración del retrieval")

# Respuesta estructurada de Q&A
class AnswerResponse(BaseModel):
    """Strcutured response for Q&A tasks"""

    question: str= Field(description= "La pregunta del usuario original")
    answer: str= Field(description="La respuesta generada")
    sources: List[str]= Field(default_factory=list, description="Lista de los IDs de los documentos usados")
    confidence: float= Field(ge=0.0, le=1.0, description="Validación entre 0 y 1")
    timestamp: datetime= Field(default_factory=datetime.utcnow, description= "Cuándo se generó la respuesta")

class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int= Field(description="Longitud del texto original")
    summary: str= Field(description="el resumen generado")
    key_points: List[str]= Field(description="lista de los puntos clave extraídos")
    document_ids: List[str]= Field(default_factory=list, description="documentos resumidos")
    timestamp: datetime= Field(default_factory=datetime.now)

class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="la expresión matemática")
    result: float = Field(description="el resultado calculado")
    explanation: str = Field(description="explicacion paso a paso")
    units: Optional[str] = Field(default=None, description="unidades (si aplica)")
    timestamp: datetime = Field(default_factory=datetime.now)

class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""
    summary:str= Field(description= "resumen de la conversación hasta este punto")
    document_ids: List[str]= Field(default_factory=list, description="lista de los documentos que son relevantes para el último mensaje del usuario")

# Clasificación de la intención---> routing del agente (QA, resumen, cálculo)

class UserIntent(BaseModel):
    """User intent classification result"""

    intent_type: Literal["qa", "summarization", "calculation", "unknown"]= Field(description="la intención del usuario clasificada") #el Literal evita valores inválidos
    confidence: float= Field(ge=0.0, le=1.0, description="Valoración de la clasiicación")
    reasoning:str= Field(description="Explicación de la clasificación")

class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: List[str]= Field(default_factory=list) ### TypedDict????
    document_context: List[str]= Field(default_factory=list, description= "IDs activos")
    created_at: datetime= Field(default_factory=datetime.now)
    last_updated: datetime= Field(default_factory=datetime.now)