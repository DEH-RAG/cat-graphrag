from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class EntityType(str, Enum):
    """Supported entity types for the knowledge graph"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    FINANCIAL = "FINANCIAL"
    UNKNOWN = "UNKNOWN"


class Entity(BaseModel):
    """Model for an entity in the knowledge graph"""
    id: str | None = None
    name: str
    type: EntityType
    embedding: List[float] | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: str
    created_at: datetime | None = None


class ExtractedEntity(BaseModel):
    """Extracted entity in a document (before being saved)"""
    name: str
    type: EntityType
    start_char: int
    end_char: int
    confidence: float = 1.0


class ExtractedRelation(BaseModel):
    """Extracted relation between two entities in a document (before being saved)"""
    source_entity: str
    target_entity: str
    relation_type: str = "RELATED_TO"
    weight: float = 1.0
    context: str | None = None


class DocumentWithEntities(BaseModel):
    """Document with extracted entities"""
    document_id: str
    content: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    metadata: Dict[str, Any]