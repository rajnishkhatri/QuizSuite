"""
Configuration Models

Pydantic models for quiz and processing configuration.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ChromaDBSettings(BaseModel):
    """ChromaDB configuration settings."""
    pdf_persist_directory: str = Field(..., description="PDF database persistence directory")
    html_persist_directory: str = Field(..., description="HTML database persistence directory")
    pdf_collection_name: str = Field(..., description="PDF collection name")
    html_collection_name: str = Field(..., description="HTML collection name")
    use_existing: bool = Field(default=True, description="Use existing database")
    create_if_not_exists: bool = Field(default=True, description="Create database if not exists")
    load_existing: bool = Field(default=True, description="Load existing data")
    use_metadata_filtering: bool = Field(default=True, description="Use metadata filtering")
    max_retrieval_results: int = Field(default=20, description="Maximum retrieval results")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")


class CategoryConfig(BaseModel):
    """Configuration for a quiz category."""
    name: str = Field(..., description="Category name")
    num_questions: int = Field(..., description="Number of questions to generate")
    description: str = Field(..., description="Category description")
    doc_paths: List[str] = Field(..., description="Document paths for this category")


class AutoTopicDistributionSettings(BaseModel):
    """Settings for automatic topic distribution."""
    enabled: bool = Field(default=True, description="Enable auto topic distribution")
    vector_database_settings: Dict[str, Any] = Field(default_factory=dict)
    coverage_settings: Dict[str, Any] = Field(default_factory=dict)
    question_generation_settings: Dict[str, Any] = Field(default_factory=dict)
    performance_settings: Dict[str, Any] = Field(default_factory=dict)


class CacheSettings(BaseModel):
    """Cache configuration settings."""
    enabled: bool = Field(default=True, description="Enable caching")
    cache_dir: str = Field(..., description="Cache directory")
    compression_ratio: int = Field(default=50, description="Compression ratio")
    retention_target: int = Field(default=95, description="Retention target percentage")


class OutputSettings(BaseModel):
    """Output configuration settings."""
    output_dir: str = Field(..., description="Output directory")
    include_timestamp: bool = Field(default=True, description="Include timestamp in output")
    include_model_name: bool = Field(default=True, description="Include model name in output")
    include_database_type: bool = Field(default=True, description="Include database type in output")


class PromptSettings(BaseModel):
    """Prompt configuration settings."""
    use_advanced_prompt: bool = Field(default=True, description="Use advanced prompts")
    include_learning_objectives: bool = Field(default=True, description="Include learning objectives")
    include_target_audience: bool = Field(default=True, description="Include target audience")
    include_difficulty_level: bool = Field(default=True, description="Include difficulty level")


class QuizConfig(BaseModel):
    """Main quiz configuration model."""
    database_type: str = Field(..., description="Database type")
    model_name: str = Field(..., description="Model name")
    embedding_model: str = Field(..., description="Embedding model")
    temperature: float = Field(default=0.0, description="Model temperature")
    target_audience: str = Field(..., description="Target audience")
    difficulty_level: str = Field(..., description="Difficulty level")
    learning_objectives: str = Field(..., description="Learning objectives")
    max_context_tokens: int = Field(default=1000, description="Maximum context tokens")
    use_memory: bool = Field(default=False, description="Use memory")
    use_reflexion: bool = Field(default=False, description="Use reflexion")
    chroma_db_settings: ChromaDBSettings = Field(..., description="ChromaDB settings")
    categories: List[CategoryConfig] = Field(..., description="Quiz categories")
    auto_topic_distribution_settings: AutoTopicDistributionSettings = Field(..., description="Auto topic distribution settings")
    cache_settings: CacheSettings = Field(..., description="Cache settings")
    output_settings: OutputSettings = Field(..., description="Output settings")
    prompt_settings: PromptSettings = Field(..., description="Prompt settings") 