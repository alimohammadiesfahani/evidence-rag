
from pydantic import BaseModel, ConfigDict, Field


class Page(BaseModel):
    """Extracted text and source location for one PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str


class Document(BaseModel):
    """A PDF document with a stable identity and page metadata."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_name: str = Field(min_length=1)
    pages: list[Page] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)



class Chunk(BaseModel):
    """A retrieval passage with traceable source metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)

    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
