from typing import Literal
from pydantic import BaseModel, Field

Category = Literal[
    "Клиенту интересен продукт",
    "Клиент не в курсе услуги",
    "Просто поинтересовался",
    "Перевел на менеджера",
    "Клиент ругался, недоволен звонком",
    "Клиент сразу положил трубку",
]

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

class Analysis(BaseModel):
    summary: str = Field(description="3–5 законченных предложений на русском языке")
    goal: str
    result: str
    category: Category
    evidence: list[str] = Field(description="1–3 дословных коротких фразы из транскрипта")
    next_action: str
    confidence: Literal["высокая", "средняя", "низкая"]
    insufficient_data: bool = False
    warnings: list[str] = Field(default_factory=list)
    speaker_roles: dict[str, str] = Field(description="Соответствие технических ID спикеров ролям: Менеджер, Клиент, Неизвестно")
