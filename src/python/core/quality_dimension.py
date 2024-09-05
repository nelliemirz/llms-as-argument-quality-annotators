from dataclasses import dataclass


@dataclass
class QualityDimension:
    dimension: str
    definition: str
    question: str
    definition_novice: str
    question_novice: str
