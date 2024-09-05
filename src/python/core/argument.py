from dataclasses import dataclass


@dataclass
class Argument:
    id: str
    issue: str
    stance: str
    conclusion: str
    argument: str

