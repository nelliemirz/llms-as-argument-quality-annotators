from enum import Enum

from core.argument import Argument
from core.quality_dimension import QualityDimension


class PromptTemplate(Enum):
    EXPERT_TEMPLATE = ("{knowledge_types} Choose one of the options below:\n"
                       "3 - High\n"
                       "2 - Medium\n"
                       "1 - Low\n"
                       "? - Cannot judge\n\n"
                       "### Your answer:")

    EXPERT_REASONING_TEMPLATE = ("{knowledge_types} Choose one of the options below and explain your reasoning:\n"
                                 "3 - High\n"
                                 "2 - Medium\n"
                                 "1 - Low\n"
                                 "? - Cannot judge\n\n"
                                 "### Your answer:")

    NOVICE_TEMPLATE = ("{knowledge_types}\n\n"
                       "### Task:\n"
                       "Choose one of the options below.\n\n"
                       "{question_novice}\n"
                       "3 - High\n"
                       "2 - Medium\n"
                       "1 - Low\n"
                       "? - Cannot judge\n\n"
                       "### Your answer:")

    NOVICE_REASONING_TEMPLATE = ("{knowledge_types}\n\n"
                                 "### Task:\n"
                                 "Choose one of the options below and explain your reasoning.\n\n"
                                 "{question_novice}\n"
                                 "3 - High\n"
                                 "2 - Medium\n"
                                 "1 - Low\n"
                                 "? - Cannot judge\n\n"
                                 "### Your answer:")


class PromptBuilder:
    def __init__(self, prompt_template: PromptTemplate):
        self.prompt_template = prompt_template
        self.default_types = []
        self.custom_types = None

    def with_knowledge(self, knowledge: str):
        if self.custom_types is None:
            self.custom_types = []

        self.custom_types.append(knowledge)

    def build(self, argument: Argument, dimension: QualityDimension):
        considered_types = self.default_types
        if self.custom_types is not None:
            considered_types = self.custom_types

        knowledge_types = ""

        for knowledge_type in considered_types:
            if knowledge_types != "":
                knowledge_types += "\n\n"

            knowledge_types += knowledge_type

        variables = {**argument.__dict__, **dimension.__dict__}
        prompt = self.prompt_template.value.format(knowledge_types=knowledge_types, **variables)
        return prompt.format(**variables)


class ExpertPromptBuilder(PromptBuilder):
    INSTRUCTION = ("### Instruction:\nPlease answer the following questions for the given comment from an online "
                   "debate forum on a given issue.")

    ISSUE = "### Issue:\n{issue}"

    STANCE = "### Stance:\n{stance}"

    ARGUMENT = "### Argument:\n{argument}"

    DEFINITION = "### Quality dimension definition:\n{dimension}: {definition}"

    QUESTION = "### Question:\n{question}"

    def __init__(self, prompt_template: PromptTemplate):
        super().__init__(prompt_template)

        self.default_types = [
            ExpertPromptBuilder.INSTRUCTION,
            ExpertPromptBuilder.ISSUE,
            ExpertPromptBuilder.STANCE,
            ExpertPromptBuilder.ARGUMENT,
            ExpertPromptBuilder.DEFINITION,
            ExpertPromptBuilder.QUESTION
        ]


class NovicePromptBuilder(PromptBuilder):
    INSTRUCTION = ("### Instruction:\nPlease rate the quality dimension of the given argument from an online debate "
                   "forum.")

    CONCLUSION = "### Conclusion:\n{conclusion}"

    ARGUMENT = "### Reason(s):\n{argument}"

    DEFINITION = "### Quality dimension definition:\n{dimension}: {definition_novice}"

    def __init__(self, prompt_template: PromptTemplate):
        super().__init__(prompt_template)

        self.default_types = [
            NovicePromptBuilder.INSTRUCTION,
            NovicePromptBuilder.CONCLUSION,
            NovicePromptBuilder.ARGUMENT,
            NovicePromptBuilder.DEFINITION
        ]
