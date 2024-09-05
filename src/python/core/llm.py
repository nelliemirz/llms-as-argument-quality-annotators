import abc
import os
from enum import Enum
from typing import List

import openai
from retry import retry

import torch
import transformers
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def generate_all(self, prompts: List[str]) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt))

        return results


class HFModel(LLM):

    def __init__(self, model_name, device_map=None):
        super().__init__()
        if device_map is None:
            device_map = {"": 0}
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            return_dict=True,
            low_cpu_mem_usage=True,
            # use_flash_attention_2=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )

        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.padding_side = "left"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

    def generate(self, prompt: str) -> str:
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=40,
            top_p=1.0,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=0.3
        )

        for seq in sequences:
            return seq["generated_text"]

    def generate_all(self, prompts: List[str]) -> List[str]:
        response = self.pipeline(
            prompts,
            do_sample=True,
            top_k=40,
            top_p=1.0,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=0.3
        )

        results = []
        for sequence in response:
            for seq in sequence:
                results.append(seq["generated_text"])

        return results


class Param(Enum):
    SEVEN_B = "7b"
    THIRTEEN_B = "13b"
    SEVENTY_B = "70b"


class LLama2(HFModel):
    def __init__(self, param: Param = Param.SEVEN_B, device_map=None):
        if device_map is None:
            device_map = {"": 0}
        name = f"meta-llama/Llama-2-{param.value}-hf"

        super().__init__(name, device_map)

    # def generate(self, prompt: str) -> str:
    #     return super().generate(f"<s>[INST]\n{prompt}\n[/INST] {{answer}}</s>")

    # def generate_all(self, prompts: List[str]) -> List[str]:
    #     return super().generate_all([f"<s>[INST]\n{prompt}\n[/INST] {{answer}}</s>" for prompt in prompts])


class LLama27b(LLama2):
    def __init__(self):
        super().__init__(param=Param.SEVEN_B)


class LLama213b(LLama2):
    def __init__(self):
        super().__init__(param=Param.THIRTEEN_B)


class LLama270b(LLama2):
    def __init__(self):
        super().__init__(param=Param.SEVENTY_B, device_map="auto")


class OpenAIModel(LLM):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20.0, max_retries=0)

    @retry((openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError), tries=10, delay=60)
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
            top_p=1.0)
        return response.choices[0].message.content


class GPT4(OpenAIModel):
    def __init__(self):
        super().__init__("gpt-4")


class GPT3(OpenAIModel):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")
