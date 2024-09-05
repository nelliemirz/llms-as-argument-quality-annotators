import copy
import datetime
import json
import re
import sys
import time
from typing import Optional

from core.argument import Argument
from core.llm import *
from core.prompts import PromptTemplate, ExpertPromptBuilder, NovicePromptBuilder
from core.quality_dimension import QualityDimension

FIRST_OCCURRENCE_PATTERN = re.compile(r"^\s*[1-3?](?: ?- ?(?:High|Medium|Low|Cannot judge))?", re.DOTALL)

ANSWER_PATTERN = re.compile(r"(?<=### Your answer:\n)^[1-3?](?: ?- ?(?:High|Medium|Low|Cannot judge))?", re.MULTILINE)

RATING_PATTERN = re.compile(r"[1-3?](?: ?- ?(?:High|Medium|Low|Cannot judge))?")


def load_arguments(path: str) -> List[Argument]:
    arguments = []
    with open(path, "r") as in_file:
        in_file.readline()

        for line in in_file:
            comp = line.strip().split("\t")
            arguments.append(Argument(*comp))

    return arguments


def load_dimension_definitions(path: str) -> List[QualityDimension]:
    dimensions = []

    with open(path, "r") as in_file:
        for line in in_file:
            data = json.loads(line)
            dimensions.append(QualityDimension(**data))

    return dimensions


def parse_response(response: str) -> Optional[str]:
    pattern_sequence = [FIRST_OCCURRENCE_PATTERN, ANSWER_PATTERN, RATING_PATTERN]

    for pattern in pattern_sequence:
        matches = re.findall(pattern, response)
        match = None
        if len(matches) > 0:
            if len(matches) > 1:
                if matches.count(matches[0]) == len(matches):
                    match = matches[0]
            else:
                match = matches[0]

        if match is not None:
            return match.split("-")[0].strip()

    return None


def main():
    arguments = load_arguments("data/arguments.tsv")
    dimensions = load_dimension_definitions("data/dimensions_definitions.jsonl")
    num_arguments = len(arguments)

    ratings = []

    llms = [LLama213b]
    begin_timestamp = datetime.datetime.now()
    os.makedirs("data/ratings/", exist_ok=True)
    os.makedirs("data/logs/", exist_ok=True)

    log_file = open(f"data/logs/log-{begin_timestamp.isoformat()}.jsonl", "w+")

    try:
        for llm in llms:
            print(
                f"[{datetime.datetime.now().isoformat()}] Initialize {llm.__name__}...",
                end="")
            llm = llm()
            print("Done.", flush=True)
            for prompt_template in PromptTemplate:
                if (prompt_template.name == PromptTemplate.NOVICE_TEMPLATE.name
                        or prompt_template.name == PromptTemplate.NOVICE_REASONING_TEMPLATE.name):
                    prompt_builder = NovicePromptBuilder(prompt_template)

                    prompt_condition = "novice"

                    if prompt_template.name == PromptTemplate.NOVICE_REASONING_TEMPLATE.name:
                        prompt_condition += "-reasoning"
                else:
                    prompt_builder = ExpertPromptBuilder(prompt_template)

                    prompt_condition = "expert"

                    if prompt_template.name == PromptTemplate.EXPERT_REASONING_TEMPLATE.name:
                        prompt_condition += "-reasoning"

                num_done = 0
                for argument in arguments:
                    prompts = []

                    dimensions_copy = copy.deepcopy(dimensions)
                    print(
                        f"[{datetime.datetime.now().isoformat()}] ({num_done + 1}/{num_arguments}) "
                        f"{llm.__class__.__name__} annotate \"{argument.id}\" with template \"{prompt_template.name}\"...",
                        end="", flush=True)

                    for dimension in dimensions_copy:
                        prompt = prompt_builder.build(argument, dimension)
                        prompts.append(prompt)

                    retries = 0
                    while len(dimensions_copy) > 0 and retries < 5:
                        timestamp = datetime.datetime.now()
                        start_time = time.time()
                        responses = llm.generate_all(prompts)
                        run_time = time.time() - start_time

                        zipped = [t for t in zip(prompts, responses, dimensions_copy)]
                        for prompt, response, dimension in zipped:
                            rating = parse_response(response)

                            log_file.write(json.dumps({
                                "timestamp": timestamp.isoformat(),
                                "model": llm.__class__.__name__,
                                "run_time": run_time,
                                "try": retries + 1,
                                "dimension": dimension.dimension,
                                "prompt": prompt,
                                "response": response,
                                "parsed_response": rating,
                            }))
                            log_file.write("\n")
                            log_file.flush()

                            if rating is not None or retries == 4:
                                ratings.append({"id": argument.id, "dimension": dimension.dimension, "rating": rating})
                                prompts.remove(prompt)
                                dimensions_copy.remove(dimension)

                        retries += 1

                    num_done += 1
                    print("Done.", flush=True)

                existing_files = [f for f in os.listdir("data/ratings")
                                  if re.match(rf"{llm.__class__.__name__}-{prompt_condition}-[0-9]+\.jsonl", f)]
                out_file_name = f"{llm.__class__.__name__}-{prompt_condition}-{len(existing_files) + 1}.jsonl"

                with open(os.path.join("data/ratings", out_file_name), "w+") as out_file:
                    for rating in ratings:
                        out_file.write(json.dumps(rating))
                        out_file.write("\n")

                    ratings.clear()

            del llm
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        log_file.close()


if __name__ == '__main__':
    main()
