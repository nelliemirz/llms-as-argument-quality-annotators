import json
import os
from calculate_alpha import get_alphas
from calculate_perfect_agreement import calculate_agreement_perfect_human_llm


def main():
    num_annotators = 10

    annotators = ["human", "GPT3", "palm2"]
    prompt_types = ["novice", "expert"]
    aggregation = ["majority", None]

    configs = []

    for annotator in annotators:
        for prompt_type in prompt_types:
            for agg in aggregation:
                config = {"annotator": annotator, "prompt_type": prompt_type, "reasoning": False, "aggregation": agg}
                configs.append(config)

                if annotator != "human":
                    config = {"annotator": annotator, "prompt_type": prompt_type, "reasoning": True, "aggregation": agg}
                    configs.append(config)

    data = []
    predictions_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'predictions'))
    print(predictions_dir)
    for i, config_i in enumerate(configs):
        for j, config_j in enumerate(configs):
            if i < j:
                print(config_i, "-", config_j)
                alphas = get_alphas(
                    [config_i, config_j],
                    num_annotators,
                    predictions_dir)

                data.append({
                    "annotators": [config_i, config_j],
                    "alphas": alphas
                })

    with open("agreements.jsonl", "w+") as out_file:
        for entry in data:
            out_file.write(json.dumps(entry))
            out_file.write("\n")


if __name__ == '__main__':
    main()
    calculate_agreement_perfect_human_llm()
