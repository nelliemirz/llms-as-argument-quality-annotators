import numpy as np
import pandas as pd
import re
import krippendorff
import glob
import os
import argparse
import statistics

QUALITY_DIMENSIONS = ['Cogency', 'Local Acceptability', 'Local Relevance',
                      'Local Sufficiency', 'Effectiveness', 'Credibility', 'Emotional Appeal',
                      'Clarity', 'Appropriateness', 'Arrangement', 'Reasonableness',
                      'Global Acceptability', 'Global Relevance', 'Global Sufficiency',
                      'Overall Quality', ]

EXCLUDED_IDS = ['44125',
                '541',
                '68938',
                '70818',
                '71467',
                '804',
                'arg155750',
                'arg219210',
                'arg219232',
                'arg219242',
                'arg219292',
                'arg230311',
                'arg234318',
                'arg236317',
                'arg236641',
                'arg317490']


def find_integers(x):
    try:
        i = re.findall(r'[1-3]', x)[0]
        return int(i)
    except:
        return None


def compute_dimension_mean(df):
    df['Cogency'] = (df[['Local Acceptability', 'Local Relevance', 'Local Sufficiency']]
                     .mean(numeric_only=True, axis=1).round()).astype("int", errors="ignore")
    df['Reasonableness'] = (df[['Global Acceptability', 'Global Relevance', 'Global Sufficiency']]
                            .mean(numeric_only=True, axis=1).round())
    df['Effectiveness'] = (df[['Credibility', 'Emotional Appeal', 'Clarity', 'Appropriateness', 'Arrangement']]
                           .mean(numeric_only=True, axis=1).round())
    df['Overall Quality'] = (df[['Cogency', 'Reasonableness', 'Effectiveness']]
                             .mean(numeric_only=True, axis=1).round())
    return df


def process_files(paths, model, prompt_type, reasoning):
    dfs = []
    response_column = 'rating'
    if reasoning:
        reasoning = "_reasoning"
    else:
        reasoning = ""

    for i, path in enumerate(paths):
        df = pd.read_json(path, lines=True)
        df = df.drop(df[df["id"].isin(EXCLUDED_IDS)].index)
        df[response_column] = df[response_column].replace("?", 0).astype("float")
        df = df.pivot(index='id', columns='dimension', values=response_column)
        if "Cogency" not in df.columns:
            df = compute_dimension_mean(df)
        df.reset_index(inplace=True)
        df['annotator'] = f'{model}_{i + 1}_{prompt_type}{reasoning}'

        dfs.append(df)
    return pd.concat(dfs)


def get_annotations(model, prompt_type, reasoning, num_annotators, path):
    if reasoning:
        path_pattern = os.path.join(path, f'{model}[-_]{prompt_type}[-_]reasoning[-_][0-9].jsonl')
    else:
        path_pattern = os.path.join(path, f'{model}[-_]{prompt_type}[-_][0-9].jsonl')

    prediction_files = sorted(glob.glob(path_pattern))[:num_annotators]
    df = process_files(prediction_files, model, prompt_type, reasoning)
    return df


def get_majority(annotations, annotator, prompt_type, reasoning):
    df = annotations
    majority = pd.DataFrame()
    for dimension in QUALITY_DIMENSIONS:
        dim = df.groupby('id')[dimension].apply(statistics.mode)
        majority[dimension] = dim

    if reasoning:
        reasoning = "_reasoning"
    else:
        reasoning = ""

    majority['annotator'] = f'{annotator}_majority_{prompt_type}{reasoning}'
    return majority.reset_index()


def get_alphas(annotator_configs, num_annotators, path):
    annotation_set = None
    for annotator_config in annotator_configs:
        annotations = get_annotations(
            annotator_config["annotator"],
            annotator_config["prompt_type"],
            annotator_config["reasoning"],
            num_annotators,
            path)
        if annotator_config["aggregation"] == "majority":
            annotations = get_majority(annotations, annotator_config["annotator"], annotator_config["prompt_type"],
                                       annotator_config["reasoning"])

        if annotation_set is None:
            annotation_set = annotations
        else:
            annotation_set = pd.concat([annotation_set, annotations], ignore_index=True)

    alpha_dict = {}
    if annotation_set is None:
        return alpha_dict

    for dimension in QUALITY_DIMENSIONS:
        annotation_per_annotator = (annotation_set.pivot(index='annotator', columns='id', values=dimension))
        alpha = krippendorff.alpha(annotation_per_annotator,
                                   level_of_measurement='ordinal')
        alpha_dict[dimension] = alpha

    overall_alpha = krippendorff.alpha(annotation_set.pivot(index='annotator', columns='id'),
                                       level_of_measurement='ordinal')
    alpha_dict["Across Dimensions"] = overall_alpha
    return alpha_dict


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotator', type=str, required=True, help='model name',
                        choices=['LLama213b', 'palm2', 'GPT3', 'human'], nargs="+", action="append")
    parser.add_argument("-agg", "--aggregation", type=str, required=False,
                        choices=["majority"], default=None, nargs="*", action="append")
    parser.add_argument('-r', '--reasoning', type=bool, nargs="*", action="append")
    parser.add_argument('-t', '--type', type=str, required=True, help='type of annotator',
                        choices=['expert', 'novice'], nargs="+", action="append")
    parser.add_argument('-k', '--num_annotators', type=int, help='number of annotators', default=3)
    parser.add_argument('-p', '--path', type=str, help='directory containing the prediction files',
                        default=data_dir)
    args = parser.parse_args()


    if len(args.annotator) != len(args.type):
        parser.error("Mismatched number of given annotators and types")

    annotator_configs = []
    for annotator, prompt_type in zip(args.annotator, args.type):
        annotator_configs.append({
            "annotator": annotator[0],
            "prompt_type": prompt_type[0],
            "reasoning": False,
            "aggregation": None})

    if args.reasoning is not None:
        for i in range(len(args.reasoning)):
            annotator_configs[i]["reasoning"] = True

    if args.aggregation is not None:
        for i, agg in enumerate(args.aggregation):
            annotator_configs[i]["aggregation"] = agg[0]

    alphas = get_alphas(annotator_configs, args.num_annotators, args.path)

    print('Annotator:', args.annotator, '\nType:', args.type, '\nReasoning:', args.reasoning,
          '\nTotal annotators:', args.num_annotators)
    print('\nKrippendorff\'s alpha')
    for k, v in alphas.items():
        print(k, round(v, 2))

    print('\nOverall Krippendorff\'s alpha:', round(alphas["Across Dimensions"], 2))
