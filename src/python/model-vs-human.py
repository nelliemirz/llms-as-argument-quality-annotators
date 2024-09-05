import calculate_alpha
import json
import pandas as pd
import re
import krippendorff
import glob
import os
import argparse
import statistics

QUALITY_DIMENSIONS = ['Appropriateness', 'Arrangement', 'Clarity', 'Credibility',
                      'Emotional Appeal', 'Global Acceptability', 'Global Relevance',
                      'Global Sufficiency', 'Local Acceptability', 'Local Relevance',
                      'Local Sufficiency', 'Cogency', 'Reasonableness', 'Effectiveness',
                      'Overall Quality']


def majority_for_dimension(df, type, dimension):
    majority = pd.DataFrame()
    dim = df.groupby('id')[dimension].apply(statistics.mode)
    majority[dimension] = dim.apply(lambda x: float(x))
    majority['annotator'] = type
    majority['id'] = dim.index
    majority.reset_index(inplace=True, drop=True)
    return majority



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--reasoning', action='store_true')
    parser.add_argument('--out', type=str, default='models-vs-humans.json')
    args = parser.parse_args()

    if args.reasoning:
        reasoning = '[-_]reasoning'
    else:
        reasoning = ''

    alphas = {}
    path = '/mnt/ceph/storage/data-in-progress/data-research/arguana/ratio24-argquality/predictions'

    for model in ['GPT3', 'palm2']:
        alphas[model] = {}
        for model_type in ['novice', 'expert']:
            alphas[model][f'{model_type}_{model}'] = {}
            for human_type in ['novice', 'expert']:
                alphas[model][f'{model_type}_{model}'][f'{human_type}_human'] = {}
                model_files = sorted(glob.glob(os.path.join(path, f'{model}-{model_type}{reasoning}-[0-9].jsonl')))[:args.k]
                model_df = calculate_alpha.process_files(model_files, model)

                human_files = sorted(glob.glob(f'/mnt/ceph/storage/data-in-progress/data-research/arguana/ratio24-argquality/perfect-agreement-{human_type}/*.tsv'))
                for f in human_files:
                    human_df = pd.read_csv(f, sep='\t')
                    dim = human_df.columns[-1]
                    human_majority = majority_for_dimension(human_df, human_type, dim)
                    model_majority = majority_for_dimension(model_df, model, dim)
                    model_majority = model_majority[model_majority['id'].isin(human_majority['id'])]

                    merged_majority = pd.concat([human_majority, model_majority])
                    overall_alpha=krippendorff.alpha(merged_majority.pivot(index='annotator', columns='id'),
                                                     level_of_measurement='ordinal')

                    alphas[model][f'{model_type}_{model}'][f'{human_type}_human'][dim]=round(overall_alpha, 2)


    with open(args.out, 'w') as f:
        json.dump(alphas, f, indent=4)