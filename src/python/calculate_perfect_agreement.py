import json
import os
import statistics
import pandas as pd
import glob
from calculate_alpha import get_alphas, get_majority,  get_annotations
import krippendorff

QUALITY_DIMENSIONS = ['Cogency', 'Local Acceptability', 'Local Relevance',
                    'Local Sufficiency', 'Effectiveness', 'Credibility',
                    'Emotional Appeal', 'Clarity', 'Appropriateness', 'Arrangement',
                    'Reasonableness', 'Global Acceptability', 'Global Relevance',
                    'Global Sufficiency', 'Overall Quality']

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

def get_llm_majority(path, prompt_type, model, reasoning, num_annotators, dimension):
    '''
    Get majority annotations for each quality dimension.
    '''
    dfs = []
    df = get_annotations(model, prompt_type, reasoning, num_annotators, path)

    # for dimension in QUALITY_DIMENSIONS:
    #     majority = pd.DataFrame()
    dim = df.groupby('id')[dimension].apply(statistics.mode)
    dim = dim.to_frame()
    # majority[dimension] = dim #.to_list()
    dim['annotator'] = model + '-majority'
    # dfs.append(majority)
    # majority_df = pd.concat(dfs)
    # return majority_df.reset_index()
    return dim.reset_index()


def perfect_human_vs_llm(human_config, model_config): #path='../data/', model=None, human_type=None, type=None, reasoning=False):
    '''
    Compare perfect human annotations with LLM annotations.
    
    Parameters:
    path (str): Path to the data directory.
    model (str): LLM model name: llama213b, palm2, GPT3.
    type (str): Type of annotator: expert, novice.
    reasoning (bool): Flag to process LLM annotations with reasoning prompts.
    '''
    perfect_human = glob.glob(f"../../data/human_annotations/perfect_agreement-{human_config['prompt_type']}/*.tsv")

    
    alphas = {}
    for human in perfect_human:
        human_df=pd.read_csv(human, sep='\t')
        human_df = human_df.drop_duplicates(subset='id') # all rows with the same id have the same annotations for a given dimension
        human_df['annotator'] = 'human'
        dimension = human_df.columns[-1]
        llm = get_llm_majority(path='../../data/predictions', prompt_type=model_config['prompt_type'], 
                           model=model_config['annotator'], reasoning=model_config['reasoning'],
                           num_annotators=10, dimension=dimension)
        model_df = llm[['id', dimension, 'annotator']]
        model_df = model_df[model_df['id'].isin(human_df['id'])]
        df = pd.concat([human_df, model_df])
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by='id', inplace=True)
        # print(dimension, krippendorff.alpha(df.pivot(index='annotator', columns='id', values=dimension), level_of_measurement='ordinal').round(2))
        alphas[dimension] = krippendorff.alpha(df.pivot(index='annotator', columns='id', values=dimension), level_of_measurement='ordinal').round(2)
    return alphas

def calculate_agreement_perfect_human_llm():
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
    
    annotation_sets = []
    human_configs = [config for config in configs if config['annotator']=='human']
    llm_configs = [config for config in configs if config['annotator']!='human']
    for human_config in human_configs:
        for llm_config in llm_configs:
            print(human_config, llm_config)
            annotations = {}
            alphas = perfect_human_vs_llm(human_config, llm_config)
            annotations['annotators'] = [human_config, llm_config]
            annotations['alphas'] = alphas
            annotation_sets.append(annotations)
    with open('agreements-perfect-human.jsonl', 'w') as f:
        for item in annotation_sets:
            f.write("%s\n" % json.dumps(item))


if __name__ == '__main__':
    calculate_agreement_perfect_human_llm()