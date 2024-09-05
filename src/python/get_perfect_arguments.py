import pandas as pd
import glob
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


def select_perfect_agreement(file_path, save=False, output_dir=None):
    """
    Selects and saves arguments with perfect agreement on quality dimensions.

    Parameters:
    file_path (str): Path to the input CSV file.
    save (bool): Flag to save the output to a file.
    output_dir (str): Path to the output directory. 

    Returns:
    None
    """
    filename = os.path.basename(file_path)
    data = pd.read_csv(file_path, sep='\t', encoding='latin-1')
    quality_dimensions = data.columns[3:]

    if save:
        output_dir = f"{output_dir}-{filename.split('.')[0]}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for dimension in quality_dimensions:
        unique_ids = data['id'].unique()
        perfect_subset = pd.DataFrame()
        for unique_id in unique_ids:
            subset_df = data.loc[data['id'] == unique_id]
            if subset_df[dimension].nunique() == 1 and subset_df['annotator'].nunique() > 1:
                subset_df['annotator'] = subset_df['annotator'].apply(lambda x: f'human-{os.path.basename(file_path).split(".")[0]}-{x}')
                # f'human-{os.path.basename(file_path).split(".")[0]}-{subset_df["annotator"]}'
                perfect_subset = perfect_subset.append(subset_df[['id', 'annotator', dimension]])
        if save:
            perfect_subset.to_csv(f"{output_dir}/{dimension}-{filename}", sep='\t', index=False)
        print(f"{dimension}: {len(perfect_subset.id.unique())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select arguments with perfect agreement on quality dimensions.')
    parser.add_argument('-i', '--input_dir', 
                        type=str, 
                        help='Path to the directory containing human annotations', 
                        default='../../data/human_annotations/')
    parser.add_argument('-o', '--output_dir', 
                        type=str, 
                        default='../../data/human_annotations/perfect_agreement',
                        help='Path to the output directory')
    parser.add_argument('--save', 
                        action='store_true', 
                        help='Flag to save the output to a file')
    args = parser.parse_args()

    for file_path in glob.glob(f"{args.input_dir}/*.tsv"):
        print('Getting perfect agreement for:', file_path)
        select_perfect_agreement(file_path, args.save, args.output_dir)
        print()
