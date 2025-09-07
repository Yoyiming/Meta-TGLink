import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import sys

# random_seed = sys.argv[1]
random_seed = 2024
# sample = sys.argv[2]
sample = 'sample3'

def generate_gene_index_df(expression_matrix_path):
    expression_matrix = pd.read_csv(expression_matrix_path, index_col=0)
    gene_names = expression_matrix.index.tolist()
    gene_index_df = pd.DataFrame({'Gene': gene_names, 'Index': range(len(gene_names))})
    return gene_index_df


def create_network_with_labels(network_path, gene_index_df):
    network_df = pd.read_csv(network_path)
    network_df['Label'] = 1
    tf_names = network_df['TF'].unique()

    all_genes = set(gene_index_df['Gene'].tolist())
    tf_target_pairs = network_df.groupby('TF')['Target'].apply(set).to_dict()

    negative_samples = []
    for tf, targets in tf_target_pairs.items():
        non_targets = list(all_genes - targets - {tf})
        num_positive_samples = len(targets)
        if num_positive_samples > len(non_targets) / 2:
            sampled_non_targets = non_targets
        else:
            sampled_non_targets = np.random.choice(non_targets, num_positive_samples, replace=False)
        for nt in sampled_non_targets:
            negative_samples.append({'TF': tf, 'Target': nt, 'Label': 0})

    negative_samples_df = pd.DataFrame(negative_samples)
    complete_network_df = pd.concat([network_df, negative_samples_df])

    # 把TF和Target转成index
    gene_index_df.set_index('Gene', inplace=True)
    complete_network_df['TF'] = complete_network_df['TF'].map(gene_index_df['Index'])
    complete_network_df['Target'] = complete_network_df['Target'].map(gene_index_df['Index'])

    return complete_network_df


def split_and_save_datasets(complete_network_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    tfs = complete_network_df['TF'].unique()
    for tf in tfs:
        tf_df = complete_network_df[complete_network_df['TF'] == tf]
        positive_samples = tf_df[tf_df['Label'] == 1]
        negative_samples = tf_df[tf_df['Label'] == 0]

        if len(positive_samples) <= 10:
            train_df = pd.concat([train_df, tf_df])
            continue
        
        pos_train, pos_temp = train_test_split(positive_samples, test_size=0.4, random_state=random_seed)
        pos_val, pos_test = train_test_split(pos_temp, test_size=0.5, random_state=random_seed)

        neg_train, neg_temp = train_test_split(negative_samples, test_size=0.4, random_state=random_seed)
        neg_val, neg_test = train_test_split(neg_temp, test_size=0.5, random_state=random_seed)

        tf_train_df = pd.concat([pos_train, neg_train])
        tf_val_df = pd.concat([pos_val, neg_val])
        tf_test_df = pd.concat([pos_test, neg_test])

        train_df = pd.concat([train_df, tf_train_df])
        val_df = pd.concat([val_df, tf_val_df])
        test_df = pd.concat([test_df, tf_test_df])

    train_df.to_csv(os.path.join(output_dir, f'{sample}_train.csv'), index=False, columns=['TF', 'Target', 'Label'])
    val_df.to_csv(os.path.join(output_dir, f'{sample}_val.csv'), index=False, columns=['TF', 'Target', 'Label'])
    test_df.to_csv(os.path.join(output_dir, f'{sample}_test.csv'), index=False, columns=['TF', 'Target', 'Label'])


def main(cell_line_name):
    expression_matrix_path = f'expression matrix/Specific Dataset/{cell_line_name}/TFs+1000/BL--ExpressionData.csv'
    network_path = f'expression matrix/Specific Dataset/{cell_line_name}/TFs+1000/BL--network.csv'
    output_dir = f'cell_line_dataset/Specific/{cell_line_name} 1000'

    gene_index_df = generate_gene_index_df(expression_matrix_path)
    complete_network_df = create_network_with_labels(network_path, gene_index_df)
    split_and_save_datasets(complete_network_df, output_dir)

    # gene_index_df.to_csv(os.path.join(output_dir, 'Genes.csv'), index=False)


if __name__ == '__main__':
    cell_line_name = ['mESC', 'mHSC-E']
    for cell_line in cell_line_name:
        main(cell_line)
    print('Done')
