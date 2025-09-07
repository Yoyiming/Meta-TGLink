import pandas as pd
import random

random.seed(2024)

cell_lines = ['A549', 'A375', 'HEK293T', 'PC3']
samples = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
for sample in samples:
    for cell_line in cell_lines:
        train_data_path = f'cell_line_dataset/oe/{cell_line}/{sample}_train.csv'
        val_data_path = f'cell_line_dataset/oe/{cell_line}/{sample}_val.csv'
        test_data_path = f'cell_line_dataset/oe/{cell_line}/{sample}_test.csv'

        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)

        dataset = pd.concat([train_data, val_data, test_data])
        tf_num = dataset['TF'].unique().shape[0]
        tf_list = dataset['TF'].unique().tolist()
        # 打乱顺序
        random.shuffle(tf_list)

        train_num = int(tf_num * 0.6)
        val_num = int(tf_num * 0.2)
        test_num = tf_num - train_num - val_num

        train_sample_tfs = tf_list[:train_num]
        val_sample_tfs = tf_list[train_num:train_num+val_num]
        test_sample_tfs = tf_list[train_num+val_num:]

        train_data_new = dataset[dataset['TF'].isin(train_sample_tfs)]
        val_data_new = dataset[dataset['TF'].isin(val_sample_tfs)]
        test_data_new = dataset[dataset['TF'].isin(test_sample_tfs)]

        train_data_new.to_csv(f'cell_line_dataset/oe/{cell_line}/{sample}_tf_train.csv', index=False)
        val_data_new.to_csv(f'cell_line_dataset/oe/{cell_line}/{sample}_tf_val.csv', index=False)
        test_data_new.to_csv(f'cell_line_dataset/oe/{cell_line}/{sample}_tf_test.csv', index=False)

print('Done!')
