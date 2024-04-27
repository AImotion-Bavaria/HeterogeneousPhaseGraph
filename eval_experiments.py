# %%
'''
TODO Experiment description
'''
import yaml
import numpy as np
import pandas as pd

from pathlib import Path

dir = Path('results/2024-04-24_10:30_experiments')
dir_res = dir / 'results'
files = list(dir_res.glob('*.yaml'))

res = {}
train_settings = []
for file_name in files:
    with open(file_name, 'r') as f:
        res_tmp = yaml.safe_load(f)
    res[file_name.name] = res_tmp
    try:
        train_settings.append(
            tuple(res_tmp['config']['data']['train_settings'])
        )
    except KeyError:
        train_settings.append(tuple(res_tmp['config']['data']['train_s']))
unique_train_settings = [list(i) for i in set(train_settings)]
pd.set_option('display.max_columns', None)

# %%
# Get test accuracies in df for each model type and parameters for each
# train setting
train_settings = []
test_settings = []
model_params = []
model_type = []
train_acc = []
test_acc = []
test_acc_s = {
    f"test_acc_{k}": [] for k in range(1, 9)
}
learning_rates = []

layer1 = []
layer2 = []
conv_pre = []
graph_type = []
phases = []
domain_adap = []

for k, v in res.items():
    train_settings.append(str(v['config']['data']['train_s']))
    test_settings.append(str(v['config']['data']['test_s']))
    model_type.append(v['config']['model_type'])
    if model_type[-1] == 'GNN':
        layer1.append(v['config']['model_params']['gnn_layer_1'])
        layer2.append(v['config']['model_params']['gnn_layer_2'])
        conv_pre.append(v['config']['model_params']['conv_pre'])
        graph_type.append(v['config']['model_params']['graph_type'])
        phases.append(v['config']['model_params']['phases'])
        domain_adap.append(np.nan)
    elif model_type[-1] == 'SWING_CNN':
        layer1.append(np.nan)
        layer2.append(np.nan)
        conv_pre.append(np.nan)
        graph_type.append(np.nan)
        phases.append(np.nan)
        domain_adap.append(v['config']['model_params']['domain_adap'])
    train_acc.append(v['metrics']['train_acc'][-1])
    test_acc.append(v['metrics']['test_acc_all'][-1])
    for k_s in test_acc_s.keys():
        try:
            test_acc_s[k_s].append(v['metrics'][k_s][-1])
        except KeyError:
            test_acc_s[k_s].append(np.nan)

df_res = pd.DataFrame({
    'model_type': model_type,
    'conv_pre': conv_pre,
    'gnn_layer_1': layer1,
    'gnn_layer_2': layer2,
    'graph_type': graph_type,
    'phases': phases,
    'domain_adap': domain_adap,
    'train_settings': train_settings,
    'test_settings': test_settings,
    'train_acc': train_acc,
    'test_acc': test_acc,
    **test_acc_s
})

agg_acc_s = {
    f'test_acc_{k}': ['mean'] for k in range(1, 9)
}

group_cols = [
    'model_type', 'conv_pre', 'graph_type',
    'phases', 'domain_adap', 'train_settings', 'test_settings'
]
df_res = df_res.groupby(
    group_cols, dropna=False
).agg({
    # 'train_acc': ['mean', 'std'],
    'test_acc': ['mean', 'std', 'count'],
    **agg_acc_s
}).reset_index().round(2)

pivot_cols = [
    'model_type', 'conv_pre', 'graph_type',
    'phases', 'domain_adap'
]
df_res[pivot_cols] = df_res[pivot_cols].astype(str)
df_res_all = pd.pivot_table(
    df_res,
    index=pivot_cols,
    values=['test_acc'],  # 'train_acc'],
    columns='train_settings'
).reset_index()
pd.set_option('display.max_colwidth', 100)
df_res_all.loc[df_res_all.model_type != 'MaxCNN', :]
df_res_all.to_excel(dir / 'results.xlsx')

# plot_cols = [col for col in df_res_all.columns if len(col[-1]) == 0]
# cols_1 = [col for col in df_res_all.columns if len(col[-1]) == 3]
# cols_1 = plot_cols + cols_1
# cols_2 = [col for col in df_res_all.columns if len(col[-1]) == 6]
# cols_2 = plot_cols + cols_2
# cols_3 = [col for col in df_res_all.columns if len(col[-1]) == 9]
# cols_3 = plot_cols + cols_3


# for i, cols in enumerate([cols_1, cols_2, cols_3]):
#     df_res_all[cols].to_excel(dir / f'results_{i+1}.xlsx')

# %%
