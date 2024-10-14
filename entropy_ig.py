import numpy as np
import pandas as pd

df = pd.read_csv('golf_data.csv')
print(df)

x = df.iloc[ :, :-1].values
y = df.iloc[ :, -1].values
print('X Part :\n',x)
print('Y Part :\n',y)

def entropy(y):
    elements, counts = np.unique(y, return_counts=True)
    entropy_val = -sum((counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements)))
    return entropy_val

print('Entropy of the Play_Golf :',entropy(y))

def info_gain(df, split_attribute_name, target_name="Play_Golf"):
    total_entropy = entropy(df[target_name])
    
    vals, counts = np.unique(df[split_attribute_name], return_counts=True)
    
    weighted_entropy = sum(
        (counts[i] / np.sum(counts)) * entropy(df[df[split_attribute_name] == vals[i]][target_name])
        for i in range(len(vals))
    )    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_node(df, target_name="Play_Golf"):
    attributes = df.columns[df.columns != target_name]
    info_gains = {attr: info_gain(df, attr, target_name) for attr in attributes}
    root_node = max(info_gains, key=info_gains.get)
    return root_node, info_gains

root_node, info_gains = find_root_node(df)
print('Information Gain for each attribute:')
for attr, gain in info_gains.items():
    print(f'{attr}: {gain}')
print(f'Root Node: {root_node}')