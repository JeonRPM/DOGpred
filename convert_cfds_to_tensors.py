#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import h5py

# Snippet for converting BLOSUM62

data = pd.read_csv('features/CFDs/BLOSUM62_Ind.csv')

ids = data['IDs']
labels = data['Label']
features = data.drop(columns=['IDs', 'Label'], axis=1)


ids_array = np.array(ids, dtype='S')
features_array = np.array(features)
labels_array = np.array(labels)

num_samples = features_array.shape[0]
new_shape = (41, int(features_array.shape[1]/41))
reshaped_features_array = features_array.reshape(features_array.shape[0], *new_shape)
print(reshaped_features_array.shape)

# Save the feature
with h5py.File("features/CFDs/BLOSUM62_Ind.h5", 'w') as f:
    f.create_dataset('ids', data=ids_array)
    f.create_dataset('features', data=reshaped_features_array)
    f.create_dataset('labels', data=labels_array)

# Load and check the feature
def load_hdf5_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        ids = np.array(f['ids'], dtype=str)
        features = np.array(f['features'])
        labels = np.array(f['labels'])
    return ids, features, labels

ids, features, labels = load_hdf5_data("features/CFDs/BLOSUM62_Ind.h5")
print(ids[0])
print(features[0].shape)
print(labels[0])

print(ids[-1])
print(features[-1].shape)
print(labels[-1])
