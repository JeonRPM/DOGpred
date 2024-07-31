import os
import pickle
import numpy as np
import pandas as pd
import torch
import time
import random
import re, json
import h5py

from bio_embeddings.embed import ESM1bEmbedder, ESM1vEmbedder, ESMEmbedder, \
    ProtTransT5UniRef50Embedder, ProtTransT5XLU50Embedder, ProtTransXLNetUniRef100Embedder, \
    ProtTransAlbertBFDEmbedder, ProtTransBertBFDEmbedder, ProtTransT5BFDEmbedder

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
################################

SEED = 0
print("Seed was: ", SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# define variables
LABELS = []
SEQUENCES = []
CNT = 0
MAX_LEN = 0

# pre-process data
with open('data/Independent.txt', 'r') as r1:
    lines = r1.readlines()
    for line in lines:
        ln = line.strip().split()
        if ">" in ln[0]:
            label = ln[0]
        else:
            seq = ln[0]
        if CNT % 2 == 0:
            LABELS.append(label)
        else:
            SEQUENCES.append(seq)
        CNT = CNT + 1
    r1.close()

print("Number of sequences: ", len(SEQUENCES))
print("Number of labels: ", len(LABELS))

print(LABELS[0])
print(LABELS[-1])

for seq in SEQUENCES:
    if len(seq) > MAX_LEN:
        MAX_LEN = len(seq)

print("Length of the longest sequence: ", MAX_LEN)

# Snippet for extracting PTB
PTB = ProtTransT5BFDEmbedder()

ids = []
features = []
labels = []
for i in range(len(SEQUENCES)):
    idn = LABELS[i]
    if "Pos_" in idn:
        label = 1
    else:
        label = 0
    seq = SEQUENCES[i]
    feature = PTB.embed(seq)
    ids.append(idn)
    features.append(feature)
    labels.append(label)

ids_array = np.array(ids, dtype='S')
features_array = np.array(features)
labels_array = np.array(labels)
print(features_array.shape)

# Save the feature
with h5py.File("features/PLMs/PTB_Ind.h5", 'w') as f:
    f.create_dataset('ids', data=ids_array)
    f.create_dataset('features', data=features_array)
    f.create_dataset('labels', data=labels_array)


# Load and check the feature
def load_hdf5_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        ids = np.array(f['ids'], dtype=str)
        features = np.array(f['features'])
        labels = np.array(f['labels'])
    return ids, features, labels

ids, features, labels = load_hdf5_data("features/PLMs/PTB_Ind.h5")
print(ids[0])
print(features[0].shape)
print(labels[0])

print(ids[-1])
print(features[-1].shape)
print(labels[-1])