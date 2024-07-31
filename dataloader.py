import h5py
import numpy as np

def load_hdf5_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        ids = np.array(f['ids'], dtype=str)
        features = np.array(f['features'])
        labels = np.array(f['labels'])
    return ids, features, labels
