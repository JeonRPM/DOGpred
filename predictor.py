import h5py
import yaml
import math
import os, random
import argparse
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, auc, roc_curve
from dataloader import load_hdf5_data
from models import cnn1d_gru_model

seed = 42
print("Seed was:", seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

gpus = tf.config.experimental.list_physical_devices('GPU')
# Using GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*20)]) # Modify memory limit based on VRAM memory_limit = 1024 * total GB

# Using CPU
# tf.config.set_visible_devices([], 'GPU')

list_conv_data = ['AAIndex', 'binary3', 'binary5', 'binary6', 'binary', 'BLOSUM62', 'OPF7', 'OPF10', 'ZScale']
list_plm_data = ['ESB', 'ESM', 'ESV', 'PTAB', 'PTBB', 'PTB', 'PTL', 'PTN', 'PTU']

list_conv_shapes = []
list_conv_sets_ind = []
list_conv_sets_imb_ind = []

list_plm_shapes = []
list_plm_sets_ind = []
list_plm_sets_imb_ind = []

y_proba_ind_testing_ls = []
y_proba_imb_ind_testing_ls = []

for conv in list_conv_data:
    _, ind_features_conv, ind_labels_conv = load_hdf5_data("features/CFDs/" + conv + "_Ind.h5")
    list_conv_sets_ind.append(ind_features_conv)
    input_shape = ind_features_conv[0].shape
    list_conv_shapes.append(input_shape)
    _, imb_ind_features_conv, imb_ind_labels_conv = load_hdf5_data("features/CFDs/" + conv + "_Imb_Ind.h5")
    list_conv_sets_imb_ind.append(imb_ind_features_conv)

for plm in list_plm_data:
    _, ind_features_plm, _ = load_hdf5_data("features/PLMs/" + plm + "_Ind.h5")
    list_plm_sets_ind.append(ind_features_plm)
    list_plm_shapes.append(ind_features_plm[0].shape)
    _, imb_ind_features_plm, _ = load_hdf5_data("features/PLMs/" + plm + "_Imb_Ind.h5")
    list_plm_sets_imb_ind.append(imb_ind_features_plm)

folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

param_config = yaml.safe_load(open('configs/cnn1d_gru.yaml', 'r'))
param_dict = param_config.get('cnn1d_gru', {})
keys = list(param_dict.keys())
values = list(param_dict.values())
for combination in product(*values):
    params = dict(zip(keys, combination))

for fold in folds:
    model = cnn1d_gru_model(list_conv_shapes, list_plm_shapes, params, fuse='cross')
    model.load_weights("final_models/" + fold + ".h5")

    # Balanced independent testing
    y_pred_proba_ind = model.predict([list_conv_sets_ind, list_plm_sets_ind])
    y_proba_ind_testing_ls.append(y_pred_proba_ind)

    # Imbalanced independent testing
    y_pred_proba_imb_ind = model.predict([list_conv_sets_imb_ind, list_plm_sets_imb_ind])
    y_proba_imb_ind_testing_ls.append(y_pred_proba_imb_ind)

# Calculating performance on the balanced independent dataset
y_pred_proba_ind_test = np.mean(y_proba_ind_testing_ls, axis=0)
y_pred_cls_ind_test = []
for i in range(len(y_pred_proba_ind_test)):
    if y_pred_proba_ind_test[i] > 0.5:
        y_pred_cls_ind_test.append(1)
    else:
        y_pred_cls_ind_test.append(0)
tn1, fp1, fn1, tp1 = confusion_matrix(ind_labels_conv, y_pred_cls_ind_test).ravel()
mcc1 = float(tp1*tn1 - fp1*fn1) / (math.sqrt((tp1+fp1) * (tp1+fn1) * (tn1+fp1) * (tn1+fn1)) + K.epsilon())
sn1 = float(tp1) / (tp1+fn1 + K.epsilon())
sp1 = float(tn1) / (tn1+fp1 + K.epsilon())
acc1 = float(tp1+tn1) / (tn1+fp1+fn1+tp1 + K.epsilon())
fpr1, tpr1, _ = roc_curve(ind_labels_conv, y_pred_proba_ind_test, pos_label=1)
auc1 = auc(fpr1, tpr1)

print("Balanced Independent Testing Scores:\n"
    f"MCC: {np.round(mcc1, 4)}\n"
    f"ACC: {np.round(acc1, 4)}\n"
    f"Sn: {np.round(sn1, 4)}\n"
    f"Sp: {np.round(sp1, 4)}\n"
    f"AUC: {np.round(auc1, 4)}")


# Calculating performance on the imbalanced independent dataset
y_pred_proba_imb_ind_test = np.mean(y_proba_imb_ind_testing_ls, axis=0)
y_pred_cls_imb_ind_test = []
for i in range(len(y_pred_proba_imb_ind_test)):
    if y_pred_proba_imb_ind_test[i] > 0.5:
        y_pred_cls_imb_ind_test.append(1)
    else:
        y_pred_cls_imb_ind_test.append(0)
tn2, fp2, fn2, tp2 = confusion_matrix(imb_ind_labels_conv, y_pred_cls_imb_ind_test).ravel()
mcc2 = float(tp2*tn2 - fp2*fn2) / (math.sqrt((tp2+fp2) * (tp2+fn2) * (tn2+fp2) * (tn2+fn2)) + K.epsilon())
sn2 = float(tp2) / (tp2+fn2 + K.epsilon())
sp2 = float(tn2) / (tn2+fp2 + K.epsilon())
acc2 = float(tp2+tn2) / (tn2+fp2+fn2+tp2 + K.epsilon())
fpr2, tpr2, _ = roc_curve(imb_ind_labels_conv, y_pred_proba_imb_ind_test, pos_label=1)
auc2 = auc(fpr2, tpr2)

print("Imbalanced Independent Testing Scores:\n"
    f"MCC: {np.round(mcc2, 4)}\n"
    f"ACC: {np.round(acc2, 4)}\n"
    f"Sn: {np.round(sn2, 4)}\n"
    f"Sp: {np.round(sp2, 4)}\n"
    f"AUC: {np.round(auc2, 4)}")

