from Config import CFG, CFG_MODEL
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

#Define paths
TRAIN_IMG_DIR = 'data\\train_input\\resnet_features\\'
train = pd.read_csv('\\data\\train_output.csv').set_index('ID')
files = sorted(set([p.split('.')[0] for p in os.listdir(TRAIN_IMG_DIR)]))
train = train.loc[files[1:]]
train = train.reset_index()

if CFG.debug:
    folds = train.sample(n=50, random_state=CFG.seed).reset_index(drop=True).copy()
else:
    folds = train.copy()

train_labels = folds[CFG_MODEL.target_col].values
kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)