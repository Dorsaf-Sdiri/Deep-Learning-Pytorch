from Config import CFG_MODEL, CFG
from Dataset import Camelyon16
from Utils import *
from Model import Model
from collections import Counter
import logging
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from albumentations import Compose
import torch
import warnings

warnings.filterwarnings("ignore")
import torchvision
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import mlflow.pytorch
import mlflow
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl

LOSS = "CE"


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))


folds = pd.read_csv("folds.csv")


def train_fn(fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"### fold: {fold} ###")

    lossTrain = []
    stepTrain = []
    lossVal = []
    stepVal = []

    step = 0

    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    # Transforms ; apply augmetations => no need to
    transformTrain = tfm = Compose(
        [
            ToTensorV2(),
        ]
    )

    transformValid = tfm = Compose(
        [
            ToTensorV2(),
        ]
    )

    # Data loaders
    train_dataset = Camelyon16(
        folds.loc[trn_idx].reset_index(drop=True), transform=transformTrain
    )
    valid_dataset = Camelyon16(
        folds.loc[val_idx].reset_index(drop=True), transform=transformValid
    )

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=0)

    # Model
    model = Model(n=CFG_MODEL.target_size, N=1000, R=CFG_MODEL.szbag)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    if CFG.onecyclepolicy == True:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG.lr,
            div_factor=100,
            pct_start=0.0,
            steps_per_epoch=len(train_loader),
            epochs=CFG.epochs,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=2, verbose=True, eps=1e-6
        )

    if LOSS == "CE":
        criterion = nn.CrossEntropyLoss()
    elif LOSS == "LabelSmoothingCE":
        criterion = label_smoothing_criterion()

    best_auc = 0
    best_loss = np.inf

    # images_train, labels_train = next(iter(train_loader))
    # images_valid, labels_valid = next(iter(valid_loader))
    i = 0

    mlflow.pytorch.autolog()

    # Train the model
    with mlflow.start_run() as run:

        for epoch in range(CFG.epochs):

            start_time = time.time()
            model.eval()

            avg_loss = 0.0

            optimizer.zero_grad()

            mlflow.pytorch.log_model(model, "model")
            tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (images, labels) in tk0:
                images = images.to(device)
                labels = labels.to(device)

                y_preds = model.double()(images)
                loss = criterion(y_preds, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                lossTrain.append(loss.item())
                stepTrain.append(step)

                if CFG.onecyclepolicy == True:
                    scheduler.step()

                avg_loss += loss.item() / len(train_loader)
                step += 1

            model.eval()
            avg_val_loss = 0.0
            preds = []
            valid_labels = []
            tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

            for i, (images, labels) in tk1:

                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    y_preds = model(images)

                preds.append(y_preds.sigmoid().to("cpu").numpy().argmax(1))
                valid_labels.append(labels.to("cpu").numpy())

                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            lossVal.append(avg_val_loss)
            stepVal.append(step)

            if CFG.onecyclepolicy == False:
                scheduler.step(avg_val_loss)

            preds = np.concatenate(preds)
            valid_labels = np.concatenate(valid_labels)
            LOGGER.debug(f"Counter preds: {Counter(preds)}")
            # fetch the auto logged parameters and metrics

            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
            try:

                auc_score = roc_auc_score(
                    valid_labels, preds
                )  # If all labels belong to one class then continue to the next epoch

            except Exception as e:
                logging.error(e)
                print(e)
                continue
            if epoch == (CFG.epochs - 1):
                print(confusion_matrix(valid_labels, preds))

            elapsed = time.time() - start_time

            LOGGER.debug(
                f"  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )
            LOGGER.debug(f"  Epoch {epoch + 1} - QWK: {auc_score}")

            if auc_score > best_auc:

                best_auc = auc_score
                print("Saving the model")
                LOGGER.debug(
                    f"  Epoch {epoch + 1} - Save Best Score: {best_auc:.4f} Model"
                )
                torch.save(model.state_dict(), f"{Weights}fold{fold}_WSI.pth")
    print("Best AUC", best_auc)
    # Plot losses
    plt.figure(figsize=(26, 6))
    plt.subplot(1, 2, 1)
    plt.plot(stepTrain, lossTrain, label="training loss")
    plt.plot(stepVal, lossVal, label="validation loss")
    plt.title("Loss")
    plt.xlabel("step")
    plt.legend(loc="center left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for fold in range(CFG.n_fold):
        train_fn(fold)
