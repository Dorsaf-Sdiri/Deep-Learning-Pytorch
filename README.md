# Chowder_Test


## Hardware/OS used for the competiion
- CPU:  4 cores
- RAM: 24GB
- OS: Windows10 and Ubuntu 20.04

## Resources

[ResNet50 extracted features](https://drive.google.com/file/d/1dncSXrycW2ncHe99ru_f8rWWRo82duK2/view?usp=sharing)

[Paper](https://arxiv.org/pdf/1802.02212.pdf)

## Training steps

To train the model using the competition data, please follow these steps:

1. Clone this repo
2. Install dependencies via `pip install -r requirements.txt`
3. Download the ResNet50 data or the raw data using src/Data_Collection.py.
4. Set the configuration file `src/config.py`. It is recommended to use the default values.

    * TRAINING_BATCH_SIZE: The training batch size
    * N_EPOCHS: The number of the training epochs


5. change TRAIN_IMG_DIR to your relative path
6. Run training script Train.py or the Dockerfile
## Data Science
*In order to reduce the dimensionality of the input arrays, I have kept the 100 most important features on the fully annotated data.
*The maximal AUC reached is 0.689
![Alt Image text](LOSS.png?raw=true "Optional Title")
## Tracking and logging
Run mlflow ui in the command line and open http://127.0.0.1:5000 in the browser
.pth files are saved for best AUC
