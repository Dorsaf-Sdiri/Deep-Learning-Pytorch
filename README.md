# Chowder_Test


## Hardware/OS used for the competiion
- CPU:  4 cores
- RAM: 24GB
- OS: Ubuntu 20.04

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
    * TEST_BATCH_SIZE: The testing/validation batch size
    * IMAGE_SIZE: The image size used during the training/infernece
    * N_EPOCHS: The number of the training epochs
    * NUM_IMAGES_3D: Number of the images/scans used to build the 3D images.
    * do_valid: bool that indicates if we want to save the model weights based on the validation score
    * n_workers: Number of workers used during the training


6. Run training script Train.py or the Dokerfile

## Tracking and logging
Run mlflow ui in the command line and open http://127.0.0.1:5000 in the browser


