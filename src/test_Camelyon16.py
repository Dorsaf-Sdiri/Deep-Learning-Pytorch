import pytest
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import pandas as pd
from Dataset import Camelyon16

folds = pd.read_csv('folds.csv')
@pytest.fixture
def camelyon():
    "Create an instance of class Camelyon16"
    transformTrain = tfm = Compose([
        ToTensorV2(),
    ])
    return Camelyon16(folds, transform=transformTrain)

def test_input_shape(camelyon):
    assert camelyon.shape[0] == 1000
    assert camelyon.shape[1] == 100
