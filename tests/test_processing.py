import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import preprocessing
import pytest


@pytest.fixture
def sample_missing_val_list():
    return [1, None, 2, "", float("nan"), 3, 0]

@pytest.fixture
def sample_repeated_val_list():
    return [1, 3, 3, 4, 1, 2]

@pytest.fixture
def sample_numeric_val_list():
    return [1, 2, 3, 4, 5]


def test_remove_missing_values(sample_missing_val_list):
    out = preprocessing.remove_missing_values(sample_missing_val_list)
    assert out == [1,2,3,0] 


def test_fill_missing_values(sample_missing_val_list):
    out = preprocessing.fill_missing_values(sample_missing_val_list)
    assert out == [1,0,2,0,0,3,0] 


def test_unique_values(sample_repeated_val_list):
    out = preprocessing.unique_values(sample_repeated_val_list)
    assert out == [1,3,4,2] 


def test_min_max_normalization(sample_numeric_val_list):
    out = preprocessing.min_max_normalization(sample_numeric_val_list)
    assert out ==  [0.0, 0.25, 0.5, 0.75, 1.0]


def test_z_score_normalization(sample_numeric_val_list):
    out = preprocessing.z_score_normalization(sample_numeric_val_list)
    assert out == pytest.approx(
        [-1.4142, -0.7071, 0.0, 0.7071, 1.4142],
        abs=1e-4
    )
