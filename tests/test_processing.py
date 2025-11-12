import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import preprocessing
import pytest
import random


@pytest.fixture
def sample_missing_val_list():
    return [1, None, 2, "", float("nan"), 3, 0]

@pytest.fixture
def sample_repeated_val_list():
    return [1, 3, 3, 4, 1, 2]

@pytest.fixture
def sample_numeric_val_list():
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_string_list():
    return ["1", "something", "3", "nothing", "5"]

@pytest.fixture
def sample_text():
    return "Hello, World! This is a test. 123"


def test_remove_missing_values(sample_missing_val_list):
    out = preprocessing.remove_missing_values(sample_missing_val_list)
    assert out == [1,2,3,0]

@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([1, None, 2, ""], [1, 2]),
        ([None, "", float("nan")], [])
    ]
)
def test_remove_missing_values_param(input_list, expected):
    assert preprocessing.remove_missing_values(input_list) == expected


def test_fill_missing_values(sample_missing_val_list):
    out = preprocessing.fill_missing_values(sample_missing_val_list)
    assert out == [1,0,2,0,0,3,0]

@pytest.mark.parametrize(
    "input_list, fill_val, expected",
    [
        ([1, None, ""], 0, [1, 0, 0]),
        ([None, 2, ""], 99, [99, 2, 99])
    ]
)
def test_fill_missing_values_param(input_list, fill_val, expected):
    assert preprocessing.fill_missing_values(input_list, fill_val) == expected


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


def test_clipping(sample_numeric_val_list):
    out = preprocessing.clipping(sample_numeric_val_list, 2, 4)
    assert out == [2, 2, 3, 4, 4]


def test_convert_to_int(sample_string_list):
    out = preprocessing.convert_to_int(sample_string_list)
    assert out == [1, 3, 5]


def test_log_transform(sample_numeric_val_list):
    out = preprocessing.log_transform(sample_numeric_val_list)
    assert out == pytest.approx(
        [0.0, 0.6931, 1.0986, 1.3863, 1.6094],
        abs=1e-4
    )

def test_log_transform_with_zero_and_negative():
    input_data = [1, 0, -1, 10]
    out = preprocessing.log_transform(input_data)
    assert out == pytest.approx([0.0, 2.3025], abs=1e-4)  


def test_tokenize_text(sample_text):
    out = preprocessing.tokenize_text(sample_text)
    assert out == ["hello", "world", "this", "is", "a", "test", "123"]


def test_select_alphanumerical_and_spaces(sample_text):
    out = preprocessing.select_alphanumerical_and_spaces(sample_text)
    assert out == "Hello World This is a test 123"


def test_stopword_removal(sample_text):
    stopwords = ["this", "is", "a"]
    out = preprocessing.stopwords_removal(sample_text, stopwords)
    assert out == "hello world test 123"

@pytest.mark.parametrize(
    "text, stopwords, expected",
    [
        ("Hello World", [], "hello world"),
        ("Hello World", ["hello"], "world"),
        ("HELLO world", ["hello"], "world")
    ]
)
def test_stopwords_removal_param(text, stopwords, expected):
    assert preprocessing.stopwords_removal(text, stopwords) == expected


def test_flatten_list():
    input_data = [[1, 2], [3, 4], [5]]
    out = preprocessing.flatten_list(input_data)
    assert out == [1, 2, 3, 4, 5]

def test_flatten_list_nested():
    input_data = [1, [2, [3, 4]], 5]
    out = preprocessing.flatten_list(input_data)
    assert out == [1, 2, 3, 4, 5]


def test_shuffle_list(sample_numeric_val_list, seed=123):
    out = preprocessing.shuffle_list(sample_numeric_val_list, seed=seed)
    random.seed(seed)
    expected = sample_numeric_val_list.copy()
    random.shuffle(expected)
    assert out == expected

def test_shuffle_list_no_seed(sample_numeric_val_list):
    out = preprocessing.shuffle_list(sample_numeric_val_list)
    assert sorted(out) == sorted(sample_numeric_val_list)
