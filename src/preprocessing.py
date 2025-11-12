"""
preprocessing.py

A collection of utility functions for data preprocessing, including:
- Handling missing values (removal, filling)
- Working with numerical data 
    (normalization, standardization, clipping, log transform, conversion to int)
- Text processing (tokenization, removing punctuation, stop words removal)
- Data structure manipulation (flattening lists, shuffling, getting unique values)

Example usage:
    >>> from preprocessing import remove_missing_values, tokenize_text
    >>> data = [1, None, 2, '', 3]
    >>> clean_data = remove_missing_values(data)
    >>> text = "Hello, World!"
    >>> tokens = tokenize_text(text)
"""


import math
import re
import random


def remove_missing_values(values):
    """Remove missing values (None, empty strings, NaN) from a list.

    Args:
        values (list): List of values to process.

    Returns:
        list: List without missing values.
    """
    output = []
    for value in values:
        if value is None:
            continue
        if value == "":
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        output.append(value)
    return output


def fill_missing_values(values, fill_val=0):
    """Fill missing values (None, empty strings, NaN) in a list with a given value.

    Args:
        values (list): List of values to process.
        fill_val (any, optional): Value to replace missing elements. Defaults to 0.

    Returns:
        list: List with missing values filled.
    """
    output = []
    for value in values:
        try:
            if math.isnan(value):
                output.append(fill_val)
                continue
        except TypeError:
            pass
        if value is None or value == "":
            output.append(fill_val)
        else:
            output.append(value)
    return output


def unique_values(values):
    """Return a list of unique values, preserving order.

    Args:
        values (list): List of values.

    Returns:
        list: List with unique elements.
    """
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def min_max_normalization(values, min_range=0.0, max_range=1.0):
    """Normalize a list of numbers to a given range using min-max normalization.

    Args:
        values (list of float): List of numerical values.
        min_range (float, optional): Minimum of target range. Defaults to 0.0.
        max_range (float, optional): Maximum of target range. Defaults to 1.0.

    Returns:
        list of float: Normalized values.
    """
    v_min = min(values)
    v_max = max(values)
    output = [
        (min_range + (((v - v_min) * (max_range - min_range)) / (v_max - v_min)))
        for v in values
    ]
    return output


def z_score_normalization(values):
    """Standardize a list of numbers using z-score normalization.

    Args:
        values (list of float): List of numerical values.

    Returns:
        list of float: Standardized values.
    """
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(variance)

    output = [(v - mean_val) / std_val for v in values]
    return output


def clipping(values, min_val, max_val):
    """Clip numerical values within a specified range.

    Args:
        values (list of float): List of numerical values.
        min_val (float): Minimum limit.
        max_val (float): Maximum limit.

    Returns:
        list of float: Clipped values.
    """
    output = [max(min(v, max_val), min_val) for v in values]
    return output


def convert_to_int(values):
    """Convert a list of values to integers, skipping invalid ones.

    Args:
        values (list): List of values to convert.

    Returns:
        list of int: List of integer values.
    """
    output = []
    for value in values:
        try:
            int_value = int(value)
            output.append(int_value)
        except ValueError:
            continue
    return output


def log_transform(values):
    """Apply logarithmic transformation to positive numbers.

    Args:
        values (list of float): List of numerical values.

    Returns:
        list of float: Log-transformed values.
    """
    output = []
    for value in values:
        if value > 0:
            log_value = math.log(value)
            output.append(log_value)
    return output


def tokenize_text(text):
    """Tokenize text into lowercase alphanumeric words.

    Args:
        text (str): Input text.

    Returns:
        list of str: List of tokens.
    """
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens


def select_alphanumerical_and_spaces(text):
    """Keep only alphanumeric characters and spaces from text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text containing only letters, numbers, and spaces.
    """
    processed_text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return processed_text


def stopwords_removal(text, stopwords):
    """Remove stop words from text (case-insensitive).

    Args:
        text (str): Input text.
        stopwords (list of str): List of stop words to remove.

    Returns:
        str: Text without stop words.
    """
    tokens = tokenize_text(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    processed_text = " ".join(filtered_tokens)
    return processed_text


def flatten_list(nested_list):
    """Flatten a nested list into a single-level list.

    Args:
        nested_list (list): List that may contain other lists.

    Returns:
        list: Flattened list.
    """
    output = []
    for item in nested_list:
        if isinstance(item, list):
            output.extend(flatten_list(item))
        else:
            output.append(item)
    return output


def shuffle_list(values, seed=42):
    """Shuffle a list of values randomly (reproducible with a seed).

    Args:
        values (list): List of values to shuffle.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list: Shuffled list.
    """
    shuffled = values.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled
