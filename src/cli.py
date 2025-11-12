"""
cli.py

Command-line interface (CLI) for the data preprocessing module.

This CLI provides access to functions for:
- Data cleaning (removing or filling missing values)
- Numerical preprocessing (normalization, standardization, clipping, 
    log transform, integer conversion)
- Text processing (tokenization, punctuation removal, stop-words removal)
- Data structure operations (flattening lists, shuffling, getting unique values)

Each group of commands is organized under a main CLI group:
- clean: Data cleaning operations
- numeric: Numerical preprocessing
- text: Text preprocessing
- struct: Data structure operations

Usage examples:
    python cli.py clean remove-missing '[1, None, "", 3]'
    python cli.py numeric normalize '[1,2,3]' --min_range 0 --max_range 1
    python cli.py text tokenize "Hello, World!"
    python cli.py struct shuffle '[1,2,3,4]' --seed 42
"""

import json
import click
from . import preprocessing


@click.group(help="Main command group for data preprocessing functionalities.")
def cli():
    """Main command group to access data preprocessing functions."""


@cli.group(help="Functions related to data cleaning.")
def clean():
    """Group of commands for cleaning data (handling missing values)."""


@clean.command(
    help="Remove missing values from a list. Example: clean remove-missing '[1, None, \"\", 3]'"
)
@click.argument("values")
def remove_missing(values):
    """Remove None, empty strings or NaN from a list."""
    values = json.loads(values)
    result = preprocessing.remove_missing_values(values)
    click.echo(json.dumps(result))


@clean.command(
    help="Fill missing values in a list. Example: clean fill-missing " +
    "'[1, None, \"\", 3]' --fill_val 0"
)
@click.argument("values")
@click.option("--fill_val", default=0, help="Value used to replace missing elements.")
def fill_missing(values, fill_val):
    """Fill missing values in a list with a given value."""
    values = json.loads(values)
    result = preprocessing.fill_missing_values(values, fill_val)
    click.echo(json.dumps(result))


@cli.group(help="Functions related to numerical attributes.")
def numeric():
    """Group of commands for numerical data preprocessing."""


@numeric.command(
    help="Normalize a list of numbers to a given range. Example: numeric normalize " +
    "'[1,2,3]' --min_range 0 --max_range 1"
)
@click.argument("values")
@click.option("--min_range", default=0.0, help="Minimum value of new range.")
@click.option("--max_range", default=1.0, help="Maximum value of new range.")
def normalize(values, min_range, max_range):
    """Apply min-max normalization to a list of numbers."""
    values = json.loads(values)
    result = preprocessing.min_max_normalization(values, min_range, max_range)
    click.echo(json.dumps(result))


@numeric.command(
    help="Standardize a list of numbers using z-score normalization. " +
    "Example: numeric standardize '[1,2,3]'"
)
@click.argument("values")
def standardize(values):
    """Apply z-score normalization to a list of numbers."""
    values = json.loads(values)
    result = preprocessing.z_score_normalization(values)
    click.echo(json.dumps(result))


@numeric.command(
    help="Clip a list of numbers within a specified range. Example: numeric clip " +
    "'[1,2,3,4]' --min_val 1 --max_val 3"
)
@click.argument("values")
@click.option("--min_val", default=0.0, help="Minimum value to clip.")
@click.option("--max_val", default=1.0, help="Maximum value to clip.")
def clip(values, min_val, max_val):
    """Clip numbers to be within a minimum and maximum range."""
    values = json.loads(values)
    result = preprocessing.clipping(values, min_val, max_val)
    click.echo(json.dumps(result))


@numeric.command(
    help='Convert list of values to integers. Example: numeric to-int \'["1", 2.5, "3"]\''
)
@click.argument("values")
def to_int(values):
    """Convert a list of values to integers, skipping invalid ones."""
    values = json.loads(values)
    result = preprocessing.convert_to_int(values)
    click.echo(json.dumps(result))


@numeric.command(
    help="Transform list of numbers to logarithmic scale. Example: numeric log '[1,10,100]'"
)
@click.argument("values")
def log(values):
    """Apply logarithmic transformation to positive numbers."""
    values = json.loads(values)
    result = preprocessing.log_transform(values)
    click.echo(json.dumps(result))


@cli.group(help="Functions to process textual information.")
def text():
    """Group of commands for text preprocessing."""


@text.command(help="Tokenize text into words. Example: text tokenize 'Hello, World!'")
@click.argument("input_text")
def tokenize(input_text):
    """Split text into lowercase alphanumeric tokens."""
    result = preprocessing.tokenize_text(input_text)
    click.echo(json.dumps(result))


@text.command(
    help="Remove punctuation from text. Example: text remove-punctuation 'Hello, World!'"
)
@click.argument("input_text")
def remove_punctuation(input_text):
    """Keep only letters, numbers and spaces in text."""
    result = preprocessing.select_alphanumerical_and_spaces(input_text)
    click.echo(result)


@text.command(
    help="Remove stop words from text. Example: " +
    "text remove-stopwords 'this is an example' --stopwords '[\"this\",\"is\"]'"
)
@click.argument("input_text")
@click.option("--stopwords", default="[]", help="List of stop words to remove.")
def remove_stopwords(input_text, stopwords):
    """Remove specified stop words from text."""
    stopwords = json.loads(stopwords)
    result = preprocessing.stopwords_removal(input_text, stopwords)
    click.echo(result)


@cli.group(help="Functions related to data structure manipulation.")
def struct():
    """Group of commands for manipulating data structures."""


@struct.command(
    help="Shuffle a list of values. Example: struct shuffle '[1,2,3,4]' --seed 42"
)
@click.argument("values")
@click.option("--seed", default=None, type=int, help="Seed for reproducibility.")
def shuffle(values, seed):
    """Shuffle a list randomly; can set a seed for reproducibility."""
    values = json.loads(values)
    result = preprocessing.shuffle_list(values, seed)
    click.echo(json.dumps(result))


@struct.command(
    help="Flatten a nested list. Example: struct flatten '[[1,2],[3,[4,5]]]'"
)
@click.argument("values")
def flatten(values):
    """Flatten a nested list into a single-level list."""
    values = json.loads(values)
    result = preprocessing.flatten_list(values)
    click.echo(json.dumps(result))


@struct.command(
    help="Get unique values from a list. Example: struct unique '[1,2,2,3]'"
)
@click.argument("values")
def unique(values):
    """Return a list of unique values, preserving order."""
    values = json.loads(values)
    result = preprocessing.unique_values(values)
    click.echo(json.dumps(result))


if __name__ == "__main__":
    cli()
