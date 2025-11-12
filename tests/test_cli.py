import json
import pytest
from click.testing import CliRunner
from src.cli import cli


@pytest.fixture
def runner():
    """Fixture to provide a Click CliRunner instance for all CLI tests."""
    return CliRunner()


def test_remove_missing_cli(runner):
    result = runner.invoke(cli, ["clean", "remove-missing", '[1, null, "", 3]'])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [1, 3]

def test_fill_missing_cli(runner):
    result = runner.invoke(cli, ["clean", "fill-missing", '[1, null, "", 3]', "--fill_val", "0"])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [1, 0, 0, 3]


def test_normalize_cli(runner):
    result = runner.invoke(cli, ["numeric", "normalize", '[1,2,3]', "--min_range", "0", "--max_range", "1"])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [0.0, 0.5, 1.0]

def test_clip_cli(runner):
    result = runner.invoke(cli, ["numeric", "clip", '[1,2,3,4]', "--min_val", "2", "--max_val", "3"])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [2, 2, 3, 3]


def test_tokenize_cli(runner):
    result = runner.invoke(cli, ["text", "tokenize", "Hello, World!"])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == ["hello", "world"]

def test_remove_punctuation_cli(runner):
    result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World!"])
    assert result.exit_code == 0
    output_str = result.output.strip()
    assert output_str == "Hello World"

def test_remove_stopwords_cli(runner):
    result = runner.invoke(cli, ["text", "remove-stopwords", "this is a test", "--stopwords", '["this","is"]'])
    assert result.exit_code == 0
    output_str = result.output.strip()
    assert output_str == "a test"


def test_flatten_cli(runner):
    result = runner.invoke(cli, ["struct", "flatten", '[[1,2],[3,[4,5]]]'])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [1,2,3,4,5]

def test_unique_cli(runner):
    result = runner.invoke(cli, ["struct", "unique", '[1,2,2,3]'])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert output_list == [1,2,3]

def test_shuffle_cli_with_seed(runner):
    result = runner.invoke(cli, ["struct", "shuffle", '[1,2,3,4]', "--seed", "42"])
    assert result.exit_code == 0
    output_list = json.loads(result.output.strip())
    assert sorted(output_list) == [1,2,3,4]
