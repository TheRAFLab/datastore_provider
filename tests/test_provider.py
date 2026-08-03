import json

import polars as pl
import pytest

from datastore_provider import DatastoreProvider
from datastore_provider.helpers import load_config

TABLES = [
    {"name": "samples", "format": "jsonl"},
    {"name": "measurements", "format": "csv"},
    {"name": "annotations", "format": "parquet"},
]


@pytest.fixture
def datastore_root(tmp_path):
    """A datastore covering every supported format, with one table sharded."""
    samples = tmp_path / "samples"
    samples.mkdir()
    for shard, ids in (("a", [1, 2]), ("b", [3])):
        (samples / f"{shard}.jsonl").write_text(
            "".join(json.dumps({"id": i, "category": "reference"}) + "\n" for i in ids)
        )

    measurements = tmp_path / "measurements"
    measurements.mkdir()
    (measurements / "a.csv").write_text("id,value\n1,0.5\n2,1.5\n")

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    pl.DataFrame({"id": [1, 2], "label": ["x", "y"]}).write_parquet(
        annotations / "a.parquet"
    )

    return tmp_path


@pytest.fixture
def config(datastore_root):
    return {"DATASTORE_ROOT": str(datastore_root), "TABLES": TABLES}


@pytest.fixture
def provider(config):
    return DatastoreProvider(config)


@pytest.mark.parametrize(
    ("table", "expected_height"),
    [("samples", 3), ("measurements", 2), ("annotations", 2)],
)
def test_every_format_scans(provider, table, expected_height):
    assert provider.load_table(table).collect().height == expected_height


def test_shards_are_concatenated(provider):
    """Both jsonl shards land in one frame."""
    assert sorted(provider.load_table("samples").collect()["id"]) == [1, 2, 3]


def test_source_file_column_is_added(provider):
    sources = provider.load_table("samples").collect()["source_file"]
    assert {s.rsplit("/", 1)[-1] for s in sources} == {"a.jsonl", "b.jsonl"}


def test_lazy_is_the_default(provider):
    assert isinstance(provider.load_table("samples"), pl.LazyFrame)


def test_eager_mode_returns_a_dataframe(provider):
    table = provider.load_table("samples", mode="eager")
    assert isinstance(table, pl.DataFrame)
    assert table.height == 3


def test_predicates_push_down_through_the_scan(provider):
    subset = provider.load_table("samples").filter(pl.col("id") > 1).collect()
    assert subset.height == 2


def test_load_config_round_trip(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_config(str(path)) == config


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(str(tmp_path / "absent.json"))


def test_load_config_invalid_json_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"DATASTORE_ROOT": "/data",}')
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_config(str(path))


def test_load_config_invalid_json_reports_position(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{\n  "DATASTORE_ROOT": "/data"\n  "TABLES": []\n}')
    with pytest.raises(ValueError, match=r"line \d+, column \d+"):
        load_config(str(path))


@pytest.mark.parametrize("payload", ["[]", '"a string"', "42", "null"])
def test_load_config_rejects_non_objects(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_config(str(path))


def test_load_config_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"DATASTORE_ROOT": "\xff\xfe"}')
    with pytest.raises(ValueError, match="is not valid UTF-8"):
        load_config(str(path))


def test_load_config_reads_utf8(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"DATASTORE_ROOT": "/données/café"}', encoding="utf-8")
    assert load_config(str(path))["DATASTORE_ROOT"] == "/données/café"


def test_build_table_path(provider, datastore_root):
    assert provider.build_table_path("samples") == f"{datastore_root}/samples"


def test_build_glob_string(provider, datastore_root):
    glob = provider.build_glob_string("samples", "jsonl")
    assert glob == f"{datastore_root}/samples/*.jsonl"


def test_unknown_table_raises(provider):
    with pytest.raises(ValueError, match="not found in configuration"):
        provider.load_table("nope")


def test_invalid_mode_raises(provider):
    with pytest.raises(ValueError, match="Invalid mode"):
        provider.load_table("samples", mode="sideways")


def test_unsupported_format_raises(datastore_root):
    provider = DatastoreProvider(
        {
            "DATASTORE_ROOT": str(datastore_root),
            "TABLES": [{"name": "samples", "format": "xml"}],
        }
    )
    with pytest.raises(ValueError, match="Unsupported format"):
        provider.load_table("samples")


def test_missing_datastore_root_raises():
    provider = DatastoreProvider({"TABLES": TABLES})
    with pytest.raises(ValueError, match="DATASTORE_ROOT is not defined"):
        provider.build_table_path("samples")


def test_storage_options_are_none_without_s3(provider):
    assert provider.storage_options is None


def test_storage_options_are_built_when_s3_enabled():
    provider = DatastoreProvider(
        {
            "DATASTORE_ROOT": "s3://bucket/datastore",
            "TABLES": TABLES,
            "S3_ENABLED": True,
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": "eu-west-2",
        }
    )
    assert provider.storage_options == {
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "aws_region": "eu-west-2",
    }


def test_s3_region_defaults_when_unset():
    provider = DatastoreProvider({"TABLES": TABLES, "S3_ENABLED": True})
    assert provider.storage_options["aws_region"] == "us-east-1"
