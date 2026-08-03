# Datastore Provider

A small Python/[Polars](https://pola.rs) provider for reading folders of data files (JSONL, CSV, Parquet) as tables.

A "datastore" here is just a directory tree: each table is a subdirectory, and each table is made up of one or more sharded files in a single format. `datastore_provider` maps a JSON config onto that layout and hands you a Polars `LazyFrame` (or `DataFrame`) per table, so you can query across the shards without caring how many files there are or where they live.

```
<DATASTORE_ROOT>/
├── <table_name>/           <- one directory per table
│   ├── <shard>.jsonl       <- one or more files, all in the table's format
│   ├── <shard>.jsonl
│   └── ...
└── <other_table>/
    ├── <shard>.parquet
    └── ...
```

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv add datastore-provider
```

Or with pip:

```bash
pip install datastore-provider
```

Requires Python 3.10 or newer. The only dependency is [Polars](https://pola.rs).

To work on the library itself rather than install it, see
[Development](#development).

## Configuration

Configuration is a plain JSON document (or any `dict`). The minimum is a datastore root and a list of tables:

```json
{
    "DATASTORE_ROOT": "/path/to/datastore",
    "TABLES": [
        { "name": "samples", "format": "jsonl" },
        { "name": "measurements", "format": "csv" },
        { "name": "annotations", "format": "parquet" }
    ]
}
```

| Key | Required | Description |
| --- | --- | --- |
| `DATASTORE_ROOT` | yes | Base path for the datastore. Local path or remote URI (e.g. `s3://my-bucket/datastore`). |
| `TABLES` | yes | List of table definitions. |
| `TABLES[].name` | yes | Table name, and the name of its subdirectory under `DATASTORE_ROOT`. |
| `TABLES[].format` | yes | One of `jsonl`, `csv`, `parquet`. All files in the table must share this format. |
| `S3_ENABLED` | no | Set `true` to pass S3 credentials through to Polars. Defaults to `false`. |
| `AWS_ACCESS_KEY_ID` | no | Used when `S3_ENABLED` is true. |
| `AWS_SECRET_ACCESS_KEY` | no | Used when `S3_ENABLED` is true. |
| `AWS_REGION` | no | Used when `S3_ENABLED` is true. Defaults to `us-east-1`. |

Files are discovered by globbing `{DATASTORE_ROOT}/{name}/*.{format}`, so adding a shard to a table is just a matter of dropping a file into the directory.

## Usage

```python
from datastore_provider import DatastoreProvider
from datastore_provider.helpers import load_config

config = load_config("config.json")
provider = DatastoreProvider(config)

# Lazy by default — nothing is read until you collect
samples = provider.load_table("samples").collect()

# Or read eagerly
annotations = provider.load_table("annotations", mode="eager")
```

The config can equally be built in code, without a file on disk:

```python
provider = DatastoreProvider({
    "DATASTORE_ROOT": "/path/to/datastore",
    "TABLES": [{"name": "samples", "format": "jsonl"}],
})
```

Because lazy mode returns a Polars `LazyFrame`, predicates and projections are pushed down into the scan — only the columns and rows you ask for are read off disk:

```python
import polars as pl

subset = (
    provider.load_table("samples")
    .filter(pl.col("category") == "reference")
    .select("id", "label")
    .collect()
)
```

Every row carries a `source_file` column recording which file it came from, which is useful when a table is sharded per-sample or per-experiment.

## API

### `DatastoreProvider(config: dict)`

Builds a provider from a config dictionary. If `S3_ENABLED` is true, AWS credentials are collected into the `storage_options` passed to every Polars scan; otherwise `storage_options` is `None`.

The configuration is validated up front, so typos fail here rather than on first use. Raises `ValueError` if `DATASTORE_ROOT` is unset, if `TABLES` is not a list, or if any table definition is not a dict, is missing `name` or `format`, or declares an unsupported format.

### `load_table(table_name: str, mode: str = "lazy")`

Scans every file belonging to `table_name` and returns a Polars `LazyFrame` (`mode="lazy"`) or `DataFrame` (`mode="eager"`). `mode` is typed as a `Literal`, so a typo is caught by a type checker before it runs.

Raises `ValueError` if the table is not in the config, or if `mode` is not `lazy` or `eager`. Table formats are validated when the provider is built.

### `build_table_path(table_name: str) -> str`

Returns `{DATASTORE_ROOT}/{table_name}`. Raises `ValueError` if the table is unknown or `DATASTORE_ROOT` is missing.

### `build_glob_string(table_name: str, format: str) -> str`

Returns the glob used to find the table's files: `{DATASTORE_ROOT}/{table_name}/*.{format}`.

### `helpers.load_config(config_path: str) -> dict`

Reads and parses a JSON config file.

## Development

```bash
git clone https://github.com/TheRAFLab/datastore_provider.git
cd datastore_provider
uv sync
uv run pytest
uv run ruff check .
```

The test suite builds a throwaway datastore under `tmp_path` covering all three
formats, so it needs no fixture data on disk. CI runs the same two commands
against Python 3.10 through 3.14.

## License

MIT — see [LICENSE](LICENSE).
