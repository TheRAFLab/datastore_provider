from collections.abc import Callable
from typing import Literal

import polars as pl

Mode = Literal["lazy", "eager"]


class DatastoreProvider:
    """
    Read a folder-based datastore as a set of Polars tables.

    A datastore is a directory tree in which each table is a subdirectory
    holding one or more sharded files in a single format. The provider maps a
    configuration dictionary onto that layout, so a table can be scanned by
    name without knowing how many files back it or where they are stored.

    Attributes:
        config (dict): The configuration the provider was built from.
        tables (dict): Table definitions, keyed by table name.
        formats (list): The file formats that can be scanned.
        storage_options (dict | None): Credentials passed to every Polars
            scan, or None when S3 is not enabled.
    """

    def __init__(self, config: dict):
        """
        Build a provider from a datastore configuration.

        Args:
            config (dict): The datastore configuration. Recognised keys are:
                DATASTORE_ROOT (str): Base path of the datastore. Either a
                    local path or a remote URI (e.g. "s3://bucket/datastore").
                TABLES (list): Table definitions, each a dict with a "name"
                    and a "format" ("jsonl", "csv" or "parquet"). Defaults to
                    an empty list.
                S3_ENABLED (bool): Whether to pass S3 credentials through to
                    Polars. Defaults to False.
                AWS_ACCESS_KEY_ID (str): Used when S3_ENABLED is true.
                AWS_SECRET_ACCESS_KEY (str): Used when S3_ENABLED is true.
                AWS_REGION (str): Used when S3_ENABLED is true. Defaults to
                    "us-east-1".

        Raises:
            ValueError: If DATASTORE_ROOT is not set, if TABLES is not a list,
                or if any table definition is missing a name, is missing a
                format, or declares an unsupported format.
        """
        self.config = config
        self.formats = ["jsonl", "csv", "parquet"]

        tables = config.get("TABLES", [])
        if not isinstance(tables, list):
            raise ValueError(  # noqa: TRY004
                f"TABLES must be a list of table definitions, "
                f"got {type(tables).__name__}."
            )
        self.tables = {}
        for position, table in enumerate(tables):
            if not isinstance(table, dict):
                raise ValueError(  # noqa: TRY004
                    f"Table definition at position {position} must be a dict, "
                    f"got {type(table).__name__}."
                )
            if "name" not in table:
                raise ValueError(
                    f"Table definition at position {position} has no 'name'."
                )
            self.tables[table["name"]] = table

        self._validate()

        self.storage_options: dict | None
        if self.config.get("S3_ENABLED", False):
            self.storage_options = {
                "aws_access_key_id": self.config.get("AWS_ACCESS_KEY_ID", "YOUR_ACCESS_KEY"),
                "aws_secret_access_key": self.config.get("AWS_SECRET_ACCESS_KEY", "YOUR_SECRET_KEY"),
                "aws_region": self.config.get("AWS_REGION", "us-east-1")
            }
        else:
            self.storage_options = None


    def _validate(self) -> None:
        """
        Check the configuration up front, so typos fail at construction.

        Raises:
            ValueError: If DATASTORE_ROOT is not set, or if any table is
                missing a format or declares an unsupported one.
        """
        if not self.config.get("DATASTORE_ROOT"):
            raise ValueError("DATASTORE_ROOT is not defined in the configuration.")

        for table_name, table_config in self.tables.items():
            table_format = table_config.get("format")
            if not table_format:
                raise ValueError(f"Table '{table_name}' has no 'format'.")
            if table_format not in self.formats:
                raise ValueError(
                    f"Unsupported format '{table_format}' for table '{table_name}'. "
                    f"Supported formats are {', '.join(self.formats)}."
                )


    def build_table_path(self, table_name: str) -> str:
        """
        Build the full path to a specific table based on the configuration.

        Args:
            table_name (str): The name of the table.

        Returns:
            str: The full path to the table, as "{DATASTORE_ROOT}/{name}".

        Raises:
            ValueError: If the table is not in the configuration, or if
                DATASTORE_ROOT is not set. The latter is checked when the
                provider is built, so it can only happen if config has been
                mutated since.
        """
        table_config = self.tables.get(table_name)
        if not table_config:
            raise ValueError(f"Table '{table_name}' not found in configuration.")
        datastore_root = self.config.get("DATASTORE_ROOT", None)
        if not datastore_root:
            raise ValueError("DATASTORE_ROOT is not defined in the configuration.")
        return f"{datastore_root}/{table_config['name']}"


    def build_glob_string(self, table_name: str, format:str) -> str:
        """
        Build a glob string matching every file in a specific table.

        Args:
            table_name (str): The name of the table.
            format (str): The file format of the table (e.g., "jsonl", "csv", "parquet").

        Returns:
            str: The glob string for the table, as
                "{DATASTORE_ROOT}/{name}/*.{format}".

        Raises:
            ValueError: If the table is not in the configuration, or if
                DATASTORE_ROOT is not set.
        """
        return f"{self.build_table_path(table_name)}/*.{format}"


    def load_table(self, table_name: str, mode: Mode = "lazy") -> pl.DataFrame | pl.LazyFrame:
        """
        Scan every file belonging to a table and return it as a Polars frame.

        The table's files are discovered by globbing, so all shards are read as
        a single frame and each row carries a "source_file" column recording
        the file it came from.

        Args:
            table_name (str): The name of the table.
            mode (str): "lazy" to return an unevaluated frame, letting Polars
                push predicates and projections down into the scan, or "eager"
                to read the table immediately. Defaults to "lazy".

        Returns:
            A polars LazyFrame when mode is "lazy", or a DataFrame when mode
            is "eager".

        Raises:
            ValueError: If the table is not in the configuration, or if mode
                is neither "lazy" nor "eager". The table's format is validated
                when the provider is built.
        """
        table_config = self.tables.get(table_name)
        if not table_config:
            raise ValueError(f"Table '{table_name}' not found in configuration.")

        if mode not in ["lazy", "eager"]:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'lazy' and 'eager'.")

        table_format = table_config["format"]
        glob_string = self.build_glob_string(table_name, table_format)

        # The format is checked in _validate, so no fallback branch is needed.
        scanners: dict[str, Callable[..., pl.LazyFrame]] = {
            "jsonl": pl.scan_ndjson,
            "csv": pl.scan_csv,
            "parquet": pl.scan_parquet,
        }
        scan = scanners[table_format](
            glob_string,
            storage_options=self.storage_options,
            include_file_paths="source_file",
        )

        return scan.collect() if mode == "eager" else scan




