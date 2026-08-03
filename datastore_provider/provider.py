import polars as pl


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

        Note:
            The configuration is not validated here. A missing DATASTORE_ROOT
            or an unsupported table format is only reported when the table is
            resolved or loaded.
        """
        self.config = config
        tables = config.get("TABLES", [])
        self.tables = {table["name"]: table for table in tables}
        self.formats = ["jsonl", "csv", "parquet"]

        if self.config.get("S3_ENABLED", False):
            self.storage_options = {
                "aws_access_key_id": self.config.get("AWS_ACCESS_KEY_ID", "YOUR_ACCESS_KEY"),
                "aws_secret_access_key": self.config.get("AWS_SECRET_ACCESS_KEY", "YOUR_SECRET_KEY"),
                "aws_region": self.config.get("AWS_REGION", "us-east-1")
            }
        else:
            self.storage_options = None


    def build_table_path(self, table_name: str) -> str:
        """
        Build the full path to a specific table based on the configuration.

        Args:
            table_name (str): The name of the table.

        Returns:
            str: The full path to the table, as "{DATASTORE_ROOT}/{name}".

        Raises:
            ValueError: If the table is not in the configuration, or if
                DATASTORE_ROOT is not set.
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


    def load_table(self, table_name: str, mode: str = "lazy") -> pl.DataFrame | pl.LazyFrame:
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
            ValueError: If the table is not in the configuration, if mode is
                neither "lazy" nor "eager", or if the table's declared format
                is not one of "jsonl", "csv" or "parquet".
        """
        table_config = self.tables.get(table_name)
        if not table_config:
            raise ValueError(f"Table '{table_name}' not found in configuration.")
        
        if mode not in ["lazy", "eager"]:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'lazy' and 'eager'.")

        if table_config.get("format") not in self.formats:
            raise ValueError(f"Unsupported format '{table_config.get('format')}' for table '{table_name}'.")
        else:
            glob_string = self.build_glob_string(table_name, table_config.get("format"))  
            
            if table_config.get("format") == "jsonl":
                scan = pl.scan_ndjson(glob_string, storage_options=self.storage_options, include_file_paths="source_file")
            elif table_config.get("format") == "csv":
                scan = pl.scan_csv(glob_string, storage_options=self.storage_options, include_file_paths="source_file")
            elif table_config.get("format") == "parquet":
                scan = pl.scan_parquet(glob_string, storage_options=self.storage_options, include_file_paths="source_file")
            else:   
                raise ValueError(f"Unsupported format '{table_config.get('format')}' for table '{table_name}'.")

            return scan.collect() if mode == "eager" else scan




