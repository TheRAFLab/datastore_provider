import json
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load the configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If no file exists at config_path.
        ValueError: If the file is not valid JSON, or does not contain a
            JSON object at the top level.

    Note:
        The contents are not validated beyond being a JSON object. Missing or
        malformed keys are reported by DatastoreProvider when a table is
        resolved or loaded.
    """
    path = Path(config_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file not found: '{path}'.") from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Configuration file '{path}' is not valid UTF-8: {error}."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Configuration file '{path}' is not valid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})."
        ) from error

    # ValueError, not TypeError: this is malformed file content rather than a
    # bad argument, and every other configuration error raises ValueError.
    if not isinstance(config, dict):
        raise ValueError(  # noqa: TRY004
            f"Configuration file '{path}' must contain a JSON object, "
            f"got {type(config).__name__}."
        )

    return config
