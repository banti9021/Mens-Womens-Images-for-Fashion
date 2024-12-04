import os
from box.exceptions import BoxValueError
import yaml
from src.cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the YAML file is empty or has invalid content.

    Returns:
        ConfigBox: YAML content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError(f"YAML file at {path_to_yaml} is empty")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path_to_yaml}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories if they do not exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): Whether to log the creation of directories. Defaults to True.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        except OSError as e:
            raise OSError(f"Error creating directory {path}: {e}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except (TypeError, OSError) as e:
        raise ValueError(f"Error saving JSON file at {path}: {e}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data loaded from the file as a ConfigBox object.
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Error loading JSON file at {path}: {e}")


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data to a binary file.

    Args:
        data (Any): Data to save.
        path (Path): Path to the binary file.
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        raise ValueError(f"Error saving binary file at {path}: {e}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Data loaded from the file.
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        raise ValueError(f"Error loading binary file at {path}: {e}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")


@ensure_annotations
def decodeImage(imgstring: str, fileName: Path):
    """Decodes a base64 image string and saves it as a file.

    Args:
        imgstring (str): Base64 encoded image string.
        fileName (Path): Path to save the decoded image.
    """
    try:
        imgdata = base64.b64decode(imgstring)
        with open(fileName, "wb") as f:
            f.write(imgdata)
        logger.info(f"Image decoded and saved at: {fileName}")
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")


@ensure_annotations
def encodeImageIntoBase64(croppedImagePath: Path) -> str:
    """Encodes an image file into a base64 string.

    Args:
        croppedImagePath (Path): Path to the image file.

    Returns:
        str: Base64 encoded string.
    """
    try:
        with open(croppedImagePath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {croppedImagePath}")
    except Exception as e:
        raise ValueError(f"Error encoding image: {e}")
