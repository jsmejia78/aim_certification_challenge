from pathlib import Path
import os
import pickle

def get_parent_dir(file_path: str) -> Path:
    """
    Returns the parent directory of the script's directory.

    Args:
        file_path (str): Typically passed as __file__ from the calling script.

    Returns:
        Path: The parent directory of the script's directory.
    """
    return Path(file_path).resolve().parent.parent

def data_file_exists(path: str, filename: str) -> bool:
    """
    Checks if a pickdata file file exists at the given path.

    Args:
        path (str): The directory path where the data file is located.
        filename (str): The name of the data file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    full_path = os.path.join(path, filename)
    return os.path.isfile(full_path)

def save_pickle(obj, filepath):
    """
    Saves an object to a pickle file.

    Args:
        obj: The object to save.
        filepath (str): The path to the pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)