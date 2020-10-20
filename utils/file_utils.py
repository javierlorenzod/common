import logging_utils as log_u
import os.path
import pickle


def check_file_existence(file_path: str) -> bool:
    """
    Returns True if file exists and False if not.
    :param file_path: file path of the file to check
    :return: bool depending of the file existence
    """
    if os.path.isfile(file_path):
        log_u.log_colorful(message=f"File {file_path} exists", color="LC")
        return True
    else:
        log_u.log_colorful(message=f"File {file_path} does not exists", color="r")
        return False

def load_pickle_file(file_path: str):
    """
    Load pickle data stored in file_path
    :param file_path:
    :return:
    """
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data