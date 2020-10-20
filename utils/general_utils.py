import os
import sys
from loguru import logger


# From https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def flatten_dict(dd, separator='_', prefix=''):
    """
    Flattens a dict of dicts to a single level dict.
    """
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}

def check_folder_existence(folder, create=False):
    """
    Checks if folder exists. If create is True, if it does not exist, it creates it
    :param folder:
    :param create:
    :return:
    """
    if not os.path.isdir(folder):
        if create is True:
            create_directory(folder)
            return True
        else:
            return False
    else:
        return True

def create_directory(folder):
    try:
        os.makedirs(folder)
    except OSError:
        logger.error(f"Creation of directory {folder} failed")
    else:
        logger.success(f"Successfully created the directory {folder}")