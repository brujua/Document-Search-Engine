from typing import List
from os import listdir
from os.path import isdir


def get_files(path: str) -> List[str]:
    """ recursively finds all the files within a directory and returns their path in a list.
    :param path: the path of the directory to do the search
    :return a list of strings corresponding to the path of each file in the specified path
    """
    if not isdir(path):
        return [path]  # its expected to return a list each time even if its a single element
    return [file for fileOrDir in listdir(path) for file in get_files(path + '/' + fileOrDir)]
    # return list of each file returned by the recursive call getFiles(fileOrDir) on
    # each fileOrDir in listdir(path)
