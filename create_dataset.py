#!/usr/bin/env python

"""
Parse all files and write to a single file
"""
import re
import string
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePath
from typing import List, NamedTuple, Set
from typing import Optional

import numpy as np
from labml import lab, monit
from labml import logger
from labml.internal.util import rm_tree

PRINTABLE = set(string.printable)


class PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


# def get_python_files(source_path):
#     """
#     Get list of python files and their paths inside `data/source` folder
#     """
#
#     # source_path = Path(lab.get_data_path() / 'source')
#     files: List[PythonFile] = []
#
#     def _add_file(path: Path):
#         """
#         Add a file to the list of tiles
#         """
#         project = path.relative_to(source_path).parents
#         relative_path = path.relative_to(source_path / project[len(project) - 3])
#
#         files.append(PythonFile(relative_path=str(relative_path),
#                                 project=str(project[len(project) - 2]),
#                                 path=path))
#
#     def _collect_python_files(path: Path):
#         """
#         Recursively collect files
#         """
#         for p in path.iterdir():
#             if p.is_dir():
#                 _collect_python_files(p)
#             else:
#                 _add_file(p)
#
#     _collect_python_files(source_path)
#
#     return files


def get_python_files(file_dir):
    file_list = []
    for file in file_dir.glob("**/*.py"):
        file_list.append(file)
    return file_list


def read_file(path: Path) -> str:
    """
    Read a file
    """
    with open(str(path), encoding='iso-8859-1') as f:
        content = f.read()

    content = ''.join(filter(lambda x: x in PRINTABLE, content))

    return content


def concat_and_save(path: PurePath, source_files: List[PythonFile]):
    with open(str(path), 'w') as f:
        for i, source in monit.enum(f"Write {path.name}", source_files):
            # f.write(f"# PROJECT: {source.project} FILE: {str(source.relative_path)}\n")
            # f.write(read_file(source.path) + "\n")
            f.write(read_file(source) + "\n")


def create_folders():
    path = Path(lab.get_data_path() / 'download')
    if not path.exists():
        path.mkdir(parents=True)
    source = Path(lab.get_data_path() / 'source')

    if not source.exists():
        source.mkdir(parents=True)


def extract_zip(file_path: Path, overwrite: bool = False):
    source = Path(lab.get_data_path() / 'source')

    with monit.section(f"Extract {file_path.stem}"):
        repo_source = source / file_path.stem
        if repo_source.exists():
            if overwrite:
                rm_tree(repo_source)
            else:
                return repo_source
        try:
            with zipfile.ZipFile(file_path, 'r') as repo_zip:
                repo_zip.extractall(repo_source)
        except zipfile.BadZipfile as e:
            print(file_path, e)

        return repo_source


def remove_files(path: Path, keep: Set[str]):
    """
    Remove files
    """

    for p in path.iterdir():
        if p.is_symlink():
            p.unlink()
            continue
        if p.is_dir():
            remove_files(p, keep)
        else:
            if p.suffix not in keep:
                p.unlink()


def main():
    try:
        create_folders()
        # batch()
    except KeyboardInterrupt:
        pass

    train_files = get_python_files(Path(lab.get_data_path() / 'pyart/train'))
    valid_files = get_python_files(Path(lab.get_data_path() / 'pyart/valid'))
    # files = get_python_files(lab.get_data_path() / 'source')
    # sample_files = get_python_files(Path(lab.get_data_path() / 'sample'))

    # np.random.shuffle(files)
    # np.random.shuffle(sample_files)

    # logger.inspect(source_files)

    # train_valid_split = int(len(files) * 0.8)
    concat_and_save(lab.get_data_path() / 'train.py', train_files)
    concat_and_save(lab.get_data_path() / 'valid.py', valid_files)
    # concat_and_save(lab.get_data_path() / 'sample.py', sample_files[int(len(sample_files) * 0.1):])


if __name__ == '__main__':
    main()
