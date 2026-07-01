import os


def create_dir(path: str):
    os.makedirs(path, exists_ok=True)
