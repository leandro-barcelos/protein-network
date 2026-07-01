import os


def create_dir(path: str):
    os.makedirs(path, exist_ok=True)
