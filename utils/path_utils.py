import os

PATH_FILE = 'saved_paths.txt'

def save_path(key, path):
    with open(PATH_FILE, 'a') as file:
        file.write(f'{key}: {path}\n')

def load_paths():
    paths = {}
    if os.path.exists(PATH_FILE):
        with open(PATH_FILE, 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                paths[key] = value
    return paths
