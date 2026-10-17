import getpass
import os
import sys

__USERNAME = getpass.getuser()
# _BASE_DIR = f'/home/joo/Cleanse'
_BASE_DIR = "..." # set your directory
# MODEL_PATH = f'{_BASE_DIR}/llama-7b/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'minsuh-dataset') # change to your directory
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'minsuh-output') # change to your directory
os.makedirs(GENERATION_FOLDER, exist_ok=True)

