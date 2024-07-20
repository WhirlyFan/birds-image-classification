import os
import torch
from pathlib import Path


def count_subdirectories(path: Path) -> int:
    """
    Counts the number of subdirectories in a given directory.
    """
    subdirectories = [f for f in path.iterdir() if f.is_dir()]
    return len(subdirectories)

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

class Config:
    # specify the paths to datasets
    ROOT_DIR = Path('data')
    TRAIN_DIR = ROOT_DIR / 'train'
    TEST_DIR = ROOT_DIR / 'test'
    VALID_DIR = ROOT_DIR / 'valid'

    # set the input height and width
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = count_subdirectories(TRAIN_DIR)
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    NUM_WORKERS = os.cpu_count()
    SUBSET_FRACTION = .1
    MODEL_STATE_DICT_PATH = ROOT_DIR / 'efficientb0_birds.pth'
