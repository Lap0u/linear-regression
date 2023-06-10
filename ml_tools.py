import os
from os import path
from pathlib import Path

def isValidPath(filePath):
  if (path.isfile(filePath) == False):
    raise Exception('File does not exist')
  if (os.access(filePath, os.R_OK)) == False:
    raise Exception('File is not readable')
  if (Path(filePath).suffix != '.csv'):
    raise Exception('File is not a csv file')


def normalize_array(X):
  return (X - X.min()) / (X.max() - X.min())