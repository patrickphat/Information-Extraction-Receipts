# This module helps save and load arbitrary objects

import pickle

def pkl_save(path,obj):

  # Input: path/to/the/file.pkl (str), obj (any type)

  with open(path, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pkl_load(path):
    
  # Input: path/to/the/pkl/file.pkl (str)
  # Output: obj (any type)

  with open(path, 'rb') as handle:
      return pickle.load(handle)