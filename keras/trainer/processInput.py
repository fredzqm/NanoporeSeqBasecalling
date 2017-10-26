import itertools
import pandas as pd
import json
import sys
import numpy as np

def read_by_tokens(fileobj):
  for line in fileobj:
    for token in line.split():
      yield token

def generator_input(input_file, chunk_size):
  with open(input_file[1]) as f:
    signals = [int(token) for token in read_by_tokens(f)]

  dataframe = pd.read_csv(open(input_file[0], 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
  genes = pd.get_dummies(dataframe['gene'])
  dataframe.drop('gene', axis = 1, inplace = True)
  dataframe.drop('prevSig', axis = 1, inplace = True)
  print(len(signals))
  print(dataframe.shape)
  print(dataframe['sig'][dataframe.shape[0]-1])
  NUM_INPUT = 10
  while True:
    for i in range(NUM_INPUT, dataframe.shape[0]):
      index = dataframe['sig'][i]
      yield (np.array([signals[index-NUM_INPUT:index+1]]), genes.iloc[[i]])


if __name__ == '__main__':
  gen = generator_input(['keras/data/propertyList.label', 'keras/data/signalFile.signal'], chunk_size=5000)
  sample = next(gen)
  print(type(sample))
  print(sample[0].shape)
  print(sample[1].shape)
