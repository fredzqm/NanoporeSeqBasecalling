import itertools
import pandas as pd
import json
import sys
import numpy as np

def generator_input(input_file, chunk_size):
  dataframe = pd.read_csv(open(input_file[0], 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
  genes = pd.get_dummies(dataframe['gene'])
  dataframe.drop('gene', axis = 1, inplace = True)
  dataframe.drop('prevSig', axis = 1, inplace = True)
  NUM_INPUT = 10
  while True:
    for i in range(NUM_INPUT, dataframe.shape[0]):
      yield (np.array(dataframe.iloc[i-NUM_INPUT:i+1,:]).transpose(), genes.iloc[[i]])


if __name__ == '__main__':
  gen = generator_input(['keras/data/probertylist.label', 'keras/data/signalFile.signal'], chunk_size=5000)
  sample = gen.next()
  print(type(sample))
  print(sample)
