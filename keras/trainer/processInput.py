import itertools
import pandas as pd
import json
import sys
import numpy as np

INPUT_SIZE = 100
OUTPUT_SIZE = 4

def read_by_tokens(fileobj):
  for line in fileobj:
    for token in line.split():
      yield token

def generator_input(input_file, chunk_size):
  with open(input_file[1]) as f:
    signals = [int(token) for token in read_by_tokens(f)]
  dataframe = pd.read_csv(open(input_file[0], 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
  expected = [None] * len(signals)
  itr = dataframe.iterrows()
  _, row = next(itr)
  end = start = row['prevSig']
  try:
    while end < len(expected):
      if end >= row['sig']:
        _, row = next(itr)
      if end >= row['prevSig']:
        expected[end] = row['gene']
      end += 1
  except StopIteration:
    pass
  expected = pd.get_dummies(expected)

  wing = int(INPUT_SIZE/2)
  while True:
    for i in range(max(start, wing), min(end-1, len(expected)-wing-chunk_size), chunk_size):
      inputSignals = [signals[i+j-wing:i+j+wing] for j in range(chunk_size)]
      ouputSignals = expected.iloc[range(i, i+chunk_size)]
      yield (np.expand_dims(np.array(inputSignals), axis=2), ouputSignals)


if __name__ == '__main__':
  gen = generator_input(['keras/data/propertyList.label', 'keras/data/signalFile.signal'], chunk_size=50)
  sample = next(gen)
  print(type(sample))
  print(sample[0].shape)
  print(sample[1].shape)
  print(type(sample[1]))
