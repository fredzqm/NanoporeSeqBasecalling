import itertools
import pandas as pd
import json
import sys
import numpy as np
import os.path
from tensorflow.python.lib.io import file_io

INPUT_DIR = 'gs://chiron-data-fred/171016_large/'

def copy_file_to(src, dest):
  with file_io.FileIO(src, mode='r') as input_f:
    with file_io.FileIO(dest, mode='w') as output_f:
        output_f.write(input_f.read())

def downloadFile(file):
  if not os.path.exists(file):
    print("downloading... " + file)
    copy_file_to(INPUT_DIR+file, file)
    print("downloaded: " + file)

wing = 150
excludeEdge = 2
INPUT_SIZE = wing*2
OUTPUT_SIZE = 4

def read_by_tokens(fileobj):
  for line in fileobj:
    for token in line.split():
      yield token

try:
  os.makedirs('train')
  os.makedirs('val')
except Exception:
  pass

def readAndParseFile(signal, label):
  with open(signal) as f:
    signals = [int(token) for token in read_by_tokens(f)]
    dataframe = pd.read_csv(open(label, 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
    # preprocess input
    expected = [None] * len(signals)
    itr = dataframe.iterrows()
    try:
      _, row = next(itr)
      end = start = row['prevSig']
      while end < len(expected):
        if end >= row['sig']:
          _, row = next(itr)
        if end >= row['prevSig']:
          expected[end] = row['gene']
        end += 1
    except StopIteration:
      pass
    return signals, expected, start, end

def generator_input(input_file, chunk_size):
  while True:
    try:
      for dataSet in range(0, len(input_file), 2):
        downloadFile(input_file[dataSet])
        downloadFile(input_file[dataSet+1])
        signals, expected, start, end = readAndParseFile(input_file[dataSet+1], input_file[dataSet])
        def filterRange(i, j):
          for x in range(i, min(j, len(expected)-wing)):
            if expected[x-excludeEdge] == expected[x] and expected[x-excludeEdge] == expected[x]:
              yield x
        expectedDummy = pd.get_dummies(expected)
        for i in range(max(start, wing), min(end, len(expected)-wing), chunk_size):
          inputSignals = [signals[j-wing:j+wing] for j in filterRange(i, i+chunk_size)]
          ouputSignals = expectedDummy.iloc[[j for j in filterRange(i, i+chunk_size)]]
          yield (np.expand_dims(np.array(inputSignals), axis=2), ouputSignals)
    except Exception as e:
      print(e)

if __name__ == '__main__':
  signals, expected = readAndParseFile('keras/data/signalFile.signal', 'keras/data/propertyList.label')
  print(expected)
