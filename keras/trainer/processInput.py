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
    labelMap = dict({
      "a":'A', "A":'A',
      "c":'C', "C":'C',
      "g":'G', "G":'G',
      "t":'T', "T":'T',
      })
    expected = [None] * len(signals)
    itr = dataframe.iterrows()
    try:
      _, row = next(itr)
      for i in range(0, len(expected)):
        while i >= row['sig'] or row['gene'] not in labelMap:
          _, row = next(itr)
        if i >= row['prevSig']:
          expected[i] = labelMap[row['gene']]
    except StopIteration:
      pass
    return signals, expected

def generator_input_record(input_file):
  for dataSet in range(0, len(input_file), 2):
    downloadFile(input_file[dataSet])
    downloadFile(input_file[dataSet+1])
    signals, expected = readAndParseFile(input_file[dataSet+1], input_file[dataSet])
    expectedDummy = pd.get_dummies(expected).as_matrix()
    for x in range(wing, len(expected)-wing):
      if expected[x] != None and expected[x-excludeEdge] == expected[x] and expected[x-excludeEdge] == expected[x]:
        yield signals[x-wing:x+wing], expectedDummy[x]

def generator_input_chunk(input_file, chunk_size):
  inputList =[]
  outputList = []
  for input, output in generator_input_record(input_file):
    inputList.append(input)
    outputList.append(output)
    if len(inputList) == chunk_size:
      yield (np.array(inputList), np.array(outputList))
      inputList =[]
      outputList = []
  yield (np.array(inputList), np.array(outputList))  

def generator_input(input_file, chunk_size):
  while True:
    try:
      for inputList, outputList in generator_input_chunk(input_file, chunk_size):
        yield inputList, outputList
    except Exception as e:
      print(e)

if __name__ == '__main__':
  train = file_io.list_directory('gs://chiron-data-fred/171016_large/train')
  val = file_io.list_directory('gs://chiron-data-fred/171016_large/val')
  files = ['train/'+s for s in train] + ['val/'+s for s in val]
  print("Found " + str(len(files)) + " data files")
  chunk_size = 100
  for input, output in generator_input_chunk(files, chunk_size=chunk_size):
    assert input.shape[1] == INPUT_SIZE
    assert output.shape[1] == OUTPUT_SIZE
    assert len(input.shape) == 2
    assert len(output.shape) == 2
    assert input.shape[0] == chunk_size
    assert output.shape[0] == chunk_size
