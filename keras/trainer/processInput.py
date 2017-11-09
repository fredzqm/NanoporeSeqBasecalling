import itertools
import pandas as pd
import json
import sys
import numpy as np
import os.path
# from google.cloud import storage
from tensorflow.python.lib.io import file_io

# client = storage.Client(project='NanoporeSequence')
# bucket = client.get_bucket(BUCKET)
# encryption_key = 'e62a27da54254884117476c7e5cbd489aa6ddb21'

def downloadFile(file):
  with file_io.FileIO('gs://chiron-data-fred/171016_large/'+file, mode='r') as input_f:
    with file_io.FileIO(file, mode='w+') as output_f:
        output_f.write(input_f.read())

# def downloadFile(file):
#   if not os.path.isfile(file):
#     blob = storage.Blob(DATA_PATH + file, bucket, encryption_key=encryption_key)
#     with open(file, 'wb') as file_obj:
#         blob.download_to_file(file_obj)
    
# Instantiates a client

wing = 50
INPUT_SIZE = wing*2
OUTPUT_SIZE = 4

def read_by_tokens(fileobj):
  for line in fileobj:
    for token in line.split():
      yield token

os.makedirs('train')
os.makedirs('test')

def generator_input(input_file, chunk_size):
  input_file = ['train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label',
                'train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal']
  while True:
    for dataSet in range(0, len(input_file), 2):
      downloadFile(input_file[dataSet])
      downloadFile(input_file[dataSet+1])
      with open(input_file[dataSet+1]) as f:
        signals = [int(token) for token in read_by_tokens(f)]
      dataframe = pd.read_csv(open(input_file[dataSet], 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
      # preprocess input
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
      # print(signals, expected)
      # generate chunks
      for i in range(max(start, wing), min(end, len(expected)-wing-chunk_size), chunk_size):
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
