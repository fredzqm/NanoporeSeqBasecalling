import itertools
import pandas as pd
import json
import sys

# csv columns in the input file
# CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
#                'marital_status', 'occupation', 'relationship', 'race',
#                'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
#                'native_country', 'income_bracket')

# CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''],
#                        [''], [0], [0], [0], [''], ['']]

# # Categorical columns with vocab size
# # native_country and fnlwgt are ignored
# CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
#                     ('relationship', 6), ('workclass', 9), ('occupation', 15),
#                     ('gender', [' Male', ' Female']), ('race', 5))

# CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
#                    'hours_per_week')

# LABELS = [' <=50K', ' >50K']
# LABEL_COLUMN = 'income_bracket'

# UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
#     zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

# def to_numeric_features(features):
#   """Convert the pandas input features to numeric values.
#      Args:
#         features: Input features in the data
#           age (continuous)
#           workclass (categorical)
#           fnlwgt (continuous)
#           education (categorical)
#           education_num (continuous)
#           marital_status (categorical)
#           occupation (categorical)
#           relationship (categorical)
#           race (categorical)
#           gender (categorical)
#           capital_gain (continuous)
#           capital_loss (continuous)
#           hours_per_week (continuous)
#           native_country (categorical)
#   """

#   for col in CATEGORICAL_COLS:
#     features = pd.concat([features, pd.get_dummies(features[col[0]], drop_first = True)], axis = 1)
#     features.drop(col[0], axis = 1, inplace = True)

#   # Remove the unused columns from the dataframe
#   for col in UNUSED_COLUMNS:
#     features.pop(col)

#   return features

def generator_input(input_file, chunk_size):
  """Generator function to produce features and labels
     needed by keras fit_generator.
  """
  input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
                           names=CSV_COLUMNS,
                           chunksize=chunk_size,
                           na_values=" ?")

  for input_data in input_reader:
    input_data = input_data.dropna()
    label = pd.get_dummies(input_data.pop(LABEL_COLUMN))

    input_data = to_numeric_features(input_data)
    n_rows = input_data.shape[0]
    return ((input_data.iloc[[index % n_rows]], label.iloc[[index % n_rows]]) for index in itertools.count() )


def generator_input(input_file, chunk_size):
  dataframe = pd.read_csv(open(input_file[0], 'r'), names=['prevSig', 'sig', 'gene'], delim_whitespace=True)
  genes = pd.get_dummies(dataframe['gene'])
  dataframe.drop('gene', axis = 1, inplace = True)
  dataframe.drop('prevSig', axis = 1, inplace = True)
  NUM_INPUT = 10
  for i in range(NUM_INPUT, dataframe.shape[0]):
    yield (dataframe.iloc[i-NUM_INPUT:i+1,:], genes.iloc[i,:])


if __name__ == '__main__':
  gen = generator_input(['keras/data/probertylist.label', 'keras/data/signalFile.signal'], chunk_size=5000)
  sample = gen.next()

  print(type(sample))
  print(sample)
