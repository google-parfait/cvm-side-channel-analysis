# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to process traces and features into datasets suitable for ML and data analysis."""

import multiprocessing
from typing import Any, Callable, Iterator, Optional

from absl import logging
from sca import feature_filters
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def merge_handcrafted_features_into_dataset(
    features_files_list: Iterator[str],
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
    merge_batch_size: int = 500,
) -> pd.DataFrame:
  """Merges features from multiple files into a single pandas DataFrame.

  This function takes a list of file paths, reads the content of each file using
  a provided reader function, and merges the resulting data into a single
  pandas DataFrame. The merging process is batched to handle large number of
  files efficiently.

  Args:
      features_files_list: An iterator of file paths to be processed.
      file_reader: A callable function that reads a file and returns a
        dictionary or pandas DataFrame.
      file_reader_args: Optional arguments to pass to the file_reader function.
      merge_batch_size: The number of files to process in each batch.

  Returns:
      A pandas DataFrame containing the merged data from all files.
  """

  todo_files = list(features_files_list)

  logging.info('Merging %d files', len(todo_files))

  merge_batches = []
  for batch_idx in range(0, len(todo_files), merge_batch_size):
    logging.info('Batch %d/%d', batch_idx, len(todo_files) / merge_batch_size)
    batch_list = []
    for instance_idx in range(
        batch_idx, min(batch_idx + merge_batch_size, len(todo_files))
    ):
      df = (
          file_reader(todo_files[instance_idx], *file_reader_args)
          if file_reader_args
          else file_reader(todo_files[instance_idx])
      )
      batch_list.append(df)

    merge_batches.append(pd.concat(batch_list))
  features_df = pd.concat(merge_batches)

  return features_df


def balanced_subsample(
    y: pd.Series, size: float | int | None = None
) -> list[int]:
  """Subsamples a dataset to ensure a balanced representation of each class.

  This function takes a series of labels (y) and returns a list of indices
  that represent a balanced subsample of the original data. It ensures that
  each class in the labels is represented equally in the subsample.

  Args:
      y: A pandas Series representing the labels of the dataset.
      size: The desired total size of the subsampled dataset. If None, the size
        will be set to the minimum number of samples in any class.

  Returns:
      A list of indices representing the balanced subsample.
  """
  subsample = []

  if size is None:
    n_smp = y.value_counts().min()
  else:
    n_smp = int(size / len(y.value_counts().index))

  for label in y.value_counts().index:
    samples = y[y == label].index.values
    index_range = range(samples.shape[0])
    indexes = np.random.choice(index_range, size=n_smp, replace=False)
    subsample += samples[indexes].tolist()

  return subsample


def get_features_and_labels_for_dataset(
    ds_merged: pd.DataFrame,
    feature_filter: Callable[..., Any] | list[Callable[..., Any]] = None,
    training_set_size: float | int | None = None,
    testing_set_size: float | int | None = None,
    fill_nulls: bool | None = True,
    standardize_features: bool | None = True,
    keep_label_with_features: bool | None = False,
    ensure_balanced_dataset: bool | None = True,
    drop_instances_with_missing_features: float | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  """Prepares a dataset for machine learning by extracting features and labels.

  This function processes a merged dataset, applies feature filtering, splits
  data into training and testing sets, handles missing values, standardizes
  features, and ensures class balance.

  Args:
      ds_merged: A pandas DataFrame containing the merged dataset.
      feature_filter: Optional; A function or list of functions to filter
        features.
      training_set_size: Optional; The size of the training set. Can be a float
        (proportion) or an int (number of samples).
      testing_set_size: Optional; The size of the testing set. Can be a float
        (proportion) or an int (number of samples).
      fill_nulls: Optional; If True, fills missing values with 0.
      standardize_features: Optional; If True, standardizes features using
        StandardScaler.
      keep_label_with_features: Optional; If True, keeps the label column in the
        feature DataFrames.
      ensure_balanced_dataset: Optional; If True, ensures a balanced
        representation of each class in the training and testing sets.
      drop_instances_with_missing_features: Optional; If set, drops instances
        that do not have at least the specified proportion of features.
      random_seed: Optional; Sets the random seed for reproducibility.

  Returns:
      A tuple containing:
      - x_training: A pandas DataFrame with training features.
      - x_testing: A pandas DataFrame with testing features.
      - y_training: A pandas Series with training labels.
      - y_testing: A pandas Series with testing labels.
  """

  if random_seed is not None:
    np.random.seed(random_seed)

  feature_names = feature_filters.filter_features(ds_merged, feature_filter)
  feature_names_plus_label = feature_names + ['label']
  ds_merged = ds_merged[feature_names_plus_label]

  if drop_instances_with_missing_features is not None:
    num_features = len(ds_merged.columns) - 1
    num_initial_instances = len(ds_merged)
    num_expected_features = drop_instances_with_missing_features * num_features
    logging.info(
        'Dropping instances that do not have at least %d/%d features',
        num_expected_features,
        num_features,
    )
    instances_to_keep = (
        num_features - ds_merged.isna().sum(axis=1)
    ) >= num_expected_features
    ds_merged = ds_merged[instances_to_keep]
    logging.info(
        'Keeping %d/%d instances', len(ds_merged), num_initial_instances
    )

  num_testing_samples = 0
  if testing_set_size is not None:
    if isinstance(testing_set_size, float):
      assert testing_set_size <= 1.0 and testing_set_size >= 0.0
      num_testing_samples = int(testing_set_size * len(ds_merged))
    else:
      assert testing_set_size >= 0 and isinstance(testing_set_size, int)
      num_testing_samples = testing_set_size

  num_training_samples = len(ds_merged) - num_testing_samples
  if training_set_size is not None:
    if isinstance(training_set_size, float):
      assert training_set_size <= 1.0 and training_set_size >= 0.0
      num_training_samples = int(training_set_size * len(ds_merged))
    else:
      assert training_set_size > 0 and isinstance(training_set_size, int)
      num_training_samples = training_set_size

  assert num_training_samples + num_testing_samples <= len(
      ds_merged
  ), 'Training and testing overlap. Readjust sizes.'
  logging.info(
      'num_testing_samples: %d/%d', num_testing_samples, len(ds_merged)
  )
  logging.info(
      'num_training_samples: %d/%d', num_training_samples, len(ds_merged)
  )

  if ensure_balanced_dataset:
    logging.info('Ensuring balanced dataset')
    assert num_training_samples % 2 == 0 and num_testing_samples % 2 == 0
    y = pd.Series(LabelEncoder().fit_transform(ds_merged['label'])).rename(
        'label'
    )
    sampled_indices = balanced_subsample(y)
    ds_merged = ds_merged.iloc[sampled_indices]

  if num_testing_samples > 0:
    y = pd.Series(LabelEncoder().fit_transform(ds_merged['label'])).rename(
        'label'
    )
    ds_training, ds_testing = train_test_split(
        ds_merged,
        train_size=num_training_samples,
        test_size=num_testing_samples,
        random_state=random_seed,
        stratify=y,
    )
  else:
    ds_training, ds_testing = ds_merged, ds_merged

  y_training = pd.Series(
      LabelEncoder().fit_transform(ds_training['label'])
  ).rename('label')
  y_testing = pd.Series(
      LabelEncoder().fit_transform(ds_testing['label'])
  ).rename('label')

  if ensure_balanced_dataset:
    # ensure every label has the same number of instances
    assert len(set(y_training.value_counts())) == 1
    assert len(set(y_testing.value_counts())) == 1

  x_training = ds_training
  x_testing = ds_testing

  if fill_nulls:
    x_training = x_training.fillna(0)
    x_testing = x_testing.fillna(0)

  if standardize_features:
    std = StandardScaler()
    std.fit(x_training[feature_names])
    x_training[feature_names] = std.transform(x_training[feature_names])
    x_testing[feature_names] = std.transform(x_testing[feature_names])

  if len(set(y_training.values)) == 2:
    assert sum(x_training['label'] == y_training.values) == len(x_training)
    assert sum(x_testing['label'] == y_testing.values) == len(x_testing)
  if not keep_label_with_features:
    x_training = x_training.drop('label', axis=1)
    x_testing = x_testing.drop('label', axis=1)

  return x_training, x_testing, y_training, y_testing


def truncate_instance(instance: str, max_sequence_length: int) -> str:
  """Truncates an instance (string) to a maximum sequence length.

  This function takes a string and truncates it to contain at most
  `max_sequence_length` words, keeping the last words of the original string.

  Args:
      instance: The string to be truncated.
      max_sequence_length: The maximum number of words allowed in the truncated
        string.

  Returns:
      The truncated string.
  """
  words = instance.split()
  if len(words) > max_sequence_length:
    return ' '.join(words[-max_sequence_length:])
  return instance


def filter_instance_based_on_min_length(
    instance: str, min_sequence_length: int
) -> bool:
  """Filters an instance based on its length.

  This function checks if the number of words in the given instance is greater
  or equal than the minimum sequence length.

  Args:
      instance: A string representing the instance to be checked.
      min_sequence_length: The minimum number of words required in the instance.

  Returns:
      True if the instance has at least the minimum required length,
      False otherwise.
  """
  return len(instance.split()) >= min_sequence_length


def transform_instance_into_sequence(instance: str) -> str:
  """Transforms a list of words into a space-separated string.

  This function takes a list of words and returns a string where each word
  is separated by a space.

  Args:
      instance: A list of words (string).

  Returns:
      A string containing the words separated by spaces.
  """
  return ' '.join(instance)


def _merge_sequence_features_into_dataset_instance(
    features_file: str,
    stage: str,
    max_sequence_length: int,
    min_sequence_length: int,
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
) -> dict[str, Any] | None:
  """Helper function to merge sequence features from a single file into a dictionary.

  This function reads a single feature file, extracts sequence features for a
  given stage, truncates or filters sequences based on length, and transforms
  them into a space-separated string. The resulting data is stored in a
  dictionary containing the processed features, label, and file name.

  Args:
      features_file: The path of the feature file to be processed.
      stage: The stage to extract features from.
      max_sequence_length: The maximum length allowed for a sequence.
      min_sequence_length: The minimum length required for a sequence.
      file_reader: A callable function that reads a file and returns a
        dictionary containing features and labels.
      file_reader_args: Optional arguments to pass to the file_reader function.

  Returns:
      A dictionary containing the processed sequence features, label, and file
      name, or None if the sequence is shorter than min_sequence_length.
  """
  inst = (
      file_reader(features_file, *file_reader_args)
      if file_reader_args
      else file_reader(features_file)
  )
  features = '\n'.join(inst['features'][stage])
  features = truncate_instance(features, max_sequence_length)
  if not filter_instance_based_on_min_length(features, min_sequence_length):
    return None
  inst = {
      'features': features,
      'label': inst['label'],
      'file_name': features_file,
  }
  return inst


def merge_sequence_features_into_dataset(
    features_files_list: Iterator[str],
    stage: str,
    max_sequence_length: int,
    min_sequence_length: int,
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
) -> list[dict[str, Any]]:
  """Merges sequence features from multiple files into a list of dictionaries.

  This function processes a list of feature files, reads sequence features from
  each file, truncates or filters sequences based on length, and transforms
  them into a space-separated string. The resulting data is stored in a list
  of dictionaries, each containing the processed features, label, and file name.

  Args:
      features_files_list: An iterator of file paths to be processed.
      stage: The stage to extract features from.
      max_sequence_length: The maximum length allowed for a sequence.
      min_sequence_length: The minimum length required for a sequence.
      file_reader: A callable function that reads a file and returns a
        dictionary containing features and labels.
      file_reader_args: Optional arguments to pass to the file_reader function.

  Returns:
      A list of dictionaries, where each dictionary contains the processed
      sequence features, label, and file name.
  """

  with multiprocessing.pool.ThreadPool() as pool:
    instances = pool.map(
        lambda x: _merge_sequence_features_into_dataset_instance(
            x,
            stage,
            max_sequence_length,
            min_sequence_length,
            file_reader,
            file_reader_args,
        ),
        features_files_list,
    )

  instances = [x for x in instances if x is not None]

  return instances
