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

"""Methods to train and evaluate ML models."""

import copy
from typing import Any, Callable, Optional
from absl import logging
from sca import data_analysis
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

copy = copy.copy


def train_and_test_classifier(
    clf: Any,
    x_training: pd.DataFrame,
    x_testing: pd.DataFrame,
    y_training: pd.Series,
    y_testing: pd.Series,
    num_random_trials: int = 5,
) -> tuple[float, float]:
  """Trains and tests a classifier, returning its accuracy and a random baseline.

  Args:
      clf: The classifier to be trained and tested.
      x_training: The training features.
      x_testing: The testing features.
      y_training: The training labels.
      y_testing: The testing labels.
      num_random_trials: The number of times to repeat the random baseline
        calculation.

  Returns:
      A tuple containing the mean random accuracy and the mean classifier
      accuracy.
  """
  mean_random_accuracy = 0.0

  for _ in range(num_random_trials):
    mean_random_accuracy += accuracy_score(y_testing, y_testing.sample(frac=1))
  mean_random_accuracy = mean_random_accuracy / num_random_trials
  logging.info('Random baseline accuracy: %.2f\n', mean_random_accuracy)

  clf.fit(x_training, y_training)

  if len(set(y_testing.values)) == 2:
    mean_accuracy = clf.score(x_testing, y_testing)
  else:
    y_pred = clf.predict(x_testing)
    mean_accuracy = balanced_accuracy_score(y_testing, y_pred)
  logging.info('Mean classifier accuracy: %.2f\n', mean_accuracy)

  return (
      mean_random_accuracy,
      mean_accuracy,
  )


def train_and_test_classifier_across_datasets_and_feature_sets(
    dataset_list: list[tuple[str, str]],
    feature_filter_list: list[Callable[..., Any]],
    feature_sets_list: list[Callable[..., Any]],
    num_trials: int,
    training_set_size: int | float,
    testing_set_size: int | float,
    drop_instances_with_missing_features: bool,
    ensure_balanced_dataset: bool,
    clf_constructor: Callable[..., Any],
    file_reader: Callable[..., pd.DataFrame],
    clf_constructor_args: Optional[Any] = None,
    file_reader_args: Optional[Any] = None,
    random_seed: int = 0,
) -> list[dict[str, dict[str, dict[str, float]]]]:
  """Trains and tests a classifier across multiple datasets and feature sets and across multiple trials.

  Args:
      dataset_list: A list of tuples, where each tuple contains the name of a
        dataset and the path to its features file.
      feature_filter_list: A list of functions that filter the features of a
        dataset.
      feature_sets_list: A list of functions that define different feature sets
        to be used for training.
      num_trials: The number of times to repeat the training and testing
        process.
      training_set_size: The size of the training set.
      testing_set_size: The size of the testing set.
      drop_instances_with_missing_features: Whether to drop instances with
        missing features.
      ensure_balanced_dataset: Whether to ensure a balanced dataset.
      clf_constructor: The constructor of the classifier to be used.
      file_reader: The function to read the features file.
      clf_constructor_args: The arguments to be passed to the classifier
        constructor.
      file_reader_args: The arguments to be passed to the file reader.
      random_seed: The seed for the random number generator.

  Returns:
      A list of dictionaries, where each dictionary contains the results of a
      single trial.
  """
  np.random.seed(random_seed)
  results_dict_list = []
  for idx in range(0, num_trials):
    logging.info('\n\nTRIAL NUMBER %d', idx)
    results_dict = {}

    for feature_filter_instance in feature_sets_list:
      logging.info(
          'feature_filter_instance: %s',
          (
              getattr(feature_filter_instance, '__name__', 'Unnamed')
              if feature_filter_instance
              else 'all'
          ),
      )
      for dataset, dataset_features_path in dataset_list:

        logging.info(
            'dataset = %s; feature_filter_instance = %s',
            dataset,
            (
                getattr(feature_filter_instance, '__name__', 'Unnamed')
                if feature_filter_instance
                else 'all'
            ),
        )

        features_df = (
            file_reader(dataset_features_path, *file_reader_args)
            if file_reader_args
            else file_reader(dataset_features_path)
        )

        feature_set_version_str = '%s' % (
            getattr(feature_filter_instance, '__name__', 'Unnamed')
            if feature_filter_instance
            else 'all'
        )

        logging.info('All v%s features:', feature_set_version_str)
        list_of_feature_filters = copy(feature_filter_list)
        if feature_filter_instance:
          list_of_feature_filters.append(feature_filter_instance)

        x_training, x_testing, y_training, y_testing = (
            data_analysis.get_features_and_labels_for_dataset(
                features_df,
                feature_filter=list_of_feature_filters,
                training_set_size=training_set_size,
                testing_set_size=testing_set_size,
                drop_instances_with_missing_features=drop_instances_with_missing_features,
                ensure_balanced_dataset=ensure_balanced_dataset,
                random_seed=idx,
            )
        )

        assert len(x_training) == len(y_training)
        if isinstance(testing_set_size, int):
          assert len(x_training) == training_set_size
        assert len(x_testing) == len(y_testing)
        if isinstance(testing_set_size, int):
          assert len(x_testing) == testing_set_size
        if isinstance(clf_constructor, type(LogisticRegression)):
          clf = LogisticRegression(random_state=idx, **clf_constructor_args)
        else:
          raise ValueError(
              'Unsupported classifier constructor: %s' % clf_constructor
          )
        random_acc, clf_acc = train_and_test_classifier(
            clf, x_training, x_testing, y_training, y_testing
        )
        if results_dict.get(feature_set_version_str, None) is None:
          results_dict[feature_set_version_str] = {}
        results_dict[feature_set_version_str][dataset] = {
            'mean_accuracy': clf_acc,
            'random_accuracy': random_acc,
        }
        logging.info('--------------------------\n')

    results_dict_list.append(results_dict)
  return results_dict_list


def compute_advantage_for_results(
    results_dict_list: list[dict[str, dict[str, dict[str, float]]]],
    num_trials: int,
) -> dict[str, pd.DataFrame]:
  """Computes the attacker advantage of a model over random guessing.

  Args:
    results_dict_list: A list of dictionaries, where each dictionary contains
      the results of a single trial.
    num_trials: The number of trials that were run.

  Returns:
    A dictionary of dataframes, where each dataframe contains the advantage
    of each feature set version over random guessing, for each dataset.
    Each dictionary has the following structure:
    {
        feature_set_version_str: pd.DataFrame
    }
    Where the pd.DataFrame has the following columns:
        'mean_accuracy': float,
        'random_accuracy': float,
        'relative_advantage': float,
        'max_advantage': float,
        'advantage': float
  """
  advantage_df_dict = {}
  for feature_set_version in results_dict_list[0].keys():
    df_results_dict_sum = None
    for results_dict in results_dict_list:
      df_results_dict = pd.DataFrame(
          results_dict[feature_set_version]
      ).transpose()
      if df_results_dict_sum is None:
        df_results_dict_sum = df_results_dict
      else:
        df_results_dict_sum += df_results_dict
    df_results_dict_sum /= num_trials

    df_results_dict_sum['relative_advantage'] = abs(
        df_results_dict_sum['mean_accuracy']
        - df_results_dict_sum['random_accuracy']
    )
    df_results_dict_sum['max_advantage'] = (
        1.0 - df_results_dict_sum['random_accuracy']
    )
    df_results_dict_sum['advantage'] = (
        df_results_dict_sum['relative_advantage']
        / df_results_dict_sum['max_advantage']
    )
    advantage_df_dict[feature_set_version] = df_results_dict_sum
  return advantage_df_dict
