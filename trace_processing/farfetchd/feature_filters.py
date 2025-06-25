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

"""Methods to extract subsets of features from a dataset."""

import types
from typing import Any, Callable
from absl import logging
import pandas as pd


def filter_features(
    features_df: pd.DataFrame, feature_filter: Callable[..., Any] | None = None
) -> list[str]:
  """Extracts subsets of features from a dataset based on the provided filter.

  Args:
    features_df: Pandas DataFrame containing the features.
    feature_filter: A function or a list of functions to filter the features. If
      None, all features will be returned.

  Returns:
    A list of feature names that passed the filter.
  """
  feature_names = _get_feature_names(features_df)
  if feature_filter is None:
    return feature_names
  if isinstance(feature_filter, list):
    for ff in feature_filter:
      feature_names = ff(feature_names)
  elif isinstance(feature_filter, types.FunctionType):
    feature_names = feature_filter(feature_names)
  logging.info('Kept features: %s', feature_names)
  return feature_names


def _get_feature_names(features_df: pd.DataFrame) -> list[str]:
  """Extracts the names of the features from the DataFrame.

  Args:
    features_df: Pandas DataFrame containing the features.

  Returns:
    A list of feature names, excluding the 'label' column.
  """
  return list(set(features_df.columns).difference(set(['label'])))


def feature_filter_f1(feature_names: list[str]) -> list[str]:
  """Filters features that start with 'f1'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that start with 'f1'.
  """
  return list(filter(lambda x: x[:2] == 'f1', feature_names))


def feature_filter_f2(feature_names: list[str]) -> list[str]:
  """Filters features that start with 'f2'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that start with 'f2'.
  """
  return list(filter(lambda x: x[:2] == 'f2', feature_names))


def feature_filter_f3(feature_names: list[str]) -> list[str]:
  """Filters features that start with 'f3'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that start with 'f3'.
  """
  return list(filter(lambda x: x[:2] == 'f3', feature_names))


def feature_filter_f4(feature_names: list[str]) -> list[str]:
  """Filters features that start with 'f4'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that start with 'f4'.
  """
  return list(filter(lambda x: x[:2] == 'f4', feature_names))


def feature_filter_f5(feature_names: list[str]) -> list[str]:
  """Filters features that start with 'f5'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that start with 'f5'.
  """
  return list(filter(lambda x: x[:2] == 'f5', feature_names))


def feature_filter_accumulate_features(feature_names: list[str]) -> list[str]:
  """Filters features that contain 'accumulate_'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that contain 'accumulate_'.
  """
  return list(filter(lambda x: 'accumulate_' in x, feature_names))


def feature_filter_report_features(feature_names: list[str]) -> list[str]:
  """Filters features that contain 'report_'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that contain 'report_'.
  """
  return list(filter(lambda x: 'report_' in x, feature_names))


def feature_filter_no_hc_cache_features(feature_names: list[str]) -> list[str]:
  """Filters features that do not contain '_hc'.

  Args:
    feature_names: A list of feature names.

  Returns:
    A list of feature names that do not contain '_hc'.
  """
  return list(filter(lambda x: '_hc' not in x, feature_names))
