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

"""Various helper functions for local file I/O."""

import json
import os
import pickle
from typing import Any
from typing import Iterator


def local_file_reader(file_path: str) -> str:
  """Reads data from a local file.

  Args:
    file_path: The path to the file to read from.

  Returns:
    The data read from the file.
  """
  with open(file_path, "r") as reader:
    return reader.read()


def local_json_file_reader(file_path: str) -> dict[str, Any]:
  """Reads data from a local file in JSON format.

  Args:
    file_path: The path to the file to read from.

  Returns:
    The data read from the file.
  """
  return json.loads(local_file_reader(file_path))


def local_json_file_writer(file_path: str, data: Any) -> None:
  """Writes data to a local file in JSON format.

  Args:
    file_path: The path to the file to write to.
    data: The data to write.
  """
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  with open(file_path, "w") as writer:
    json.dump(data, writer)


def local_pickle_file_reader(file_path: str) -> dict[str, Any]:
  """Reads data from a local file in pickle format.

  Args:
    file_path: The path to the file to read from.

  Returns:
    The data read from the file.
  """
  with open(file_path, "rb") as reader:
    return pickle.load(reader)


def local_pickle_file_writer(file_path: str, data: Any) -> None:
  """Writes data to a local file in pickle format.

  Args:
    file_path: The path to the file to write to.
    data: The data to write.
  """
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  with open(file_path, "wb") as writer:
    pickle.dump(data, writer)


def filter_textfiles_from_dataset(
    dataset: Iterator[str],
    dataset_path: str,
    filename_filter_include_set: set[str] | None = None,
    filename_filter_string: str | None = None,
    filename_filter_range: list[int] | None = None,
    string_separator: str = "_",
    example_index_in_string: int = 1,
) -> Iterator[str]:
  """Filters text files from a dataset based on filename criteria.

  Args:
    dataset: An iterator of filenames.
    dataset_path: The path to the dataset.
    filename_filter_include_set: A set of filenames to include. If None, all
      files are included.
    filename_filter_string: A string to filter filenames by. If None, no
      filtering is done based on string.
    filename_filter_range: A list of two integers representing the start and end
      indices of the range to filter filenames by. If None, no filtering is done
      based on range.
    string_separator: The string used to separate parts of the filename.
    example_index_in_string: The index of the example ID in the filename when
      split by the string_separator.

  Yields:
    An iterator of filtered file paths.
  """
  for filename in dataset:
    if (
        filename_filter_include_set is not None
        and filename not in filename_filter_include_set
    ):
      continue
    if (
        filename_filter_string is not None
        and filename_filter_string not in filename
    ):
      continue
    if filename_filter_range is not None:
      example_idx = int(
          filename.split("/")[-1].split(string_separator)[
              example_index_in_string
          ]
      )
      if filename_filter_range[0] and example_idx < filename_filter_range[0]:
        continue
      if filename_filter_range[1] and example_idx >= filename_filter_range[1]:
        continue
    yield os.path.join(dataset_path, filename)
