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

"""Tools to process traces and extract useful representations."""

import re
from typing import Any, Callable, Iterator, Optional
from absl import logging

trace_line_pattern_dict = {
    "PMC": {
        "pattern": (
            r"PMCs: (?P<num1>\d+) (?P<num2>\d+) (?P<num3>\d+) (?P<num4>\d+)"
            r" (?P<num5>\d+) (?P<num6>\d+)"
        ),
        "name": "PMC",
    },
    "CF": {
        "pattern": (
            r"(?P<page_type>CF)\s+(?P<guest_page_address>\w+)\s+npf_num:(?P<code_fetch_count>\d+)"
        ),
        "name": "CF",
    },
    "CI_BK": {
        "pattern": (
            r"(?P<ma_page_number>\w+) ci_bk_(?P<ci_block_number>\w+)"
            r" B:(?P<B_2bytes_ciphertext>\w+) A:(?P<A_2bytes_ciphertext>\w+)"
        ),
        "name": "CI_BK",
    },
    "MA": {
        "pattern": r"(?P<page_type>MA)\s+(?P<guest_page_address>\w+)",
        "name": "MA",
    },
    "MA_CACHE": {
        "pattern": (
            r"MA_(?P<guest_page_address>\w+):\s*CL(?P<to_parse>(\s\d+_\d+)+)"
        ),
        "name": "MA_CACHE",
    },
}


def read_dataset(
    files_list: Iterator[str],
    file_reader: Callable[..., str],
    file_reader_args: Optional[Any] = None,
    log_every_n: int = 20,
) -> Iterator[dict[str, Any]]:
  """Reads a dataset of files and yields their content.

  Args:
    files_list: An iterator of file paths to read.
    file_reader: A function that takes a file path and arguments and returns the
      content of the file as a string.
    file_reader_args: Additional arguments to pass to the file_reader function.
    log_every_n: The interval at which to log the number of files read.

  Yields:
    An iterator of dictionaries, where each dictionary represents a file and
    contains the "file_name" and "file_content" keys.
  """
  num_files = 0
  for example_file in files_list:
    yield {
        "file_name": example_file,
        "file_content": (
            file_reader(example_file, *file_reader_args)
            if file_reader_args
            else file_reader(example_file)
        ),
    }
    num_files += 1
    if num_files % log_every_n == 0:
      logging.info("[read_dataset] # files processed: %d", num_files)
  logging.info("[read_dataset] Done. Total # files processed: %d", num_files)


def extract_dataset_labels_from_filename(
    dataset: Iterator[dict[str, Any]],
    string_separator: str,
    label_index_in_string: int,
    label_is_numeric: bool = True,
    log_every_n: int = 20,
) -> Iterator[dict[str, Any]]:
  """Extracts labels from filenames in a dataset.

  Args:
    dataset: An iterator of dictionaries, where each dictionary represents an
      example file and contains the key "file_name" with the filename.
    string_separator: The string used to separate parts of the filename.
    label_index_in_string: The index of the label in the filename when split by
      the string_separator.
    label_is_numeric: Whether the label should be converted to a numeric type.
    log_every_n: The interval at which to log the number of files processed.

  Yields:
    An iterator of dictionaries, where each dictionary is the same as the input
    dictionary but with additional "label" and "example_id" keys.
  """
  num_files = 0
  example_id = 0
  for example in dataset:
    label = (
        example["file_name"]
        .split("/")[-1]
        .split(string_separator)[label_index_in_string]
    )
    if label_is_numeric:
      label = int(label)
    example["label"] = label
    example["example_id"] = example_id
    example_id += 1
    yield example
    num_files += 1
    if num_files % log_every_n == 0:
      logging.info(
          "[extract_dataset_labels_from_filename] # files processed: %d",
          num_files,
      )
  logging.info(
      "[extract_dataset_labels_from_filename] Done. Total # files"
      " processed: %d",
      num_files,
  )


def _get_current_stage(
    line: str, current_stage: str, stages_names: list[str]
) -> str:
  """Determines the current stage based on the trace line and stage names.

  Args:
    line: The current trace line.
    current_stage: The current stage.
    stages_names: A list of stage names.

  Returns:
    The updated current stage.
  """
  if "Start tracking" in line:
    return stages_names[0]
  if "Second Part Starts" in line:
    return stages_names[1]
  if "End tracking" in line:
    return ""
  return current_stage


def _get_monitoring_status(line: str, is_currently_monitoring: bool) -> bool:
  """Determines the monitoring status based on the trace line.

  Args:
    line: The current trace line.
    is_currently_monitoring: The current monitoring status.

  Returns:
    The updated monitoring status.
  """
  if "Start tracking" in line:
    return True
  if "End tracking" in line:
    return False
  return is_currently_monitoring


def _filter_trace_line(line: str) -> bool:
  """Filters trace lines based on specific criteria for the fourth+ dataset.

  Args:
    line: The trace line to filter.

  Returns:
    True if the line should be kept, False otherwise.
  """
  if line.startswith("PMCs:"):
    return True
  if line.startswith("MA"):
    if line.endswith("CL"):
      return False
    return True
  if line.startswith("CF"):
    return True
  if "ci_bk_" in line:
    return True
  if (
      "[DEBUG]" in line
      or "Start tracking" in line
      or "Second Part Starts" in line
  ):
    return False
  logging.warning("[_filter_trace_line] Possible missing line: >%s<", line)
  return False


def _do_pattern_matching(
    trace_pattern: dict[str, str], line: str, line_dict: dict[str, str]
) -> dict[str, str]:
  """Performs pattern matching on a line and updates the line dictionary.

  Args:
    trace_pattern: The dictionary containing the regex pattern and name.
    line: The line to match against.
    line_dict: The dictionary to update with the matched groups.

  Returns:
    The updated line dictionary.
  """
  match = re.search(trace_pattern["pattern"], line)
  if match is None:
    logging.warning(
        "[_do_pattern_matching] Possible missing regex pattern for line: >%s<",
        line,
    )
    return line_dict
  line_dict.update(match.groupdict())
  line_dict.update(trace_pattern)
  return line_dict


def _process_trace_line(line: str, stage: str) -> dict[str, str]:
  """Processes a single trace line, extracting relevant information.

  Args:
    line: The trace line to process.
    stage: The current stage of the trace.

  Returns:
    A dictionary containing the processed trace line information.
  """
  line_dict = {"line": line, "name": "", "stage": stage}
  if "PMCs:" == line[:5]:
    return _do_pattern_matching(trace_line_pattern_dict["PMC"], line, line_dict)
  if "CF" == line[:2]:
    return _do_pattern_matching(trace_line_pattern_dict["CF"], line, line_dict)
  if "MA " == line[:3]:
    return _do_pattern_matching(trace_line_pattern_dict["MA"], line, line_dict)
  if "MA_" == line[:3]:
    return _do_pattern_matching(
        trace_line_pattern_dict["MA_CACHE"], line, line_dict
    )
  if "ci_bk_" in line:
    return _do_pattern_matching(
        trace_line_pattern_dict["CI_BK"], line, line_dict
    )
  logging.warning(
      "[_process_trace_line] Possible missing line type: >%s<", line
  )
  return line_dict


def _extract_trace_from_file_content(
    file_content: str, stages_names: list[str]
) -> Iterator[dict[str, Any]]:
  """Extracts traces from a file content string.

  Args:
    file_content: The content of the trace file as a string.
    stages_names: An optional list of stage names. If provided, the function
      will track the current stage based on the provided names.

  Yields:
    An iterator of dictionaries, where each dictionary represents a trace line.
  """
  current_stage = ""
  currently_monitoring = False
  for line in file_content.split("\n"):
    currently_monitoring = _get_monitoring_status(line, currently_monitoring)
    if not currently_monitoring:
      continue
    if stages_names:
      current_stage = _get_current_stage(line, current_stage, stages_names)
      if not current_stage:
        continue
    if _filter_trace_line(line):
      yield _process_trace_line(line, current_stage)


def extract_dataset_traces(
    dataset: Iterator[dict[str, Any]],
    stages_names: list[str],
    log_every_n: int = 20,
) -> Iterator[dict[str, Any]]:
  """Extracts traces from a dataset of files.

  Args:
    dataset: An iterator of dictionaries, where each dictionary represents a
      file and contains the key "file_content" with the file's content.
    stages_names: An optional list of stage names. If provided, the function
      will track the current stage based on the provided names.
    log_every_n: The interval at which to log the number of files processed.

  Yields:
    An iterator of dictionaries, where each dictionary is the same as the input
    dictionary but with an additional "traces" key containing an iterator of
    trace dictionaries.
  """
  num_files = 0
  for example in dataset:
    example["traces"] = _extract_trace_from_file_content(
        example["file_content"], stages_names
    )
    yield example
    num_files += 1
    if num_files % log_every_n == 0:
      logging.info("[extract_dataset_traces] # files processed: %d", num_files)
  logging.info(
      "[extract_dataset_traces] Done. Total # files processed: %d", num_files
  )
