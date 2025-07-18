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

"""Methods to extract features from traces."""

import multiprocessing
import re
from typing import Any, Callable, Iterator, Optional
from absl import logging
import numpy as np
import pandas as pd


def _add_value_to_dict_set(
    dict_set: dict[str, set[Any]], k: str, v: Any
) -> None:
  """Adds a value to the set pointed at by k in dict_set.

  If the set does not exist for k, it is created.

  Args:
      dict_set: A dictionary to store the sets.
      k: The key to the dictionary.
      v: The value to add to the set.
  """
  if dict_set.get(k, None) is None:
    dict_set[k] = set()
  dict_set[k].add(v)


def _parse_cache_accesses_traceline(
    traceline: dict[str, Any],
) -> list[tuple[int, int]]:
  """Parses a MA_CACHE traceline.

  Args:
    traceline: A parsed traceline with all the relevant fields.

  Returns:
    A list of tuples, where each tuple contains the index and confidence of a
    cache line accessed in a MA_CACHE event.
  """
  cache_accesses = []
  if traceline['name'] == 'MA_CACHE':
    accesses = traceline['to_parse'].strip().split(' ')
    for el in accesses:
      idx, confidence = el.split('_')
      idx, confidence = int(idx), int(confidence)
      cache_accesses.append((idx, confidence))
  return cache_accesses


def _get_feature_set_1_from_traceline(
    stage: str,
    name: str,
    traceline: dict[str, Any],
    row: dict[str, Any],
    row_set_accumulator: dict[str, set[Any]],
) -> None:
  """Feature Set 1.

  - Number of total & unique code pages fetched.
  - Number of total & unique data pages accessed.

  This function updates the provided row dictionary with counts of 'MA' and 'CF'
  events, and keeps track of unique guest page addresses for each event type
  using the row_set_accumulator.

  Args:
      stage: The stage of the traceline.
      name: The type of traceline.
      traceline: A parsed traceline with all the relevant fields.
      row: A dictionary to store the total counts of 'MA' and 'CF' encountered
        in the current stage.
      row_set_accumulator: A dictionary to store unique guest page addresses for
        each MA/CF.
  """
  k1 = 'f1_%s_%s' % (stage, name)
  if name in ['MA', 'CF']:
    row[k1] = row.get(k1, 0) + 1
    k1u = '%s_unique' % (k1)
    _add_value_to_dict_set(
        row_set_accumulator, k1u, traceline['guest_page_address']
    )
  return None


def _get_feature_set_2_from_traceline(
    stage: str,
    name: str,
    traceline: dict[str, Any],
    cache_accesses: list[tuple[int, int]],
    max_ci_bk_number: int,
    max_cl_line_number: int,
    row: dict[str, Any],
    row_set_accumulator: dict[str, set[Any]],
) -> str | None:
  """Feature Set 2.

  - Number of total & unique intercepted cache lines in cache attacks.
  - Number of total & unique modified ciphertext blocks in memory.

  This function updates the provided row dictionary with counts of 'CI_BK' and
  'MA_CACHE'
  events, and keeps track of unique cache lines and ciphertext blocks for each
  event type
  using the row_set_accumulator.

  Args:
      stage: The stage of the traceline.
      name: The type of traceline.
      traceline: A parsed traceline with all the relevant fields.
      cache_accesses: A list of tuples, where each tuple contains the index and
        confidence of a cache line accessed in a MA_CACHE event. This is used to
        avoid parsing the traceline multiple times for other feature sets.
      max_ci_bk_number: The number of ciphertext blocks to track.
      max_cl_line_number: The number of cache lines to track.
      row: A dictionary to store the total counts of 'CI_BK' and 'MA_CACHE'
        encountered in the current stage.
      row_set_accumulator: A dictionary to store unique cache lines/ciphertext
        blocks for each MA_CACHE/CI_BK.

  Returns:
      An error message if the traceline is malformed, otherwise None.
  """
  k2 = 'f2_%s_%s' % (stage, name)
  try:
    if traceline['name'] == 'CI_BK':
      assert (
          len(re.findall(r'ci_bk_', traceline['line'].lower())) == 1
      ), f"Malformed CI_BK traceline: {traceline['line']}"
      ci_block_number = int(traceline['ci_block_number'])
      assert ci_block_number < max_ci_bk_number, (
          f'CI bk number exceeded. max = {max_ci_bk_number}, encountered ='
          f' {ci_block_number}'
      )
      row[k2] = row.get(k2, 0) + 1
      k2u = '%s_unique' % (k2)
      _add_value_to_dict_set(row_set_accumulator, k2u, ci_block_number)

    if traceline['name'] == 'MA_CACHE':
      for idx, confidence in cache_accesses:
        assert idx < max_cl_line_number, (
            f'Cache line number exceeded. max = {max_cl_line_number},'
            f' encountered = {idx}'
        )
        row[k2] = row.get(k2, 0) + 1
        k2u = '%s_unique' % (k2)
        _add_value_to_dict_set(row_set_accumulator, k2u, idx)
        if confidence > 1:
          k2hc = '%s_hc' % (k2)
          row[k2hc] = row.get(k2hc, 0) + 1
          k2hcu = '%s_unique' % (k2hc)
          _add_value_to_dict_set(row_set_accumulator, k2hcu, idx)
  except AssertionError as e:
    return e.args[0]


def _accumulate_feature_sets_1_2_for_trace(row, row_set_accumulator):
  """Accumulates Feature Sets 1 and 2 counts into the row dictionary.

  This function iterates through the row_set_accumulator dictionary, which
  contains sets of unique items for different features, and updates the row
  dictionary with the number of unique items in each set.

  Args:
      row: A dictionary to store the accumulated counts.
      row_set_accumulator: A dictionary containing sets of unique items for
        different features.
  """
  for k in row_set_accumulator:
    row[k] = len(row_set_accumulator[k])


def _get_feature_set_3_from_traceline(
    stage: str,
    name: str,
    traceline: dict[str, Any],
    cache_accesses: list[tuple[int, int]],
    max_ci_bk_number: int,
    max_cl_line_number: int,
    ci_bk_dict: dict[str, list[int]],
    cl_line_dict_all: dict[str, list[int]],
    cl_line_dict_hc: dict[str, list[int]],
) -> None:
  """Feature Set 3.

  This function updates the provided dictionaries with counts of 'CI_BK'
  (ciphertext block access) and
  'MA_CACHE' (cache line access) events, and keeps track of the number of times
  each ci_block_number
  and cache line are accessed.

  Args:
      stage: The stage of the traceline.
      name: The type of traceline.
      traceline: A parsed traceline with all the relevant fields.
      cache_accesses: A list of tuples, where each tuple contains the index and
        confidence of a cache line accessed in a MA_CACHE event. This is used to
        avoid parsing the traceline multiple times for other feature sets.
      max_ci_bk_number: The number of ciphertext blocks to track.
      max_cl_line_number: The number of cache lines to track.
      ci_bk_dict: A dictionary to store the counts of CI_BK events for each
        ciphertext block.
      cl_line_dict_all: A dictionary to store the counts of MA_CACHE events for
        each cache line.
      cl_line_dict_hc: A dictionary to store the counts of MA_CACHE events
        encountered with high confidence, for each cache line.
  """
  k3 = 'f3_%s_%s' % (stage, name)
  if traceline['name'] == 'CI_BK':
    ci_block_number = int(traceline['ci_block_number'])
    if ci_bk_dict.get(k3, None) is None:
      ci_bk_dict[k3] = [0] * max_ci_bk_number
    ci_bk_dict[k3][ci_block_number] += 1

  if traceline['name'] == 'MA_CACHE':
    for idx, confidence in cache_accesses:
      if cl_line_dict_all.get(k3, None) is None:
        cl_line_dict_all[k3] = [0] * max_cl_line_number
        cl_line_dict_hc[k3] = [0] * max_cl_line_number
      cl_line_dict_all[k3][idx] += 1
      if confidence > 1:
        cl_line_dict_hc[k3][idx] += 1
  return None


def _accumulate_feature_set_3_for_trace(
    row: dict[str, Any],
    ci_bk_dict: dict[str, list[int]],
    cl_line_dict_all: dict[str, list[int]],
    cl_line_dict_hc: dict[str, list[int]],
) -> None:
  """Accumulates Feature Set 3 counts into the row dictionary.

  This function iterates through the ci_bk_dict, cl_line_dict_all, and
  cl_line_dict_hc dictionaries, which contain counts of accesses for each
  ciphertext block and cache line, and adds these counts to the row dictionary.

  Args:
      row: A dictionary to store the accumulated counts.
      ci_bk_dict: A dictionary containing the counts of CI_BK events for each
        ciphertext block.
      cl_line_dict_all: A dictionary containing the counts of MA_CACHE events
        for each cache line.
      cl_line_dict_hc: A dictionary containing the counts of MA_CACHE events
        encountered with high confidence, for each cache line.
  """

  for k in ci_bk_dict:
    hist = ci_bk_dict[k]
    for idx in range(len(hist)):
      row['%s_%s' % (k, idx)] = hist[idx]
  for k in cl_line_dict_all:
    hist = cl_line_dict_all[k]
    for idx in range(len(hist)):
      row['%s_%s_all' % (k, idx)] = hist[idx]
    hist = cl_line_dict_hc[k]
    for idx in range(len(hist)):
      row['%s_%s_hc' % (k, idx)] = hist[idx]


def _get_feature_set_4_from_traceline(
    stage: str,
    name: str,
    traceline: dict[str, Any],
    cache_accesses: list[tuple[int, int]],
    cf_to_ma_list_stage_dict: dict[str, list[int]],
    ma_to_ci_list_stage_dict: dict[str, dict[str, list[Any]]],
    ma_to_cl_list_stage_dict: dict[str, dict[str, list[Any]]],
) -> None:
  """Feature Set 4.

  - Stats over the number of data page accesses following a code page fetch.
  - Stats over the number of total & unique cache lines accessed for a page.
  - Stats over the number of total & unique blocks accessed for a page.

  This function extracts features related to the sequences of CF, MA, CI_BK, and
  MA_CACHE events,
  tracking the relationships between these events within each stage. It
  populates dictionaries to store
  lists of:
  - The number of MA events after each CF event.
  - The CI_BK events after each MA event.
  - The MA_CACHE events after each MA event, differentiating between all
    accesses and high-confidence accesses.

  Args:
      stage: The stage of the traceline.
      name: The type of traceline.
      traceline: A parsed traceline with all the relevant fields.
      cache_accesses: A list of tuples, where each tuple contains the index and
        confidence of a cache line accessed in a MA_CACHE event. This is used to
        avoid parsing the traceline multiple times for other feature sets.
      cf_to_ma_list_stage_dict: A dictionary to store the number of MA events
        after each CF event.
      ma_to_ci_list_stage_dict: A dictionary to store the CI_BK events after
        each MA event.
      ma_to_cl_list_stage_dict: A dictionary to store the MA_CACHE events after
        each MA event.
  """
  k4 = 'f4_%s' % (stage)
  if cf_to_ma_list_stage_dict.get(k4, None) is None:
    cf_to_ma_list_stage_dict[k4] = []

  if ma_to_ci_list_stage_dict['all'].get(k4, None) is None:
    ma_to_ci_list_stage_dict['all'][k4] = []
    ma_to_ci_list_stage_dict['unique'][k4] = []

  if ma_to_cl_list_stage_dict['all_all'].get(k4, None) is None:
    ma_to_cl_list_stage_dict['all_all'][k4] = []
    ma_to_cl_list_stage_dict['all_hc'][k4] = []
    ma_to_cl_list_stage_dict['unique_all'][k4] = []
    ma_to_cl_list_stage_dict['unique_hc'][k4] = []

  if name == 'CF':
    cf_to_ma_list_stage_dict[k4].append(0)
  if name == 'MA':
    cf_to_ma_list_stage_dict[k4][-1] += 1

    ma_to_ci_list_stage_dict['all'][k4].append([])
    ma_to_ci_list_stage_dict['unique'][k4].append(set())

    ma_to_cl_list_stage_dict['all_all'][k4].append([])
    ma_to_cl_list_stage_dict['unique_all'][k4].append(set())
    ma_to_cl_list_stage_dict['all_hc'][k4].append([])
    ma_to_cl_list_stage_dict['unique_hc'][k4].append(set())
  if name == 'CI_BK':
    code_fetch_count = int(traceline['ci_block_number'])
    ma_to_ci_list_stage_dict['all'][k4][-1].append(code_fetch_count)
    ma_to_ci_list_stage_dict['unique'][k4][-1].add(code_fetch_count)
  if name == 'MA_CACHE':
    for idx, confidence in cache_accesses:
      ma_to_cl_list_stage_dict['all_all'][k4][-1].append(idx)
      ma_to_cl_list_stage_dict['unique_all'][k4][-1].add(idx)
      if confidence > 1:
        ma_to_cl_list_stage_dict['all_hc'][k4][-1].append(idx)
        ma_to_cl_list_stage_dict['unique_hc'][k4][-1].add(idx)
  return None


def _accumulate_feature_set_4_for_trace(
    row: dict[str, Any],
    cf_to_ma_list_stage_dict: dict[str, list[int]],
    ma_to_ci_list_stage_dict: dict[str, dict[str, list[int]]],
    ma_to_cl_list_stage_dict: dict[str, dict[str, list[int]]],
    num_bins_histogram: int,
) -> None:
  """Accumulates Feature Set 4 counts into the row dictionary.

  This function processes the data collected in the cf_to_ma_list_stage_dict,
  ma_to_ci_list_stage_dict, and ma_to_cl_list_stage_dict dictionaries,
  which contain sequences of events, by computing histograms and quantiles
  of the collected data and adding them to the row dictionary.

  Args:
      row: A dictionary to store the accumulated counts.
      cf_to_ma_list_stage_dict: A dictionary storing the number of MA events
        after each CF event.
      ma_to_ci_list_stage_dict: A dictionary storing the CI_BK events after each
        MA event.
      ma_to_cl_list_stage_dict: A dictionary storing the MA_CACHE events after
        each MA event.
      num_bins_histogram: The number of bins to use for the histograms.
  """
  for dict_in, dict_name in [
      (cf_to_ma_list_stage_dict, 'CF_to_MA'),
      (ma_to_ci_list_stage_dict['all'], 'MA_to_CI_all'),
      (ma_to_ci_list_stage_dict['unique'], 'MA_to_CI_unique'),
      (ma_to_cl_list_stage_dict['all_all'], 'MA_to_CL_all_all'),
      (ma_to_cl_list_stage_dict['all_hc'], 'MA_to_CL_all_hc'),
      (ma_to_cl_list_stage_dict['unique_all'], 'MA_to_CL_unique_all'),
      (ma_to_cl_list_stage_dict['unique_hc'], 'MA_to_CL_unique_hc'),
  ]:
    for k in dict_in:
      val = dict_in[k]
      if type(val) in [list, set]:
        val = len(val)
      hist, _ = np.histogram(val, density=False, bins=num_bins_histogram)
      for idx in range(len(hist)):
        row['%s_%s_hist_%s' % (k, dict_name, idx)] = int(hist[idx])
      quantiles = np.quantile(
          val, list(map(lambda e: e / 10.0, range(0, 11, 1)))
      )
      for idx in range(len(quantiles)):
        row['%s_%s_quantiles_%d' % (k, dict_name, idx)] = int(quantiles[idx])


def _get_feature_set_5_from_traceline(
    stage: str,
    name: str,
    traceline: dict[str, Any],
    cf_addr_to_index_dict: dict[str, dict[str, int]],
    cf_counts: dict[str, list[int]],
    ma_addr_to_index_dict: dict[str, dict[str, int]],
    ma_counts: dict[str, list[int]],
) -> None:
  """Feature Set 5.

  - Frequency of page accesses for individual data pages.
  - Frequency of code fetches for individual code pages.

  This function updates dictionaries to track the frequency of code fetch (CF)
  and memory access (MA) for each unique encountered address.

  Args:
      stage: The stage of the traceline.
      name: The type of traceline.
      traceline: A parsed traceline with all the relevant fields.
      cf_addr_to_index_dict: A dictionary to store the mapping of CF page
        addresses to indices.
      cf_counts: A dictionary to store the access counts for each CF page.
      ma_addr_to_index_dict: A dictionary to store the mapping of MA page
        addresses to indices.
      ma_counts: A dictionary to store the access counts for each MA page.
  """
  k5 = 'f5_%s_%s' % (stage, name)
  if name == 'CF':
    if cf_addr_to_index_dict.get(k5, None) is None:
      cf_addr_to_index_dict[k5] = {}
      cf_counts[k5] = []
    cf_addr = traceline['guest_page_address']
    if cf_addr_to_index_dict[k5].get(cf_addr, None) is None:
      cf_addr_to_index_dict[k5][cf_addr] = len(cf_counts[k5])
      cf_counts[k5].append(0)
    cf_counts[k5][cf_addr_to_index_dict[k5][cf_addr]] += 1

  if name == 'MA':
    if ma_addr_to_index_dict.get(k5, None) is None:
      ma_addr_to_index_dict[k5] = {}
      ma_counts[k5] = []
    ma_addr = traceline['guest_page_address']
    if ma_addr_to_index_dict[k5].get(ma_addr, None) is None:
      ma_addr_to_index_dict[k5][ma_addr] = len(ma_counts[k5])
      ma_counts[k5].append(0)
    ma_counts[k5][ma_addr_to_index_dict[k5][ma_addr]] += 1
  return None


def _accumulate_feature_set_5_for_trace(
    row: dict[str, Any],
    cf_counts: dict[str, Any],
    ma_counts: dict[str, Any],
    max_num_cf_counts: int,
    max_num_ma_counts: int,
) -> None:
  """Accumulates Feature Set 5 counts into the row dictionary.

  This function iterates through the CF (code fetch) and MA (memory access)
  counts dictionaries, and adds the counts for each page up to the maximum
  allowed number of pages to the row dictionary.

  Args:
      row: A dictionary to store the accumulated counts.
      cf_counts: A dictionary containing the counts for each CF page.
      ma_counts: A dictionary containing the counts for each MA page.
      max_num_cf_counts: The maximum number of CF pages to include.
      max_num_ma_counts: The maximum number of MA pages to include.
  """
  for k in cf_counts:
    for idx in range(len(cf_counts[k][:max_num_cf_counts])):
      row['%s_%s' % (k, idx)] = cf_counts[k][idx]
  for k in ma_counts:
    for idx in range(len(ma_counts[k][:max_num_ma_counts])):
      row['%s_%s' % (k, idx)] = ma_counts[k][idx]


def extract_handcrafted_features_v1(
    traces_files_list: Iterator[str],
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
    max_ci_bk_number: int = 256,
    max_cl_line_number: int = 64,
    num_bins_histogram: int = 50,
    max_num_cf_counts: int = 100,
    max_num_ma_counts: int = 100,
    log_every_n: int = 20,
) -> Iterator[pd.DataFrame]:
  """Extracts handcrafted features from trace files.

  Args:
    traces_files_list: Iterator of trace file paths.
    file_reader: Function to read the content of a trace file.
    file_reader_args: Optional arguments to pass to the file reader.
    max_ci_bk_number: Maximum number of ciphertext blocks to track.
    max_cl_line_number: Maximum number of distinct cache lines to track.
    num_bins_histogram: Number of bins for the histograms.
    max_num_cf_counts: Maximum number of CF pages to count frequency for.
    max_num_ma_counts: Maximum number of MA pages to count frequency for.
    log_every_n: Log progress every n files.

  Yields:
    A pandas DataFrame for each trace file, containing the extracted features.
  """

  num_files = 0

  for trace_file in traces_files_list:
    num_files += 1

    example_file = (
        file_reader(trace_file, *file_reader_args)
        if file_reader_args
        else file_reader(trace_file)
    )
    if not example_file:
      logging.warning(
          '[extract_handcrafted_features_v1] Error reading reading traces'
          ' file: %s',
          trace_file,
      )
      continue

    try:
      label = example_file['label']
      traces = example_file['traces']
      row = {}
      # Feature set f1,f2
      row_set_accumulator = {}
      # Feature set f3
      ci_bk_dict = {}
      cl_line_dict_all = {}
      cl_line_dict_hc = {}
      # Feature set f4
      cf_to_ma_list_stage_dict = {}
      ma_to_ci_list_stage_dict = {'all': {}, 'unique': {}}
      ma_to_cl_list_stage_dict = {
          'all_all': {},
          'all_hc': {},
          'unique_all': {},
          'unique_hc': {},
      }
      # Feature set f5
      cf_addr_to_index_dict = {}
      cf_counts = {}
      ma_addr_to_index_dict = {}
      ma_counts = {}

      for traceline in traces:
        name = traceline['name']
        stage = traceline['stage']

        if name is None:
          logging.warning(
              '[extract_handcrafted_features_v1] Possible missing traceline'
              ' name: >%s<',
              traceline,
          )

        feature_extraction_errors = []

        feature_extraction_errors.append(
            _get_feature_set_1_from_traceline(
                stage,
                name,
                traceline,
                row,
                row_set_accumulator,
            )
        )

        cache_accesses = _parse_cache_accesses_traceline(traceline)

        feature_extraction_errors.append(
            _get_feature_set_2_from_traceline(
                stage,
                name,
                traceline,
                cache_accesses,
                max_ci_bk_number,
                max_cl_line_number,
                row,
                row_set_accumulator,
            )
        )

        feature_extraction_errors.append(
            _get_feature_set_3_from_traceline(
                stage,
                name,
                traceline,
                cache_accesses,
                max_ci_bk_number,
                max_cl_line_number,
                ci_bk_dict,
                cl_line_dict_all,
                cl_line_dict_hc,
            )
        )

        feature_extraction_errors.append(
            _get_feature_set_4_from_traceline(
                stage,
                name,
                traceline,
                cache_accesses,
                cf_to_ma_list_stage_dict,
                ma_to_ci_list_stage_dict,
                ma_to_cl_list_stage_dict,
            )
        )

        feature_extraction_errors.append(
            _get_feature_set_5_from_traceline(
                stage,
                name,
                traceline,
                cf_addr_to_index_dict,
                cf_counts,
                ma_addr_to_index_dict,
                ma_counts,
            )
        )

        for feature_set_idx, feature_extraction_error in enumerate(
            feature_extraction_errors
        ):
          if feature_extraction_error:
            logging.warning(
                '[extract_features_v1] Error extracting Feature Set %d from'
                ' traceline: %s. Error: %s',
                feature_set_idx,
                traceline,
                feature_extraction_error,
            )

      # Feature Sets f1,f2
      _accumulate_feature_sets_1_2_for_trace(row, row_set_accumulator)

      # Feature Set f3
      _accumulate_feature_set_3_for_trace(
          row, ci_bk_dict, cl_line_dict_all, cl_line_dict_hc
      )

      # Feature Set f4
      _accumulate_feature_set_4_for_trace(
          row,
          cf_to_ma_list_stage_dict,
          ma_to_ci_list_stage_dict,
          ma_to_cl_list_stage_dict,
          num_bins_histogram,
      )

      # Feature Set f5
      _accumulate_feature_set_5_for_trace(
          row, cf_counts, ma_counts, max_num_cf_counts, max_num_ma_counts
      )

      # Add label and filename metadata.
      row['label'] = label
      row['file_name'] = example_file['file_name']
      df_row = pd.DataFrame([row])
      df_row.set_index('file_name', inplace=True)

      if num_files % log_every_n == 0:
        logging.info(
            '[extract_handcrafted_features_v1] # files processed: %d', num_files
        )

      yield df_row
    except (AssertionError, IndexError) as e:
      logging.warning(
          '[extract_handcrafted_features_v1] Error extracting features from'
          ' traces file: %s. Error: %s',
          trace_file,
          e,
      )
  logging.info(
      '[extract_handcrafted_features_v1] Done. Total # files processed: %d',
      num_files,
  )


def extract_sequence_features_v1(
    traces_files_list: Iterator[str],
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
    log_every_n: int = 20,
) -> Iterator[dict[str, Any]]:
  """Extracts Page-level sequence features from trace files.

  For each trace trace, this function extracts the sequence of CF and MA events.
  It also normalizes the MA addresses, replacing the page addresses with indices
  representing the order of the page occurrences.

  Args:
    traces_files_list: Iterator of trace file paths.
    file_reader: Function to read the content of a trace file.
    file_reader_args: Optional arguments to pass to the file reader.
    log_every_n: Log progress every n files.

  Yields:
    A dictionary for each trace file, containing the extracted features. The
    features are stored in a dictionary with the stage as the key and the
    sequence of CF and MA events as the value.
  """

  num_files = 0

  for trace_file in traces_files_list:
    num_files += 1

    example_file = (
        file_reader(trace_file, *file_reader_args)
        if file_reader_args
        else file_reader(trace_file)
    )
    if not example_file:
      logging.warning(
          '[extract_features_v1] Error reading reading traces file: %s',
          trace_file,
      )
      continue
    try:
      label = example_file['label']
      traces = example_file['traces']

      ma_page_address_set_dict = {}
      for traceline in traces:
        name = traceline['name']
        stage = traceline['stage']

        if name is None:
          logging.warning(
              '[extract_sequence_features_v1] Possible missing traceline'
              ' name: >%s<',
              traceline,
          )

        if ma_page_address_set_dict.get(stage, None) is None:
          ma_page_address_set_dict[stage] = set()
        if name == 'MA':
          ma_addr = traceline['guest_page_address']
          ma_page_address_set_dict[stage].add(ma_addr)

      ma_addr_to_index_dict = {}
      for stage in ma_page_address_set_dict:
        ma_addr_to_index_dict[stage] = {}
        running_idx = 0
        for ma_addr in sorted(
            ma_page_address_set_dict[stage], key=lambda e: int(e, 16)
        ):
          ma_addr_to_index_dict[stage][ma_addr] = running_idx
          running_idx += 1

      row_dict = {}

      for traceline in traces:
        name = traceline['name']
        stage = traceline['stage']

        if name is None:
          logging.warning(
              '[extract_sequence_features_v1] Possible missing traceline'
              ' name: >%s<',
              traceline,
          )

        if row_dict.get(stage, None) is None:
          row_dict[stage] = []

        if name == 'CF':
          row_dict[stage].append('CF')
        if name == 'MA':
          ma_addr = traceline['guest_page_address']
          row_dict[stage].append(
              'MA_%d' % (ma_addr_to_index_dict[stage][ma_addr])
          )

      row = {}
      row['label'] = label
      row['file_name'] = example_file['file_name']
      row['features'] = row_dict
      yield row

      if num_files % log_every_n == 0:
        logging.info(
            '[extract_sequence_features_v1] # files processed: %d', num_files
        )
    except (AssertionError, IndexError) as e:
      logging.warning(
          '[extract_sequence_features_v1] Error extracting features from traces'
          ' file: %s. Error: %s',
          trace_file,
          e,
      )
  logging.info(
      '[extract_sequence_features_v1] Done. Total # files processed: %d',
      num_files,
  )


def replace_hex_with_unique_numbers(text: str, context: str) -> str:
  """Replaces hexadecimal strings in a text with unique numbers.

  This function identifies 4/6-character hexadecimal strings in the input text
  and
  replaces them with unique numbers, maintaining a consistent mapping for
  repeated hexadecimal strings.

  Args:
      text: The input string containing hexadecimal strings to be replaced.
      context: The context of the replacement, either a page address or a
        ciphertext.

  Returns:
      A string with hexadecimal strings replaced by unique numbers.
  """
  if context == 'page_address':
    hex_pattern = r'[0-9a-fA-F]{6}'
  elif context == 'ciphertext':
    hex_pattern = r'[0-9a-fA-F]{4}'
  else:
    raise ValueError('Unsupported context: %s' % context)

  hex_to_num = {}
  next_num = 0

  # find all matches of pattern and put them in a list.
  all_hex_strings = re.findall(hex_pattern, text)
  for hex_str in all_hex_strings:
    if hex_str not in hex_to_num:
      hex_to_num[hex_str] = (
          str(next_num)
          if context == 'page_address'
          else f'ciphertext{str(next_num)}'
      )
      next_num += 1

  def replace_hex(match):
    hex_str = match.group(0)
    return hex_to_num[hex_str]

  return re.sub(hex_pattern, replace_hex, text)


def _extract_sequence_features_v2_for_trace_file(
    trace_file: str,
    traceline_patterns: str,
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
) -> dict[str, Any] | None:
  """Extracts sequence features from a single trace file.

  This function processes a single trace file, extracting sequences of events
  that match a given pattern. It normalizes page addresses and ciphertexts,
  replacing them with unique indices.

  Args:
      trace_file: Path to the trace file.
      traceline_patterns: Regular expressions to match against the traceline
        line.
      file_reader: Function to read the content of a trace file.
      file_reader_args: Optional arguments to pass to the file reader.

  Returns:
      A dictionary containing the extracted features, or None if an error
      occurs. The features are stored in a dictionary with the stage as the key
      and the sequence of events as the value.
  """
  example_file = (
      file_reader(trace_file, *file_reader_args)
      if file_reader_args
      else file_reader(trace_file)
  )
  if not example_file:
    logging.warning(
        '[extract_features_v2] Error reading reading traces file: %s',
        trace_file,
    )
    return None
  try:
    label = example_file['label']
    traces = example_file['traces']

    dict_unique_addresses_ma = {}
    dict_indices_ma = {}
    dict_unique_addresses_cf = {}
    dict_indices_cf = {}
    dict_unique_ciphertexts = {}
    dict_indices_ciphertexts = {}
    row_dict = {}
    for traceline in traces:
      name = traceline['name']
      stage = traceline['stage']
      line = traceline['line']
      if re.search(traceline_patterns, line) is None:
        continue

      if name is None:
        logging.warning(
            '[extract_sequence_features_v2] Possible missing traceline'
            ' name: >%s<',
            traceline,
        )
      if row_dict.get(stage, None) is None:
        row_dict[stage] = []

      if dict_unique_addresses_ma.get(stage, None) is None:
        dict_unique_addresses_ma[stage] = {}
        dict_indices_ma[stage] = 0
      if dict_unique_addresses_cf.get(stage, None) is None:
        dict_unique_addresses_cf[stage] = {}
        dict_indices_cf[stage] = 0
      if dict_unique_ciphertexts.get(stage, None) is None:
        dict_unique_ciphertexts[stage] = {}
        dict_indices_ciphertexts[stage] = 0
      processed_line = None
      if name == 'CF':
        addr = traceline['guest_page_address']
        if addr not in dict_unique_addresses_cf[stage]:
          idx_cf = dict_indices_cf[stage]
          dict_indices_cf[stage] += 1
          dict_unique_addresses_cf[stage][addr] = idx_cf
        processed_line = 'CF CF_%d' % (dict_unique_addresses_cf[stage][addr])
      if name == 'MA':
        addr = traceline['guest_page_address']
        if addr not in dict_unique_addresses_ma[stage]:
          idx_ma = dict_indices_ma[stage]
          dict_indices_ma[stage] += 1
          dict_unique_addresses_ma[stage][addr] = idx_ma
        processed_line = 'MA MA_%d' % (dict_unique_addresses_ma[stage][addr])
      if name == 'CI_BK':
        code_fetch_count = int(traceline['ci_block_number'])
        ciphertext_a = traceline['A_2bytes_ciphertext']
        ciphertext_b = traceline['B_2bytes_ciphertext']
        for ciphertext in [ciphertext_a, ciphertext_b]:
          if ciphertext not in dict_unique_ciphertexts[stage]:
            idx_ciphertext = dict_indices_ciphertexts[stage]
            dict_indices_ciphertexts[stage] += 1
            dict_unique_ciphertexts[stage][ciphertext] = idx_ciphertext
        processed_line = 'CI_BK CI_BK_%d CI_A_%d CI_B_%d' % (
            code_fetch_count,
            dict_unique_ciphertexts[stage][ciphertext_a],
            dict_unique_ciphertexts[stage][ciphertext_b],
        )
      if processed_line is not None:
        row_dict[stage].append(processed_line)

    row = {}
    row['label'] = label
    row['file_name'] = example_file['file_name']
    row['features'] = row_dict
    return row
  except (AssertionError, IndexError) as e:
    logging.warning(
        '[extract_sequence_features_v2] Error extracting features from traces'
        ' file: %s. Error: %s',
        trace_file,
        e,
    )
    return None


def extract_sequence_features_v2(
    traces_files_list: Iterator[str],
    traceline_patterns: str,
    file_reader: Callable[..., dict[str, Any]],
    file_reader_args: Optional[Any] = None,
) -> list[dict[str, Any]]:
  """Extracts sequence features from trace files.

  For each trace trace, this function extracts the sequence of events that match
  a given pattern.
  It also normalizes the page addresses, replacing the page addresses with
  unique numbers representing the order of the page occurrences.

  Args:
    traces_files_list: Iterator of trace file paths.
    traceline_patterns: Regular expressions to match against the traceline line.
    file_reader: Function to read the content of a trace file.
    file_reader_args: Optional arguments to pass to the file reader.

  Returns:
    A dictionary for each trace file, containing the extracted features. The
    features are stored in a dictionary with the stage as the key and the
    sequence of CF and MA events as the value.
  """

  with multiprocessing.pool.ThreadPool() as pool:
    instances = pool.map(
        lambda x: _extract_sequence_features_v2_for_trace_file(
            x,
            traceline_patterns,
            file_reader,
            file_reader_args,
        ),
        traces_files_list,
    )

  instances = [x for x in instances if x is not None]

  logging.info(
      '[extract_sequence_features_v2] Done. Total # files processed: %d',
      len(instances),
  )
  return instances
