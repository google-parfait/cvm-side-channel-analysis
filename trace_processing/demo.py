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

"""A demo for running the SCA framework."""
from absl import app
from absl import flags
from sca import data_analysis
from sca import feature_extraction
from sca import feature_filters
from sca import trace_processing
from sca import util
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "parse_traces",
    False,
    "Whether to parse traces.",
)
flags.DEFINE_boolean(
    "extract_features",
    False,
    "Whether to extract features.",
)
flags.DEFINE_boolean(
    "merge_features",
    False,
    "Whether to merge features.",
)
flags.DEFINE_string(
    "input_dir",
    "./demo_data/phh",
    "The directory containing the input traces collected from the guest VM.",
)
flags.DEFINE_string(
    "pipeline_dir",
    "./tmp",
    "The directory where all the intermediate pipeline files will be stored.",
)
def main(argv):
  stages_phh = ["accumulate", "report"]
  example_files = ["0_1003", "1_993"]
  #### Parse traces:
  if FLAGS.parse_traces:
    examples_filenames = [f"{FLAGS.input_dir}/{fn}" for fn in example_files]
    examples_with_content = trace_processing.read_dataset(
        iter(examples_filenames), util.local_file_reader
    )
    examples_with_labels = (
        trace_processing.extract_dataset_labels_from_filename(
            examples_with_content, "_", 0, label_is_numeric=True
        )
    )
    examples_with_traces = trace_processing.extract_dataset_traces(
        examples_with_labels, stages_phh
    )
    for example_file in examples_with_traces:
      fn = example_file["file_name"].split("/")[-1]
      example_file["traces"] = list(example_file["traces"])
      util.local_pickle_file_writer(
          f"{FLAGS.pipeline_dir}/traces/traces_{fn}.pickle", example_file
      )
  #### Extract features:
  if FLAGS.extract_features:
    examples_filenames_traces = [
        f"{FLAGS.pipeline_dir}/traces/traces_{fn}.pickle"
        for fn in example_files
    ]
    examples_features_v1 = feature_extraction.extract_handcrafted_features_v1(
        iter(examples_filenames_traces), util.local_pickle_file_reader
    )
    for example_features in examples_features_v1:
      fn = example_features.index.values[0].split("/")[-1]
      util.local_pickle_file_writer(
          f"{FLAGS.pipeline_dir}/features/features_{fn}.pickle",
          example_features,
      )
  #### Merge features into a single dataset:
  if FLAGS.merge_features:
    examples_filenames_features = [
        f"{FLAGS.pipeline_dir}/features/features_{fn}.pickle"
        for fn in example_files
    ]
    merged_features = data_analysis.merge_handcrafted_features_into_dataset(
        iter(examples_filenames_features), util.local_pickle_file_reader
    )
    util.local_pickle_file_writer(
        f"{FLAGS.pipeline_dir}/datasets/merged_features.pickle", merged_features
    )
    print(merged_features["label"])
    x, _, y, _ = data_analysis.get_features_and_labels_for_dataset(
        merged_features, feature_filter=feature_filters.feature_filter_f1
    )
    print(x)
    print(y)
if __name__ == "__main__":
  app.run(main)
