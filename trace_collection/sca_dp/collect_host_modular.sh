#!/bin/bash
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


# Number of iterations to run
VM_COMMAND=$1
START=$2
NUM_ITERATIONS=$3
OUTPUT_PATH=$4
LABEL=$5
SHUFFLE_MEMORY_AFTER=$6

mkdir -p  $OUTPUT_PATH

echo "Collecting ${NUM_ITERATIONS} traces starting at index ${START} via '${VM_COMMAND}' at ${OUTPUT_PATH}/${LABEL}_*"

# Cleanup
sudo ./sca_dp 0 0 100000 1

for ((i = START; i < START+NUM_ITERATIONS; i++)); do
        # Clean the log buffer
        sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'

        # Run the controller
        sudo ./sca_dp 0 0 200000 0 &
        MAIN_HOST_PID=$!

        # SSH to the VM and execute the VM script
        ssh snp-vm "${VM_COMMAND}" > /dev/null 2>&1

        # Wait for the controller to finish
        wait $MAIN_HOST_PID

        # Output the trace from the log buffer
	sudo cp /sys/kernel/debug/tracing/trace "${OUTPUT_PATH}/${LABEL}_${i}"
        sudo chmod 644 "${OUTPUT_PATH}/${LABEL}_${i}"

        echo "$((i+1)) traces collected"
        if [ $(((i+1)%${SHUFFLE_MEMORY_AFTER})) -eq 0 ]; then
                sudo ./sca_dp 0 0 200000 1

		ssh snp-vm 'echo 3 | sudo tee /proc/sys/vm/drop_caches' > /dev/null

                # Wait for the cleanup to finish
                echo "Memory was shuffled. sleep 3s"
                sleep 3
        fi
done

# Cleanup
sudo ./sca_dp 0 0 100000 1

# reset the terminal
reset
