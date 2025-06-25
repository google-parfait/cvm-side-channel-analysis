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

echo "printk-msg-only" | sudo tee /sys/kernel/debug/tracing/trace_options
sudo sh -c 'echo 98304 > /sys/kernel/debug/tracing/buffer_size_kb'

echo 0 | sudo tee /sys/devices/system/cpu/cpu23/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu22/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu21/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu20/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu4/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu5/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online

sudo cpufreq-set -c 7 -g performance
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
sudo wrmsr -p 7 0xc001109a 0xffff # L3 cache range reservation - MSR