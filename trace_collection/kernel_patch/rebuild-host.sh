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


SCRIPT_DIR="$(dirname "$0")"
. "${SCRIPT_DIR}"/common.sh
. "${SCRIPT_DIR}"/stable-commits
[[ -e /etc/os-release ]] && . /etc/os-release

OUTPUT_DIR="snp-release-2024-08-27"

build_host_kernel
rm -rf $OUTPUT_DIR/linux/host/*

if [[ "$ID" = "debian" ]] || [[ "$ID_LIKE" = "debian" ]]; then
	cp linux/linux-*-host-*.deb $OUTPUT_DIR/linux/host -v
else
	cp linux/kernel-*.rpm $OUTPUT_DIR/linux -v
fi