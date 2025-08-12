/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CONFIG_H
#define CONFIG_H

/* CPU Core definitions */
#define ASSIST_CORE 1
#define ISOLATED_CPU_CORE 7

/* Physical address of the kernel sync region */
#define KERNEL_SYNC_PA 0x6B7707000

/* Configuration to instruct collection */
#define INT_ALLOWED 1
#define CLEAN_CACHE 1
#define CACHE_ATTACKS 1
#define CIPHERTEXT_SCA 1
#define PMC_SCA 1
#define DYNAMIC_QUEUE 1
#define PMC_WRAPPER 1

/* Default values for configurable parameters */
#define MEM_MAPPED_QUEUE_SIZE 5
#define PMC_CLF_SW_THRESHOLD 8
#define PMC1 0x2600
#define PMC2 0xc200
#define PMC3 0xc400
#define PMC4 0xc800
#define PMC5 0xc100

#define V8_COVERT_CHANNEL 0
#if V8_COVERT_CHANNEL
#define JIT_SWITCH 0x77
#else
#define JIT_SWITCH 0
#endif

/* No need to care */

/* Offsets into the kernel sync area (in bytes) */
#define SYNC_OFFSET_FLAG 0
#define SYNC_OFFSET_STEP_NUM 8
#define SYNC_OFFSET_MEM_MAPPED_STACK_SIZE 96
#define SYNC_OFFSET_JIT 2048
#define SYNC_OFFSET_WX_TIMES 2049
#define SYNC_OFFSET_PMC_CLF_SW_THRESHOLD 1048
#define SYNC_OFFSET_PMC1 1040
#define SYNC_OFFSET_PMC2 1056
#define SYNC_OFFSET_PMC3 1072
#define SYNC_OFFSET_PMC4 1088
#define SYNC_OFFSET_PMC5 1104

/* Bit positions for building the sync flag */
#define SYNC_FLAG_CODE_PAGE_NUM_SHIFT 32
#define SYNC_FLAG_INTERVAL_SHIFT 16
#define SYNC_FLAG_INT_ALLOWED_SHIFT 8
#define SYNC_FLAG_CLEAN_CACHE_SHIFT 9
#define SYNC_FLAG_CACHE_ATTACKS_SHIFT 10
#define SYNC_FLAG_CIPHERTEXT_SCA_SHIFT 11
#define SYNC_FLAG_PMC_SCA_SHIFT 12
#define SYNC_FLAG_DYNAMIC_QUEUE_SHIFT 13
#define SYNC_FLAG_PMC_WRAPPER_SHIFT 15

/* The hook switch value */
#define HOOK_SWITCH 0x77

#endif /* CONFIG_H */
