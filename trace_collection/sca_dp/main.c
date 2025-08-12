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

#include <inttypes.h>
#include <unistd.h>

#include "../libtea.h"
#include "config.h"

static uint64_t flag = 0;
static void *sev_step_kernel_sync_addr_p;

/* Inline helper functions to write to the sync area */
static inline void write64(void *base, size_t offset, uint64_t value) {
  *((volatile uint64_t *)((char *)base + offset)) = value;
}
static inline void write8(void *base, size_t offset, uint8_t value) {
  *((volatile uint8_t *)((char *)base + offset)) = value;
}

void pin_to_core(int core) {
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

/* Control thread: hooks the kernel exit handler and waits until the kernel
 * clears the sync flag. */
void *ctrl_thread(void *arg) {
  pin_to_core(ASSIST_CORE);
  /* Write our flag into the sync area */
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_FLAG, flag);

  /* Wait until the kernel clears the sync flag */
  while (*(volatile uint8_t *)sev_step_kernel_sync_addr_p) sched_yield();

  return NULL;
}

/* Print usage information */
void print_usage(const char *prog) {
  printf("Usage: %s <apic_interval> <step_num> <code_page_num> <clean_all>\n",
         prog);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  /* Parse mandatory command-line arguments */
  int interval = atoi(argv[1]);       // Not used - single step
  uint64_t step_num = atoi(argv[2]);  // Not used - single step
  uint64_t code_page_num = atoi(argv[3]);
  int clean_all = atoi(argv[4]);

  /* Initialize libtea */
  libtea_instance *instance = libtea_init();
  if (!instance) {
    libtea_info("Libtea test init failed.");
    return 1;
  }
  libtea_pin_to_core(getpid(), ISOLATED_CPU_CORE);

  /* Map the kernel sync area (4096 bytes) for read/write */
  sev_step_kernel_sync_addr_p = libtea_map_physical_address_range(
      instance, KERNEL_SYNC_PA, 4096, PROT_READ | PROT_WRITE, true);
  memset(sev_step_kernel_sync_addr_p, 0, 4096);

  if (clean_all) {
    goto cleanup;
  }

  /* Build the sync flag.
   * The flag encodes:
   *   - The number of code pages to record (shifted by
   * SYNC_FLAG_CODE_PAGE_NUM_SHIFT)
   *   - And various configuration bits.
   */
  flag = (code_page_num << SYNC_FLAG_CODE_PAGE_NUM_SHIFT) |
         (interval << SYNC_FLAG_INTERVAL_SHIFT) |
         (INT_ALLOWED << SYNC_FLAG_INT_ALLOWED_SHIFT) |
         (CLEAN_CACHE << SYNC_FLAG_CLEAN_CACHE_SHIFT) |
         (CACHE_ATTACKS << SYNC_FLAG_CACHE_ATTACKS_SHIFT) |
         (CIPHERTEXT_SCA << SYNC_FLAG_CIPHERTEXT_SCA_SHIFT) |
         (PMC_SCA << SYNC_FLAG_PMC_SCA_SHIFT) |
         (DYNAMIC_QUEUE << SYNC_FLAG_DYNAMIC_QUEUE_SHIFT) |
         (PMC_WRAPPER << SYNC_FLAG_PMC_WRAPPER_SHIFT) | HOOK_SWITCH;

  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_STEP_NUM, step_num);

  /* Write the configuration values from config.h into the sync area */
  write8(sev_step_kernel_sync_addr_p, SYNC_OFFSET_MEM_MAPPED_STACK_SIZE,
         MEM_MAPPED_QUEUE_SIZE);
  write8(sev_step_kernel_sync_addr_p, SYNC_OFFSET_JIT, JIT_SWITCH);
  write8(sev_step_kernel_sync_addr_p, SYNC_OFFSET_WX_TIMES, 1);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC_CLF_SW_THRESHOLD,
          PMC_CLF_SW_THRESHOLD);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC1, PMC1);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC2, PMC2);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC3, PMC3);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC4, PMC4);
  write64(sev_step_kernel_sync_addr_p, SYNC_OFFSET_PMC5, PMC5);

  /* Create the control thread that hooks the kernel exit handler. */
  pthread_t ctrl_tid;
  if (pthread_create(&ctrl_tid, NULL, ctrl_thread, NULL)) {
    perror("pthread_create");
    exit(EXIT_FAILURE);
  }
  sched_yield();
  pthread_join(ctrl_tid, NULL);

cleanup:
  /* Cleanup: reset APIC parameters and free libtea resources */
  libtea_apic_lvtt = libtea_apic_tdcr = 0;
  libtea_cleanup(instance);
  return 0;
}
