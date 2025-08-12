# Collecting traces from the CVM

Before proceeding to trace collection, the host kernel must be instrumented
according to the instructions in `kernel_patch`.

### Compile libtea

```bash
git clone https://github.com/Jdkhnjggf/frameworks.git
cd frameworks/libtea

make libtea-x86-interrupts
sudo insmod module/libtea.ko
```

### Compile the userspace-controller

```bash
cp -r sca_dp ./
cd sca_dp && make
```

### Configure the env and the runtime

```bash
sudo ./env.sh


> vim config.h

#define CLEAN_CACHE     0  //  Set to 1 for Cachelines traces
#define CACHE_ATTACKS   0  //  Set to 1 for Cachelines traces
#define CIPHERTEXT_SCA  0  //  Set to 1 for Cipherblock traces
#define PMC_SCA         0  //  Set to 1 for PMC traces

# Page Faults (CF/MA) are collected by default

> make
```

### Prepare the guest environment

The framework is collecting two stages for each traces. The test case needs to
be instrumented as follows:

```c
#define REP8(X) X X X X
int foo;

/* A Page contains `clflush` to indicate START */
REP4(asm volatile("clflush 0(%0)\n\n"::"c"(&foo):"rax");)
asm volatile(".align 4096\n");

<Part One we want to test, e.g., accumulate>

/* A Page contains `clflush` to indicate PART TWO */
asm volatile(".align 4096\n");
REP4(asm volatile("clflush 0(%0)\n\n"::"c"(&foo):"rax"););
asm volatile(".align 4096\n");

<Part Two we want to test, e.g., noisy report>

/* A Page contains `clflush` to indicate STOP */
asm volatile(".align 4096\n");
REP4(asm volatile("clflush 0(%0)\n\n"::"c"(&foo):"rax"););
```

### Collect a single trace

```bash
hv> sudo ./sca_dp 0 0 1000000 0
vm> <target>

# Output the trace from the log buffer
hv> sudo cat /sys/kernel/debug/tracing/trace > <OUTPUT_FILE>
# Clean the log buffer
hv> sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'
```

### Collect a batch of traces

A helper script `collect_batch.sh` can be used to collect a batch of
`<NUM_TRACES>` examples using a single `<VM_COMMAND>`. The examples are saved at
`<OUTPUT_PATH>/<TRACE_LABEL>_[idx]`, where `idx` starts at `TRACE_START_IDX`.
The CVM memory is reshuffled every `<SHUFFLE_MEMORY_AFTER>` traces.

```bash
collect_batch.sh <VM_COMMAND> <TRACE_START_IDX> <NUM_TRACES> <OUTPUT_PATH> <TRACE_LABEL> <SHUFFLE_MEMORY_AFTER>
```
