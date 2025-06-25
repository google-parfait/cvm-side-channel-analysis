
### compile libtea
```bash
git clone https://github.com/Jdkhnjggf/frameworks.git
cd frameworks/libtea

make libtea-x86-interrupts
sudo insmod module/libtea.ko
```

### compile the userspace-controller
```bash
cp -r sca_dp ./
cd sca_dp && make
```

### env and runtime configuration
```bash
sudo ./env.sh


> vim main.c

#define CLEAN_CACHE     0  //  Set to 1 for Cachelines traces
#define CACHE_ATTACKS   0  //  Set to 1 for Cachelines traces
#define CIPHERTEXT_SCA  0  //  Set to 1 for Cipherblock traces
#define PMC_SCA         0  //  Set to 1 for PMC traces

# Page Faults (CF/MA) are collected by default

> make
```

### Prepare Guest Environment
Currently, the framework is looking for two different parts in traces. Please slightly change the test case as follows: 

``` C
#define REP4(X) X X X X
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

### Collect Data Trace
```bash
hv> sudo ./sca_dp 0 0 1000000 0
vm> <target> 

# Output the trace from the log buffer
hv> sudo cat /sys/kernel/debug/tracing/trace > <OUTPUT_FILE>
# Clean the log buffer
hv> sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'
```