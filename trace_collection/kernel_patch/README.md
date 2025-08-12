# Patching the host kernel

This modification is based on commit `85ef1ac03` (AMDSEV/linux/host) and is used
to implement a malicious hypervisor.

## Preparation

### Install a benign hypervisor

```bash
git clone https://github.com/AMDESE/AMDSEV.git
git checkout snp-latest
./build.sh --package
sudo cp kvm.conf /etc/modprobe.d/
```

> On successful build, the binaries will be available in `snp-release-<DATE>`.

```bash
cd snp-release-<date>
./install.sh
sudo reboot now
#... follow the instructions from AMD
```

### Apply patches

Note: please make sure the current commit is 85ef1ac. ```bash cd
~/AMDSEV/linux/host

patch -p1 < sevsca_host_kernel.patch ```

### Rebuild the host kernel

```bash
# copy "common.sh" and "rebuild-host.sh" to //AMDSEV
cd ../../
chmod +x ./rebuild-host.sh
./rebuild-host.sh
```

### Load new host kernel images

```bash
cd snp-release-<DATE>/linux/host
sudo dpkg -i *.deb
```

### Reboot the host machine into the new kernel

```bash
sudo reboot now
```

## Notes on the patch

In `svm_vcpu_create()`, the malicious hypervisor allocates 512 contiguous 4kB
pages at (0x780000000) for the eviction buffer and PAGE at 0x6B7707000 for
communicating with the user-space controller.

`svm_vcpu_enter_exit()` is the place where the VM exits and resumes.
Specifically, `__svm_sev_es_vcpu_run` contains the assembly instruction `vmrun`.

When the first byte in this page `sev_step_page_va` is magic value `0x77`, the
hypervisor should start tracking and initialize stuff (PMCs, MSRs for cache) via
`sev_step_initialize()`.

`kvm_unpre_all()` iterates over the nested page table, clear the present bits of
all guest pages. Then `svm_flush_tlb_current(vcpu);` flushes TLB for the vcpu.

`npf_interception()` is the handler for nested page fault. `is code_page` and
`is_data_page` determines what type is the faulted page.
