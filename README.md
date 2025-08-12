# CVM Side Channel Analysis Framework

This repo hosts the framework for auditing privacy applications running inside
Confidential VM, described in the paper [A Side-Channel Analysis Framework for
Privacy Applications on Confidential Virtual
Machines](https://arxiv.org/abs/2506.15924).

The repo is organized in two parts: `trace_collection` and `trace_processing`.
`trace_collection` contains the tools to setup a CVM environment,
instrument the host, and collect traces from the applications running inside
the guest VMs. `trace_processing` implements basic pipelines for trace
processing, feature extraction, data analysis and machine learning.

## Disclaimers

This project is intended for demonstration purposes only. It is not intended for
use in a production environment.
