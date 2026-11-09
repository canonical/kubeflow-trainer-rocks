# GPU Diagnostics

Smoke tests for validating NVIDIA GPU access inside the
`pytorch` rock when deployed on Kubernetes with the
NVIDIA device plugin/runtime class.

## Contents

- `check_gpu.py` — single-GPU smoke test: checks CUDA/cuDNN availability and
  runs a matrix multiplication on `cuda:0`.
- `check_distributed.py` — multi-GPU smoke test: spawns one process per GPU
  and runs an NCCL `all_reduce` to verify distributed communication.
- `single-gpu-test.yaml` — Pod manifest requesting 1 GPU (templated: `${IMAGE}`, `${HOST_PATH}`).
- `distributed-gpu-test.yaml` — Pod manifest requesting 2 GPUs (templated: `${IMAGE}`, `${HOST_PATH}`).
- `run-test.sh` — renders a manifest, creates the pod, waits for readiness,
  runs the diagnostic, and cleans up.

## Prerequisites

- Bash, `envsubst` (`gettext-base` on Ubuntu), and `kubectl` configured for
  the target cluster and namespace on the machine running the script.
- A Kubernetes cluster (e.g. via Juju/charmed-kubernetes) with GPU worker
  nodes, each with a working NVIDIA driver (`nvidia-smi` reports all GPUs).
- The NVIDIA GPU Operator installed, with all its pods (including the
  `nvidia-cuda-validator` and `nvidia-operator-validator`) healthy — this is
  what provides the `nvidia` `RuntimeClass` and `nvidia.com/gpu` resource used
  by the manifests.
  > On nodes with NVSwitch-connected GPUs (e.g. DGX/HGX-class multi-GPU
  > systems), the host driver and `nvidia-fabricmanager` package must be the
  > exact same version — a mismatch will leave the GPU Operator validators
  > stuck in `CrashLoopBackOff`.
- The rock image already imported into the node's containerd (not pulled
  from a registry) and tagged with a fully-qualified reference matching
  `IMAGE` below (manifests use `imagePullPolicy: Never`).

## Usage

1. Copy this folder to a directory on the node, matching `HOST_PATH` (default
   `/home/ubuntu/gpu-tests`; override with the `HOST_PATH` env var).
2. Set `IMAGE` to the fully-qualified reference imported on the node, then
  run a test:
  ```bash
  export IMAGE=docker.io/library/pytorch:2.7.1-cuda12.8-cudnn9-runtime
    ./run-test.sh
   # or
    ./run-test.sh --distributed
   ```
  When running from the repository checkout, `IMAGE` can be omitted: it
  defaults to `docker.io/library/<name>:<version>` read from
  `../rockcraft.yaml`. An explicit `IMAGE` is used without reading that
  file, so copying the diagnostics folder alone is sufficient.

Without arguments, the runner executes the single-GPU test. Pass
`--distributed` to run the two-GPU distributed test instead.

After creating the pod, the runner waits up to `300s` for readiness (override
with `POD_READY_TIMEOUT`), runs the diagnostic, and deletes the pod on exit.
Readiness or diagnostic failures print the pod description before cleanup.
The runner returns a nonzero status if creation, readiness, the diagnostic,
or cleanup fails. An existing pod with the same name is left untouched;
remove any leftover test pod before rerunning.

## Tox

From the repository's `pytorch` directory, run the single-GPU test (default)
or the distributed test:

```bash
tox -e gpu
tox -e gpu -- --distributed
```

The `gpu` environment is opt-in and is not included in the default `envlist`.
It uses the same prerequisites and passes through `IMAGE`, `HOST_PATH`,
`POD_READY_TIMEOUT`, and your Kubernetes configuration. It does not build or
import the image.
