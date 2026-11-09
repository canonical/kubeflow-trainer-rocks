# GPU Diagnostics

Smoke tests for validating NVIDIA GPU access inside the
`pytorch` rock when deployed on Kubernetes with the
NVIDIA device plugin/runtime class.

## Contents

- `check_gpu.py` — single-GPU smoke test: checks CUDA/cuDNN availability and
  runs a matrix multiplication on `cuda:0`.
- `check_distributed.py` — multi-GPU smoke test: spawns one process per GPU
  and runs an NCCL `all_reduce` to verify distributed communication.
- `single-gpu-test.yaml` — Pod manifest requesting 1 GPU.
- `distributed-gpu-test.yaml` — Pod manifest requesting 2 GPUs.

## Prerequisites

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

1. Copy this folder to a directory on the node. The manifests currently
   hardcode `/home/ubuntu/gpu-tests` as the `hostPath.path` mounted into the
   pod at `/gpu-tests` — either copy the files there, or edit both manifests'
   `hostPath.path` to match the directory you used.
2. Apply a manifest:
   ```
   kubectl apply -f single-gpu-test.yaml
   # or
   kubectl apply -f distributed-gpu-test.yaml
   ```
3. Run the matching script inside the pod:
   ```
   kubectl exec gpu-test -- python /gpu-tests/check_gpu.py
   kubectl exec gpu-test-distributed -- python /gpu-tests/check_distributed.py
   ```
4. Clean up: `kubectl delete pod gpu-test gpu-test-distributed`
