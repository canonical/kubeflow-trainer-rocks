#!/usr/bin/env bash
# Renders single-gpu-test.yaml / distributed-gpu-test.yaml with IMAGE and
# HOST_PATH substituted, runs the diagnostic, then removes the pod.
#
# Usage: ./run-test.sh [--distributed]
# Env overrides: IMAGE (default: derived from ../rockcraft.yaml),
#                HOST_PATH (default: /home/ubuntu/gpu-tests),
#                POD_READY_TIMEOUT (default: 300s)
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--distributed]" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  "")
    MANIFEST="$SCRIPT_DIR/single-gpu-test.yaml"
    POD_NAME="gpu-test"
    DIAGNOSTIC="check_gpu.py"
    ;;
  --distributed)
    MANIFEST="$SCRIPT_DIR/distributed-gpu-test.yaml"
    POD_NAME="gpu-test-distributed"
    DIAGNOSTIC="check_distributed.py"
    ;;
  *)
    echo "Unknown option: $1 (usage: $0 [--distributed])" >&2
    exit 1
    ;;
esac

ROCKCRAFT_YAML="$SCRIPT_DIR/../rockcraft.yaml"
if [[ -z "${IMAGE:-}" ]]; then
  if [[ ! -r "$ROCKCRAFT_YAML" ]]; then
    echo "Cannot read $ROCKCRAFT_YAML; set IMAGE to the imported image reference." >&2
    exit 1
  fi
  NAME=$(grep -m1 -E '^name:' "$ROCKCRAFT_YAML" | sed -E "s/^name:[[:space:]]*[\"']?([^\"'[:space:]]*)[\"']?.*/\1/")
  VERSION=$(grep -m1 -E '^version:' "$ROCKCRAFT_YAML" | sed -E "s/^version:[[:space:]]*[\"']?([^\"'[:space:]]*)[\"']?.*/\1/")
  IMAGE="docker.io/library/${NAME}:${VERSION}"
fi
export IMAGE
export HOST_PATH="${HOST_PATH:-/home/ubuntu/gpu-tests}"

cleanup() {
  local exit_status=$?
  if [[ "$exit_status" -ne 0 ]]; then
    kubectl describe pod "$POD_NAME" >&2 || true
  fi
  if ! kubectl delete pod "$POD_NAME" --ignore-not-found --timeout=60s; then
    if [[ "$exit_status" -eq 0 ]]; then
      exit_status=1
    fi
  fi
  exit "$exit_status"
}

echo "Creating $MANIFEST with IMAGE=$IMAGE HOST_PATH=$HOST_PATH" >&2
envsubst '${IMAGE} ${HOST_PATH}' < "$MANIFEST" | kubectl create -f -
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl wait --for=condition=Ready "pod/$POD_NAME" --timeout="${POD_READY_TIMEOUT:-300s}"
kubectl exec "$POD_NAME" -- python "/gpu-tests/$DIAGNOSTIC"
