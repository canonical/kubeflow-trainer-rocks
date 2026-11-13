# PyTorch Rock

Rock for a CUDA-enabled PyTorch distributed runtime, based on the [upstream PyTorch Dockerfile](https://github.com/pytorch/pytorch/blob/v2.7.1/Dockerfile).

- Base: `ubuntu@22.04`, `amd64` only
- Contents: `torch` 2.7.1, `torchvision` 0.22.1, `torchaudio` 2.7.1 (CUDA 12.8 wheels) and `torchelastic`, in a uv-managed Python 3.11 venv at `/opt/venv`
- Runs as the non-root `_daemon_` user (UID 584792)
- NVIDIA driver libraries are expected to be injected by the host runtime; no system CUDA toolkit is installed

## Build and test

```bash
tox -e pack             # build the rock
tox -e export-to-docker # load the rock into the local Docker daemon
tox -e sanity           # run the rock tests
```

## GPU Testing

This section records the GPU validation performed on the Testflinger host blanka, using the hardware and software versions listed below. The commands reflect that specific environment and may require adaptation for other machines.

- **Host:** `blanka` (Testflinger), Ubuntu 24.04.5 LTS, kernel `6.8.0-149-generic`
- **GPUs:** 8× NVIDIA A100-SXM4-40GB, NVSwitch fabric (6× `NVIDIA GA100 [A100 NVSwitch]` bridges on `lspci`, classic DGX/HGX A100 topology)
- **Driver:** `595.91.07`, Ubuntu `595-server` branch. 

All commands were run as `ubuntu` on the host. Use `sudo` where shown.

### 1. Copy the repositories to the host

Run from your local machine. Excludes built rocks and tox environments.

```bash
FOLDER=<repo-folder>
HOST=<gpu-host-ip>

rsync -avz --progress --exclude='*.rock' --exclude='.tox' \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  ./${FOLDER}/ ubuntu@${HOST}:~/${FOLDER}/
```

### 2. Install the NVIDIA driver and Fabric Manager

On the host:

1. Stop background package churn so apt is not locked or upgraded mid-install:

   ```bash
   sudo systemctl stop unattended-upgrades.service packagekit.service
   sudo systemctl mask unattended-upgrades.service
   sudo sed -i 's/"1"/"0"/' /etc/apt/apt.conf.d/20auto-upgrades
   ```

2. Confirm the available archive versions. Do not assume `595.91.07` is still present; **check every time**. These version worked specifically for this machine, it may not work on you machine:

   ```bash
   apt-cache madison nvidia-driver-595-server nvidia-fabricmanager-595
   ```

3. Install only the two top-level packages and let apt resolve the rest. Pin the driver and Fabric Manager to the same version, then hold them and reboot:

   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     nvidia-driver-595-server=595.91.07-0ubuntu0.24.04.1 \
     nvidia-fabricmanager-595=595.91.07-0ubuntu0.24.04.1

   sudo apt-mark hold nvidia-driver-595-server nvidia-fabricmanager-595
   sudo reboot
   ```

4. After the reboot, verify that all GPUs are listed and both `nvidia-fabricmanager` and `nvidia-persistenced` are `active (running)`:

   ```bash
   nvidia-smi
   systemctl status nvidia-fabricmanager nvidia-persistenced --no-pager
   journalctl -u nvidia-fabricmanager -n 10 --no-pager
   ```

### 3. Set up the Kubernetes environment

Installs Terraform, Concierge (which prepares Juju and a Kubernetes cluster) and Helm.

```bash
sudo snap install terraform --channel latest/stable --classic
sudo snap install concierge --classic

sudo concierge prepare --trace

sudo apt-get install curl gpg apt-transport-https --yes
curl -fsSL https://packages.buildkite.com/helm-linux/helm-debian/gpgkey | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/helm.gpg] https://packages.buildkite.com/helm-linux/helm-debian/any/ any main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

Check health; all pods should be `Running`:

```bash
kubectl get pods -A
```

### 4. Set up the GPU Operator

Installs the NVIDIA GPU Operator with Helm and polls until its validator reports success.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --generate-name -n gpu-operator-resources --create-namespace nvidia/gpu-operator

# Wait for gpu operator components to be ready
while ! kubectl logs -n gpu-operator-resources -l app=nvidia-operator-validator | grep "all validations are successful"
do
  echo "waiting for validations"
  kubectl get pods -n gpu-operator-resources
  sleep 60
done
```

Check health; all pods should be `Running`, and `nvidia-cuda-validator-xxxx` should be `Completed`:

```bash
kubectl get pods -A
```

### 5. Build and import the rock

1. Pack the rock.

2. Import it into the k8s containerd. The `ctr` binary must be invoked directly from the snap's `k8s` folder. The charm's relevant ClusterTrainingRuntime (see rock-ci-metadata.yaml) image used must match the name of the rock:

   ```bash
   sudo /snap/k8s/current/bin/ctr --address /run/containerd/containerd.sock --namespace k8s.io \
     images import --base-name localhost/pytorch pytorch_2.7.1-cuda12.8-cudnn9-runtime_amd64.rock
   ```

3. Check that the image exists:

   ```bash
   sudo /snap/k8s/current/bin/ctr --address /run/containerd/containerd.sock --namespace k8s.io \
     images ls -q | grep pytorch
   ```

### 6. Build and deploy the charm

1. Pack the charm.

2. Make sure Juju is using the Kubernetes controller, not the LXD (machine charm) one:

   ```bash
   juju controllers                 # confirm a k8s controller exists (likely "concierge-k8s")
   juju switch concierge-k8s:testing  # or whatever the k8s controller/model is named
   ```

3. Deploy the charm:

   ```bash
   juju deploy ./kubeflow-trainer_ubuntu@24.04-amd64.charm kubeflow-trainer --trust
   ```

### 7. Run the UATs

Use any virtualenv manager; `uv` is shown here as an example.
 
Replace `<absolute-notebook-path>` with the absolute path to the desired notebook. This validation used `tests/notebooks/gpu/kubeflow-trainer/kubeflow-trainer-gpu-integration.ipynb` from the UAT repository.

```bash
git clone https://github.com/canonical/charmed-kubeflow-uats
cd charmed-kubeflow-uats
uv venv
source .venv/bin/activate
uv pip install ipykernel pip
uv pip install -r tests/requirements.txt
cd tests
python3 run_notebook.py <absolute-notebook-path>   # must be the full path
```
