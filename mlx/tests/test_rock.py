# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

# The ClusterTrainingRuntime runs as mpiuser rather than the rock's run-user.
MPI_USER_ID = "1000"


def test_container_is_running(rock_container):
    """Test that the container keeps running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", rock_container[1]],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "true"


def test_mpiuser_uid(rock_container):
    """Test that mpiuser exists with the UID the ClusterTrainingRuntime runs as."""
    result = subprocess.run(
        ["docker", "exec", rock_container[1], "id", "-u", "mpiuser"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == MPI_USER_ID


@pytest.mark.parametrize(
    "path", ["/home/mpiuser", "/home/mpiuser/.local", "/home/mpiuser/.cache"]
)
def test_mpiuser_home_ownership(rock_container, path):
    """Test that mpiuser owns its home, packages and cache directories."""
    result = subprocess.run(
        ["docker", "exec", rock_container[1], "stat", "-c", "%u", path],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == MPI_USER_ID


@pytest.mark.parametrize(
    "binary",
    [
        "/usr/bin/python",
        "/usr/bin/mpirun",
        "/usr/bin/ssh",
        "/usr/sbin/sshd",
        "/usr/local/cuda/bin/nvcc",
        "/home/mpiuser/.local/bin/mlx.launch",
        "/home/mpiuser/.local/bin/mlx_lm.lora",
    ],
)
def test_binaries(rock_container, binary):
    """Test that expected binaries are executable by mpiuser."""
    subprocess.run(
        ["docker", "exec", rock_container[1], "test", "-x", binary],
        check=True,
    )


def test_pip_available(rock_container):
    """Test that the Kubeflow SDK's `python -m pip` uses the upgraded pip in mpiuser's ~/.local."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "python",
            "-m",
            "pip",
            "--version",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    # Ubuntu's pip 22.0.2 in /usr/lib/python3/dist-packages means the upgrade didn't ship.
    assert "/home/mpiuser/.local/lib/python3.10/site-packages/pip" in result.stdout


@pytest.mark.parametrize(
    "option",
    ["port 2222", "stricthostkeychecking false", "userknownhostsfile /dev/null"],
)
def test_ssh_client_config(rock_container, option):
    """
    Test the ssh client options the MPI launcher relies on to reach workers.
    Set in the rock's ssh client configuration.
    """
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "ssh",
            "-G",
            "localhost",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert option in result.stdout.splitlines()


def test_sshd_as_mpiuser(rock_container):
    """
    Test that a worker's sshd, started as in the ClusterTrainingRuntime,
    accepts the launcher's key login.
    """
    # Stand-in for the Secret Trainer mounts at ~/.ssh: one key pair is both
    # sshd's HostKey (set in ~/.sshd_config) and the launcher's login key.
    subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "bash",
            "-c",
            textwrap.dedent(
                """\
                set -eu
                mkdir -p ~/.ssh
                ssh-keygen -q -t rsa -N "" -f ~/.ssh/id_rsa
                cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
                """
            ),
        ],
        check=True,
    )

    # Exact command and args of the runtime's node container. -D keeps sshd in
    # the foreground (it is the pod's main process), so the exec is detached.
    subprocess.run(
        [
            "docker",
            "exec",
            "--detach",
            rock_container[1],
            "/usr/sbin/sshd",
            "-De",
            "-f",
            "/home/mpiuser/.sshd_config",
        ],
        check=True,
    )

    # Log in as the MPI launcher would. BatchMode fails instead of hanging on a
    # password prompt; ConnectionAttempts retries until the detached sshd
    # is listening.
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=5",
            "localhost",
            "whoami",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=60,
    )
    # sshd rejects the key with "account is locked" if useradd's default '!'
    # password is back (see the useradd comment in rockcraft.yaml).
    assert result.stdout.strip() == "mpiuser"


def test_security_manifest(rock_container):
    """Test that the dpkg/os-release manifest required by the security team is present."""
    subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "test",
            "-s",
            "/usr/share/rocks/dpkg.query",
        ],
        check=True,
    )


def test_environment_variables():
    """Test that rockcraft.yaml sets the env vars the rock relies on; it may define more."""
    expected = {
        "HOME": "/home/mpiuser",
        "PATH": (
            "/home/mpiuser/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",
        "CUDA_VERSION": "12.8.1",
        "NCCL_VERSION": "2.25.1-1",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }
    environment = yaml.safe_load(Path("rockcraft.yaml").read_text())["environment"]
    assert {var: environment.get(var) for var in expected} == expected
