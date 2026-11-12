# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess

import pytest


def test_rock_filename(rock_container):
    """Test that the built rock's name/version match rockcraft.yaml."""
    check_rock = rock_container[0]
    assert check_rock.get_name() == "pytorch"
    assert check_rock.get_version() == "2.7.1-cuda12.8-cudnn9-runtime"


def test_container_is_running(rock_container):
    """Test that the container keeps running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", rock_container[1]],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "true"


def test_non_root_user(rock_container):
    """Test that the container runs as the non-root _daemon_ user."""
    result = subprocess.run(
        ["docker", "exec", rock_container[1], "id", "-u"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "584792"


@pytest.mark.parametrize("path", ["/opt/venv", "/opt/uv"])
def test_venv_ownership(rock_container, path):
    """Test that the venv and its managed interpreter are owned by the non-root user."""
    result = subprocess.run(
        ["docker", "exec", rock_container[1], "stat", "-c", "%u:%g", path],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "584792:584792"


@pytest.mark.parametrize("binary", ["/opt/venv/bin/python", "/opt/venv/bin/torchrun"])
def test_venv_artifacts(rock_container, binary):
    """Test that expected binaries are installed in the uv-managed venv."""
    subprocess.run(
        ["docker", "exec", rock_container[1], "test", "-x", binary], check=True
    )


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


@pytest.mark.parametrize(
    "env_var,expected",
    [
        ("VIRTUAL_ENV", "/opt/venv"),
        ("NVIDIA_VISIBLE_DEVICES", "all"),
        ("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
        ("PYTORCH_VERSION", "2.7.1"),
    ],
)
def test_environment_variables(rock_container, env_var, expected):
    """Test env vars relied on by the NVIDIA runtime hooks are set on the image."""
    result = subprocess.run(
        ["docker", "exec", rock_container[1], "printenv", env_var],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
