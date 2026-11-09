# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess

import pytest


@pytest.mark.parametrize(
    "module", ["torch", "torchvision", "torchaudio", "torchelastic", "numpy"]
)
def test_import(rock_container, module):
    """Test that a Python module imports in the rock's virtual environment."""
    subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "/opt/venv/bin/python",
            "-c",
            f"import {module}",
        ],
        check=True,
    )


def test_torch_version(rock_container):
    """Test installed torch version matches the pinned PYTORCH_VERSION."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "/opt/venv/bin/python",
            "-c",
            "import torch; print(torch.__version__)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip().startswith("2.7.1")


def test_torch_compiled_with_cuda(rock_container):
    """Test torch built with CUDA support (doesn't require a GPU on the test host)."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "/opt/venv/bin/python",
            "-c",
            "import torch; print(torch.cuda._is_compiled())",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "True"
