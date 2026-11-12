# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess
import textwrap

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


def test_torch_compile(rock_container):
    """Test runtime Inductor compilation as the non-root user without a GPU."""
    subprocess.run(
        [
            "docker",
            "exec",
            "--user",
            "584792:584792",
            "--env",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
            "--env",
            "TRITON_CACHE_DIR=/tmp/triton",
            rock_container[1],
            "/opt/venv/bin/python",
            "-c",
            textwrap.dedent(
                """\
                import torch

                torch._dynamo.config.suppress_errors = False

                def model(tensor):
                    return tensor.sin() + tensor.square()

                inputs = torch.randn(1024, device="cpu")
                expected = model(inputs)
                compiled = torch.compile(
                    model, backend="inductor", fullgraph=True
                )
                torch.testing.assert_close(compiled(inputs), expected)
                """
            ),
        ],
        check=True,
    )
