# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess
import textwrap

import pytest


@pytest.mark.parametrize(
    "package,version",
    [
        ("torch", "2.7.1"),
        ("deepspeed", "0.17.4"),
        ("mpi4py", "4.1.0"),
        ("datasets", "4.0.0"),
        ("transformers", "4.55.0"),
        ("sentencepiece", "0.2.0"),
    ],
)
def test_package_version(rock_container, package, version):
    """Test installed package versions match the pins in requirements.txt."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "python",
            "-c",
            f"from importlib.metadata import version; print(version('{package}'))",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == version


def test_torch_compiled_with_cuda(rock_container):
    """Test torch built with CUDA support (doesn't require a GPU on the test host)."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "python",
            "-c",
            "import torch; print(torch.cuda._is_compiled())",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "True"


def test_mpirun_mpi4py(rock_container):
    """
    Test that mpirun launches mpi4py ranks that can communicate, as mpiuser.

    mpi4py comes from its PyPI binary wheel, which isn't built against the
    rock's MPI: the dynamic linker finds the MPI library (libmpi.so.*) at
    runtime, here Ubuntu's Open MPI from overlay-packages' libopenmpi-dev.
    test_import's `import mpi4py` doesn't load libmpi; `from mpi4py import MPI`
    loads it and initializes MPI.
    See https://mpi4py.readthedocs.io/en/4.1.0/install.html (Wheel packages, Linux)
    and https://mpi4py.readthedocs.io/en/4.1.0/mpi4py.html (MPI4PY_RC_INITIALIZE).
    """
    subprocess.run(
        [
            "docker",
            "exec",
            rock_container[1],
            "mpirun",
            "--oversubscribe",
            "-np",
            "2",
            "python",
            "-c",
            textwrap.dedent(
                """\
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                assert comm.Get_size() == 2
                assert comm.allreduce(comm.Get_rank() + 1) == 3
                """
            ),
        ],
        check=True,
        timeout=60,
    )
