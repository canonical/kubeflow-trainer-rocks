# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess
import textwrap

import pytest

# libcuda.so.1 for mlx-cuda without a GPU; on GPU nodes the NVIDIA runtime mounts the host's.
CUDA_COMPAT_LD_LIBRARY_PATH = "LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64"

@pytest.mark.parametrize(
    "package,version",
    [
        ("mlx", "0.28.0"),
        # Installed by the mlx[cuda] extra.
        ("mlx-cuda", "0.28.0"),
        ("mlx-data", "0.1.0"),
        ("mlx-lm", "0.26.3"),
        ("datasets", "4.0.0"),
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


def test_mpirun_mlx(rock_container):
    """
    Test that mpirun launches MLX ranks that communicate over MLX's MPI backend, as mpiuser.

    MLX loads the MPI library (libmpi.so, from overlay-packages' libopenmpi-dev)
    at runtime, so only a distributed init catches a missing or broken MPI.
    See "Getting Started with MPI" in
    https://ml-explore.github.io/mlx/build/html/usage/distributed.html
    """
    subprocess.run(
        [
            "docker",
            "exec",
            "--env",
            CUDA_COMPAT_LD_LIBRARY_PATH,
            rock_container[1],
            "mpirun",
            "--oversubscribe",
            "-np",
            "2",
            "python",
            "-c",
            textwrap.dedent(
                """\
                import mlx.core as mx

                # No GPU on the test host.
                mx.set_default_device(mx.cpu)
                world = mx.distributed.init(backend="mpi")
                assert world.size() == 2
                x = mx.distributed.all_sum(mx.ones(10))
                assert (x == 2).all().item()
                """
            ),
        ],
        check=True,
        timeout=60,
    )
