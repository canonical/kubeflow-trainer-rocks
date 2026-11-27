# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess
import pytest

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
    """Test installed package versions match the pins in upstream requirements.txt."""
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