# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess

import pytest
from charmed_kubeflow_chisme.rock import CheckRock

# The ClusterTrainingRuntime runs as mpiuser rather than the rock's run-user.
MPIUSER = "1000:1000"


@pytest.fixture(scope="session")
def rock_container():
    """Run the rock as mpiuser once for the whole test session, then remove it."""
    check_rock = CheckRock("rockcraft.yaml")
    rock_image = check_rock.get_name()
    rock_version = check_rock.get_version()
    local_rock_image = f"{rock_image}:{rock_version}"

    # No pebble services are defined for this rock, so the container is
    # kept alive by pebble itself (the default rockcraft entrypoint).
    container_id = subprocess.run(
        ["docker", "run", "-d", "--user", MPIUSER, local_rock_image],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    yield check_rock, container_id

    subprocess.run(["docker", "rm", "--force", container_id])
