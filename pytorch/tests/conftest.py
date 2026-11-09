# Copyright 2025 Canonical Ltd.
# See LICENSE file for licensing details.
import subprocess

import pytest
from charmed_kubeflow_chisme.rock import CheckRock


@pytest.fixture(scope="session")
def rock_container():
    """Run the rock once for the whole test session, then remove it."""
    check_rock = CheckRock("rockcraft.yaml")
    rock_image = check_rock.get_name()
    rock_version = check_rock.get_version()
    local_rock_image = f"{rock_image}:{rock_version}"

    # No pebble services are defined for this rock, so the container is
    # kept alive by pebble itself (the default rockcraft entrypoint).
    container_id = subprocess.run(
        ["docker", "run", "-d", local_rock_image],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    yield check_rock, container_id

    subprocess.run(["docker", "rm", "--force", container_id])
