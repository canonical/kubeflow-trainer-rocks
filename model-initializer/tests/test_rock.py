# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
#
#

import random
import string
import subprocess

import pytest
from charmed_kubeflow_chisme.rock import CheckRock


@pytest.mark.parametrize("module", ["pkg.initializers.model", "huggingface_hub", "opendal"])
def test_rock(module):
    """Test that the venv interpreter can import the initializer and its dependencies."""
    check_rock = CheckRock("rockcraft.yaml")
    rock_image = check_rock.get_name()
    rock_version = check_rock.get_version()
    LOCAL_ROCK_IMAGE = f"{rock_image}:{rock_version}"

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            LOCAL_ROCK_IMAGE,
            "exec",
            "/app/.venv/bin/python",
            "-c",
            f"import {module}",
        ],
        check=True,
    )
