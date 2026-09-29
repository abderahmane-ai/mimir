"""The Vercel AI SDK example's own test (`examples/vercel_ai/agent.test.ts`), run by Node against
a real `mimir mcp` process on the test release."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

from tests.conftest import MIMIR_SERVER

pytestmark = pytest.mark.integration

EXAMPLE: Final = Path(__file__).parents[3] / "examples" / "vercel_ai"


def test_the_typescript_agent_routes_a_ticket_through_the_server(hub_release: Path) -> None:
    npm = shutil.which("npm")
    assert npm is not None, "npm is not on PATH; install Node 22 or later"
    assert (EXAMPLE / "node_modules").is_dir(), f"{EXAMPLE}: no node_modules; run make install"
    environment = {
        **os.environ,
        "MIMIR_SERVER": MIMIR_SERVER,
        "MIMIR_TEST_RELEASE": str(hub_release),
    }
    finished = subprocess.run(
        [npm, "test"],
        cwd=EXAMPLE,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "pass 1\n" in finished.stdout
    assert "fail 0\n" in finished.stdout
