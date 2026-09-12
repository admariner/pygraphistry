"""The existing-suite replay is a failing gate and includes every route switch."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from graphistry.tests.compute.gfql.routes.switch import ROUTES


@pytest.mark.parametrize("failure", [False, True])
def test_replay_broadcasts_existing_tests_and_propagates_failures(tmp_path, failure):
    root = Path(__file__).resolve().parents[5]
    fake = tmp_path / "python"
    # Use the real registry import, replacing only pytest execution with a recorder.
    fake.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        'echo "$GFQL_ROUTES_OFF|$*" >> "$REPLAY_CALLS"\n'
        'if [ "$REPLAY_FAIL" = 1 ] && [ "$GFQL_ROUTES_OFF" = point-rows ]; then exit 2; fi\n'
        'echo "1 passed"\n'
    )
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "OUT": str(tmp_path / "logs"), "REPLAY_CALLS": str(calls),
           "REPLAY_FAIL": str(int(failure)),
           "SUITES": "graphistry/tests/compute/test_chain.py"}
    env.pop("MODES", None)
    result = subprocess.run(["bash", str(root / "bin/test-routes-off.sh")], env=env,
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == int(failure), result.stderr
    records = [line.split("|", 1) for line in calls.read_text().splitlines()]
    assert [record[0] for record in records] == [*ROUTES, ",".join(ROUTES)]
    assert all("graphistry/tests/compute/test_chain.py" in record[1] for record in records)
    assert all((tmp_path / "logs" / f"{mode}.log").is_file() for mode in (*ROUTES, "all-off"))
