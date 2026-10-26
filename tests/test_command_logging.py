"""Global: external diagnostics must not corrupt machine-readable release metadata.

Local: exercise a real child process without external services or credentials.
Source: https://docs.python.org/3/library/subprocess.html.
"""
import json  # Local: require actual parseable stdout, not a hand-filtered approximation.
import sys  # Local: use the current locked Python executable as a deterministic child.
from gphyt_attnres_study.runtime import run_logged  # Global: cover the same helper used by Hugging Face publication.


def test_logged_command_keeps_stderr_out_of_returned_stdout(tmp_path, capsys):
    """Local: retain the warning in logs while parsing JSON; global: avoid duplicate publication after a successful CLI call."""
    command = [sys.executable, "-c", "import sys; print('diagnostic warning', file=sys.stderr); print('{\"slug\": \"fixture\"}')"]  # Local: reproduce the observed HF stream separation.
    output = run_logged(command, cwd=tmp_path)  # Local: use the production subprocess wrapper.
    assert json.loads(output) == {"slug": "fixture"}  # Global: non-JSON stderr must never enter the returned stdout.
    captured = capsys.readouterr()  # Local: inspect the full forwarded terminal text.
    assert "diagnostic warning" in captured.out + captured.err  # Global: fixing parsing must not hide the diagnostic.
