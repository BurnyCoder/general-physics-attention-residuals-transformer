"""Global context: copy only required, pinned upstream modules with their licenses.

Local context: GitHub's raw commit URLs make every copied byte attributable.
Sources: https://docs.python.org/3/library/urllib.request.html and docs/sources.md.
"""

import hashlib  # Local: SHA-256 identifies copied bytes in the provenance manifest.
import json  # Local: serialize provenance using a standard portable format.
from pathlib import Path  # Local: keep all writes rooted in this repository.
from urllib.request import urlopen  # Local: download public raw source, without credentials.

ROOT = Path(__file__).resolve().parents[1]  # Global: the repository owns its vendor tree.
GPHYT = "df1527c4773ababc6f8af0d117fb5803fa3bf74b"  # Global: reviewed upstream revision.
FLA = "a400aff6ec0eb81cc88721aedba782cd7d535357"  # Global: reviewed reference implementation.
FILES = [  # Local: include transitive model dependencies, excluding unused training machinery.
    "gphyt/__init__.py", "gphyt/models/__init__.py", "gphyt/models/model_specs.py",
    "gphyt/models/tokenizer/__init__.py", "gphyt/models/tokenizer/tokenizer.py",
    "gphyt/models/tokenizer/tokenizer_utils.py", "gphyt/models/transformer/__init__.py",
    "gphyt/models/transformer/model.py", "gphyt/models/transformer/attention.py",
    "gphyt/models/transformer/pos_encodings.py", "gphyt/models/transformer/derivatives.py",
    "gphyt/models/transformer/num_integration.py",
]


def main():
    """Local: download exact files; global: preserve reproducibility and MIT notices."""
    entries = [("FloWsnr/General-Physics-Transformer", GPHYT, p, "vendor/" + p) for p in FILES]  # Local: model imports remain unchanged.
    entries += [("FloWsnr/General-Physics-Transformer", GPHYT, "LICENSE", "vendor/GPHYT-LICENSE")]  # Local: retain the original license.
    entries += [("fla-org/flash-linear-attention", FLA, "fla/ops/attnres/naive.py", "vendor/fla_attnres.py")]  # Local: reuse the standalone operator.
    entries += [("fla-org/flash-linear-attention", FLA, "LICENSE", "vendor/FLA-LICENSE")]  # Local: preserve FLA attribution.
    manifest = []  # Local: collect source URL and content digest for every vendor file.
    for repo, revision, source, destination in entries:  # Global: identical rules cover both upstream projects.
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{source}"  # Local: commit-qualified URLs cannot drift with a branch.
        with urlopen(url, timeout=60) as response:  # Local: bound failed network requests.
            content = response.read()  # Local: preserve the source bytes without rewriting comments.
        path = ROOT / destination  # Local: resolve the destination inside the current project.
        path.parent.mkdir(parents=True, exist_ok=True)  # Local: create only required package directories.
        path.write_bytes(content)  # Local: copy reviewed upstream code verbatim.
        manifest.append(dict(path=destination, url=url, sha256=hashlib.sha256(content).hexdigest()))  # Global: make reuse auditable.
        print(destination, flush=True)  # Local: surface progress immediately.
    (ROOT / "vendor/provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")  # Global: publish exact provenance.


if __name__ == "__main__":  # Local: importing this utility never downloads anything.
    main()  # Global: explicit invocation performs the vendoring phase.
