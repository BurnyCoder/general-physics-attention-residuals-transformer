"""Global: export measured artifacts for a static GitHub Pages viewer.

Local: no GPU execution, private files, credentials, or synthetic replacement results.
Sources: https://plotly.com/python/interactive-html-export/ and GitHub Pages documentation.
"""
import json  # Local: strict JSON is portable to the browser.
import shutil  # Local: copy only explicitly public media and the paper.
import numpy as np  # Local: deterministic visual precision reduction, without altering evaluation metrics.
import pandas as pd  # Local: read report tables with explicit null preservation.
from plotly.offline import get_plotlyjs  # Local: bundle the pinned Plotly renderer without a third-party CDN dependency.
from .comparisons import MODES  # Global: public field examples cover every trained architecture in the matched comparison.
from .runtime import PROJECT, read_json, root, stamp, write_json  # Global: one source of artifact paths and timestamps.


def table_records(filename):
    """Local: CSV to JSON-safe rows; global: undefined estimates remain null, not zero."""
    path = PROJECT / "reports" / filename  # Local: allowlisted generated report table.
    return json.loads(pd.read_csv(path).to_json(orient="records")) if path.exists() else []  # Local: pandas represents missing numerical values as JSON null.


def export_site():
    """Local: build static public data; global: local and hosted viewers share measured evidence."""
    site = PROJECT / "site"  # Local: GitHub deploys only this directory.
    destination = site / "data"  # Local: separate generated data from hand-authored interface code.
    destination.mkdir(parents=True, exist_ok=True)  # Local: make the explicit public export directory.
    curves, routing, rollouts = {}, {}, []  # Local: actual saved evidence only.
    for path in sorted((root() / "runs").glob("*/metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]  # Local: export after a completed pipeline phase, not midway through an append.
        if rows:
            frame = pd.DataFrame(rows).drop_duplicates("step", keep="last")  # Local: retain replayed updates once while full logs remain in the model release.
            grouped = frame.groupby((frame.step - 1) // 50).agg(step=("step", "max"), loss=("loss", "mean"), gradient_norm=("gradient_norm", "mean"), seconds=("seconds", "mean"))  # Local: fixed disjoint fifty-update means include every update.
            curves[path.parent.name] = json.loads(grouped.to_json(orient="records"))  # Global: public curves disclose their aggregation explicitly.
    for path in sorted((root() / "evaluation").glob("*.json")):
        result = read_json(path)  # Local: raw evaluation provides routing diagnostics.
        if result.get("routing") and result.get("intervention") is None:
            routing[path.stem] = result["routing"][0]  # Local: fixed first diagnostic window, not a selected interesting pattern.
    for path in sorted((root() / "visuals").glob("*.npz")):
        if not path.stem.startswith(tuple(f"{mode}_seed0_" for mode in MODES)) or "_uniform_" in path.stem or "_shuffle_" in path.stem:
            continue  # Global: public field cases are prespecified seed-zero examples.
        arrays, payload = np.load(path), {}  # Local: use actual saved rollouts without rerunning models.
        channel = 2 if arrays["active"][2] else 1  # Local: buoyancy or density are the fixed representative fields.
        for kind in ("truth", "prediction"):
            values = np.round(arrays[kind][..., channel], 5)  # Local: five-decimal display precision reduces file size; full metrics remain unchanged.
            objects = values.astype(object)  # Local: JSON requires explicit null for divergent fields.
            objects[~np.isfinite(values)] = None  # Global: never replace nonfinite predictions with plausible finite values.
            payload[kind] = objects.tolist()  # Local: nested arrays are directly usable by Plotly heatmaps.
        payload.update(field="buoyancy" if channel == 2 else "density", visual_decimal_places=5, resolution=list(arrays["truth"].shape[1:3]))  # Global: preview precision and actual grid are disclosed.
        filename = f"{path.stem}.json"  # Local: known experiment identifiers define stable public paths.
        write_json(destination / filename, payload)  # Local: one case is fetched only when selected.
        rollouts.append(dict(name=path.stem, file=f"data/{filename}", field=payload["field"]))  # Local: browser selects only exported files.
    ledger = read_json(root() / "gpu_budget.json", {"sessions": []})  # Local: sum actual recorded usage, not the requested allocation.
    payload = dict(updated=stamp(), primary=read_json(PROJECT / "reports/primary_analysis.json", {}), runs=table_records("experiments.csv"), metrics=table_records("metrics.csv"), interventions=table_records("interventions.csv"), power=read_json(PROJECT / "reports/power_sensitivity.json", {}).get("rows", []), gpu_hours=sum(row["seconds"] for row in ledger["sessions"]) / 3600, curves=curves, routing=routing, rollouts=rollouts, datasets=list(read_json(PROJECT / "configs/study.json")["data"]))  # Global: every browser claim points back to saved evidence.
    payload["accuracy"] = table_records("accuracy_summary.csv")  # Global: the prominent baseline table uses the paper's measured reductions, without a second statistical calculation.
    comparison = PROJECT / "reports/figures/field_comparison.png"  # Local: exported scientific figure comes from the same full-precision saved arrays as the paper.
    payload["comparison_image"] = "field_comparison.png" if comparison.exists() else None  # Global: the viewer hides unavailable evidence instead of displaying a broken or synthetic example.
    if comparison.exists():
        shutil.copy2(comparison, site / "field_comparison.png")  # Local: make the matched all-model figure available without cross-origin requests.
    write_json(destination / "study.json", payload)  # Local: publish complete strict JSON atomically.
    (site / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")  # Local: retain the library's bundled copyright/license header.
    shutil.copy2(PROJECT / "assets/architecture.svg", site / "architecture.svg")  # Local: reuse the same controlled-architecture diagram.
    if (PROJECT / "paper/main.pdf").exists():
        shutil.copy2(PROJECT / "paper/main.pdf", site / "paper.pdf")  # Local: public paper link resolves to an actually compiled artifact.
    return payload  # Local: export tests can inspect integrity without a browser.
