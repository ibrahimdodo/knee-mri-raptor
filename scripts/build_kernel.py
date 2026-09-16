"""Assemble a Kaggle script kernel from src/raptor_core.py + kaggle/<job>/main.py.

Kaggle script kernels are a single file, so the core module is pasted in front of the job's main
and the job's `import raptor_core as rc` is pointed at the pasted copy. Usage:

    python scripts/build_kernel.py gold_run        # or gold_headers
    kaggle kernels push -p kaggle/gold_run/build
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOBS = {
    "gold_run": {
        "id": "ibrahimdodo/raptor-knee-gold-reproduction",
        "title": "Raptor Knee Gold Reproduction",
        "dataset_sources": ["dreaddevelopment/raptor-knee-native384dense"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
    },
    "gold_headers": {
        "id": "ibrahimdodo/raptor-knee-gold-headers",
        "title": "Raptor Knee Gold Headers",
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "gpu": False,
    },
}


def build(job: str) -> Path:
    spec = JOBS[job]
    core = (ROOT / "src" / "raptor_core.py").read_text()
    main = (ROOT / "kaggle" / job / "main.py").read_text()
    marker = "import raptor_core as rc  # replaced by the build script"
    assert marker in main, f"{job}/main.py must import the core with the marker line"
    main = main.replace(marker, "rc = sys.modules[__name__]")

    out = ROOT / "kaggle" / job / "build"
    out.mkdir(exist_ok=True)
    code_file = spec["id"].split("/")[1] + ".py"
    (out / code_file).write_text(core + "\n\n# " + "=" * 76 + "\n# JOB: " + job + "\n# " + "=" * 76
                                 + "\nimport sys\n" + main)
    meta = {
        "id": spec["id"], "title": spec["title"], "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": spec.get("gpu", True), "enable_tpu": False, "enable_internet": False,
        **({"machine_shape": "NvidiaTeslaT4"} if spec.get("gpu", True) else {}),
        "dataset_sources": spec["dataset_sources"], "competition_sources": spec["competition_sources"],
        "kernel_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out / code_file


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "gold_run"))
