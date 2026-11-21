"""Freeze serving's direct dependencies from the active, tested conda environment.

Run from the project root, with the mibici environment active:
    python tools/prepare_serving_requirements.py

The exported MLflow model declares precise versions of its serialization dependencies.
We refuse to generate an image with a different version of any declared package.
"""

from importlib import metadata
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "models/serving/champion/requirements.txt"
OUTPUT = ROOT / "requirements-serving.txt"

# The serialized model's full direct requirements plus all API/snapshot imports.
# Explicit names are PyPI distribution names; `pip` resolves indirect dependencies.
SERVING_DISTRIBUTIONS = (
    "mlflow",
    "cloudpickle",
    "loguru",
    "numpy",
    "packaging",
    "pandas",
    "psutil",
    "python-dotenv",
    "PyYAML",
    "rich",
    "scikit-learn",
    "scipy",
    "tqdm",
    "fastapi",
    "uvicorn",
    "pyarrow",
    "typer",
    "requests",
)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    if not CHAMPION.is_file():
        raise SystemExit(f"Missing MLflow model requirements: {CHAMPION}")

    model_versions: dict[str, str] = {}
    for line in CHAMPION.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" not in stripped:
            raise SystemExit(f"Champion requires an unpinned dependency: {stripped}")
        name, version = stripped.split("==", 1)
        model_versions[normalize(name)] = version.strip()

    lines = [
        "# Generated from the tested mibici environment by",
        "# python tools/prepare_serving_requirements.py",
        "# Exact direct dependency pins; install without conda in Docker.",
    ]
    for name in sorted(SERVING_DISTRIBUTIONS, key=normalize):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SystemExit(
                f"Required package {name!r} is not installed. "
                "Run `make requirements` and retry."
            ) from exc
        expected = model_versions.get(normalize(name))
        if expected is not None and version != expected:
            raise SystemExit(
                f"Version mismatch for {name}: current={version}, "
                f"champion requires={expected}. Update your conda environment first."
            )
        lines.append(f"{name}=={version}")

    missing = set(model_versions) - {normalize(x) for x in SERVING_DISTRIBUTIONS}
    if missing:
        raise SystemExit(f"Missing model dependencies from SERVING_DISTRIBUTIONS: {missing}")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(SERVING_DISTRIBUTIONS)} pinned direct dependencies)")
    for line in lines[3:]:
        print("  " + line)


if __name__ == "__main__":
    main()
