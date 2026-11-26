"""Build the Moodle submission zip. Run from the project root: python tools/make_zip.py"""

import zipfile
from pathlib import Path

TARGET = Path("dl-har-project-hcy41e-moodle.zip")
FILES = ["README.md", "LICENSE", "requirements.txt"]
FOLDERS = ["configs", "src", "docs", "outputs/checkpoints"]
RESULT_GLOB = "outputs/*.json"
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}

paths = [Path(f) for f in FILES]
for folder in FOLDERS:
    paths += [p for p in sorted(Path(folder).rglob("*")) if p.is_file()]
paths += sorted(Path(".").glob(RESULT_GLOB))
paths = [p for p in paths if not SKIP_PARTS & set(p.parts) and not p.name.startswith("~$")]

with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.as_posix())
print(f"Wrote {TARGET} with {len(paths)} files:")
for path in paths:
    print("  ", path.as_posix())
