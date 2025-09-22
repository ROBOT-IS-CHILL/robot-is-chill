import tomllib
from pathlib import Path
import sys


def main():
    # For now, all we're checking is if they parse
    failures = []
    for file in Path("data/custom").glob("*.toml"):
        try:
            with open(file, "rb") as f:
                tomllib.load(f)
        except Exception as err:
            failures.append((file, err))
    for path, err in failures:
        print(f"File {path} failed to parse: {err}")
    return 1 if len(failures) else 0


if __name__ == "__main__":
    sys.exit(main())
