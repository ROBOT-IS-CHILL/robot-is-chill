import tomllib
from pathlib import Path
import sys


def main():
    # For now, all we're checking is if they parse
    failed = False
    for file in Path("data/custom").glob("*.toml"):
        print(f"Testing {file}...")
        try:
            with open(file, "rb") as f:
                tomllib.load(f)
        except Exception as err:
            print(f"File {file} failed to parse!")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
