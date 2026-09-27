"""Write the server's OpenAPI document to a file."""

import json
import sys
from pathlib import Path

from mimir.server.app import openapi_document


def write_document(path: Path) -> None:
    text = json.dumps(openapi_document(), indent=2, ensure_ascii=False, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


if __name__ == "__main__":
    write_document(Path(sys.argv[1]))
