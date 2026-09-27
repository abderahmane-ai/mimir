"""Tools files: the YAML that declares a server's decision tools.

```yaml
tools:
  - name: route_ticket
    description: Route a support ticket to the team that owns it.
    decision:
      type: choice
      question: Which team should handle this ticket?
      options:
        billing: "Billing: payments, refunds and invoices"
        security: "Security: account access and fraud"
    risk: 0.01
```

`decision` is the `decision` object of `POST /v1/decide`; `risk` and `alpha` are optional.
Aliases and repeated keys are refused.
"""

from collections.abc import Hashable
from pathlib import Path
from typing import Final

import yaml
from pydantic import JsonValue, ValidationError
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

from mimir.core.errors import MimirError
from mimir.core.tools import ToolDefinitions

MAX_FILE_BYTES: Final = 1024 * 1024


class ToolFileError(MimirError, ValueError):
    """A tools file cannot be read, is not YAML, or declares invalid tools."""


class _Loader(yaml.SafeLoader):
    def compose_node(self, parent: Node | None, index: int) -> Node | None:
        if self.check_event(yaml.AliasEvent):
            message = "aliases are not allowed in a tools file"
            raise ComposerError(None, None, message, self.get_mark())
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, object]:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        repeated = sorted({str(key) for key in keys if keys.count(key) > 1})
        if repeated:
            message = f"keys {repeated} are repeated"
            raise ConstructorError(None, None, message, node.start_mark)
        return super().construct_mapping(node, deep=deep)


def _parse(text: str) -> JsonValue:
    loader = _Loader(text)
    try:
        document: JsonValue = loader.get_single_data()
    finally:
        loader.dispose()
    return document


def read_tools(path: Path) -> ToolDefinitions:
    """Read and validate a tools file. Errors name the file and the failing line or field."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            message = f"{path}: {size} bytes; a tools file holds at most {MAX_FILE_BYTES}"
            raise ToolFileError(message)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        message = f"{path}: {error}"
        raise ToolFileError(message) from error
    try:
        return ToolDefinitions.model_validate(_parse(text))
    except (yaml.YAMLError, ValidationError) as error:
        message = f"{path}: {error}"
        raise ToolFileError(message) from error
