from __future__ import annotations

import re

_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f-\x9f]")


def safe(text: str) -> str:
    return _CONTROLS.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
