from __future__ import annotations

ART = r"""
     _        _                     ___    ___   _____
  __| |  ___ | |__   _   _   __ _  / _ \  / _ \ |___  |
 / _` | / _ \| '_ \ | | | | / _` || (_) || | | |   / /
| (_| ||  __/| |_) || |_| || (_| | \__, || |_| |  / /
 \__,_| \___||_.__/  \__,_| \__, |   /_/  \___/  /_/
                            |___/
""".strip("\n").splitlines()

TAGLINE = "natural-language code retrieval"
CREDITS = ["By Aadvik Chawla", "   Kamneev Singh"]
MAX_WIDTH = 80
INDENT = "  "


def render() -> str:
    lines = [INDENT + line for line in ART]
    lines += ["", INDENT + TAGLINE, *(INDENT + line for line in CREDITS)]
    return "\n" + "\n".join(lines) + "\n"


def show() -> None:
    print(render())


if __name__ == "__main__":
    show()
