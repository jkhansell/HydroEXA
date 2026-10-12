"""HydroEXA input file parser.

Parses the dot-separated parameter syntax used by HydroEXA `.in` files.
Handles comments, quoted strings, integers, floats, scientific notation,
negative values, scalar values, vectors (space-separated), inline comments,
and arbitrary whitespace.

Unknown parameters are silently ignored — only the parameters needed by
preprocessing are extracted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Regex that matches one token at a time:
#   1) a quoted string (double-quoted)
#   2) an inline comment (# ...)
#   3) a number (integer or float, possibly negative or scientific notation)
#   4) a dot-separated key (e.g. amr.blocking_factor)
#   5) any other non-whitespace (skipped — unknown tokens)
_TOKEN_RE = re.compile(
    r"""
    (?P<quoted>"[^"]*")
  | (?P<comment>#.*$)
  | (?P<number>-?(?:0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?))
  | (?P<key>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
  | (?P<other>\S+)
    """,
    re.VERBOSE | re.MULTILINE,
)


def _parse_value(token: str):
    """Convert a single token string to its Python type."""
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    # Try integer first
    try:
        return int(token)
    except ValueError:
        pass
    # Try float
    try:
        return float(token)
    except ValueError:
        pass
    # Return as string (shouldn't happen for valid tokens)
    return token


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedInput:
    """Immutable result of parsing a HydroEXA input file.

    Attributes:
        raw: The full parsed dict, keyed by dotted parameter name.
        hydroexa: HydroEXA.* parameters.
        amr: amr.* parameters.
        io: io.* parameters.
    """

    raw: dict[str, object] = field(repr=False)
    hydroexa: dict[str, object] = field(default_factory=dict)
    amr: dict[str, object] = field(default_factory=dict)
    io: dict[str, object] = field(default_factory=dict)

    @property
    def max_time(self) -> float:
        return float(self.hydroexa.get("max_time", 1.0))

    @property
    def model(self) -> str:
        return str(self.hydroexa.get("model", "Roe"))

    @property
    def cfl(self) -> float:
        return float(self.hydroexa.get("cfl", 0.5))

    @property
    def h_grad_thresh(self) -> list[float]:
        v = self.hydroexa.get("h_grad_thresh", 0.005)
        return [_to_float(x) for x in _ensure_list(v)]

    @property
    def umag_grad_thresh(self) -> list[float]:
        v = self.hydroexa.get("umag_grad_thresh", 0.005)
        return [_to_float(x) for x in _ensure_list(v)]

    @property
    def z_grad_thresh(self) -> list[float]:
        v = self.hydroexa.get("z_grad_thresh", 0.005)
        return [_to_float(x) for x in _ensure_list(v)]

    @property
    def mass_diag_freq(self) -> float:
        return float(self.hydroexa.get("mass_diag_freq", -1.0))

    @property
    def lo_bc(self) -> tuple[int, ...]:
        v = self.hydroexa.get("lo_bc", (0, 0))
        return tuple(int(x) for x in _ensure_list(v))

    @property
    def hi_bc(self) -> tuple[int, ...]:
        v = self.hydroexa.get("hi_bc", (0, 0))
        return tuple(int(x) for x in _ensure_list(v))

    # --- AMR parameters ---

    @property
    def max_level(self) -> int:
        return int(self.amr.get("max_level", 0))

    @property
    def regrid_int(self) -> int:
        return int(self.amr.get("regrid_int", 2))

    @property
    def do_reflux(self) -> int:
        return int(self.amr.get("do_reflux", 1))

    @property
    def do_subcycle(self) -> int:
        return int(self.amr.get("do_subcycle", 1))

    @property
    def blocking_factor(self) -> int:
        return int(self.amr.get("blocking_factor", 16))

    @property
    def terrain_ref_lev(self) -> int:
        return int(self.amr.get("terrain_ref_lev", 0))

    @property
    def ref_ratio(self) -> tuple[int, ...]:
        v = self.amr.get("ref_ratio", (2, 2, 2))
        return tuple(int(x) for x in _ensure_list(v))

    # --- IO parameters ---

    @property
    def io_freq(self) -> float:
        return float(self.io.get("io_freq", 1.0))

    @property
    def chk_freq(self) -> float:
        return float(self.io.get("chk_freq", -1.0))

    @property
    def plot_freq(self) -> float:
        return float(self.io.get("plot_freq", -1.0))

    @property
    def async_io(self) -> bool:
        v = self.io.get("async_io", 0)
        return bool(int(v))

    @property
    def chk_type(self) -> str:
        return str(self.io.get("chk_type", "native"))

    @property
    def plot_file(self) -> str:
        return str(self.io.get("plot_file", "plt"))

    @property
    def plot_type(self) -> str:
        return str(self.io.get("plot_type", "native"))

    @property
    def chk_file(self) -> str:
        return str(self.io.get("chk_file", "chk"))

    @property
    def input_file(self) -> str:
        return str(self.io.get("input_file", "input.h5"))

    @property
    def restart_chkfile(self) -> str:
        return str(self.io.get("restart_chkfile", "restart"))


def _ensure_list(value: object) -> list[object]:
    """Wrap a scalar in a list; pass through lists/tuples."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_float(value: object) -> float:
    return float(value)


def parse_input_file(path: str | Path) -> ParsedInput:
    """Parse a HydroEXA input file and return a :class:`ParsedInput`.

    Args:
        path: Path to the input file.

    Returns:
        A :class:`ParsedInput` with parameters grouped by prefix.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return _parse_text(text)


def _parse_text(text: str) -> ParsedInput:
    """Parse input text and return a :class:`ParsedInput`."""
    raw: dict[str, object] = {}
    current_key: str | None = None
    current_values: list[object] = []
    in_vector: bool = False

    def _flush() -> None:
        nonlocal current_key, current_values, in_vector
        if current_key is not None:
            if in_vector and current_values:
                raw[current_key] = current_values
            elif current_values:
                raw[current_key] = current_values[0] if len(current_values) == 1 else current_values
            else:
                raw[current_key] = True  # bare key with no value
            current_key = None
            current_values = []
            in_vector = False

    for match in _TOKEN_RE.finditer(text):
        group = match.lastgroup
        token = match.group()

        if group == "comment":
            continue

        if group == "quoted":
            val = _parse_value(token)
            if current_key is not None:
                current_values.append(val)
            continue

        if group == "number":
            val = _parse_value(token)
            if current_key is not None:
                current_values.append(val)
            continue

        if group == "key":
            _flush()
            current_key = token
            continue

        # 'other' — skip unknown tokens
        continue

    _flush()

    # Group by prefix
    hydroexa: dict[str, object] = {}
    amr: dict[str, object] = {}
    io: dict[str, object] = {}

    for key, val in raw.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            prefix, name = parts
            if prefix == "HydroEXA":
                hydroexa[name] = val
            elif prefix == "amr":
                amr[name] = val
            elif prefix == "io":
                io[name] = val
        # Keys without a recognised prefix are silently ignored.

    return ParsedInput(raw=raw, hydroexa=hydroexa, amr=amr, io=io)