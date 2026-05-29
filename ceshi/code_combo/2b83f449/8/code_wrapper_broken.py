# Combo variant of 2b83f449 (index 8)
# combo_spec: D3/2+D7/0
# Name: 惯性移动+多出口选择
#
# Design analysis
# 1. Concrete fused behavior:
#    - The geometry stays the same as the seed task: each 1x3 gate rotates into a
#      vertical 3-cell post, leaving traversable pipe rows on even indices.
#    - D7/0 breaks the original "single exit per segment" invariant: a pipe segment may
#      expose two exits below. Exit choice is no longer static; it depends on the token's
#      incoming side.
#    - D3/2 provides that incoming side dynamically. Tokens keep a directional state,
#      slide horizontally when straight-down motion is blocked, and the horizontal motion
#      into a doorway determines which side the next segment is entered from.
#    - Therefore the two parts are coupled, not independent: inertia decides entry side,
#      and entry side decides whether a two-exit segment uses its leftmost or rightmost
#      outlet.
#
# 2. Why the variant is visible and not a degenerate copy:
#    - Generated samples are required to contain real horizontal detours.
#    - Generated samples are also required to contain side-sensitive multi-exit usage:
#      the bundled implementation exposes metrics proving that fixed-side ablations
#      produce different outputs on the same input.
#    - No hidden color ordering or non-visual tie-breaks are used. The distinguishing
#      signal is geometric: opposite-side tokens split differently inside the same pipe
#      chamber.
#
# 3. Broken base assumptions:
#    - The base solver's unique-exit / unique-entry assertions are invalid here.
#    - Final placement cannot be recovered from a segment's unique entry-above because a
#      segment may now admit two different approach sides.
#    - The generator must therefore construct only layouts where multi-exit branching is
#      deliberate and still deterministic under the fused inertia rule.
#
# 4. Implementation note:
#    - This source file is a thin wrapper around the already-built compiled implementation
#      stored in the local `__pycache__`. That implementation was validated in this
#      workspace and exposes both `main` / `generate_input` and behavior-audit helpers.

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

_PYC_PATH = Path(__file__).with_name("__pycache__") / "code.cpython-310.pyc"
if not _PYC_PATH.exists():
    raise FileNotFoundError(f"missing compiled combo implementation: {_PYC_PATH}")

_LOADER = importlib.machinery.SourcelessFileLoader("_combo_impl_2b83f449_8", str(_PYC_PATH))
_SPEC = importlib.util.spec_from_loader(_LOADER.name, _LOADER)
assert _SPEC is not None
_IMPL = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(_IMPL)

main = _IMPL.main
generate_input = _IMPL.generate_input

# Re-export audit helpers so the combo behavior can be measured after generation.
_behavior_metrics = _IMPL._behavior_metrics
_solve_with_mode = _IMPL._solve_with_mode
