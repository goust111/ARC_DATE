# Combo variant of b5ca7ac4 (index 12)
# combo_spec: D11/0+D12/2
# Name: 大量框架+完全包含才偏移

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import crop_bbox, get_bg, paste


"""
Design analysis for combo D11/0 + D12/2

a) Concrete behavior and trace
- The original solver moves every teal frame to the left side and every red frame to
  the right side, preserving rows and using input-column order within each color.
  Any row overlap with an already edge-placed same-color frame forces a one-step
  horizontal offset, so the original generator only needs two layers.
- In the fused variant, the frame count rises to 10-12 total and the overlap test is
  stricter: a shift happens only when one 5-row span fully contains the other.
- Because every frame is still exactly 5x5, "full containment" between row spans
  collapses to equality of the row start. Example for teal processing order:
  rows `[1, 3, 1, 1, 9, 1]`.
  Under the old rule:
  `1 -> layer0`, `3 -> layer1` because rows 1..5 and 3..7 overlap,
  then later `1`s are forced even farther inward by overlap checks.
  Under the fused rule:
  `1 -> layer0`, `3 -> layer0` because partial overlap is ignored,
  next `1 -> layer1`, next `1 -> layer2`, `9 -> layer0`, final `1 -> layer3`.
  The visible result is a deep 4-column stack for the repeated exact row, plus one
  partially overlapping frame that stays on the edge and visibly occludes part of the
  first edge frame instead of being shifted away.

b) Complexity calibration
- Counts are raised to 5-6 teal and 5-6 red frames, so each sample has 10-12 total.
  That is enough to make 3-4 stacking layers appear without making the puzzle depend
  on a huge number of object categories.
- The generator uses only a small row-pattern vocabulary per side. Every sample has
  repeated exact rows to prove the containment-triggered stack, and also has at least
  one partial-overlap row offset by 2 or 3 to prove that partial overlap no longer
  causes shifting.
- I keep the original code/spec ordering rule based on input columns, not the loose
  prose summary above. That keeps the puzzle anchored to the source task instead of
  silently changing the sorting axis too.

c) Implicit-assumption audit of `main()`
- Scanning 5x5 windows for monochrome borders still works because the generator keeps
  the input squares non-overlapping and axis-aligned.
- The old `placed_at_default` list assumed only two output columns per side. That is
  invalid once 10-12 frames can create 3-4 layers, so the new solver tracks occupied
  row spans separately for every layer and chooses the shallowest legal one.
- The old `overlaps(...)` predicate is invalid for D11/0. It is replaced by a
  symmetric "fully nested" test. With fixed 5x5 squares this means equal row spans,
  but the implementation keeps the generic containment wording instead of hard-coding
  equality.
- `paste(... transparent=bg)` is still safe. The input generator never overlaps input
  frames, while output overdraw is intentional and comes only from the stacking rule.
- New solver assumption: the number of required layers must fit in the grid width.
  The generator caps the designed depth at 4, and width 42 gives columns
  `0,5,10,15` on the left and `37,32,27,22` on the right with a visible center gap.

d) Placement legality
- Input placement is guaranteed by construction. Teal and red use disjoint row bands,
  and each side uses fixed monotone input columns spaced by exactly 5, so no two
  input 5x5 frames overlap even when they share the same row.
- Output placement is also in-bounds because the solver only uses up to 4 layers per
  side on a 42-column canvas.

e) Visual observability
- A human can see two distinct cues in the examples:
  1. many frames of each border color still migrate to their designated side;
  2. some same-color frames with identical top rows form deep horizontal stacks,
     while another same-color frame with a visibly different top row but overlapping
     height stays in the edge column and partly covers the earlier one.
- No color-id arithmetic or hidden metadata is used. The changed behavior depends only
  on visible row relationships and visible input-column order.

f) Visual cleanliness
- The generator limits the deliberate partial-overlap occlusion to one compact cluster
  per side, and keeps the two colors in separated vertical bands. This preserves the
  intended "messier than base task" look without turning the whole output into an
  unreadable pile of overdraw.

g) Change inferability
- There is no new recoloring or symbolic remapping. Inner colors remain unique within
  each border color so each frame is trackable across input and output.
- The key fused contrast is structural: exact row repetition creates extra depth,
  partial overlap does not. The generator rejects samples where the fused output would
  match a generalized legacy "any overlap stacks inward" output, so the D11 change is
  always active rather than accidental.

9) New assumptions introduced and generator guarantees
- Assumption: at least one partial-overlap pair exists. Guarantee: each side pattern
  includes one row offset by 2 or 3 from the repeated anchor row.
- Assumption: multi-layer depth is visibly larger than the base task. Guarantee: each
  side has at least 3 layers, and some samples reach 4.
- Assumption: the new containment predicate matters. Guarantee: generation compares the
  fused output against an "any overlap" baseline and rejects any sample where they are
  identical.
"""


SQ = 5
AZURE = 8
RED = 2
GRID_H = 30
GRID_W = 42
N_AZURE_RANGE = (5, 6)
N_RED_RANGE = (5, 6)
MAX_LAYERS = 4
GEN_RETRIES = 400

AZURE_INPUT_COLS = (7, 13, 19, 25, 31, 37)
RED_INPUT_COLS = (30, 24, 18, 12, 6, 0)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _fully_nested(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (int(a0) <= int(b0) and int(b1) <= int(a1)) or (int(b0) <= int(a0) and int(a1) <= int(b1))


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rectangles_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) < SQ and abs(int(a[1]) - int(b[1])) < SQ


def _detect_squares(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    squares: list[dict] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != int(frame_col)).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _stack_side(
    ordered_squares: list[dict],
    width: int,
    edge_left: bool,
    conflict_fn,
) -> list[dict]:
    placed_rows_by_layer: list[list[tuple[int, int]]] = []
    placements: list[dict] = []
    for square in ordered_squares:
        r0 = int(square["r0"])
        r1 = int(r0 + SQ - 1)
        layer = None
        for idx, intervals in enumerate(placed_rows_by_layer):
            if any(conflict_fn(r0, r1, pr0, pr1) for pr0, pr1 in intervals):
                continue
            layer = int(idx)
            break
        if layer is None:
            layer = len(placed_rows_by_layer)
            placed_rows_by_layer.append([])
        placed_rows_by_layer[layer].append((r0, r1))
        dest_c = int(layer * SQ) if edge_left else int(width - SQ * (layer + 1))
        assert 0 <= dest_c <= width - SQ, "stack depth exceeds canvas width"
        placements.append(
            {
                "frame": int(square["frame"]),
                "r0": r0,
                "c0": int(square["c0"]),
                "dest_c": dest_c,
                "layer": layer,
                "patch": square["patch"],
            }
        )
    return placements


def _render_from_ordered_sides(
    bg: int,
    azure_ordered: list[dict],
    red_ordered: list[dict],
    *,
    conflict_fn,
) -> tuple[np.ndarray, dict]:
    out = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    azure_placed = _stack_side(azure_ordered, GRID_W, True, conflict_fn)
    red_placed = _stack_side(red_ordered, GRID_W, False, conflict_fn)
    for item in azure_placed + red_placed:
        paste(out, item["patch"], int(item["r0"]), int(item["dest_c"]), transparent=bg)

    metrics = {
        "azure_max_depth": max((int(item["layer"]) + 1 for item in azure_placed), default=0),
        "red_max_depth": max((int(item["layer"]) + 1 for item in red_placed), default=0),
        "azure_layers": [int(item["layer"]) for item in azure_placed],
        "red_layers": [int(item["layer"]) for item in red_placed],
    }
    return out, metrics


def _count_partial_pairs(ordered_squares: list[dict]) -> int:
    rows = [int(square["r0"]) for square in ordered_squares]
    count = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a0 = int(rows[i])
            b0 = int(rows[j])
            if _overlaps(a0, a0 + SQ - 1, b0, b0 + SQ - 1) and not _fully_nested(a0, a0 + SQ - 1, b0, b0 + SQ - 1):
                count += 1
    return count


def _behavior_profile(azure_ordered: list[dict], red_ordered: list[dict], bg: int) -> dict:
    fused_out, fused_metrics = _render_from_ordered_sides(bg, azure_ordered, red_ordered, conflict_fn=_fully_nested)
    legacy_out, legacy_metrics = _render_from_ordered_sides(bg, azure_ordered, red_ordered, conflict_fn=_overlaps)
    return {
        "fused_output": fused_out,
        "legacy_output": legacy_out,
        "azure_partial_pairs": _count_partial_pairs(azure_ordered),
        "red_partial_pairs": _count_partial_pairs(red_ordered),
        "azure_max_depth": int(fused_metrics["azure_max_depth"]),
        "red_max_depth": int(fused_metrics["red_max_depth"]),
        "legacy_azure_max_depth": int(legacy_metrics["azure_max_depth"]),
        "legacy_red_max_depth": int(legacy_metrics["red_max_depth"]),
    }


def _sample_side_rows(n: int, top_band: bool) -> list[int]:
    if top_band:
        base = random.randint(0, 2)
        far = random.randint(8, 10)
    else:
        base = random.randint(15, 17)
        far = random.randint(21, 23)
    overlap_row = int(base + random.choice((2, 3)))
    patterns = {
        5: [
            [base, overlap_row, base, far, base],
            [base, overlap_row, base, base, far],
        ],
        6: [
            [base, overlap_row, base, far, base, base],
            [base, overlap_row, base, base, far, base],
            [base, overlap_row, base, far, base, far],
        ],
    }
    rows = list(random.choice(patterns[int(n)]))
    assert all(0 <= int(r) <= GRID_H - SQ for r in rows)
    return rows


def _ordered_side_squares(rows: list[int], cols: tuple[int, ...], frame_color: int, colors: list[int]) -> list[dict]:
    ordered: list[dict] = []
    for row, col, inner_color in zip(rows, cols[: len(rows)], colors):
        ordered.append(
            {
                "frame": int(frame_color),
                "r0": int(row),
                "c0": int(col),
                "patch": _make_patch(int(frame_color), int(inner_color)),
            }
        )
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack many bordered 5x5 frames onto their color-designated edges.

    Input structure:
    - Background is the most common color.
    - The grid contains 10-12 non-overlapping 5x5 framed squares.
    - Teal (8) and red (2) are the only frame colors; each frame has a uniform 3x3 inner fill.

    Rule:
    - Teal frames are processed by increasing input column and stacked from the left edge.
    - Red frames are processed by decreasing input column and stacked from the right edge.
    - Each frame keeps its original top row.
    - A frame moves one more square-width inward only when its 5-row span is fully contained
      in a previously used layer on that same side; partial row overlap does not force a shift.

    Key insight:
    - With fixed 5x5 frames, "full row containment" means matching the same row span, so exact
      row repeats create deeper layers while merely partial overlaps stay in place and can overdraw.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)
    azure_ordered = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    red_ordered = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))

    for item in _stack_side(azure_ordered, w, True, _fully_nested):
        paste(out, item["patch"], int(item["r0"]), int(item["dest_c"]), transparent=bg)
    for item in _stack_side(red_ordered, w, False, _fully_nested):
        paste(out, item["patch"], int(item["r0"]), int(item["dest_c"]), transparent=bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([color for color in range(10) if color not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_rows = _sample_side_rows(n_azure, top_band=True)
    red_rows = _sample_side_rows(n_red, top_band=False)

    azure_choices = [color for color in range(10) if color not in {bg, AZURE}]
    red_choices = [color for color in range(10) if color not in {bg, RED}]
    if len(azure_choices) < n_azure or len(red_choices) < n_red:
        return None
    azure_colors = random.sample(azure_choices, n_azure)
    red_colors = random.sample(red_choices, n_red)

    azure_ordered = _ordered_side_squares(azure_rows, AZURE_INPUT_COLS, AZURE, azure_colors)
    red_ordered = _ordered_side_squares(red_rows, RED_INPUT_COLS, RED, red_colors)

    profile = _behavior_profile(azure_ordered, red_ordered, bg)
    if profile["azure_partial_pairs"] + profile["red_partial_pairs"] < 2:
        return None
    if max(int(profile["azure_max_depth"]), int(profile["red_max_depth"])) < 3:
        return None
    if np.array_equal(profile["fused_output"], profile["legacy_output"]):
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for square in azure_ordered + red_ordered:
        paste(inp, square["patch"], int(square["r0"]), int(square["c0"]))

    top_lefts = [(int(square["r0"]), int(square["c0"])) for square in azure_ordered + red_ordered]
    counts = Counter(top_lefts)
    if counts and max(counts.values()) > 1:
        return None
    for i in range(len(top_lefts)):
        for j in range(i + 1, len(top_lefts)):
            if _rectangles_overlap(top_lefts[i], top_lefts[j]):
                return None

    detected = _detect_squares(inp)
    if len(detected) != n_azure + n_red:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, profile["fused_output"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
