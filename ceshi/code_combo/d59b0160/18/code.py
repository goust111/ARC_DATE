# Combo variant of d59b0160 (index 18)
# combo_spec: D3/1+D11/1
# Name: 调色板条带图例+单行/单列图例
#
# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    Original: read an unordered color set from a 3x3 square at the top-left, then erase every
#    non-bg 4-connected component that contains all legend colors anywhere inside it.
#    Fused: the legend is now a 2-cell-thick strip on one edge; each column/row is duplicated
#    across the strip thickness, so the visible strip encodes a 1D ordered sequence. Components
#    are still solid 0-rectangles with colored marks, but deletion now requires two conditions:
#    the component contains every legend color, and those colors appear in the same left-to-right
#    or top-to-bottom order as the strip.
# b) Complexity calibration:
#    The legend keeps only 3 colors and the component family stays rectangular, so the task is
#    still learnable from a few examples. The main visible novelty is that some components have
#    all three target colors yet survive because their order is wrong; this avoids collapsing
#    back to the original set-only rule.
# c) Broken old assumptions:
#    `_find_legend` in the seed assumes a top-left square with an L-border. That no longer exists.
#    The seed also labels every non-bg cell as a candidate component; here the strip itself is
#    non-bg, so the solver must detect and mask it out before component analysis. Finally, the
#    old subset check ignores ordering, which would make D11/1 irrelevant, so ordering must enter
#    the predicate explicitly.
# d) Placement legality:
#    The generator places a duplicated 2x3 / 3x2 legend on one edge, keeps one bg clearance
#    inward from the strip, and uses `random_free_location(..., padding=1)` for the remaining
#    rectangles so objects do not overlap or merge.
# e) Visual observability:
#    Ordered components place the three target colors in monotone slots along the strip axis.
#    Wrong-order kept components also contain all three colors, so the order cue is visible in
#    the input/output pairs and cannot be explained by the original rule.
# f) Visual cleanliness:
#    Target colors appear as compact domino markers on top of solid 0-rectangles; extra noise
#    uses only non-target colors, so the ordered markers remain easy to see.
# g) Inference support:
#    Every generated sample contains at least one kept component with the full target set in the
#    wrong order. That guarantees the fused behavior differs from the original set-based rule on
#    the same input and gives the learner repeated evidence that sequence, not just membership,
#    matters.

from __future__ import annotations

import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask, paste, random_free_location


GRID_H, GRID_W = 16, 16
BG = 7
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
LEGEND_THICKNESS = 2
LEGEND_LEN = 3
NUM_KEPT = 3
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
DENSITY_LO = 0.08
DENSITY_HI = 0.22
MAX_RETRIES = 1600


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(mask.tolist()):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _legend_candidates(g: np.ndarray, bg: int) -> list[tuple[int, str, int, int, list[int], np.ndarray]]:
    """
    Return candidate edge strips as:
      (score, axis, side, start, seq, mask)
    axis is "h" for left->right reading and "v" for top->bottom reading.
    """
    h, w = g.shape
    out: list[tuple[int, str, int, int, list[int], np.ndarray]] = []

    if h >= 3:
        top_eq = (g[0, :] == g[1, :]) & (g[0, :] != bg) & (g[2, :] == bg)
        for c0, c1 in _runs(top_eq):
            seq = [int(x) for x in g[0, c0:c1].tolist()]
            if c1 - c0 >= NUM_TARGETS and len(set(seq)) == len(seq):
                mask = np.zeros_like(g, dtype=bool)
                mask[0:2, c0:c1] = True
                out.append((c1 - c0, "h", 0, c0, seq, mask))

        bot_eq = (g[h - 1, :] == g[h - 2, :]) & (g[h - 1, :] != bg) & (g[h - 3, :] == bg)
        for c0, c1 in _runs(bot_eq):
            seq = [int(x) for x in g[h - 1, c0:c1].tolist()]
            if c1 - c0 >= NUM_TARGETS and len(set(seq)) == len(seq):
                mask = np.zeros_like(g, dtype=bool)
                mask[h - 2 : h, c0:c1] = True
                out.append((c1 - c0, "h", h - 2, c0, seq, mask))

    if w >= 3:
        left_eq = (g[:, 0] == g[:, 1]) & (g[:, 0] != bg) & (g[:, 2] == bg)
        for r0, r1 in _runs(left_eq):
            seq = [int(x) for x in g[r0:r1, 0].tolist()]
            if r1 - r0 >= NUM_TARGETS and len(set(seq)) == len(seq):
                mask = np.zeros_like(g, dtype=bool)
                mask[r0:r1, 0:2] = True
                out.append((r1 - r0, "v", r0, 0, seq, mask))

        right_eq = (g[:, w - 1] == g[:, w - 2]) & (g[:, w - 1] != bg) & (g[:, w - 3] == bg)
        for r0, r1 in _runs(right_eq):
            seq = [int(x) for x in g[r0:r1, w - 1].tolist()]
            if r1 - r0 >= NUM_TARGETS and len(set(seq)) == len(seq):
                mask = np.zeros_like(g, dtype=bool)
                mask[r0:r1, w - 2 : w] = True
                out.append((r1 - r0, "v", r0, w - 2, seq, mask))

    return out


def _find_legend_strip(g: np.ndarray, bg: int) -> tuple[str, list[int], np.ndarray]:
    candidates = _legend_candidates(g, bg)
    assert candidates, "no edge strip legend found"
    best = max(candidates, key=lambda item: item[0])
    _score, axis, _r0, _c0, seq, mask = best
    return axis, seq, mask


def _sequence_positions(
    g: np.ndarray, pts: np.ndarray, legend_seq: list[int], axis: str
) -> list[float] | None:
    pos: list[float] = []
    for color in legend_seq:
        vals = []
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            if int(g[rr, cc]) == color:
                vals.append(cc if axis == "h" else rr)
        if not vals:
            return None
        pos.append(float(sum(vals)) / len(vals))
    return pos


def _matches_legend_order(g: np.ndarray, pts: np.ndarray, legend_seq: list[int], axis: str) -> bool:
    pos = _sequence_positions(g, pts, legend_seq, axis)
    if pos is None:
        return False
    return all(a < b for a, b in zip(pos, pos[1:], strict=False))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove connected components whose target colors follow the legend strip order.

    Input structure:
    - The background fills most cells.
    - One grid edge contains a 2-cell-thick palette strip; each strip column/row is duplicated
      across the thickness, so the strip visually encodes a 1D sequence of target colors.
    - Away from the strip are solid non-bg rectangles with scattered colored markers.

    Rule:
    - Detect the edge strip and read its color sequence left-to-right or top-to-bottom.
    - Ignore the strip itself when finding 4-connected non-background components.
    - Erase a component iff it contains every legend color and those colors appear in the same
      monotone order along the strip axis.

    Key insight:
    - The strip is not just a palette set anymore: its linear order is the deciding signal.
      Components with the full color set can survive when that order is wrong.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    axis, legend_seq, legend_mask = _find_legend_strip(g, bg)
    assert legend_seq, "legend strip contains no colors"

    out = g.copy()
    comp_mask = (g != bg) & (~legend_mask)
    comps = components_mask(comp_mask, connectivity=4)
    for pts in comps:
        if _matches_legend_order(g, pts, legend_seq, axis):
            for r, c in pts:
                out[int(r), int(c)] = bg
    return out


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> tuple[str, str]:
    side = random.choice(["top", "bottom", "left", "right"])
    axis = "h" if side in {"top", "bottom"} else "v"

    if axis == "h":
        start = random.randint(0, GRID_W - LEGEND_LEN)
        rows = slice(0, 2) if side == "top" else slice(GRID_H - 2, GRID_H)
        for offset, color in enumerate(targets):
            inp[rows, start + offset] = color
            out[rows, start + offset] = color
    else:
        start = random.randint(0, GRID_H - LEGEND_LEN)
        cols = slice(0, 2) if side == "left" else slice(GRID_W - 2, GRID_W)
        for offset, color in enumerate(targets):
            inp[start + offset, cols] = color
            out[start + offset, cols] = color

    return side, axis


def _component_dims(axis: str) -> tuple[int, int]:
    if axis == "h":
        return random.randint(3, 5), random.randint(5, 8)
    return random.randint(5, 8), random.randint(3, 5)


def _slots(length: int) -> list[int]:
    buckets = np.array_split(np.arange(length), NUM_TARGETS)
    return [int(random.choice(bucket.tolist())) for bucket in buckets]


def _paint_marker(patch: np.ndarray, axis: str, major: int, color: int, occupied: set[tuple[int, int]]) -> None:
    h, w = patch.shape
    if axis == "h":
        r0 = random.randint(0, h - 2)
        coords = [(r0, major), (r0 + 1, major)]
    else:
        c0 = random.randint(0, w - 2)
        coords = [(major, c0), (major, c0 + 1)]

    for rc in coords:
        occupied.add(rc)
        patch[rc] = color


def _noise_count(area: int) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _component_patch(axis: str, targets: list[int], mode: str) -> np.ndarray:
    h, w = _component_dims(axis)
    patch = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    if mode == "ordered":
        target_order = list(targets)
    elif mode == "wrong_order":
        target_order = random.sample(targets, len(targets))
        while target_order == targets:
            target_order = random.sample(targets, len(targets))
    elif mode == "missing":
        keep_n = random.randint(1, NUM_TARGETS - 1)
        target_order = random.sample(targets, keep_n)
    else:
        raise ValueError(f"unknown component mode: {mode}")

    majors = _slots(w if axis == "h" else h)
    for major, color in zip(majors, target_order, strict=False):
        _paint_marker(patch, axis, major, color, occupied)

    non_targets = [color for color in PALETTE if color not in targets]
    area = h * w
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    random.shuffle(free)
    for r, c in free[: min(len(free), _noise_count(area))]:
        patch[r, c] = random.choice(non_targets)
    return patch


def _placement_canvas(inp: np.ndarray, side: str) -> np.ndarray:
    canvas = inp.copy()
    if side == "top":
        canvas[0:3, :] = 1
    elif side == "bottom":
        canvas[GRID_H - 3 : GRID_H, :] = 1
    elif side == "left":
        canvas[:, 0:3] = 1
    else:
        canvas[:, GRID_W - 3 : GRID_W] = 1
    return canvas


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    side, axis = _place_legend(inp, out, targets)
    canvas = _placement_canvas(inp, side)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    modes = ["ordered"] * deleted_count + ["wrong_order", "missing", "missing"]
    random.shuffle(modes)

    for mode in modes:
        patch = _component_patch(axis, targets, mode)
        try:
            r, c = random_free_location(
                canvas, patch, transparent=BG, padding=1, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        paste(canvas, patch, r, c)
        if mode != "ordered":
            paste(out, patch, r, c)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            inp, _out = pair
            return inp
    raise RuntimeError("failed to generate valid input")
