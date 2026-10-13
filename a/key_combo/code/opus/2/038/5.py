from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, paste


BG = Color.BLACK
ANCHOR = Color.MAROON
SEGMENT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
ORIENTATIONS = ("H", "V")
LENGTHS = tuple(range(3, 10))
MAX_ATTEMPTS = 400
BORDER_MARGIN = 4


def _run_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    steps = 0
    rr = r + dr
    cc = c + dc
    while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(color):
        steps += 1
        rr += dr
        cc += dc
    return steps


def _extract_segment(grid: np.ndarray, anchor: tuple[int, int]) -> dict:
    r, c = anchor
    left = int(grid[r, c - 1]) if c - 1 >= 0 else BG
    right = int(grid[r, c + 1]) if c + 1 < grid.shape[1] else BG
    up = int(grid[r - 1, c]) if r - 1 >= 0 else BG
    down = int(grid[r + 1, c]) if r + 1 < grid.shape[0] else BG

    h_ok = left == right and left not in {BG, ANCHOR}
    v_ok = up == down and up not in {BG, ANCHOR}
    assert h_ok ^ v_ok, "each anchor must split exactly one straight segment"

    if h_ok:
        color = left
        length = 1 + _run_length(grid, r, c, 0, -1, color) + _run_length(grid, r, c, 0, 1, color)
        orientation = "H"
    else:
        color = up
        length = 1 + _run_length(grid, r, c, -1, 0, color) + _run_length(grid, r, c, 1, 0, color)
        orientation = "V"

    return {
        "anchor": (int(r), int(c)),
        "color": int(color),
        "length": int(length),
        "orientation": orientation,
    }


def _stamp_patch(rank: int, color: int) -> np.ndarray:
    side = 2 * int(rank) - 1
    radius = side // 2
    patch = np.full((side, side), BG, dtype=int)

    rows, cols = np.indices((side, side))
    dist = np.maximum(np.abs(rows - radius), np.abs(cols - radius))
    patch[(dist > 0) & (((radius - dist) % 2) == 0)] = int(color)
    patch[radius, radius] = ANCHOR
    return patch


def _segment_cells(anchor: tuple[int, int], orientation: str, length: int) -> list[tuple[int, int]]:
    r, c = anchor
    mid = (int(length) - 1) // 2
    cells: list[tuple[int, int]] = []
    if orientation == "H":
        start_c = c - mid
        for offset in range(int(length)):
            if offset == mid:
                continue
            cells.append((r, start_c + offset))
    else:
        start_r = r - mid
        for offset in range(int(length)):
            if offset == mid:
                continue
            cells.append((start_r + offset, c))
    return cells


def _render_output(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for seg in sorted(segments, key=lambda item: int(item["rank"]), reverse=True):
        r, c = seg["anchor"]
        patch = _stamp_patch(int(seg["rank"]), int(seg["color"]))
        radius = int(seg["rank"]) - 1
        paste(out, patch, r - radius, c - radius, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each anchored straight segment by a centered concentric-ring stamp whose size is set by the segment's global length rank.

    Input structure:
    - The grid has a black background and 3 to 5 monochrome horizontal or vertical segments.
    - Each segment uses a distinct non-black color and has exactly one maroon anchor pixel replacing its midpoint.
    - Segment lengths are distinct, and each anchor splits one contiguous straight segment of a single color.

    Rule:
    - Recover each segment's full length by combining its colored cells with the maroon anchor.
    - Rank the segments by decreasing length, with rank 1 for the longest segment.
    - Clear the canvas to black, then at each anchor draw a concentric square stamp of side 2*rank-1:
      outer layer in the segment color, next layer black, then alternating inward, with the center pixel maroon.
    - Draw stamps from largest rank to smallest rank so later smaller-rank stamps overwrite earlier ones.

    Key insight:
    - The visible maroon anchor identifies each segment's center, while the segment's relative length determines the number of concentric layers rather than the drawing order.
    """
    g = np.asarray(grid, dtype=int)
    anchors = np.argwhere(g == ANCHOR)
    assert 3 <= len(anchors) <= 5, "expected 3 to 5 anchor pixels"

    segments = [_extract_segment(g, (int(r), int(c))) for r, c in anchors]
    lengths = [int(seg["length"]) for seg in segments]
    colors = [int(seg["color"]) for seg in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be distinct"
    assert len(colors) == len(set(colors)), "segment colors must be distinct"

    ranked = sorted(segments, key=lambda seg: int(seg["length"]), reverse=True)
    for rank, seg in enumerate(ranked, start=1):
        seg["rank"] = rank

    return _render_output(g.shape, ranked)


def _sample_lengths(count: int, rng: random.Random) -> list[int]:
    for _ in range(100):
        lengths = rng.sample(LENGTHS, count)
        if count > 1 and (all(length % 2 == 0 for length in lengths) or all(length % 2 == 1 for length in lengths)):
            continue
        return lengths
    raise RuntimeError("failed to sample mixed segment lengths")


def _sample_orientations(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(ORIENTATIONS)]

    horizontal = rng.randint(1, count - 1)
    orientations = ["H"] * horizontal + ["V"] * (count - horizontal)
    rng.shuffle(orientations)
    return orientations


def _slot_shape(count: int, rng: random.Random) -> tuple[int, int]:
    options = [(rows, cols) for rows in (2, 3) for cols in (2, 3) if rows * cols >= count]
    return rng.choice(options)


def _build_specs(rng: random.Random) -> tuple[list[dict], int]:
    count = rng.randint(3, 5)
    lengths = _sample_lengths(count, rng)
    colors = rng.sample(SEGMENT_COLORS, count)
    orientations = _sample_orientations(count, rng)

    specs = []
    for length, color, orientation in zip(lengths, colors, orientations):
        specs.append({"length": int(length), "color": int(color), "orientation": orientation})

    ranked = sorted(specs, key=lambda item: int(item["length"]), reverse=True)
    for rank, spec in enumerate(ranked, start=1):
        spec["rank"] = rank
    return specs, max(lengths)


def _anchor_grid(count: int, sep: int, rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    slot_rows, slot_cols = _slot_shape(count, rng)
    min_h = max(18, BORDER_MARGIN + 1 + sep * (slot_rows - 1) + BORDER_MARGIN)
    min_w = max(18, BORDER_MARGIN + 1 + sep * (slot_cols - 1) + BORDER_MARGIN)
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    row_start = rng.randint(BORDER_MARGIN, height - BORDER_MARGIN - 1 - sep * (slot_rows - 1))
    col_start = rng.randint(BORDER_MARGIN, width - BORDER_MARGIN - 1 - sep * (slot_cols - 1))
    rows = [row_start + sep * i for i in range(slot_rows)]
    cols = [col_start + sep * i for i in range(slot_cols)]
    anchors = [(r, c) for r in rows for c in cols]
    return (height, width), anchors


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        anchor = spec["anchor"]
        for r, c in _segment_cells(anchor, str(spec["orientation"]), int(spec["length"])):
            grid[r, c] = int(spec["color"])
        ar, ac = anchor
        grid[ar, ac] = ANCHOR
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs, max_length = _build_specs(rng)
    sep = max(8, int(max_length))
    shape, anchor_candidates = _anchor_grid(len(specs), sep, rng)
    if len(anchor_candidates) < len(specs):
        return None

    rng.shuffle(anchor_candidates)
    for spec, anchor in zip(specs, anchor_candidates):
        spec["anchor"] = anchor

    inp = _render_input(shape, specs)
    expected = _render_output(shape, specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
