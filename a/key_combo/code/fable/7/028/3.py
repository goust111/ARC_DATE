import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, paint_coords


MIN_BAR_LEN = 4
MAX_DIM = 30
GEN_ATTEMPTS = 400
PALETTE = [Color.BLACK, Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
NOISE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
)


def _bar_signature(coords: np.ndarray):
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0 = int(rs.min())
    r1 = int(rs.max())
    c0 = int(cs.min())
    c1 = int(cs.max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(coords.shape[0])
    if h == 1 and area == w and w >= MIN_BAR_LEN:
        return ("H", w, (r0, c0))
    if w == 1 and area == h and h >= MIN_BAR_LEN:
        return ("V", h, (r0, c0))
    return None


def _find_instruction_bars(grid: np.ndarray):
    bg = int(get_bg(grid))
    bars = []
    for coords in components_mask(grid == bg, connectivity=4):
        sig = _bar_signature(coords)
        if sig is not None:
            orient, length, anchor = sig
            bars.append(
                {
                    "orient": orient,
                    "length": int(length),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                    "coords": {(int(r), int(c)) for r, c in coords},
                }
            )
    bars.sort(key=lambda item: item["anchor"])
    return bars


def _trace_path(start, bars):
    r, c = map(int, start)
    path = [(r, c)]
    for bar in bars:
        if bar["orient"] == "H":
            for _ in range(int(bar["length"])):
                c += 1
                path.append((int(r), int(c)))
        else:
            for _ in range(int(bar["length"])):
                r += 1
                path.append((int(r), int(c)))
    return path


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode long blank corridor segments as ordered drawing instructions and paint the resulting polyline green.

    Input structure:
    - The grid has one dominant background color, one denser clutter color, and a single gray start pixel.
    - Inside the clutter field are several isolated one-cell-wide pure-background straight bars, plus shorter or bent background gaps.
    - The true instruction bars are background connected components that remain straight and have length at least 4.

    Rule:
    - Find every background connected component that is exactly a horizontal or vertical bar of width 1 and length >= 4.
    - Sort those bars by the row-major order of their upper-left endpoint.
    - Starting from the gray pixel, execute the sorted bars in sequence: a horizontal bar means move right by its length, and a vertical bar means move down by its length.
    - Recolor the whole visited polyline, including the start pixel, to green and leave every other cell unchanged.

    Key insight:
    - The instructions are encoded by negative space: only the long isolated blank bars matter, while the rest of the background is either outside the clutter field or too short or crooked to count.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    start_pixels = np.argwhere(grid == Color.GREY)
    assert start_pixels.shape[0] == 1, "expected exactly one gray start pixel"
    start = tuple(map(int, start_pixels[0]))

    bars = _find_instruction_bars(grid)
    assert bars, "expected at least one instruction bar"

    path = _trace_path(start, bars)
    h, w = grid.shape
    for r, c in path:
        assert 0 <= r < h and 0 <= c < w, "decoded path must stay in bounds"

    out = grid.copy()
    paint_coords(out, path, Color.GREEN)
    return out


def _segment_cells(orient: str, length: int, anchor):
    r0, c0 = map(int, anchor)
    if orient == "H":
        return [(r0, c0 + dc) for dc in range(int(length))]
    return [(r0 + dr, c0) for dr in range(int(length))]


def _expanded(cells):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _panel_interior(panel):
    r0, c0, r1, c1 = map(int, panel)
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _all_bar_candidates(panel, orient: str, length: int):
    ir0, ic0, ir1, ic1 = _panel_interior(panel)
    candidates = []
    if orient == "H":
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 - int(length) + 2):
                candidates.append((int(r), int(c)))
    else:
        for r in range(ir0, ir1 - int(length) + 2):
            for c in range(ic0, ic1 + 1):
                candidates.append((int(r), int(c)))
    return candidates


def _choose_bars(rng: np.random.Generator, panel, spec):
    attempts = 120
    for _ in range(attempts):
        bars = []
        forbidden = set()
        prev_anchor = (-1, -1)
        for i, (orient, length) in enumerate(spec):
            candidates = []
            same_row = []
            for anchor in _all_bar_candidates(panel, orient, length):
                if anchor <= prev_anchor:
                    continue
                cells = _segment_cells(orient, length, anchor)
                if any(cell in forbidden for cell in cells):
                    continue
                candidates.append((anchor, cells))
                if bars and anchor[0] == prev_anchor[0]:
                    same_row.append((anchor, cells))
            if not candidates:
                break
            pool = same_row if same_row and rng.random() < 0.35 else candidates
            anchor, cells = pool[int(rng.integers(len(pool)))]
            bars.append(
                {
                    "orient": orient,
                    "length": int(length),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                    "cells": cells,
                }
            )
            forbidden |= _expanded(cells)
            prev_anchor = anchor
        if len(bars) == len(spec):
            return bars, forbidden
    return None, None


def _noise_positions(panel, shape):
    ir0, ic0, ir1, ic1 = _panel_interior(panel)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    for r0 in range(ir0, ir1 - max_r + 1):
        for c0 in range(ic0, ic1 - max_c + 1):
            yield [(int(r0 + dr), int(c0 + dc)) for dr, dc in shape]


def _add_noise(rng: np.random.Generator, grid: np.ndarray, panel, forbidden):
    noise_target = int(rng.integers(7, 15))
    placed = 0
    for _ in range(noise_target * 8):
        shape = NOISE_SHAPES[int(rng.integers(len(NOISE_SHAPES)))]
        candidates = [cells for cells in _noise_positions(panel, shape) if all(cell not in forbidden for cell in cells)]
        if not candidates:
            continue
        cells = candidates[int(rng.integers(len(candidates)))]
        for r, c in cells:
            grid[int(r), int(c)] = int(get_bg(grid))
        forbidden |= _expanded(cells)
        placed += 1
        if placed >= noise_target:
            break
    return placed


def _make_case(rng: np.random.Generator):
    seg_count = int(rng.integers(2, 6))
    while True:
        orientations = rng.choice(np.array(["H", "V"]), size=seg_count, replace=True).tolist()
        if seg_count == 2 or rng.random() < 0.15 or len(set(orientations)) == 2:
            break
    lengths = [int(rng.integers(MIN_BAR_LEN, 9)) for _ in range(seg_count)]
    total_right = sum(length for orient, length in zip(orientations, lengths) if orient == "H")
    total_down = sum(length for orient, length in zip(orientations, lengths) if orient == "V")

    min_h = max(18, total_down + 5)
    min_w = max(18, total_right + 5)
    if min_h > MAX_DIM or min_w > MAX_DIM:
        return None
    h = int(rng.integers(min_h, MAX_DIM + 1))
    w = int(rng.integers(min_w, MAX_DIM + 1))

    start_r_max = h - total_down - 1
    start_c_max = w - total_right - 1
    if start_r_max < 1 or start_c_max < 1:
        return None
    start = (int(rng.integers(0, start_r_max + 1)), int(rng.integers(0, start_c_max + 1)))

    panel_h_min = max(10, max((length for orient, length in zip(orientations, lengths) if orient == "V"), default=4) + 4)
    panel_w_min = max(10, max((length for orient, length in zip(orientations, lengths) if orient == "H"), default=4) + 4)
    panel_h_max = min(h - 4, max(panel_h_min, int(0.68 * h)))
    panel_w_max = min(w - 4, max(panel_w_min, int(0.68 * w)))
    if panel_h_min > panel_h_max or panel_w_min > panel_w_max:
        return None
    panel_h = int(rng.integers(panel_h_min, panel_h_max + 1))
    panel_w = int(rng.integers(panel_w_min, panel_w_max + 1))
    panel_r0 = int(rng.integers(2, h - panel_h))
    panel_c0 = int(rng.integers(2, w - panel_w))
    panel = (panel_r0, panel_c0, panel_r0 + panel_h - 1, panel_c0 + panel_w - 1)

    spec = list(zip(orientations, lengths))
    bars, forbidden = _choose_bars(rng, panel, spec)
    if bars is None:
        return None

    colors = rng.choice(np.array(PALETTE), size=2, replace=False)
    bg = int(colors[0])
    clutter = int(colors[1])
    grid = np.full((h, w), bg, dtype=int)
    r0, c0, r1, c1 = panel
    grid[r0 : r1 + 1, c0 : c1 + 1] = clutter
    for bar in bars:
        for r, c in bar["cells"]:
            grid[int(r), int(c)] = bg

    if _add_noise(rng, grid, panel, forbidden) < 4:
        return None

    sr, sc = start
    grid[int(sr), int(sc)] = Color.GREY

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == clutter):
        return None
    if Color.GREEN in set(map(int, np.unique(grid))):
        return None

    planned = [{"orient": bar["orient"], "length": int(bar["length"]), "anchor": bar["anchor"]} for bar in bars]
    found = [{"orient": bar["orient"], "length": int(bar["length"]), "anchor": bar["anchor"]} for bar in _find_instruction_bars(grid)]
    if planned != found:
        return None

    out = main(grid)
    path = _trace_path(start, planned)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
        return None
    expected = grid.copy()
    paint_coords(expected, path, Color.GREEN)
    if not np.array_equal(out, expected):
        return None

    green_count = int(np.count_nonzero(out == Color.GREEN))
    if green_count < len(planned) + 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative-space polyline input")
