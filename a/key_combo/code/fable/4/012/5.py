from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flat))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _border_coords(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    coords: set[Coord] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _strictly_inside(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < int(r) < r1 and c0 < int(c) < c1


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, BBox, Coord]:
    candidates: list[tuple[int, BBox, Coord]] = []
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        box = (r0, c0, r1, c1)
        border = _border_coords(box)
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        present = {tuple(map(int, rc)) for rc in coords}

        if len(present) != len(border) - 1:
            continue
        if not present.issubset(border):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue

        missing = list(border - present)
        if len(missing) != 1 or missing[0] in corners:
            continue
        candidates.append((int(color), box, missing[0]))

    assert len(candidates) == 1, "expected exactly one hollow frame with one non-corner gap"
    return candidates[0]


def _project_output(grid: np.ndarray, box: BBox, gap: Coord, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    frame_cells = _border_coords(box) - {gap}
    hr, hc = gap
    r0, c0, r1, c1 = box

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value == bg:
                continue
            if (r, c) in frame_cells:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue

            rr = 2 * hr - r
            cc = 2 * hc - c
            if _strictly_inside(box, rr, cc):
                out[rr, cc] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the frame gap as a pinhole and draw each outside pixel's point-reflected image inside the box.

    Input structure:
    - The grid has a dominant background, a one-pixel-thick monochrome hollow rectangle,
      and exactly one missing non-corner border cell in that rectangle.
    - The missing border cell is background-colored and acts as the pinhole center.
    - Colored pixels may appear outside the frame, while the frame interior starts as background.

    Rule:
    - Identify the unique hollow rectangular frame and its single border gap.
    - For every non-background pixel outside the frame, compute its point reflection through
      the gap cell: `(r, c) -> (2*gr-r, 2*gc-c)`.
    - Paint the reflected cell with the source color only when it lands strictly inside the
      frame interior; otherwise leave the grid unchanged.

    Key insight:
    - The missing frame cell is not damage to repair: it is the symmetry center of a 180-degree
      projection from the outside scene into the empty box interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, box, gap = _find_frame(g, bg)
    return _project_output(g, box, gap, bg)


# --------------- generator (o5x) ---------------

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
FRAME_H_RANGE = (5, 12)
FRAME_W_RANGE = (5, 12)
MARGIN = 2
PROJECTED_COUNT_RANGE = (2, 6)
DISTRACTOR_COUNT_RANGE = (1, 5)
MAX_ATTEMPTS = 400
PALETTE = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")


def _sample_box(h: int, w: int) -> BBox:
    frame_h = random.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], h - 2 * MARGIN))
    frame_w = random.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], w - 2 * MARGIN))
    r0 = random.randint(MARGIN, h - frame_h - MARGIN)
    c0 = random.randint(MARGIN, w - frame_w - MARGIN)
    return r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _sample_gap(box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    side = random.choice(SIDES)
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _valid_projected_pairs(shape: tuple[int, int], box: BBox, gap: Coord) -> list[tuple[Coord, Coord]]:
    pairs: list[tuple[Coord, Coord]] = []
    hr, hc = gap
    r0, c0, r1, c1 = box
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            sr = 2 * hr - rr
            sc = 2 * hc - cc
            if not _in_bounds(shape, sr, sc):
                continue
            if r0 <= sr <= r1 and c0 <= sc <= c1:
                continue
            pairs.append(((rr, cc), (sr, sc)))
    return pairs


def _outside_cells(shape: tuple[int, int], box: BBox) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            cells.append((r, c))
    return cells


def _branch_counts(grid: np.ndarray, box: BBox, gap: Coord, bg: int) -> tuple[int, int]:
    frame_cells = _border_coords(box) - {gap}
    projected = 0
    discarded = 0
    r0, c0, r1, c1 = box
    hr, hc = gap

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value == bg:
                continue
            if (r, c) in frame_cells:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            rr = 2 * hr - r
            cc = 2 * hc - c
            if _strictly_inside(box, rr, cc):
                projected += 1
            else:
                discarded += 1
    return projected, discarded


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if h < FRAME_H_RANGE[0] + 2 * MARGIN or w < FRAME_W_RANGE[0] + 2 * MARGIN:
        return None

    bg, frame_color = random.sample(PALETTE, 2)
    other_colors = [c for c in PALETTE if c not in (bg, frame_color)]

    box = _sample_box(h, w)
    gap = _sample_gap(box)
    valid_pairs = _valid_projected_pairs((h, w), box, gap)
    if len(valid_pairs) < PROJECTED_COUNT_RANGE[0]:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for r, c in _border_coords(box) - {gap}:
        grid[r, c] = frame_color

    projected_count = random.randint(PROJECTED_COUNT_RANGE[0], min(PROJECTED_COUNT_RANGE[1], len(valid_pairs)))
    chosen_pairs = random.sample(valid_pairs, projected_count)
    used_sources = set()
    for _target, source in chosen_pairs:
        color = random.choice(other_colors)
        sr, sc = source
        grid[sr, sc] = color
        used_sources.add(source)

    distractor_candidates = []
    hr, hc = gap
    for source in _outside_cells((h, w), box):
        if source in used_sources:
            continue
        rr = 2 * hr - source[0]
        cc = 2 * hc - source[1]
        if _strictly_inside(box, rr, cc):
            continue
        distractor_candidates.append(source)
    if len(distractor_candidates) < DISTRACTOR_COUNT_RANGE[0]:
        return None

    distractor_count = random.randint(
        DISTRACTOR_COUNT_RANGE[0],
        min(DISTRACTOR_COUNT_RANGE[1], len(distractor_candidates)),
    )
    for sr, sc in random.sample(distractor_candidates, distractor_count):
        grid[sr, sc] = random.choice(other_colors)

    if _get_bg(grid) != bg:
        return None

    projected, discarded = _branch_counts(grid, box, gap, bg)
    if projected < PROJECTED_COUNT_RANGE[0] or discarded < DISTRACTOR_COUNT_RANGE[0]:
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    if np.any(grid[box[0] + 1 : box[2], box[1] + 1 : box[3]] != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pinhole-frame input")
