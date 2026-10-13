from __future__ import annotations

import numpy as np


def _most_common_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_unique_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == int(bg):
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        frame = _frame_mask(g.shape, (r0, c0, r1, c1))
        if int(coords.shape[0]) != int(frame.sum()):
            continue
        if np.array_equal(g == color, frame):
            candidates.append((color, (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected exactly one hollow rectangular frame"
    return candidates[0]


def _bar_length(coords: np.ndarray) -> int:
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] >= 1
    rows = coords[:, 0]
    cols = coords[:, 1]
    assert int(rows.min()) == int(rows.max()), "reference bar must be horizontal"
    c0 = int(cols.min())
    c1 = int(cols.max())
    assert int(coords.shape[0]) == c1 - c0 + 1, "reference bar must be contiguous"
    return int(coords.shape[0])


def _direction_for_seed(
    row: int,
    col: int,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    if row < center_r:
        dr = -1
    elif row > center_r:
        dr = 1
    else:
        raise AssertionError("seed may not lie on the horizontal center line")

    if col < center_c:
        dc = -1
    elif col > center_c:
        dc = 1
    else:
        raise AssertionError("seed may not lie on the vertical center line")

    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow diagonal colored traces inside a frame using seed quadrants and matching outside bar lengths.

    Input structure:
    - The background is uniform and one non-background color forms a unique
      one-pixel-thick hollow rectangular frame.
    - Inside the frame, each remaining colored cell is a single-pixel seed
      placed strictly inside one of the four quadrants defined by the frame
      center.
    - Outside the frame, every seed color appears exactly once as a horizontal
      contiguous reference bar.

    Rule:
    - Keep the background, the frame, and the outside reference bars unchanged.
    - For each seed, find the outside bar of the same color and read its length
      L.
    - Use the seed quadrant to choose a diagonal growth direction: upper-left,
      upper-right, lower-left, or lower-right.
    - Starting at the seed itself, paint up to L cells of that color along the
      chosen diagonal, stopping before the path would leave the frame interior.

    Key insight:
    The transformation combines two separate cues: the seed position relative to
    the frame center gives the diagonal direction, while the same-colored bar
    outside the frame gives the growth distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_unique_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox

    inside = np.zeros_like(g, dtype=bool)
    inside[r0 + 1 : r1, c0 + 1 : c1] = True
    outside = ~inside

    seed_mask = inside & (g != int(bg)) & (g != int(frame_color))
    seed_coords = np.argwhere(seed_mask)
    assert seed_coords.shape[0] >= 1, "expected at least one seed"
    seed_colors = {int(g[r, c]) for r, c in seed_coords}

    bar_mask = outside & (g != int(bg)) & (g != int(frame_color))
    bar_coords = np.argwhere(bar_mask)
    bar_colors = {int(g[r, c]) for r, c in bar_coords}
    assert seed_colors == bar_colors, "seed colors must match outside bar colors"

    out = g.copy()
    for color in sorted(seed_colors):
        seed_points = np.argwhere(seed_mask & (g == color))
        ref_points = np.argwhere(bar_mask & (g == color))
        assert seed_points.shape[0] == 1, "each color must have exactly one seed"
        length = _bar_length(ref_points)
        sr, sc = map(int, seed_points[0])
        dr, dc = _direction_for_seed(sr, sc, frame_bbox)

        for step in range(length):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
                break
            out[rr, cc] = color

    return out


# --------------- generator ---------------

BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("ul", "ur", "ll", "lr")
GENERATION_ATTEMPTS = 400

INTERIOR_HEIGHT_RANGE = (5, 15)
INTERIOR_WIDTH_RANGE = (5, 15)
MARGIN_RANGE = (2, 6)
SEED_COUNT_RANGE = (1, 4)
BAR_LENGTH_RANGE = (1, 5)


def _quadrant_cells(
    frame_bbox: tuple[int, int, int, int],
    quadrant: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if quadrant == "ul" and r < center_r and c < center_c:
                cells.append((r, c))
            elif quadrant == "ur" and r < center_r and c > center_c:
                cells.append((r, c))
            elif quadrant == "ll" and r > center_r and c < center_c:
                cells.append((r, c))
            elif quadrant == "lr" and r > center_r and c > center_c:
                cells.append((r, c))
    return cells


def _growth_capacity(
    seed: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = frame_bbox
    sr, sc = seed
    dr, dc = direction
    if dr < 0:
        row_cap = sr - (r0 + 1) + 1
    else:
        row_cap = (r1 - 1) - sr + 1
    if dc < 0:
        col_cap = sc - (c0 + 1) + 1
    else:
        col_cap = (c1 - 1) - sc + 1
    return int(min(row_cap, col_cap))


def _paint_expected(
    grid: np.ndarray,
    seeds: dict[int, tuple[int, int]],
    lengths: dict[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = frame_bbox
    for color, (sr, sc) in seeds.items():
        dr, dc = _direction_for_seed(sr, sc, frame_bbox)
        for step in range(int(lengths[color])):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
                break
            out[rr, cc] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    interior_h = int(rng.integers(INTERIOR_HEIGHT_RANGE[0], INTERIOR_HEIGHT_RANGE[1] + 1))
    interior_w = int(rng.integers(INTERIOR_WIDTH_RANGE[0], INTERIOR_WIDTH_RANGE[1] + 1))
    top = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    bottom = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

    height = top + interior_h + 2 + bottom
    width = left + interior_w + 2 + right
    if max(height, width) > 30:
        return None

    seed_count = int(rng.integers(SEED_COUNT_RANGE[0], SEED_COUNT_RANGE[1] + 1))
    if top + bottom < seed_count:
        return None

    frame_bbox = (top, left, top + interior_h + 1, left + interior_w + 1)
    fr0, fc0, fr1, fc1 = frame_bbox

    quadrants = list(rng.choice(np.array(QUADRANTS), size=seed_count, replace=False))
    colors = [int(x) for x in rng.choice(np.array(PALETTE), size=seed_count, replace=False)]
    lengths = {
        color: int(rng.integers(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1] + 1))
        for color in colors
    }

    grid = np.full((height, width), BG, dtype=int)
    frame = _frame_mask(grid.shape, frame_bbox)
    grid[frame] = FRAME_COLOR

    seeds: dict[int, tuple[int, int]] = {}
    extended_any = False
    for quadrant, color in zip(quadrants, colors, strict=False):
        candidates = _quadrant_cells(frame_bbox, quadrant)
        if not candidates:
            return None
        seed = candidates[int(rng.integers(len(candidates)))]
        seeds[color] = seed
        grid[seed] = color
        capacity = _growth_capacity(seed, frame_bbox, _direction_for_seed(seed[0], seed[1], frame_bbox))
        if min(capacity, lengths[color]) >= 2:
            extended_any = True

    if not extended_any:
        return None

    available_rows = list(range(0, fr0)) + list(range(fr1 + 1, height))
    if len(available_rows) < seed_count:
        return None
    chosen_rows = [int(x) for x in rng.choice(np.array(available_rows), size=seed_count, replace=False)]

    for row, color in zip(chosen_rows, colors, strict=False):
        length = lengths[color]
        start = int(rng.integers(0, width - length + 1))
        grid[row, start : start + length] = color

    expected = _paint_expected(grid, seeds, lengths, frame_bbox)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
