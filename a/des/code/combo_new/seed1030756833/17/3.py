import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = tuple(range(10))


def _grey_components(g: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(g == GREY):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == GREY:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _frame_bbox(g: np.ndarray, pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(pts) != expected:
        return None
    if not np.all(g[r0, c0 : c1 + 1] == GREY):
        return None
    if not np.all(g[r1, c0 : c1 + 1] == GREY):
        return None
    if not np.all(g[r0 : r1 + 1, c0] == GREY):
        return None
    if not np.all(g[r0 : r1 + 1, c1] == GREY):
        return None
    if (c1 - c0 - 1) % 2 != 1:
        return None
    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GREY):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _read_seed_colors(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    colors: list[int] = []
    r = r1 - 1
    while r > r0:
        row = g[r, c0 + 1 : c1]
        vals = np.unique(row)
        if len(vals) != 1:
            break
        color = int(vals[0])
        if color in (BLACK, GREY):
            break
        colors.append(color)
        r -= 1

    assert 1 <= len(colors) <= 3, "each frame must have one to three seed rows"
    assert len(set(colors)) == len(colors), "seed row colors must be distinct within a frame"

    if r >= r0 + 1:
        upper = g[r0 + 1 : r + 1, c0 + 1 : c1]
        assert np.all(upper == BLACK), "space above seed rows must be black"
    return colors


def _paint_tower(out: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    interior_w = c1 - c0 - 1
    max_inset = interior_w // 2
    seed_top = r1 - len(colors)

    for i, r in enumerate(range(seed_top - 1, r0, -1)):
        color = colors[i % len(colors)]
        inset = min(i // len(colors), max_inset)
        out[r, c0 + 1 : c1] = BLACK
        out[r, c0 + 1 + inset : c1 - inset] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build striped shrinking towers inside grey rectangular frames from bottom seed rows.

    Input structure:
    - Black background contains one or more non-overlapping grey rectangular frames with one-pixel borders.
    - Each frame has an odd interior width and a black interior space above its seed rows.
    - At the bottom of each frame interior are one to three full-width seed rows, each a distinct non-black, non-grey color.

    Rule:
    - For each frame, read the seed row colors from bottom to top as C1, C2, C3.
    - Fill the black rows above the seed block upward with the repeating color sequence C1, C2, C3, C1...
    - The stripe width starts at the full interior width and shrinks by one cell from both left and right after each complete color cycle.
    - Once the stripe has shrunk to one cell, continue the same color cycle in the center column up to the top inner wall.

    Key insight: The geometric narrowing is tied to completed color cycles, not to every single row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    frames: list[tuple[int, int, int, int]] = []
    for pts in _grey_components(g):
        bbox = _frame_bbox(g, pts)
        if bbox is not None:
            frames.append(bbox)
    assert frames, "at least one grey frame is required"

    for bbox in sorted(frames):
        colors = _read_seed_colors(g, bbox)
        _paint_tower(out, bbox, colors)
    return out


# --------------- generator ---------------

_GRID_SIZE_RANGE = (12, 30)
_FRAME_COUNT_RANGE = (1, 4)
_INTERIOR_WIDTHS = (3, 5, 7, 9, 11, 13)
_MAX_RETRIES = 500
_PLACE_RETRIES = 80
_PAD = 1


def _can_place(occupied: np.ndarray, r0: int, c0: int, fh: int, fw: int) -> bool:
    h, w = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + fh > h - 1 or c0 + fw > w - 1:
        return False
    rr0 = max(0, r0 - _PAD)
    cc0 = max(0, c0 - _PAD)
    rr1 = min(h, r0 + fh + _PAD)
    cc1 = min(w, c0 + fw + _PAD)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _draw_input_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    fh: int,
    fw: int,
    seed_colors: list[int],
) -> None:
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for i, color in enumerate(seed_colors):
        row = r1 - 1 - i
        grid[row, c0 + 1 : c1] = int(color)


def _pick_frame_shape(h: int, w: int) -> tuple[int, int, int, int, bool] | None:
    seed_count = random.choice([1, 2, 3])
    max_fill_h = h - 4 - seed_count
    if max_fill_h < 1:
        return None

    max_iw = min(max(_INTERIOR_WIDTHS), w - 4)
    widths = [iw for iw in _INTERIOR_WIDTHS if iw <= max_iw]
    if not widths:
        return None

    want_plateau = random.random() < 0.5
    random.shuffle(widths)
    for interior_w in widths:
        first_plateau_height = (interior_w // 2) * seed_count + 1
        if want_plateau:
            if first_plateau_height > max_fill_h:
                continue
            fill_h = random.randint(first_plateau_height, max_fill_h)
        else:
            max_short = min(max_fill_h, first_plateau_height - 1)
            if max_short < 1:
                continue
            fill_h = random.randint(1, max_short)

        fh = seed_count + fill_h + 2
        fw = interior_w + 2
        return fh, fw, seed_count, fill_h, want_plateau
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIZE_RANGE)
    w = random.randint(*_GRID_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    placed = 0
    plateau_seen = False
    short_seen = False

    for _ in range(frame_count):
        shape = _pick_frame_shape(h, w)
        if shape is None:
            return None
        fh, fw, seed_count, _fill_h, is_plateau = shape
        seed_colors = random.sample([c for c in COLORS if c not in (BLACK, GREY)], seed_count)

        loc = None
        for _try in range(_PLACE_RETRIES):
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if _can_place(occupied, r0, c0, fh, fw):
                loc = (r0, c0)
                break
        if loc is None:
            return None

        r0, c0 = loc
        _draw_input_frame(grid, r0, c0, fh, fw, seed_colors)
        occupied[r0 : r0 + fh, c0 : c0 + fw] = True
        placed += 1
        plateau_seen = plateau_seen or is_plateau
        short_seen = short_seen or not is_plateau

    if placed == 0:
        return None

    # Multi-frame scenes are most useful when both narrowing regimes are visible.
    if frame_count >= 2 and not (plateau_seen and short_seen):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid striped-frame input")
