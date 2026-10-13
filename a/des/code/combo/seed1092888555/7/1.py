import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_RETRIES = 400


def _extract_legend(grid: np.ndarray) -> dict[int, int]:
    row = grid[-1]
    mapping: dict[int, int] = {}
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == BG:
            c += 1
            continue
        assert c + 1 < row.shape[0], "legend entries must come in adjacent pairs"
        border = int(row[c])
        fill = int(row[c + 1])
        assert fill != BG, "legend fill colors must be non-background"
        assert border not in mapping, "legend border colors must be unique"
        mapping[border] = fill
        c += 2
    assert mapping, "bottom-row legend must not be empty"
    return mapping


def _paint_line(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        out[r0, lo : hi + 1] = color
        return
    assert c0 == c1
    lo, hi = sorted((r0, r1))
    out[lo : hi + 1, c0] = color


def _paint_rectangle(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, border: int, fill: int) -> None:
    out[r0, c0 : c1 + 1] = border
    out[r1, c0 : c1 + 1] = border
    out[r0 : r1 + 1, c0] = border
    out[r0 : r1 + 1, c1] = border
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        out[r0 + 1 : r1, c0 + 1 : c1] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """Draw hidden rectangles or line segments from same-color marker pairs, using the bottom legend as border->fill mapping."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 4 and g.shape[1] >= 6

    mapping = _extract_legend(g)
    upper = g[:-1]
    colors = [int(color) for color in np.unique(upper) if int(color) != BG]
    assert colors, "there must be at least one marker pair above the legend"

    out = g.copy()
    out[-1, :] = BG

    for color in colors:
        coords = np.argwhere(upper == color)
        assert coords.shape == (2, 2), "each marker color must occur exactly twice above the legend"
        assert color in mapping, "every marker color must have a legend mapping"

        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1 or c0 == c1:
            _paint_line(out, r0, c0, r1, c1, color)
            continue

        rr0, rr1 = sorted((r0, r1))
        cc0, cc1 = sorted((c0, c1))
        _paint_rectangle(out, rr0, cc0, rr1, cc1, color, mapping[color])

    return out


def _reserve_clear(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
        return False
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
    return True


def _place_object(rng: random.Random, occupied: np.ndarray, branch: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = occupied.shape
    if branch == "hline":
        max_len = min(8, w)
        if max_len < 2:
            return None
        for _ in range(80):
            length = rng.randint(2, max_len)
            row = rng.randrange(h)
            col0 = rng.randrange(w - length + 1)
            col1 = col0 + length - 1
            if _reserve_clear(occupied, row, col0, row, col1):
                return (row, col0), (row, col1)
        return None

    if branch == "vline":
        max_len = min(8, h)
        if max_len < 2:
            return None
        for _ in range(80):
            length = rng.randint(2, max_len)
            row0 = rng.randrange(h - length + 1)
            row1 = row0 + length - 1
            col = rng.randrange(w)
            if _reserve_clear(occupied, row0, col, row1, col):
                return (row0, col), (row1, col)
        return None

    assert branch == "rect"
    max_rh = min(7, h)
    max_rw = min(8, w)
    if max_rh < 3 or max_rw < 3:
        return None
    for _ in range(120):
        rect_h = rng.randint(3, max_rh)
        rect_w = rng.randint(3, max_rw)
        row0 = rng.randrange(h - rect_h + 1)
        col0 = rng.randrange(w - rect_w + 1)
        row1 = row0 + rect_h - 1
        col1 = col0 + rect_w - 1
        if not _reserve_clear(occupied, row0, col0, row1, col1):
            continue
        if rng.random() < 0.5:
            return (row0, col0), (row1, col1)
        return (row0, col1), (row1, col0)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(3, 4)
    border_colors = rng.sample(PALETTE, num_objects)
    remaining = [c for c in PALETTE if c not in border_colors]
    fill_colors = rng.sample(remaining, num_objects)
    pairs = list(zip(border_colors, fill_colors))

    min_width = 2 * num_objects + 4
    width = rng.randint(max(12, min_width), 22)
    height = rng.randint(9, 18)
    object_rows = height - 2
    if object_rows < 3:
        return None

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((object_rows, width), dtype=bool)

    branches = ["hline", "vline", "rect"]
    while len(branches) < num_objects:
        branches.append(rng.choice(["hline", "vline", "rect"]))
    rng.shuffle(branches)

    markers: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for (border, _fill), branch in zip(pairs, branches):
        placed = _place_object(rng, occupied, branch)
        if placed is None:
            return None
        p0, p1 = placed
        markers.append((border, p0, p1))
        grid[p0] = border
        grid[p1] = border

    legend_order = pairs[:]
    rng.shuffle(legend_order)
    legend_span = 2 * num_objects
    start = rng.randrange(width - legend_span + 1)
    for i, (border, fill) in enumerate(legend_order):
        grid[-1, start + 2 * i] = border
        grid[-1, start + 2 * i + 1] = fill

    upper_colors = [int(c) for c in np.unique(grid[:-1]) if int(c) != BG]
    if sorted(upper_colors) != sorted(border_colors):
        return None
    if any(np.count_nonzero(grid[:-1] == color) != 2 for color in border_colors):
        return None
    if np.any(grid[-2] != BG):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.all(out[-1] == BG):
        return None
    if not any(p0[0] != p1[0] and p0[1] != p1[1] for _, p0, p1 in markers):
        return None
    if not any(p0[0] == p1[0] for _, p0, p1 in markers):
        return None
    if not any(p0[1] == p1[1] for _, p0, p1 in markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
