import random
from typing import Iterable

import numpy as np


BG = 0
OCCLUDER = 1
ARROW = 2
DIVIDER = 5
PATTERN_COLORS = (3, 4, 6, 7, 8, 9)
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
GEN_RETRIES = 300


def _full_line_color(vec: np.ndarray) -> int | None:
    first = int(vec[0])
    if first == BG:
        return None
    if np.all(vec == first):
        return first
    return None


def _find_divider_color(grid: np.ndarray) -> int:
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}
    for r in range(grid.shape[0]):
        color = _full_line_color(grid[r, :])
        if color is not None:
            row_colors[color] = row_colors.get(color, 0) + 1
    for c in range(grid.shape[1]):
        color = _full_line_color(grid[:, c])
        if color is not None:
            col_colors[color] = col_colors.get(color, 0) + 1
    common = set(row_colors) & set(col_colors)
    assert common, "expected a divider color spanning full rows and columns"
    return max(common, key=lambda color: row_colors[color] + col_colors[color])


def _split_intervals(indices: list[int], n: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for idx in indices:
        if start < idx:
            intervals.append((start, idx))
        start = idx + 1
    if start < n:
        intervals.append((start, n))
    return intervals


def _closest_corner(cells: list[tuple[int, int]], size: int) -> tuple[int, int]:
    corners = ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1))
    return min(
        corners,
        key=lambda corner: sum(abs(r - corner[0]) + abs(c - corner[1]) for r, c in cells),
    )


def _decode_direction(panel: np.ndarray) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(panel == ARROW)]
    assert len(coords) == 2, "each panel must have exactly two arrow pixels"
    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "arrow pixels must be adjacent"
    corner = _closest_corner(coords, panel.shape[0])
    tail, head = sorted(
        coords,
        key=lambda rc: abs(rc[0] - corner[0]) + abs(rc[1] - corner[1]),
    )
    dr = head[0] - tail[0]
    dc = head[1] - tail[1]
    assert (dr, dc) in DIRS, "arrow must be cardinal"
    return dr, dc


def _reflect(size: int, r: int, c: int, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dc != 0:
        return r, size - 1 - c
    return size - 1 - r, c


def _restore_panel(panel: np.ndarray) -> np.ndarray:
    size = panel.shape[0]
    assert panel.shape[1] == size, "panels must be square"
    direction = _decode_direction(panel)
    out = np.full_like(panel, BG)

    visible = (
        (panel != BG)
        & (panel != OCCLUDER)
        & (panel != ARROW)
    )
    for r, c in np.argwhere(visible):
        rr = int(r)
        cc = int(c)
        color = int(panel[rr, cc])
        mr, mc = _reflect(size, rr, cc, direction)
        for tr, tc in ((rr, cc), (mr, mc)):
            prev = int(out[tr, tc])
            assert prev in (BG, color), "reflected colors cannot conflict"
            out[tr, tc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore symmetric panel motifs hidden behind edge occluders.

    The grid is split by uniform divider rows and columns into equal square
    panels. Each panel contains exactly one two-pixel red arrow in a corner.
    The arrow's direction says which side's occluder gets pushed away. The
    remaining visible motif is symmetric across the panel axis perpendicular to
    that direction, so reflecting visible motif pixels across the whole panel
    restores the hidden side. Arrow and occluder pixels are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_color(g)
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == divider)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == divider)]
    row_intervals = _split_intervals(row_seps, g.shape[0])
    col_intervals = _split_intervals(col_seps, g.shape[1])
    assert row_intervals and col_intervals, "expected at least one panel"

    heights = {r1 - r0 for r0, r1 in row_intervals}
    widths = {c1 - c0 for c0, c1 in col_intervals}
    assert len(heights) == 1 and len(widths) == 1, "panels must be equal-sized"
    size = next(iter(heights))
    assert heights == widths and size >= 4, "panels must be square"

    out = np.full_like(g, BG)
    for r in row_seps:
        out[r, :] = divider
    for c in col_seps:
        out[:, c] = divider

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _restore_panel(panel)

    return out


def _arrow_cells(size: int, direction: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        tail = rng.choice(((0, 0), (size - 1, 0)))
        return [tail, (tail[0], tail[1] + 1)]
    if (dr, dc) == (0, -1):
        tail = rng.choice(((0, size - 1), (size - 1, size - 1)))
        return [tail, (tail[0], tail[1] - 1)]
    if (dr, dc) == (1, 0):
        tail = rng.choice(((0, 0), (0, size - 1)))
        return [tail, (tail[0] + 1, tail[1])]
    tail = rng.choice(((size - 1, 0), (size - 1, size - 1)))
    return [tail, (tail[0] - 1, tail[1])]


def _arrow_corner_mask(size: int, arrow_cells: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    cells = list(arrow_cells)
    corner = _closest_corner(cells, size)
    cr = 0 if corner[0] == 0 else size - 2
    cc = 0 if corner[1] == 0 else size - 2
    mask[cr:cr + 2, cc:cc + 2] = True
    return mask


def _occluder_side(direction: tuple[int, int]) -> str:
    return {
        (0, 1): "right",
        (0, -1): "left",
        (1, 0): "bottom",
        (-1, 0): "top",
    }[direction]


def _stripe_mask(size: int, side: str, width: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if side == "left":
        mask[:, :width] = True
    elif side == "right":
        mask[:, size - width:] = True
    elif side == "top":
        mask[:width, :] = True
    else:
        mask[size - width:, :] = True
    return mask


def _seed_domain(size: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction[1] != 0:
        half = size // 2
        return [(r, c) for r in range(size) for c in range(half + 1)]
    half = size // 2
    return [(r, c) for r in range(half + 1) for c in range(size)]


def _grow_connected_subset(
    valid: set[tuple[int, int]],
    target: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    start = rng.choice(tuple(valid))
    chosen = {start}
    frontier = [start]
    while len(chosen) < target:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in valid and (nr, nc) not in chosen:
                    candidates.add((nr, nc))
        if not candidates:
            break
        nxt = rng.choice(tuple(candidates))
        chosen.add(nxt)
        frontier.append(nxt)
    return chosen


def _make_symmetric_mask(
    size: int,
    direction: tuple[int, int],
    arrow_cells: list[tuple[int, int]],
    stripe_width: int,
    rng: random.Random,
) -> np.ndarray | None:
    banned = _arrow_corner_mask(size, arrow_cells)
    full_valid: set[tuple[int, int]] = set()
    for r, c in _seed_domain(size, direction):
        mr, mc = _reflect(size, r, c, direction)
        if banned[r, c] or banned[mr, mc]:
            continue
        full_valid.add((r, c))
    if len(full_valid) < 4:
        return None

    low = max(4, size - 1)
    high = min(len(full_valid), size + 4)
    target = rng.randint(low, high)
    seed = _grow_connected_subset(full_valid, target, rng)
    if len(seed) < 4:
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in seed:
        mask[r, c] = True
        mr, mc = _reflect(size, r, c, direction)
        mask[mr, mc] = True

    stripe = _stripe_mask(size, _occluder_side(direction), stripe_width)
    if not np.any(mask & stripe):
        return None
    if not np.any(mask & ~stripe):
        return None
    if np.any(mask & banned):
        return None
    visible = mask & ~stripe
    if np.array_equal(mask, visible):
        return None
    return mask


def _build_panel(size: int, rng: random.Random) -> np.ndarray | None:
    panel = np.full((size, size), BG, dtype=int)
    direction = rng.choice(DIRS)
    stripe_width = rng.choice((1, 2))
    if stripe_width >= size - 2:
        stripe_width = 1
    arrow_cells = _arrow_cells(size, direction, rng)
    mask = _make_symmetric_mask(size, direction, arrow_cells, stripe_width, rng)
    if mask is None:
        return None

    color = rng.choice(PATTERN_COLORS)
    panel[mask] = color
    stripe = _stripe_mask(size, _occluder_side(direction), stripe_width)
    panel[stripe & mask] = OCCLUDER
    for r, c in arrow_cells:
        panel[r, c] = ARROW
    return panel


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_rows = rng.randint(2, 3)
    panel_cols = rng.randint(2, 4)
    size = rng.randint(7, 10)
    h = panel_rows * size + (panel_rows - 1)
    w = panel_cols * size + (panel_cols - 1)
    grid = np.full((h, w), BG, dtype=int)

    for r in range(1, panel_rows):
        grid[r * size + (r - 1), :] = DIVIDER
    for c in range(1, panel_cols):
        grid[:, c * size + (c - 1)] = DIVIDER

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            for _ in range(40):
                panel = _build_panel(size, rng)
                if panel is not None:
                    break
            else:
                return None
            r0 = pr * (size + 1)
            c0 = pc * (size + 1)
            grid[r0:r0 + size, c0:c0 + size] = panel

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
