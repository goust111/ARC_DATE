import random

import numpy as np


GEN_BG = 5
GEN_DIVIDER = 1
GEN_BORDER_COLORS = [0, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("ul", "ur", "ll", "rr")
QUAD_DIRS = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "ll": (1, -1),
    "rr": (1, 1),
}
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel())
    return int(np.argmax(counts))


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        rows = np.flatnonzero(np.all(grid == color, axis=1))
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((color, int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, "expected one full-row/full-column divider color"
    return candidates[0]


def _quadrant_bounds(name: str, height: int, width: int, split_r: int, split_c: int) -> tuple[int, int, int, int]:
    if name == "ul":
        return 0, split_r, 0, split_c
    if name == "ur":
        return 0, split_r, split_c + 1, width
    if name == "ll":
        return split_r + 1, height, 0, split_c
    if name == "rr":
        return split_r + 1, height, split_c + 1, width
    raise AssertionError(f"unknown quadrant: {name}")


def _quadrant_boundary_coords(name: str, height: int, width: int, split_r: int, split_c: int) -> list[tuple[int, int]]:
    if name == "ul":
        coords = {(0, c) for c in range(split_c)} | {(r, 0) for r in range(split_r)}
    elif name == "ur":
        coords = {(0, c) for c in range(split_c + 1, width)} | {(r, width - 1) for r in range(split_r)}
    elif name == "ll":
        coords = {(height - 1, c) for c in range(split_c)} | {(r, 0) for r in range(split_r + 1, height)}
    elif name == "rr":
        coords = {(height - 1, c) for c in range(split_c + 1, width)} | {(r, width - 1) for r in range(split_r + 1, height)}
    else:
        raise AssertionError(f"unknown quadrant: {name}")
    return sorted(coords)


def _border_info(
    grid: np.ndarray,
    bg: int,
    divider_color: int,
    name: str,
    split_r: int,
    split_c: int,
) -> tuple[int, int, set[tuple[int, int]]]:
    boundary = _quadrant_boundary_coords(name, grid.shape[0], grid.shape[1], split_r, split_c)
    counts: dict[int, int] = {}
    color_cells: dict[int, set[tuple[int, int]]] = {}
    for r, c in boundary:
        color = int(grid[r, c])
        if color in (bg, divider_color):
            continue
        counts[color] = counts.get(color, 0) + 1
        color_cells.setdefault(color, set()).add((r, c))
    assert len(counts) == 1, "expected exactly one border color per quadrant"
    color = next(iter(counts))
    return color, int(counts[color]), color_cells[color]


def _seed_positions(
    grid: np.ndarray,
    name: str,
    split_r: int,
    split_c: int,
    color: int,
    border_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _quadrant_bounds(name, grid.shape[0], grid.shape[1], split_r, split_c)
    seeds: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if int(grid[r, c]) != color:
                continue
            if (r, c) not in border_cells:
                seeds.append((r, c))
    return seeds


def _draw_diagonal(out: np.ndarray, start_r: int, start_c: int, length: int, dr: int, dc: int, divider_color: int) -> None:
    height, width = out.shape
    for step in range(length):
        r = int(start_r + dr * step)
        c = int(start_c + dc * step)
        if not (0 <= r < height and 0 <= c < width):
            break
        if int(out[r, c]) == divider_color:
            continue
        out[r, c] = int(out[start_r, start_c])


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each seed along its quadrant's diagonal using the matching border segment length.

    Input structure:
    - The background is grey.
    - One blue full-width row and one blue full-height column cross inside the grid and split it into four quadrants.
    - Each quadrant has exactly one colored line segment on its outer boundary, and interior single-pixel seeds share that border color.

    Rule:
    - Detect the quadrant of every seed and use that quadrant's boundary segment as its match.
    - Let L be the matched boundary segment length.
    - Extend the seed into a diagonal line of length L including the seed itself:
      upper-left grows to upper-left, upper-right to upper-right, lower-left to lower-left, and lower-right to lower-right.
    - Stop each line at the grid boundary if needed, while keeping the blue divider and the original boundary segments unchanged.

    Key insight:
    - Color links each seed to its quadrant's border segment, the border length transfers a numeric property, and the quadrant location determines the hidden diagonal direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    divider_color, split_r, split_c = _find_divider(g)

    out = g.copy()
    for name in QUADRANTS:
        color, length, border_cells = _border_info(g, bg, divider_color, name, split_r, split_c)
        dr, dc = QUAD_DIRS[name]
        for r, c in _seed_positions(g, name, split_r, split_c, color, border_cells):
            _draw_diagonal(out, r, c, length, dr, dc, divider_color)

    return out


def _available_steps(r: int, c: int, dr: int, dc: int, height: int, width: int) -> int:
    steps = 1
    rr, cc = int(r), int(c)
    while True:
        rr += int(dr)
        cc += int(dc)
        if not (0 <= rr < height and 0 <= cc < width):
            return steps
        steps += 1


def _full_candidates(name: str, split_r: int, split_c: int, height: int, width: int, length: int) -> list[tuple[int, int]]:
    dr, dc = QUAD_DIRS[name]
    return [
        (r, c)
        for r, c in _interior_cells(name, split_r, split_c, height, width)
        if _available_steps(r, c, dr, dc, height, width) >= length
    ]


def _truncated_candidates(name: str, split_r: int, split_c: int, height: int, width: int, length: int) -> list[tuple[int, int]]:
    dr, dc = QUAD_DIRS[name]
    return [
        (r, c)
        for r, c in _interior_cells(name, split_r, split_c, height, width)
        if _available_steps(r, c, dr, dc, height, width) < length
    ]


def _diag_key(name: str, r: int, c: int) -> int:
    if name in {"ul", "rr"}:
        return int(r - c)
    return int(r + c)


def _interior_cells(name: str, split_r: int, split_c: int, height: int, width: int) -> list[tuple[int, int]]:
    if name == "ul":
        rows = range(1, split_r)
        cols = range(1, split_c)
    elif name == "ur":
        rows = range(1, split_r)
        cols = range(split_c + 1, width - 1)
    elif name == "ll":
        rows = range(split_r + 1, height - 1)
        cols = range(1, split_c)
    elif name == "rr":
        rows = range(split_r + 1, height - 1)
        cols = range(split_c + 1, width - 1)
    else:
        raise AssertionError(f"unknown quadrant: {name}")
    return [(r, c) for r in rows for c in cols]


def _segment_cells(side: str, start: int, length: int, split_r: int, split_c: int, height: int, width: int) -> list[tuple[int, int]]:
    if side == "top_left":
        return [(0, c) for c in range(start, start + length)]
    if side == "top_right":
        return [(0, c) for c in range(start, start + length)]
    if side == "bottom_left":
        return [(height - 1, c) for c in range(start, start + length)]
    if side == "bottom_right":
        return [(height - 1, c) for c in range(start, start + length)]
    if side == "left_top":
        return [(r, 0) for r in range(start, start + length)]
    if side == "left_bottom":
        return [(r, 0) for r in range(start, start + length)]
    if side == "right_top":
        return [(r, width - 1) for r in range(start, start + length)]
    if side == "right_bottom":
        return [(r, width - 1) for r in range(start, start + length)]
    raise AssertionError(f"unknown segment side: {side}")


def _place_border_segment(name: str, grid: np.ndarray, color: int, length: int, split_r: int, split_c: int) -> None:
    height, width = grid.shape
    if name == "ul":
        side = random.choice(["top_left", "left_top"])
        if side == "top_left":
            start = random.randint(0, split_c - length)
        else:
            start = random.randint(0, split_r - length)
    elif name == "ur":
        side = random.choice(["top_right", "right_top"])
        if side == "top_right":
            start = random.randint(split_c + 1, width - length)
        else:
            start = random.randint(0, split_r - length)
    elif name == "ll":
        side = random.choice(["bottom_left", "left_bottom"])
        if side == "bottom_left":
            start = random.randint(0, split_c - length)
        else:
            start = random.randint(split_r + 1, height - length)
    elif name == "rr":
        side = random.choice(["bottom_right", "right_bottom"])
        if side == "bottom_right":
            start = random.randint(split_c + 1, width - length)
        else:
            start = random.randint(split_r + 1, height - length)
    else:
        raise AssertionError(f"unknown quadrant: {name}")

    for r, c in _segment_cells(side, start, length, split_r, split_c, height, width):
        grid[r, c] = int(color)


def _pick_seed_positions(name: str, split_r: int, split_c: int, height: int, width: int, length: int) -> tuple[list[tuple[int, int]], bool, bool] | None:
    full = _full_candidates(name, split_r, split_c, height, width, length)
    trunc = _truncated_candidates(name, split_r, split_c, height, width, length)
    if not full or not trunc:
        return None

    seed_count = random.randint(2, 3)
    picks: list[tuple[int, int]] = []
    used_keys: set[int] = set()
    seen_full = False
    seen_trunc = False

    pools = [full, trunc]
    random.shuffle(pools)
    for pool in pools:
        options = [(r, c) for r, c in pool if _diag_key(name, r, c) not in used_keys]
        if not options:
            return None
        r, c = random.choice(options)
        picks.append((r, c))
        used_keys.add(_diag_key(name, r, c))
        if pool is full:
            seen_full = True
        else:
            seen_trunc = True

    combined = full + trunc
    while len(picks) < seed_count:
        options = [
            (r, c)
            for r, c in combined
            if _diag_key(name, r, c) not in used_keys
        ]
        if not options:
            break
        r, c = random.choice(options)
        picks.append((r, c))
        used_keys.add(_diag_key(name, r, c))
        if _available_steps(r, c, *QUAD_DIRS[name], height, width) >= length:
            seen_full = True
        else:
            seen_trunc = True

    if len(picks) < 2:
        return None
    return picks, seen_full, seen_trunc


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(17, 28)
    width = random.randint(17, 28)
    split_r = random.randint(8, height - 9)
    split_c = random.randint(8, width - 9)

    grid = np.full((height, width), GEN_BG, dtype=int)
    grid[split_r, :] = GEN_DIVIDER
    grid[:, split_c] = GEN_DIVIDER

    lengths = random.sample(range(3, 9), 4)
    colors = random.sample(GEN_BORDER_COLORS, 4)

    any_full = False
    any_trunc = False
    for name, color, length in zip(QUADRANTS, colors, lengths):
        _place_border_segment(name, grid, color, length, split_r, split_c)

        picked = _pick_seed_positions(name, split_r, split_c, height, width, length)
        if picked is None:
            return None
        seeds, seen_full, seen_trunc = picked
        any_full |= seen_full
        any_trunc |= seen_trunc
        for r, c in seeds:
            grid[r, c] = int(color)

    if not (any_full and any_trunc):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out[split_r, :] != GEN_DIVIDER) or np.any(out[:, split_c] != GEN_DIVIDER):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
