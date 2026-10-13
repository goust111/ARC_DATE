from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 2
AVAILABLE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        column = grid[:, c]
        color = int(column[0])
        if color == int(bg):
            continue
        if not np.all(column == color):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append((c, color))
    assert len(candidates) == 1, f"expected exactly one separator column, got {candidates}"
    return candidates[0]


def _ordered_upper_left(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if a != b and a[0] <= b[0] and a[1] <= b[1]:
        return a, b
    if a != b and b[0] <= a[0] and b[1] <= a[1]:
        return b, a
    return (a, b) if a < b else (b, a)


def _path_cells(start: tuple[int, int], end: tuple[int, int], pattern: str) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []

    if pattern == "HV":
        step_c = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step_c, step_c):
            cells.append((r0, c))
        step_r = 1 if r1 >= r0 else -1
        for r in range(r0 + step_r, r1 + step_r, step_r):
            cells.append((r, c1))
        return cells

    assert pattern == "VH", f"unknown pattern {pattern}"
    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        cells.append((r, c0))
    step_c = 1 if c1 >= c0 else -1
    for c in range(c0 + step_c, c1 + step_c, step_c):
        cells.append((r1, c))
    return cells


def _instruction_pattern(coords: np.ndarray) -> str:
    cells = {tuple(map(int, rc)) for rc in coords}
    endpoints: list[tuple[int, int]] = []
    elbows: list[tuple[int, int]] = []

    for r, c in cells:
        neighbors = [(r + dr, c + dc) for dr, dc in NEIGHBORS4 if (r + dr, c + dc) in cells]
        degree = len(neighbors)
        if degree == 1:
            endpoints.append((r, c))
        elif degree == 2:
            (r1, c1), (r2, c2) = neighbors
            if r1 != r2 and c1 != c2:
                elbows.append((r, c))
        else:
            raise AssertionError("instruction fragment must be a one-cell-wide path")

    assert len(endpoints) == 2, f"expected two endpoints, got {endpoints}"
    assert len(elbows) == 1, f"expected one elbow, got {elbows}"
    start, _ = _ordered_upper_left(endpoints[0], endpoints[1])
    elbow = elbows[0]
    if start[0] == elbow[0]:
        return "HV"
    if start[1] == elbow[1]:
        return "VH"
    raise AssertionError("elbow must share a row or column with the starting endpoint")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color anchor pair in the workspace using the matching L-turn template from the instruction pane.

    Input structure:
    - The background is black, and one full-height vertical separator line splits the grid into a left instruction pane and a right workspace.
    - The left pane contains one monochrome one-turn path fragment per active color; each fragment is a single-pixel-wide L with exactly one right-angle turn.
    - The right pane contains exactly two isolated anchor pixels for each active color, and those colors match the instruction fragments one-to-one.

    Rule:
    - Read the turn order encoded by each left-pane fragment of a given color: from its upper-left endpoint to its other endpoint, the first segment is either horizontal or vertical.
    - For each same-color anchor pair on the right, start from the upper-left anchor and connect it to the other anchor using the same turn order.
    - Draw the resulting single-pixel-wide L-shaped line in that color, while leaving the instruction pane and separator untouched.

    Key insight:
    - Color matches each workspace pair to a left-side template, and the template's first segment direction transfers an abstract turn pattern onto concrete anchor coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    instruction_colors = {
        int(color)
        for color in np.unique(left)
        if int(color) not in (int(bg), int(sep_color))
    }
    anchor_colors = {
        int(color)
        for color in np.unique(right)
        if int(color) != int(bg)
    }
    assert instruction_colors == anchor_colors, "left instructions and right anchors must use the same colors"

    patterns: dict[int, str] = {}
    for color in instruction_colors:
        coords = np.argwhere(left == int(color))
        patterns[color] = _instruction_pattern(coords)

    out = g.copy()
    for color in sorted(anchor_colors):
        coords = np.argwhere(right == int(color))
        assert len(coords) == 2, f"color {color} must have exactly two anchors"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        start, end = _ordered_upper_left(a, b)
        for r, c in _path_cells(start, end, patterns[color]):
            out[r, sep_col + 1 + c] = int(color)
    return out


def _sample_bands(total: int, count: int, min_size: int, gap: int = 1) -> list[tuple[int, int]] | None:
    required = count * min_size + (count + 1) * gap
    if required > total:
        return None

    sizes = [min_size] * count
    gaps = [gap] * (count + 1)
    extra = total - required
    buckets = [("size", i) for i in range(count)] + [("gap", i) for i in range(count + 1)]
    for _ in range(extra):
        kind, idx = random.choice(buckets)
        if kind == "size":
            sizes[idx] += 1
        else:
            gaps[idx] += 1

    bands: list[tuple[int, int]] = []
    r = gaps[0]
    for i, size in enumerate(sizes):
        bands.append((r, r + size - 1))
        r += size + gaps[i + 1]
    return bands


def _make_instruction_fragment(height: int, width: int, pattern: str, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if pattern == "HV":
        patch[0, :] = int(color)
        patch[:, -1] = int(color)
    else:
        patch[:, 0] = int(color)
        patch[-1, :] = int(color)
    return patch


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _background_color(grid)
    try:
        sep_col, sep_color = _find_separator(grid, bg)
    except AssertionError:
        return False

    left = grid[:, :sep_col]
    right = grid[:, sep_col + 1 :]
    left_colors = {
        int(color)
        for color in np.unique(left)
        if int(color) not in (int(bg), int(sep_color))
    }
    right_colors = {
        int(color)
        for color in np.unique(right)
        if int(color) != int(bg)
    }
    if not left_colors or left_colors != right_colors:
        return False

    for color in left_colors:
        if np.count_nonzero(right == int(color)) != 2:
            return False
        if np.count_nonzero(left == int(color)) < 3:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 28)
    left_width = random.randint(6, 10)
    right_width = random.randint(8, min(19, 29 - left_width))
    width = left_width + 1 + right_width
    if width > 30:
        return None

    max_pairs = min(5, (height - 1) // 4, len(AVAILABLE_COLORS))
    if max_pairs < 2:
        return None
    pair_count = random.randint(2, max_pairs)

    right_bands = _sample_bands(height, pair_count, min_size=3, gap=1)
    left_bands = _sample_bands(height, pair_count, min_size=2, gap=1)
    if right_bands is None or left_bands is None:
        return None

    colors = random.sample(AVAILABLE_COLORS, pair_count)
    hv_count = pair_count // 2 + random.randint(0, pair_count % 2)
    patterns = ["HV"] * hv_count + ["VH"] * (pair_count - hv_count)
    random.shuffle(patterns)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_width] = SEPARATOR_COLOR

    for color, pattern, band in zip(colors, patterns, right_bands):
        r_lo, r_hi = band
        start_r = random.randint(r_lo, r_hi - 1)
        end_r = random.randint(start_r + 1, r_hi)
        start_c = random.randint(1, right_width - 4)
        end_c = random.randint(start_c + 2, right_width - 2)
        grid[start_r, left_width + 1 + start_c] = int(color)
        grid[end_r, left_width + 1 + end_c] = int(color)

    left_specs = list(zip(colors, patterns))
    random.shuffle(left_specs)
    for (color, pattern), band in zip(left_specs, left_bands):
        r_lo, r_hi = band
        max_h = min(5, r_hi - r_lo + 1)
        max_w = min(5, left_width - 1)
        if max_h < 2 or max_w < 2:
            return None
        frag_h = random.randint(2, max_h)
        frag_w = random.randint(2, max_w)
        row0 = random.randint(r_lo, r_hi - frag_h + 1)
        col_limit = left_width - frag_w - 1
        col0 = random.randint(0, max(0, col_limit))
        patch = _make_instruction_fragment(frag_h, frag_w, pattern, color)
        mask = patch != BG
        target = grid[row0 : row0 + frag_h, col0 : col0 + frag_w]
        if np.any(target[mask] != BG):
            return None
        target[mask] = patch[mask]

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
