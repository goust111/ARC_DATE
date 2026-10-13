from __future__ import annotations

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(cells: set[Coord]) -> set[Coord]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _translate(cells: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(cells)


def _random_polyomino(size: int, rng: np.random.Generator) -> set[Coord]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[Coord] = []
        seen = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) not in cells and (rr, cc) not in seen:
                    frontier.append((rr, cc))
                    seen.add((rr, cc))
        cells.add(frontier[int(rng.integers(len(frontier)))])

    transformed = set(cells)
    if bool(rng.integers(2)):
        transformed = {(r, -c) for r, c in transformed}
    for _ in range(int(rng.integers(4))):
        transformed = {(-c, r) for r, c in transformed}
    return _normalize(transformed)


def _all_origins(cells: set[Coord], height: int, width: int) -> list[Coord]:
    min_r, min_c, max_r, max_c = _bbox(cells)
    origins = []
    for r0 in range(-min_r, height - max_r):
        for c0 in range(-min_c, width - max_c):
            origins.append((r0, c0))
    return origins


def _roomy_origins(cells: set[Coord], height: int, width: int, origins: list[Coord]) -> list[Coord]:
    roomy = []
    for r0, c0 in origins:
        placed = _translate(cells, r0, c0)
        min_r, min_c, max_r, max_c = _bbox(placed)
        if min_r >= 1 and min_c >= 1 and max_r <= height - 2 and max_c <= width - 2:
            roomy.append((r0, c0))
    return roomy or origins


def _separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    assert w % 2 == 1, "grid width must split evenly around a 1-cell separator"
    mid = w // 2
    column = grid[:, mid]
    assert int(column[0]) != bg and np.all(column == column[0]), "missing separator column"
    return int(mid)


def _color_cells(panel: np.ndarray, color: int) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in np.argwhere(panel == color)}


def _infer_translation(left_panel: np.ndarray, bg: int) -> tuple[int, int]:
    height, width = left_panel.shape
    colors = [int(color) for color in np.unique(left_panel) if int(color) != bg]
    assert len(colors) == 2, "left panel must contain exactly two non-background colors"

    candidates: list[tuple[int, int, int]] = []
    for original_color in colors:
        shadow_color = colors[0] if colors[1] == original_color else colors[1]
        original = _color_cells(left_panel, original_color)
        shadow_visible = _color_cells(left_panel, shadow_color)
        if not (3 <= len(original) <= 6 and _is_connected(original)):
            continue
        if not (1 <= len(shadow_visible) < len(original)):
            continue

        for dr in range(-(height - 1), height):
            for dc in range(-(width - 1), width):
                if dr == 0 and dc == 0:
                    continue
                shifted = _translate(original, dr, dc)
                if any(r < 0 or r >= height or c < 0 or c >= width for r, c in shifted):
                    continue
                visible_shadow = shifted - original
                overlap = len(original) - len(visible_shadow)
                if overlap <= 0:
                    continue
                if visible_shadow == shadow_visible:
                    candidates.append((dr, dc, overlap))

    unique = {(dr, dc) for dr, dc, _overlap in candidates}
    assert len(unique) == 1, "left panel must imply a unique translation"
    dr, dc = next(iter(unique))
    return int(dr), int(dc)


def _paint(grid: np.ndarray, cells: set[Coord], color: int, col_offset: int = 0) -> None:
    for r, c in cells:
        grid[r, c + col_offset] = int(color)


def _expected_output(grid: np.ndarray, right_cells: set[Coord], right_color: int, dr: int, dc: int) -> np.ndarray:
    out = grid.copy()
    sep_c = _separator_col(grid, _get_bg(grid))
    right_width = grid.shape[1] - sep_c - 1
    for r, c in right_cells:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < right_width:
            out[rr, sep_c + 1 + cc] = int(right_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the shadow-offset on the left side and duplicate the right-side shape with the same translation.

    Input structure:
    - The background is black, and a 1-pixel-wide vertical separator line splits the grid into equal left and right subgrids.
    - The left subgrid contains one small connected shape in one color and a shadow copy of the same shape in a second color.
    - The shadow is a pure translation of the original shape, and any overlap is hidden by the original color.
    - The right subgrid contains a different small connected shape in a third color and no shadow.

    Rule:
    - Read the left subgrid and find the translation vector `(dr, dc)` whose shifted original shape produces the visible shadow-only pixels.
    - Keep the left subgrid and the separator line unchanged.
    - In the right subgrid, keep the original shape and draw a second copy shifted by the same `(dr, dc)`.
    - Clip the copied right-side pixels to the right subgrid bounds if the translation would send them out of bounds.

    Key insight:
    - The left panel is an instruction panel: the visible difference between the original shape and its occluded shadow directly encodes the translation to reuse on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sep_c = _separator_col(g, bg)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]

    dr, dc = _infer_translation(left, bg)

    right_colors = [int(color) for color in np.unique(right) if int(color) != bg]
    assert len(right_colors) == 1, "right panel must contain exactly one non-background color"
    right_color = right_colors[0]
    right_cells = _color_cells(right, right_color)
    assert 3 <= len(right_cells) <= 6 and _is_connected(right_cells)

    out = g.copy()
    right_width = right.shape[1]
    for r, c in right_cells:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < g.shape[0] and 0 <= cc < right_width:
            out[rr, sep_c + 1 + cc] = int(right_color)
    return out


def _sample_vector(branch: int, rng: np.random.Generator) -> tuple[int, int]:
    steps = [-2, -1, 1, 2]
    if branch == 0:
        return 0, int(rng.choice(steps))
    if branch == 1:
        return int(rng.choice(steps)), 0
    return int(rng.choice(steps)), int(rng.choice(steps))


def _build_left_scene(
    height: int,
    width: int,
    rng: np.random.Generator,
    original_color: int,
    shadow_color: int,
    branch: int,
) -> tuple[set[Coord], set[Coord], int, int] | None:
    for _ in range(120):
        shape = _random_polyomino(int(rng.integers(3, 7)), rng)
        dr, dc = _sample_vector(branch, rng)
        shifted = _translate(shape, dr, dc)
        overlap = shape & shifted
        if not (0 < len(overlap) < len(shape)):
            continue

        union = shape | shifted
        origins = _all_origins(union, height, width)
        if not origins:
            continue
        origins = _roomy_origins(union, height, width, origins)
        r0, c0 = origins[int(rng.integers(len(origins)))]

        original = _translate(shape, r0, c0)
        shadow_visible = _translate(shifted, r0, c0) - original
        if len(shadow_visible) >= len(original):
            continue

        panel = np.full((height, width), BG, dtype=int)
        _paint(panel, shadow_visible, shadow_color)
        _paint(panel, original, original_color)

        try:
            found_dr, found_dc = _infer_translation(panel, BG)
        except AssertionError:
            continue
        if (found_dr, found_dc) != (dr, dc):
            continue
        return original, shadow_visible, dr, dc
    return None


def _build_right_scene(
    height: int,
    width: int,
    dr: int,
    dc: int,
    clip_copy: bool,
    rng: np.random.Generator,
) -> set[Coord] | None:
    for _ in range(120):
        shape = _random_polyomino(int(rng.integers(3, 7)), rng)
        origins = _all_origins(shape, height, width)
        if not origins:
            continue

        valid: list[Coord] = []
        for r0, c0 in origins:
            placed = _translate(shape, r0, c0)
            shifted = _translate(placed, dr, dc)
            in_bounds = {
                (r, c)
                for r, c in shifted
                if 0 <= r < height and 0 <= c < width
            }
            if clip_copy:
                if not (0 < len(in_bounds) < len(shape)):
                    continue
            else:
                if len(in_bounds) != len(shape):
                    continue
            if not (in_bounds - placed):
                continue
            valid.append((r0, c0))

        if not valid:
            continue
        chosen = valid[int(rng.integers(len(valid)))]
        return _translate(shape, chosen[0], chosen[1])
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sub_width = int(rng.integers(8, 15))
    height = int(rng.integers(8, 16))
    full_width = 2 * sub_width + 1
    if max(height, full_width) > 30:
        return None

    colors = rng.choice(PALETTE, size=3, replace=False).tolist()
    original_color, shadow_color, right_color = [int(color) for color in colors]
    branch = int(rng.integers(3))
    clip_copy = bool(rng.integers(2))

    left_scene = _build_left_scene(height, sub_width, rng, original_color, shadow_color, branch)
    if left_scene is None:
        return None
    left_original, left_shadow_visible, dr, dc = left_scene

    right_shape = _build_right_scene(height, sub_width, dr, dc, clip_copy, rng)
    if right_shape is None:
        return None

    grid = np.full((height, full_width), BG, dtype=int)
    grid[:, sub_width] = SEPARATOR
    _paint(grid, left_shadow_visible, shadow_color)
    _paint(grid, left_original, original_color)
    _paint(grid, right_shape, right_color, col_offset=sub_width + 1)

    expected = _expected_output(grid, right_shape, right_color, dr, dc)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
