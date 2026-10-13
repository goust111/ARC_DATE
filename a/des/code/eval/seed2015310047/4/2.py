from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
RED = 2
ORANGE = 7
FRAGMENT_COLORS = (1, 3, 4, 6)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

HEIGHT_RANGE = (9, 18)
WIDTH_RANGE = (11, 19)
MAX_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected exactly one full-height wall column"
    return candidates[0]


def _row_projection_colors(grid: np.ndarray, wall_c: int, wall_color: int, bg: int, side: str) -> dict[int, int]:
    assert side in {"left", "right"}
    cols = grid[:, :wall_c] if side == "left" else grid[:, wall_c + 1 :]
    colors_by_row: dict[int, int] = {}
    for r in range(grid.shape[0]):
        row = cols[r]
        colors = np.unique(row[(row != bg) & (row != wall_color)])
        if colors.size == 0:
            continue
        assert colors.size == 1, "each side may project at most one color per row"
        colors_by_row[r] = int(colors[0])
    return colors_by_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project left/right fragments horizontally onto the central wall.

    Single-sided projections keep the fragment color, overlapping left/right
    projections become red, and untouched wall cells stay orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    wall_c, wall_color = _find_wall(g, bg)

    left = _row_projection_colors(g, wall_c, wall_color, bg, "left")
    right = _row_projection_colors(g, wall_c, wall_color, bg, "right")

    out = g.copy()
    for r in range(g.shape[0]):
        if r in left and r in right:
            out[r, wall_c] = RED
        elif r in left:
            out[r, wall_c] = left[r]
        elif r in right:
            out[r, wall_c] = right[r]
        else:
            out[r, wall_c] = wall_color
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_connected_shape(
    rng: random.Random,
    size: int,
    *,
    min_rows: int = 1,
    max_rows: int = 3,
    max_cols: int = 4,
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.append(cand)
            cells.add(rng.choice(frontier))
        shape = _normalize_shape(cells)
        row_count = len({r for r, _ in shape})
        col_count = len({c for _, c in shape})
        if min_rows <= row_count <= max_rows and col_count <= max_cols:
            return shape
    return None


def _side_bounds(width: int, wall_c: int, side: str) -> tuple[int, int]:
    if side == "left":
        return 0, wall_c - 2
    return wall_c + 2, width - 1


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BACKGROUND:
            return False
    for r, c in cell_set:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BACKGROUND:
                return False
    return True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_row_segment(
    grid: np.ndarray,
    wall_c: int,
    side: str,
    row: int,
    color: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    c0, c1 = _side_bounds(grid.shape[1], wall_c, side)
    usable = c1 - c0 + 1
    if usable < 2:
        return None
    lengths = list(range(2, min(5, usable) + 1))
    rng.shuffle(lengths)
    for length in lengths:
        starts = list(range(c0, c1 - length + 2))
        rng.shuffle(starts)
        for start in starts:
            cells = [(row, c) for c in range(start, start + length)]
            if _can_place(grid, cells):
                _paint(grid, cells, color)
                return cells
    return None


def _place_random_fragment(
    grid: np.ndarray,
    wall_c: int,
    side: str,
    used_rows: set[int],
    color: int,
    rng: random.Random,
) -> set[int] | None:
    c0, c1 = _side_bounds(grid.shape[1], wall_c, side)
    if c1 - c0 + 1 < 2:
        return None
    for _ in range(120):
        size = rng.randint(2, 5)
        min_rows = 2 if rng.random() < 0.75 else 1
        shape = _random_connected_shape(rng, size, min_rows=min_rows)
        if shape is None:
            continue
        shape_h = 1 + max(r for r, _ in shape)
        shape_w = 1 + max(c for _, c in shape)
        tops = list(range(0, grid.shape[0] - shape_h + 1))
        rng.shuffle(tops)
        lefts = list(range(c0, c1 - shape_w + 2))
        rng.shuffle(lefts)
        for top in tops:
            rows = {top + r for r, _ in shape}
            if rows & used_rows:
                continue
            for left in lefts:
                cells = [(top + r, left + c) for r, c in shape]
                if _can_place(grid, cells):
                    _paint(grid, cells, color)
                    return rows
    return None


def _components(grid: np.ndarray, wall_c: int, wall_color: int, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (bg, wall_color):
                continue
            side = "left" if c < wall_c else "right"
            assert c != wall_c
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "side": side, "coords": coords})
    return comps


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert np.any(grid == BACKGROUND)

    bg = _dominant_color(grid)
    assert bg == BACKGROUND
    wall_c, wall_color = _find_wall(grid, bg)
    assert wall_color == ORANGE
    assert 2 <= wall_c <= grid.shape[1] - 3

    comps = _components(grid, wall_c, wall_color, bg)
    assert comps, "expected fragments"

    left_rows: set[int] = set()
    right_rows: set[int] = set()
    sides_present = set()
    for comp in comps:
        color = int(comp["color"])
        side = str(comp["side"])
        coords = comp["coords"]
        rows = {r for r, _ in coords}
        assert color in FRAGMENT_COLORS
        assert 2 <= len(coords) <= 5
        sides_present.add(side)
        target_rows = left_rows if side == "left" else right_rows
        assert rows.isdisjoint(target_rows), "same-side projections must not share rows"
        target_rows.update(rows)
    assert sides_present == {"left", "right"}
    assert left_rows & right_rows
    assert left_rows - right_rows
    assert right_rows - left_rows
    assert set(range(grid.shape[0])) - (left_rows | right_rows)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[:, :wall_c], grid[:, :wall_c])
    assert np.array_equal(out[:, wall_c + 1 :], grid[:, wall_c + 1 :])

    left = _row_projection_colors(grid, wall_c, wall_color, bg, "left")
    right = _row_projection_colors(grid, wall_c, wall_color, bg, "right")
    for r in range(grid.shape[0]):
        if r in left and r in right:
            assert int(out[r, wall_c]) == RED
        elif r in left:
            assert int(out[r, wall_c]) == left[r]
        elif r in right:
            assert int(out[r, wall_c]) == right[r]
        else:
            assert int(out[r, wall_c]) == wall_color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    wall_c = min(max(width // 2 + rng.randint(-1, 1), 3), width - 4)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, wall_c] = ORANGE

    overlap_count = rng.randint(1, 2)
    left_only_count = rng.randint(1, 2)
    right_only_count = rng.randint(1, 2)
    total_special = overlap_count + left_only_count + right_only_count
    rows = rng.sample(range(height), total_special)
    overlap_rows = rows[:overlap_count]
    left_only_rows = rows[overlap_count : overlap_count + left_only_count]
    right_only_rows = rows[overlap_count + left_only_count :]

    used_left: set[int] = set()
    used_right: set[int] = set()

    for row in overlap_rows:
        if _place_row_segment(grid, wall_c, "left", row, rng.choice(FRAGMENT_COLORS), rng) is None:
            return None
        if _place_row_segment(grid, wall_c, "right", row, rng.choice(FRAGMENT_COLORS), rng) is None:
            return None
        used_left.add(row)
        used_right.add(row)

    for row in left_only_rows:
        if _place_row_segment(grid, wall_c, "left", row, rng.choice(FRAGMENT_COLORS), rng) is None:
            return None
        used_left.add(row)

    for row in right_only_rows:
        if _place_row_segment(grid, wall_c, "right", row, rng.choice(FRAGMENT_COLORS), rng) is None:
            return None
        used_right.add(row)

    max_extra_left = min(2, max(0, (height - len(used_left)) // 2))
    max_extra_right = min(2, max(0, (height - len(used_right)) // 2))
    extra_left = rng.randint(0, max_extra_left)
    extra_right = rng.randint(0, max_extra_right)
    if extra_left + extra_right == 0 and (max_extra_left > 0 or max_extra_right > 0) and rng.random() < 0.75:
        if max_extra_left > 0 and max_extra_right > 0:
            if rng.random() < 0.5:
                extra_left = 1
            else:
                extra_right = 1
        elif max_extra_left > 0:
            extra_left = 1
        else:
            extra_right = 1

    for _ in range(extra_left):
        rows_used = _place_random_fragment(grid, wall_c, "left", used_left, rng.choice(FRAGMENT_COLORS), rng)
        if rows_used is None:
            return None
        used_left.update(rows_used)

    for _ in range(extra_right):
        rows_used = _place_random_fragment(grid, wall_c, "right", used_right, rng.choice(FRAGMENT_COLORS), rng)
        if rows_used is None:
            return None
        used_right.update(rows_used)

    try:
        _validate_generated_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
