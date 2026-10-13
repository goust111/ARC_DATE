from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEPARATOR = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

INPUT_SHARED_SAME = YELLOW
INPUT_SHARED_DIFF_LEFT = BLUE
INPUT_SHARED_DIFF_RIGHT = GREEN
TERRAIN_COLORS = (PINK, ORANGE, TEAL, MAROON)

HEIGHT_RANGE = (9, 15)
WIDTH_RANGE = (7, 11)
SHELF_LEN_RANGE = (2, 3)
BOTTOM_COUNT_RANGE = (2, 4)
SUPPORT_COUNT_MAX = 3
GEN_RETRIES = 400

RIGHT_BLOB_TEMPLATES = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _separator_col(grid: np.ndarray, bg: int) -> int:
    cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(cols) == 1, "expected exactly one solid separator column"
    return int(cols[0])


def _fold_layers(left: np.ndarray, right: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    left_fg = left != bg
    right_fg = right != bg
    terrain = np.full(left.shape, bg, dtype=int)
    left_only = left_fg & ~right_fg
    right_only = right_fg & ~left_fg
    terrain[left_only] = left[left_only]
    terrain[right_only] = right[right_only]
    sand = left_fg & right_fg
    return terrain, sand


def _drop_sand(
    terrain: np.ndarray, sand_mask: np.ndarray, bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str]]]:
    out = np.array(terrain, copy=True)
    H, W = terrain.shape
    landings: list[tuple[int, int, int, str]] = []

    for c in range(W):
        occupied = out[:, c] != bg
        sand_rows = np.flatnonzero(sand_mask[:, c]).tolist()
        for src_r in reversed(sand_rows):
            dst_r = int(src_r)
            while dst_r + 1 < H and not occupied[dst_r + 1]:
                dst_r += 1
            if dst_r == H - 1:
                support = "bottom"
            elif terrain[dst_r + 1, c] != bg:
                support = "terrain"
            else:
                support = "sand"
            out[dst_r, c] = RED
            occupied[dst_r] = True
            landings.append((int(src_r), int(dst_r), int(c), support))
    return out, landings


def _solve_from_halves(left: np.ndarray, right: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, str]]]:
    terrain, sand = _fold_layers(left, right, bg)
    out, landings = _drop_sand(terrain, sand, bg)
    return terrain, sand, out, landings


def main(grid: np.ndarray) -> np.ndarray:
    """Fold across the vertical separator, keep singly covered cells as terrain, and let doubly covered cells fall as red sand.

    Input structure:
    - A single non-background vertical separator column spans the full height.
    - The separator splits the grid into equal-width left and right panels.
    - Red is absent from the input; foreground pixels may appear on either panel.

    Rule:
    - Remove the separator and mirror the right panel horizontally into the left-panel coordinate system.
    - A cell covered by exactly one side keeps that side's color as fixed terrain.
    - A cell covered by both sides becomes a red sand grain at that folded coordinate.
    - In each column, sand grains fall straight down, with lower source grains settling first,
      and they stop on the bottom edge, fixed terrain, or previously settled sand.

    Key insight:
    Folding is only a coverage census: single coverage defines the immobile terrain, while
    double coverage defines the sand sources whose final positions are determined afterward by gravity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    sep = _separator_col(g, bg)
    assert 2 * sep + 1 == g.shape[1], "separator must split equal-width panels"

    left = g[:, :sep]
    right = np.fliplr(g[:, sep + 1 :])
    assert left.shape == right.shape

    _terrain, _sand, out, _landings = _solve_from_halves(left, right, bg)
    return out


def _paint(panel: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        panel[r, c] = int(color)


def _mark_used(used: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        used[r, c] = True


def _deterministic_rows(limit_exclusive: int, count: int) -> list[int]:
    rows = list(range(0, limit_exclusive + 1, 2))
    assert len(rows) >= count
    return rows[:count]


def _free_cells(
    used: np.ndarray,
    cells: list[tuple[int, int]],
    *,
    allowed_cols: set[int] | None = None,
    margin: int = 0,
) -> bool:
    H, W = used.shape
    seen = set()
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if allowed_cols is not None and c not in allowed_cols:
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))
        if used[r, c]:
            return False
    if margin <= 0:
        return True
    for r, c in cells:
        r0 = max(0, r - margin)
        r1 = min(H - 1, r + margin)
        c0 = max(0, c - margin)
        c1 = min(W - 1, c + margin)
        if np.any(used[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _random_blob_cells(
    H: int,
    W: int,
    used: np.ndarray,
    *,
    allowed_cols: set[int],
    margin: int = 1,
    attempts: int = 100,
) -> list[tuple[int, int]] | None:
    for _ in range(attempts):
        sprite = np.array(random.choice(RIGHT_BLOB_TEMPLATES), copy=True)
        if random.random() < 0.5:
            sprite = np.fliplr(sprite)
        sprite = np.rot90(sprite, k=random.randrange(4))
        coords = np.argwhere(sprite)
        sh, sw = sprite.shape
        r0 = random.randrange(0, H - sh + 1)
        c0 = random.randrange(0, W - sw + 1)
        cells = [(int(r0 + r), int(c0 + c)) for r, c in coords]
        if _free_cells(used, cells, allowed_cols=allowed_cols, margin=margin):
            return cells
    return None


def _build_case() -> np.ndarray | None:
    H = random.randint(*HEIGHT_RANGE)
    W = random.randint(*WIDTH_RANGE)

    left = np.full((H, W), BG, dtype=int)
    right = np.full((H, W), BG, dtype=int)
    used = np.zeros((H, W), dtype=bool)

    shelf_len = random.randint(SHELF_LEN_RANGE[0], min(SHELF_LEN_RANGE[1], W - 4))
    shelf_start = random.randint(1, W - shelf_len - 2)
    shelf_cols = list(range(shelf_start, shelf_start + shelf_len))
    shelf_row = random.randint(H // 2, H - 3)
    shelf_cells = [(shelf_row, c) for c in shelf_cols]
    _paint(left, shelf_cells, random.choice(TERRAIN_COLORS))
    _mark_used(used, shelf_cells)

    support_col = random.choice(shelf_cols)
    support_rows_all = _deterministic_rows(shelf_row - 2, SUPPORT_COUNT_MAX if shelf_row >= 6 else 2)
    support_count = random.randint(1, min(SUPPORT_COUNT_MAX, len(support_rows_all)))
    support_rows = support_rows_all[:support_count]
    support_cells = [(r, support_col) for r in support_rows]
    _paint(left, support_cells, INPUT_SHARED_SAME)
    _paint(right, support_cells, INPUT_SHARED_SAME)
    _mark_used(used, support_cells)

    distant_cols = [
        c
        for c in range(W)
        if c not in shelf_cols and (c < shelf_start - 1 or c > shelf_cols[-1] + 1)
    ]
    if not distant_cols:
        distant_cols = [c for c in range(W) if c not in shelf_cols]
    if not distant_cols:
        return None
    bottom_col = random.choice(distant_cols)
    bottom_rows_all = _deterministic_rows(H - 3, min(BOTTOM_COUNT_RANGE[1], len(list(range(0, H - 2, 2)))))
    bottom_count = random.randint(BOTTOM_COUNT_RANGE[0], min(BOTTOM_COUNT_RANGE[1], len(bottom_rows_all)))
    bottom_rows = bottom_rows_all[:bottom_count]
    bottom_cells = [(r, bottom_col) for r in bottom_rows]
    _paint(left, bottom_cells, INPUT_SHARED_DIFF_LEFT)
    _paint(right, bottom_cells, INPUT_SHARED_DIFF_RIGHT)
    _mark_used(used, bottom_cells)

    blob_allowed_cols = set(range(W)) - set(shelf_cols) - {bottom_col}
    if len(blob_allowed_cols) < 2:
        return None
    blob_cells = _random_blob_cells(H, W, used, allowed_cols=blob_allowed_cols, margin=1)
    if blob_cells is None:
        return None
    terrain_palette = [color for color in TERRAIN_COLORS if color != left[shelf_cells[0]]]
    _paint(right, blob_cells, random.choice(terrain_palette))
    _mark_used(used, blob_cells)

    if random.random() < 0.5:
        extra_allowed_cols = set(range(W)) - set(shelf_cols) - {bottom_col}
        extra_cells = _random_blob_cells(H, W, used, allowed_cols=extra_allowed_cols, margin=1)
        if extra_cells is not None:
            _paint(left, extra_cells, left[shelf_cells[0]])
            _mark_used(used, extra_cells)

    sep_col = np.full((H, 1), SEPARATOR, dtype=int)
    grid = np.concatenate([left, sep_col, np.fliplr(right)], axis=1)
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    if grid is None:
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.any(grid == RED):
        return False
    if np.count_nonzero(grid == SEPARATOR) != grid.shape[0]:
        return False

    bg = _get_bg(grid)
    if bg != BG:
        return False
    sep = _separator_col(grid, bg)
    left = grid[:, :sep]
    right = np.fliplr(grid[:, sep + 1 :])
    terrain, sand, expected, landings = _solve_from_halves(left, right, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    left_only = (left != bg) & (right == bg)
    right_only = (right != bg) & (left == bg)
    if left_only.sum() == 0 or right_only.sum() == 0 or sand.sum() == 0:
        return False

    overlap_same = sand & (left == right)
    overlap_diff = sand & (left != right)
    if overlap_same.sum() == 0 or overlap_diff.sum() == 0:
        return False

    if not any(src_r != dst_r for src_r, dst_r, _c, _support in landings):
        return False

    supports = {support for _src_r, _dst_r, _c, support in landings}
    if not {"bottom", "terrain", "sand"} <= supports:
        return False

    if max(int(sand[:, c].sum()) for c in range(sand.shape[1])) < 2:
        return False

    pre_gravity = np.array(terrain, copy=True)
    pre_gravity[sand] = RED
    if np.array_equal(actual, terrain) or np.array_equal(actual, pre_gravity):
        return False

    if np.count_nonzero(actual == RED) != int(sand.sum()):
        return False

    if np.count_nonzero(grid != BG) >= grid.size // 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid fold-survey gravity example")
