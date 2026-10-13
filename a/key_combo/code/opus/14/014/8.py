from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500
H_RANGE = (10, 17)  # randint upper bound is exclusive
HALF_W_RANGE = (8, 13)

_LINE_TEMPLATES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
]

_BLOB_TEMPLATES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
]


def _normalize_shape(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = np.array(list(cells), dtype=int)
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = np.array(cells, dtype=int)
    variants: set[tuple[tuple[int, int], ...]] = set()
    for flip in (1, -1):
        cur = pts.copy()
        cur[:, 1] *= flip
        for _ in range(4):
            variants.add(_normalize_shape(cur))
            cur = np.column_stack((-cur[:, 1], cur[:, 0]))
    return list(variants)


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_marker_side(grid: np.ndarray) -> int:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape == (2, 2), "expected exactly two green markers"
    assert int(coords[0, 0]) == int(coords[1, 0]), "green markers must share a row"
    width = int(grid.shape[1])
    assert width % 2 == 0, "input width must be even"
    half_w = width // 2
    cols = coords[:, 1]
    if np.all(cols < half_w):
        return 0
    if np.all(cols >= half_w):
        return 1
    raise AssertionError("green markers must lie in the same half")


def _transform_half(half: np.ndarray) -> np.ndarray:
    core = np.array(half, dtype=int, copy=True)
    core[core == GREEN] = BLACK

    blank_rows = np.all(core == BLACK, axis=1)
    blank_cols = np.all(core == BLACK, axis=0)

    out = core.copy()
    out[blank_rows, :] = YELLOW
    out[:, blank_cols] = YELLOW

    to_red = []
    for r, c in np.argwhere(core == BLUE):
        if any(out[rr, cc] == YELLOW for rr, cc in _neighbors4(out.shape, int(r), int(c))):
            to_red.append((int(r), int(c)))
    for r, c in to_red:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the green-marked half, paint its fully blank rows and columns yellow, then recolor adjacent blue pixels red.

    Input structure:
    - The grid has an even width and a black background with blue and red foreground patterns.
    - Exactly two green marker pixels share one row and both lie in the same left or right half.
    - The kept half contains at least one row and one column that become fully black after the green markers are erased.

    Rule:
    - Keep the left or right half that contains both green markers.
    - Erase the green markers to black inside that kept half.
    - Find every row and column in the kept half that is entirely black, and paint all of those cells yellow.
    - Recolor each blue pixel to red if at least one of its 4-neighbors is yellow; leave red pixels unchanged.

    Key insight:
    - The green markers choose the half, that half determines which rows and columns are completely empty, and those new yellow guide lines determine exactly which blue pixels change color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side = _find_marker_side(g)
    half_w = g.shape[1] // 2
    kept = g[:, side * half_w : (side + 1) * half_w]
    return _transform_half(kept)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    allowed: np.ndarray,
    color: int,
    templates: list[tuple[tuple[int, int], ...]],
) -> bool:
    h, w = grid.shape
    choices: list[list[tuple[int, int]]] = []
    for base in templates:
        for variant in _shape_variants(base):
            shape_h = max(r for r, _ in variant) + 1
            shape_w = max(c for _, c in variant) + 1
            for r0 in range(h - shape_h + 1):
                for c0 in range(w - shape_w + 1):
                    cells = [(r0 + r, c0 + c) for r, c in variant]
                    if all(allowed[r, c] and not blocked[r, c] for r, c in cells):
                        choices.append(cells)
    if not choices:
        return False

    cells = random.choice(choices)
    for r, c in cells:
        grid[r, c] = color
    _mark_blocked(blocked, cells)
    return True


def _actual_blank_sets(half_no_markers: np.ndarray) -> tuple[set[int], set[int]]:
    rows = {int(r) for r, is_blank in enumerate(np.all(half_no_markers == BLACK, axis=1)) if is_blank}
    cols = {int(c) for c, is_blank in enumerate(np.all(half_no_markers == BLACK, axis=0)) if is_blank}
    return rows, cols


def _classify_blue_adjacency(half_no_markers: np.ndarray) -> tuple[int, int, int]:
    blank_rows, blank_cols = _actual_blank_sets(half_no_markers)
    row_only = 0
    col_only = 0
    safe = 0
    for r, c in np.argwhere(half_no_markers == BLUE):
        rr = int(r)
        cc = int(c)
        adj_row = any(nr in blank_rows for nr, nc in _neighbors4(half_no_markers.shape, rr, cc))
        adj_col = any(nc in blank_cols for nr, nc in _neighbors4(half_no_markers.shape, rr, cc))
        if adj_row and not adj_col:
            row_only += 1
        elif adj_col and not adj_row:
            col_only += 1
        elif not adj_row and not adj_col:
            safe += 1
    return row_only, col_only, safe


def _has_red_adjacent_to_yellow(half_no_markers: np.ndarray) -> bool:
    transformed = _transform_half(half_no_markers)
    for r, c in np.argwhere(half_no_markers == RED):
        if any(transformed[rr, cc] == YELLOW for rr, cc in _neighbors4(transformed.shape, int(r), int(c))):
            return True
    return False


def _build_masks(h: int, w: int, blank_rows: set[int], blank_cols: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    line_mask = np.zeros((h, w), dtype=bool)
    for r in blank_rows:
        line_mask[r, :] = True
    for c in blank_cols:
        line_mask[:, c] = True

    row_only = np.zeros((h, w), dtype=bool)
    col_only = np.zeros((h, w), dtype=bool)
    safe = np.zeros((h, w), dtype=bool)
    adj_any = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if line_mask[r, c]:
                continue
            adj_row = any(abs(r - br) == 1 for br in blank_rows)
            adj_col = any(abs(c - bc) == 1 for bc in blank_cols)
            if adj_row and not adj_col:
                row_only[r, c] = True
            if adj_col and not adj_row:
                col_only[r, c] = True
            if not adj_row and not adj_col:
                safe[r, c] = True
            if adj_row or adj_col:
                adj_any[r, c] = True
    return row_only, col_only, safe, adj_any


def _build_selected_half(h: int, w: int) -> np.ndarray | None:
    if h < 10 or w < 8:
        return None

    marker_row = random.randint(2, h - 3)
    marker_cols = sorted(random.sample(range(1, w - 1), 2))
    if marker_cols[1] - marker_cols[0] < 2:
        return None

    blank_rows = {marker_row}
    row_candidates = [r for r in range(1, h - 1) if abs(r - marker_row) >= 3]
    if row_candidates and random.random() < 0.75:
        blank_rows.add(random.choice(row_candidates))

    blank_cols = set(marker_cols)
    col_candidates = [c for c in range(1, w - 1) if min(abs(c - mc) for mc in marker_cols) >= 3]
    if col_candidates and random.random() < 0.55:
        blank_cols.add(random.choice(col_candidates))

    row_only, col_only, safe, adj_any = _build_masks(h, w, blank_rows, blank_cols)
    if row_only.sum() < 3 or col_only.sum() < 3 or safe.sum() < 8:
        return None

    half = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for c in marker_cols:
        blocked[marker_row, c] = True

    if not _place_shape(half, blocked, row_only, BLUE, _LINE_TEMPLATES + _BLOB_TEMPLATES[:3]):
        return None
    if not _place_shape(half, blocked, col_only, BLUE, _LINE_TEMPLATES + _BLOB_TEMPLATES[:3]):
        return None
    if not _place_shape(half, blocked, safe, BLUE, _BLOB_TEMPLATES):
        return None
    if not _place_shape(half, blocked, adj_any, RED, _LINE_TEMPLATES + _BLOB_TEMPLATES):
        return None

    extra_templates = _BLOB_TEMPLATES + _LINE_TEMPLATES
    for _ in range(random.randint(0, 2)):
        extra_color = random.choice([BLUE, RED])
        extra_mask = safe if random.random() < 0.65 else adj_any
        _place_shape(half, blocked, extra_mask, extra_color, extra_templates)

    for c in marker_cols:
        if half[marker_row, c] != BLACK:
            return None
        half[marker_row, c] = GREEN
    return half


def _build_other_half(h: int, w: int) -> np.ndarray | None:
    half = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    allowed = np.ones((h, w), dtype=bool)

    colors = [BLUE, RED]
    random.shuffle(colors)
    base_count = random.randint(3, 5)
    for idx in range(base_count):
        color = colors[idx % 2]
        templates = _BLOB_TEMPLATES if random.random() < 0.7 else _LINE_TEMPLATES + _BLOB_TEMPLATES
        if not _place_shape(half, blocked, allowed, color, templates):
            return None
    return half


def _validate_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    side = _find_marker_side(grid)
    half_w = grid.shape[1] // 2
    kept = grid[:, side * half_w : (side + 1) * half_w].copy()
    other = grid[:, (1 - side) * half_w : (2 - side) * half_w].copy()
    kept_no_markers = kept.copy()
    kept_no_markers[kept_no_markers == GREEN] = BLACK

    if not np.array_equal(out, _transform_half(kept)):
        return False
    if np.array_equal(out, _transform_half(other)):
        return False

    blank_rows, blank_cols = _actual_blank_sets(kept_no_markers)
    if not blank_rows or not blank_cols:
        return False

    row_only, col_only, safe = _classify_blue_adjacency(kept_no_markers)
    if row_only == 0 or col_only == 0 or safe == 0:
        return False

    changed_blue = int(np.sum((kept_no_markers == BLUE) & (out == RED)))
    stayed_blue = int(np.sum((kept_no_markers == BLUE) & (out == BLUE)))
    if changed_blue == 0 or stayed_blue == 0:
        return False

    if not _has_red_adjacent_to_yellow(kept_no_markers):
        return False

    if np.any(kept_no_markers[np.array(sorted(blank_rows)), :] != BLACK):
        return False
    if np.any(kept_no_markers[:, np.array(sorted(blank_cols))] != BLACK):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(*H_RANGE)
    half_w = random.randrange(*HALF_W_RANGE)
    selected_half = _build_selected_half(h, half_w)
    if selected_half is None:
        return None

    other_half = _build_other_half(h, half_w)
    if other_half is None:
        return None

    side = random.randint(0, 1)
    if side == 0:
        grid = np.concatenate([selected_half, other_half], axis=1)
    else:
        grid = np.concatenate([other_half, selected_half], axis=1)

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
