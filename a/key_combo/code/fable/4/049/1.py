from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLORS = (BLUE, GREEN, GREY, PINK, ORANGE, TEAL, MAROON)
INPUT_COLORS = set(MARKER_COLORS) | {BLACK, RED}

GEN_ATTEMPTS = 500
GEN_TYPES = ("row_row", "col_col", "row_col")
GEN_SIZE_RANGE = (10, 22)
GEN_EXTRA_AUTH_RANGE = (0, 2)
GEN_MISMATCH_RANGE = (1, 4)
GEN_MIN_PATH_CELLS = 6


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _read_authorized_lines(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    row_auth = np.zeros(h, dtype=bool)
    col_auth = np.zeros(w, dtype=bool)

    for r in range(h):
        left = int(grid[r, 0])
        right = int(grid[r, w - 1])
        if left == BLACK and right == BLACK:
            continue
        assert left != BLACK and right != BLACK, "row markers must appear in pairs"
        row_auth[r] = left == right

    for c in range(w):
        top = int(grid[0, c])
        bottom = int(grid[h - 1, c])
        if top == BLACK and bottom == BLACK:
            continue
        assert top != BLACK and bottom != BLACK, "column markers must appear in pairs"
        col_auth[c] = top == bottom

    return row_auth, col_auth


def _walkable_mask(grid: np.ndarray, row_auth: np.ndarray, col_auth: np.ndarray) -> np.ndarray:
    walkable = row_auth[:, None] | col_auth[None, :]
    red_cells = grid == RED
    assert np.all(walkable[red_cells]), "each red endpoint must lie on an authorized row or column"
    return walkable


def _unique_shortest_path(grid: np.ndarray) -> list[tuple[int, int]]:
    red_coords = np.argwhere(grid == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red_coords[0]))
    goal = tuple(map(int, red_coords[1]))

    row_auth, col_auth = _read_authorized_lines(grid)
    walkable = _walkable_mask(grid, row_auth, col_auth)
    h, w = map(int, grid.shape)

    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc) or not walkable[nr, nc]:
                continue
            nd = dist[r, c] + 1
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                ways[nr, nc] = min(2, ways[nr, nc] + ways[r, c])

    assert dist[goal] >= 0, "expected the red endpoints to be connected"
    assert ways[goal] == 1, "expected a unique shortest path between the red endpoints"

    full_path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if _in_bounds((h, w), nr, nc) and dist[nr, nc] == dist[r, c] - 1 and ways[nr, nc] > 0
        ]
        assert len(preds) == 1, "shortest-path reconstruction should be unique"
        cur = preds[0]
        full_path.append(cur)
    full_path.reverse()
    return full_path


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the unique shortest hidden-corridor path between the two red endpoints yellow.

    Input structure:
    - The grid has a black background, exactly two red interior endpoints, and optional colored marker pairs on some boundary rows and columns.
    - A marked row has one color on its left edge and one color on its right edge; a marked column has one color on its top edge and one color on its bottom edge.
    - The red endpoints each lie on at least one row or column whose two boundary markers match.

    Rule:
    - A row or column is traversable exactly when its two boundary markers have the same color; all other interior cells are blocked.
    - Find the unique shortest 4-neighbor path through the resulting union of full authorized rows and full authorized columns.
    - Preserve the input and recolor only the black interior cells of that path to yellow.

    Key insight:
    - The corridors are invisible in the input: equality of a boundary marker pair is the only clue that unlocks an entire row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 3 <= g.shape[0] <= 30 and 3 <= g.shape[1] <= 30, "grid must respect ARC size limits"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "unexpected input colors"
    assert not np.any(g == YELLOW), "yellow should only appear in the output"

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    for r, c in red_coords:
        assert 0 < int(r) < g.shape[0] - 1 and 0 < int(c) < g.shape[1] - 1, "endpoints must be interior pixels"

    path = _unique_shortest_path(g)
    out = g.copy()
    for r, c in path[1:-1]:
        if out[r, c] == BLACK:
            out[r, c] = YELLOW
    return out


def _sample_distinct(pool: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count > len(pool):
        count = len(pool)
    return random.sample(pool, count)


def _random_equal_pair() -> tuple[int, int]:
    color = random.choice(MARKER_COLORS)
    return color, color


def _random_mismatch_pair() -> tuple[int, int]:
    a, b = random.sample(MARKER_COLORS, 2)
    return a, b


def _extra_count() -> int:
    return random.randint(*GEN_EXTRA_AUTH_RANGE)


def _mismatch_count() -> int:
    return random.randint(*GEN_MISMATCH_RANGE)


def _split_candidates(candidates: list[int], forbidden_lo: int, forbidden_hi: int) -> list[int]:
    return [x for x in candidates if x < forbidden_lo or x > forbidden_hi]


def _choose_row_row_layout(h: int, w: int) -> tuple[set[int], set[int], tuple[int, int], tuple[int, int]] | None:
    row_pairs = [(r1, r2) for r1 in range(2, h - 4) for r2 in range(r1 + 3, h - 1)]
    if not row_pairs:
        return None
    connector_cols = [c for c in range(3, w - 3)]
    if not connector_cols:
        return None

    r1, r2 = random.choice(row_pairs)
    c_star = random.choice(connector_cols)
    left_cols = [c for c in range(1, c_star - 1)]
    right_cols = [c for c in range(c_star + 2, w - 1)]
    if not left_cols or not right_cols:
        return None

    if random.random() < 0.5:
        a_col = random.choice(left_cols)
        b_col = random.choice(right_cols)
    else:
        a_col = random.choice(right_cols)
        b_col = random.choice(left_cols)

    auth_rows = {r1, r2}
    auth_cols = {c_star}

    extra_col_pool = _split_candidates(list(range(1, w - 1)), min(a_col, b_col), max(a_col, b_col))
    extra_col_pool = [c for c in extra_col_pool if c != c_star]
    extra_row_pool = [r for r in range(1, h - 1) if r not in auth_rows]
    auth_cols.update(_sample_distinct(extra_col_pool, _extra_count()))
    auth_rows.update(_sample_distinct(extra_row_pool, _extra_count()))

    return auth_rows, auth_cols, (r1, a_col), (r2, b_col)


def _choose_col_col_layout(h: int, w: int) -> tuple[set[int], set[int], tuple[int, int], tuple[int, int]] | None:
    col_pairs = [(c1, c2) for c1 in range(2, w - 4) for c2 in range(c1 + 3, w - 1)]
    if not col_pairs:
        return None
    connector_rows = [r for r in range(3, h - 3)]
    if not connector_rows:
        return None

    c1, c2 = random.choice(col_pairs)
    r_star = random.choice(connector_rows)
    top_rows = [r for r in range(1, r_star - 1)]
    bottom_rows = [r for r in range(r_star + 2, h - 1)]
    if not top_rows or not bottom_rows:
        return None

    if random.random() < 0.5:
        a_row = random.choice(top_rows)
        b_row = random.choice(bottom_rows)
    else:
        a_row = random.choice(bottom_rows)
        b_row = random.choice(top_rows)

    auth_rows = {r_star}
    auth_cols = {c1, c2}

    extra_row_pool = _split_candidates(list(range(1, h - 1)), min(a_row, b_row), max(a_row, b_row))
    extra_row_pool = [r for r in extra_row_pool if r != r_star]
    extra_col_pool = [c for c in range(1, w - 1) if c not in auth_cols]
    auth_rows.update(_sample_distinct(extra_row_pool, _extra_count()))
    auth_cols.update(_sample_distinct(extra_col_pool, _extra_count()))

    return auth_rows, auth_cols, (a_row, c1), (b_row, c2)


def _choose_row_col_layout(h: int, w: int) -> tuple[set[int], set[int], tuple[int, int], tuple[int, int]] | None:
    row_choices = [r for r in range(2, h - 2)]
    col_choices = [c for c in range(2, w - 2)]
    if not row_choices or not col_choices:
        return None

    r_star = random.choice(row_choices)
    c_star = random.choice(col_choices)
    left_cols = [c for c in range(1, c_star - 1)]
    right_cols = [c for c in range(c_star + 2, w - 1)]
    top_rows = [r for r in range(1, r_star - 1)]
    bottom_rows = [r for r in range(r_star + 2, h - 1)]
    if not (left_cols or right_cols) or not (top_rows or bottom_rows):
        return None

    if left_cols and right_cols:
        a_col = random.choice(left_cols if random.random() < 0.5 else right_cols)
    else:
        a_col = random.choice(left_cols or right_cols)
    if top_rows and bottom_rows:
        b_row = random.choice(top_rows if random.random() < 0.5 else bottom_rows)
    else:
        b_row = random.choice(top_rows or bottom_rows)

    auth_rows = {r_star}
    auth_cols = {c_star}

    col_lo, col_hi = sorted((a_col, c_star))
    row_lo, row_hi = sorted((b_row, r_star))
    extra_row_pool = _split_candidates(list(range(1, h - 1)), row_lo, row_hi)
    extra_row_pool = [r for r in extra_row_pool if r != r_star]
    extra_col_pool = _split_candidates(list(range(1, w - 1)), col_lo, col_hi)
    extra_col_pool = [c for c in extra_col_pool if c != c_star]
    auth_rows.update(_sample_distinct(extra_row_pool, _extra_count()))
    auth_cols.update(_sample_distinct(extra_col_pool, _extra_count()))

    return auth_rows, auth_cols, (r_star, a_col), (b_row, c_star)


def _choose_layout(h: int, w: int, kind: str) -> tuple[set[int], set[int], tuple[int, int], tuple[int, int]] | None:
    if kind == "row_row":
        return _choose_row_row_layout(h, w)
    if kind == "col_col":
        return _choose_col_col_layout(h, w)
    if kind == "row_col":
        return _choose_row_col_layout(h, w)
    raise ValueError(f"unknown layout kind: {kind}")


def _render_grid(
    h: int,
    w: int,
    auth_rows: set[int],
    auth_cols: set[int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    protected_rows = set(auth_rows)
    protected_cols = set(auth_cols)

    for r in sorted(auth_rows):
        left, right = _random_equal_pair()
        grid[r, 0] = left
        grid[r, w - 1] = right

    for c in sorted(auth_cols):
        top, bottom = _random_equal_pair()
        grid[0, c] = top
        grid[h - 1, c] = bottom

    row_pool = [r for r in range(1, h - 1) if r not in protected_rows]
    col_pool = [c for c in range(1, w - 1) if c not in protected_cols]
    mismatch_rows = _sample_distinct(row_pool, _mismatch_count())
    mismatch_cols = _sample_distinct(col_pool, _mismatch_count())

    for r in mismatch_rows:
        left, right = _random_mismatch_pair()
        grid[r, 0] = left
        grid[r, w - 1] = right

    for c in mismatch_cols:
        top, bottom = _random_mismatch_pair()
        grid[0, c] = top
        grid[h - 1, c] = bottom

    (r1, c1), (r2, c2) = endpoints
    grid[r1, c1] = RED
    grid[r2, c2] = RED
    return grid


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    kind = random.choice(GEN_TYPES)
    layout = _choose_layout(h, w, kind)
    if layout is None:
        return None
    auth_rows, auth_cols, start, goal = layout
    grid = _render_grid(h, w, auth_rows, auth_cols, (start, goal))

    try:
        path = _unique_shortest_path(grid)
    except AssertionError:
        return None

    if len(path) < GEN_MIN_PATH_CELLS:
        return None
    if _path_turns(path) < 1:
        return None
    out = main(grid)
    if np.count_nonzero(out == YELLOW) != len(path) - 2:
        return None
    if not any(int(grid[r, c]) == BLACK for r, c in path[1:-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-corridor input")
