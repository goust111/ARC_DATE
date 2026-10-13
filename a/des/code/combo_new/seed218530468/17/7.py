from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _detect_walls(grid: np.ndarray) -> tuple[set[int], set[int]]:
    wall_rows = {r for r in range(grid.shape[0]) if np.all(grid[r, :] != BLACK)}
    wall_cols = {c for c in range(grid.shape[1]) if np.all(grid[:, c] != BLACK)}
    return wall_rows, wall_cols


def _find_sources(
    grid: np.ndarray, wall_rows: set[int], wall_cols: set[int]
) -> list[tuple[int, int, int]]:
    sources: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == BLACK:
                continue
            if r in wall_rows or c in wall_cols:
                continue
            sources.append((r, c, color))
    sources.sort()
    return sources


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_rows, wall_cols = _detect_walls(g)
    sources = _find_sources(g, wall_rows, wall_cols)

    assert len(sources) > 0, "expected at least one signal source"
    assert len({r for r, _c, _color in sources}) == len(sources), "sources must have distinct rows"
    assert len({c for _r, c, _color in sources}) == len(sources), "sources must have distinct columns"

    best_key: dict[tuple[int, int], tuple[int, int, int]] = {}
    best_color: dict[tuple[int, int], int] = {}

    for sr, sc, source_color in sources:
        for dr, dc in DIRS:
            cur_color = int(source_color)
            step = 0
            r, c = sr, sc
            while True:
                r += dr
                c += dc
                step += 1
                if not (0 <= r < h and 0 <= c < w):
                    break

                cell = int(g[r, c])
                if r in wall_rows or c in wall_cols:
                    if cell == GREY:
                        break
                    cur_color = cell
                    continue

                if cell != BLACK:
                    break

                pos = (r, c)
                key = (step, sr, sc)
                if pos not in best_key or key < best_key[pos]:
                    best_key[pos] = key
                    best_color[pos] = cur_color

    out = g.copy()
    for (r, c), color in best_color.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each source pixel into orthogonal rays that stop on gray walls and recolor when passing through filter points.

    Input structure:
    - The background is black.
    - One-cell signal sources are colored non-black pixels that are not on any complete wall row or wall column.
    - Gray full rows and full columns form walls; some wall cells are recolored into non-gray filter points.

    Rule:
    - Every source emits four straight rays along its row and column, starting with the source color.
    - Rays paint only black cells. A gray wall cell blocks the ray, while a colored filter point on a wall lets the ray pass straight through and changes the ray color to the filter color.
    - Existing sources, walls, and filter points stay unchanged.
    - If multiple rays reach the same black cell, keep the color from the nearer source; ties go to the source with the smaller row index.

    Key insight:
    - The wall geometry is encoded by complete non-black rows and columns, and filter points act as one-cell color-changing gates embedded in those walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_grid(g)


# --------------- generator ---------------

MIN_SIZE = 15
MAX_SIZE = 26
MIN_REGION = 3
GEN_RETRIES = 400
EXTRA_SOURCE_PROB = 0.65
EXTRA_FILTER_PROB = 0.55


def _sample_wall_positions(length: int, count: int, rng: random.Random) -> list[int] | None:
    need = (count + 1) * MIN_REGION + count
    if need > length:
        return None

    band_sizes = [MIN_REGION] * (count + 1)
    slack = length - need
    for _ in range(slack):
        band_sizes[rng.randrange(count + 1)] += 1

    positions: list[int] = []
    cur = band_sizes[0]
    for i in range(count):
        positions.append(cur)
        cur += 1 + band_sizes[i + 1]
    return positions


def _bands_from_walls(length: int, walls: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for wall in sorted(walls):
        bands.append((start, wall - 1))
        start = wall + 1
    bands.append((start, length - 1))
    return bands


def _choose_counts(rng: random.Random) -> tuple[int, int]:
    mode = rng.choice(("simple", "multi_h", "multi_v", "multi_both"))
    if mode == "simple":
        return 1, 1
    if mode == "multi_h":
        return rng.choice((2, 3)), rng.choice((1, 2, 3))
    if mode == "multi_v":
        return rng.choice((1, 2, 3)), rng.choice((2, 3))
    return rng.choice((2, 3)), rng.choice((2, 3))


def _relation_ok(dist_a: int, dist_b: int, relation: str) -> bool:
    if min(dist_a, dist_b) < 2:
        return False
    if relation == "tie":
        return dist_a == dist_b
    if relation == "a_near":
        return dist_a < dist_b
    return dist_b < dist_a


def _pick_conflict_pair(
    row_band_a: tuple[int, int],
    row_band_b: tuple[int, int],
    col_band_a: tuple[int, int],
    col_band_b: tuple[int, int],
    relation: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    rows_a = list(range(row_band_a[0], row_band_a[1] + 1))
    rows_b = list(range(row_band_b[0], row_band_b[1] + 1))
    cols_a = list(range(col_band_a[0], col_band_a[1] + 1))
    cols_b = list(range(col_band_b[0], col_band_b[1] + 1))
    rng.shuffle(rows_a)
    rng.shuffle(rows_b)
    rng.shuffle(cols_a)
    rng.shuffle(cols_b)

    candidates: list[tuple[int, int, int, int]] = []
    for ra in rows_a:
        for rb in rows_b:
            dist_b = abs(rb - ra)
            for ca in cols_a:
                for cb in cols_b:
                    dist_a = abs(cb - ca)
                    if _relation_ok(dist_a, dist_b, relation):
                        candidates.append((ra, ca, rb, cb))

    if not candidates:
        return None
    return rng.choice(candidates)


def _next_filter_color(rng: random.Random, current: int) -> int:
    choices = [color for color in PALETTE if color != current]
    return int(rng.choice(choices))


def _ordered_between(values: Iterable[int], a: int, b: int) -> list[int]:
    inside = [v for v in values if min(a, b) < v < max(a, b)]
    inside.sort(reverse=a > b)
    return inside


def _add_filter_chain(
    filters: dict[tuple[int, int], int],
    fixed: int,
    walls: list[int],
    start_color: int,
    horizontal: bool,
    rng: random.Random,
) -> None:
    cur = int(start_color)
    for wall in walls:
        cur = _next_filter_color(rng, cur)
        pos = (fixed, wall) if horizontal else (wall, fixed)
        filters[pos] = cur


def _pick_extra_source(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    rng: random.Random,
) -> tuple[int, int] | None:
    band_pairs = [(ri, ci) for ri in range(len(row_bands)) for ci in range(len(col_bands))]
    rng.shuffle(band_pairs)
    for ri, ci in band_pairs:
        rows = [r for r in range(row_bands[ri][0], row_bands[ri][1] + 1) if r not in used_rows]
        cols = [c for c in range(col_bands[ci][0], col_bands[ci][1] + 1) if c not in used_cols]
        if not rows or not cols:
            continue
        return int(rng.choice(rows)), int(rng.choice(cols))
    return None


def _add_extra_filters(
    filters: dict[tuple[int, int], int],
    source: tuple[int, int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    rng: random.Random,
) -> None:
    sr, sc, source_color = source
    paths: list[tuple[str, list[int]]] = []

    left_cols = [c for c in wall_cols if c < sc]
    right_cols = [c for c in wall_cols if c > sc]
    up_rows = [r for r in wall_rows if r < sr]
    down_rows = [r for r in wall_rows if r > sr]

    if left_cols:
        paths.append(("left", sorted(left_cols, reverse=True)))
    if right_cols:
        paths.append(("right", sorted(right_cols)))
    if up_rows:
        paths.append(("up", sorted(up_rows, reverse=True)))
    if down_rows:
        paths.append(("down", sorted(down_rows)))
    if not paths:
        return

    direction, walls = rng.choice(paths)
    count = rng.randint(1, len(walls))
    chosen = walls[:count]
    cur = int(source_color)
    for wall in chosen:
        cur = _next_filter_color(rng, cur)
        if direction in ("left", "right"):
            filters[(sr, wall)] = cur
        else:
            filters[(wall, sc)] = cur


def _build_input_grid(
    shape: tuple[int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    sources: list[tuple[int, int, int]],
    filters: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r in wall_rows:
        grid[r, :] = GREY
    for c in wall_cols:
        grid[:, c] = GREY
    for (r, c), color in filters.items():
        grid[r, c] = int(color)
    for r, c, color in sources:
        grid[r, c] = int(color)
    return grid


def _expected_from_plan(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    sources: list[tuple[int, int, int]],
    filters: dict[tuple[int, int], int],
) -> np.ndarray:
    h, w = grid.shape
    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)
    source_cells = {(r, c): color for r, c, color in sources}
    best_key: dict[tuple[int, int], tuple[int, int, int]] = {}
    best_color: dict[tuple[int, int], int] = {}

    for sr, sc, source_color in sources:
        for dr, dc in DIRS:
            cur = int(source_color)
            step = 0
            r, c = sr, sc
            while True:
                r += dr
                c += dc
                step += 1
                if not (0 <= r < h and 0 <= c < w):
                    break
                if r in wall_row_set or c in wall_col_set:
                    filter_color = filters.get((r, c))
                    if filter_color is None:
                        break
                    cur = int(filter_color)
                    continue
                if (r, c) in source_cells:
                    break
                key = (step, sr, sc)
                pos = (r, c)
                if pos not in best_key or key < best_key[pos]:
                    best_key[pos] = key
                    best_color[pos] = cur

    out = grid.copy()
    for (r, c), color in best_color.items():
        out[r, c] = color
    return out


def _is_valid_input(grid: np.ndarray, sources: list[tuple[int, int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len({r for r, _c, _color in sources}) != len(sources):
        return False
    if len({c for _r, c, _color in sources}) != len(sources):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    n_h, n_v = _choose_counts(rng)

    wall_rows = _sample_wall_positions(h, n_h, rng)
    wall_cols = _sample_wall_positions(w, n_v, rng)
    if wall_rows is None or wall_cols is None:
        return None

    row_bands = _bands_from_walls(h, wall_rows)
    col_bands = _bands_from_walls(w, wall_cols)
    relation = rng.choice(("tie", "a_near", "b_near"))

    a_top = bool(rng.getrandbits(1))
    a_left = bool(rng.getrandbits(1))
    a_row_idx = 0 if a_top else len(row_bands) - 1
    b_row_idx = len(row_bands) - 1 - a_row_idx
    a_col_idx = 0 if a_left else len(col_bands) - 1
    b_col_idx = len(col_bands) - 1 - a_col_idx

    pair = _pick_conflict_pair(
        row_bands[a_row_idx],
        row_bands[b_row_idx],
        col_bands[a_col_idx],
        col_bands[b_col_idx],
        relation,
        rng,
    )
    if pair is None:
        return None
    ra, ca, rb, cb = pair

    source_colors = list(rng.sample(PALETTE, 3))
    sources = [(ra, ca, source_colors[0]), (rb, cb, source_colors[1])]
    filters: dict[tuple[int, int], int] = {}

    h_crossings = _ordered_between(wall_cols, ca, cb)
    v_crossings = _ordered_between(wall_rows, rb, ra)
    _add_filter_chain(filters, ra, h_crossings, source_colors[0], horizontal=True, rng=rng)
    _add_filter_chain(filters, cb, v_crossings, source_colors[1], horizontal=False, rng=rng)

    used_rows = {ra, rb}
    used_cols = {ca, cb}
    if rng.random() < EXTRA_SOURCE_PROB:
        extra = _pick_extra_source(row_bands, col_bands, used_rows, used_cols, rng)
        if extra is not None:
            er, ec = extra
            sources.append((er, ec, source_colors[2]))
            used_rows.add(er)
            used_cols.add(ec)
            if rng.random() < EXTRA_FILTER_PROB:
                _add_extra_filters(filters, sources[-1], wall_rows, wall_cols, rng)

    grid = _build_input_grid((h, w), wall_rows, wall_cols, sources, filters)
    if not _is_valid_input(grid, sources):
        return None

    expected = _expected_from_plan(grid, wall_rows, wall_cols, sources, filters)
    if np.array_equal(grid, expected):
        return None
    if np.count_nonzero(expected != BLACK) > int(grid.size * 0.7):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
