from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))

LOGICAL_HEIGHT_RANGE = (5, 10)
LOGICAL_WIDTH_RANGE = (6, 11)
SOURCE_COUNT_OPTIONS = (3, 4, 5)
MAX_ENDPOINTS = 3

OVERLAP_PROB = 0.45
MIXED_TURN_PROB = 0.7
DISTINCT_ROW_PROB = 0.75
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250


def _downsample_even_lattice(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "expected even input dimensions"
    assert np.all(g[1::2, :] == BACKGROUND), "odd rows must be separator rows"
    assert np.all(g[:, 1::2] == BACKGROUND), "odd columns must be separator columns"
    return g[::2, ::2].copy()


def _source_columns(compact: np.ndarray) -> list[int]:
    cols = [int(c) for c in np.flatnonzero(compact[0] != BACKGROUND)]
    assert cols, "expected at least one top-row source pixel"
    assert all(cols[i + 1] == cols[i] + 1 for i in range(len(cols) - 1)), "top sources must form one run"
    return cols


def _path_cells(
    height: int,
    src_col: int,
    endpoint: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    if endpoint is None:
        end_row, end_col = height - 1, src_col
    else:
        end_row, end_col = endpoint

    cells = [(r, src_col) for r in range(end_row + 1)]
    if endpoint is not None:
        step = 1 if end_col > src_col else -1
        for c in range(src_col + step, end_col + step, step):
            cells.append((end_row, c))
    return cells


def _render_compact_paths(compact: np.ndarray) -> np.ndarray:
    g = np.asarray(compact, dtype=int)
    h, _ = g.shape
    src_cols = _source_columns(g)
    src_colors = [int(g[0, c]) for c in src_cols]
    assert len(set(src_colors)) == len(src_colors), "top-row source colors must be distinct"
    assert {int(v) for v in np.unique(g) if int(v) != BACKGROUND} == set(src_colors), "all non-bg colors must come from the top run"

    out = np.full_like(g, BACKGROUND)
    for src_col, color in zip(src_cols, src_colors):
        coords = np.argwhere(g == color)
        endpoints = [(int(r), int(c)) for r, c in coords if not (int(r) == 0 and int(c) == src_col)]
        assert len(endpoints) <= 1, "each source color may have at most one endpoint"
        endpoint = endpoints[0] if endpoints else None

        for r, c in _path_cells(h, src_col, endpoint):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the sparse lattice and draw each top source as a compressed vertical-or-L path to its same-colored endpoint.

    Input structure:
    - The input grid has black background, even height, and even width.
    - Every odd-indexed row and odd-indexed column is an all-black separator, while meaningful pixels only appear at even/even crossings.
    - After removing the separators, the compact top row contains one contiguous run of distinct source colors, and lower compact cells may contain at most one same-colored endpoint per source.

    Rule:
    - Delete every odd row and odd column to obtain the compact grid.
    - For each top-row source, search the compact grid for a lower cell of the same color.
    - If a matching endpoint exists, paint that color vertically from the source down to the endpoint row and then horizontally along that row to the endpoint column.
    - If no endpoint exists, paint the color straight down to the bottom row; later sources are painted after earlier ones from left to right and therefore overwrite overlaps.

    Key insight:
    - The separator rows and columns hide the true adjacency: only after downsampling do the source columns, endpoint offsets, and left-to-right overwrite order define the intended path geometry.
    """
    compact = _downsample_even_lattice(grid)
    return _render_compact_paths(compact)


def _expand_with_separators(compact: np.ndarray) -> np.ndarray:
    h, w = compact.shape
    sparse = np.full((2 * h, 2 * w), BACKGROUND, dtype=int)
    sparse[::2, ::2] = compact
    return sparse


def _paint_stats(
    shape: tuple[int, int],
    source_cols: list[int],
    source_colors: list[int],
    endpoints_by_color: dict[int, tuple[int, int]],
) -> dict[str, object]:
    occupied: set[tuple[int, int]] = set()
    overlaps = 0
    turn_lengths = []
    endpoint_rows = []
    turn_dirs = []

    for src_col, color in zip(source_cols, source_colors):
        endpoint = endpoints_by_color.get(color)
        if endpoint is not None:
            end_row, end_col = endpoint
            turn_lengths.append(abs(end_col - src_col))
            endpoint_rows.append(end_row)
            turn_dirs.append(-1 if end_col < src_col else 1)

        for cell in _path_cells(shape[0], src_col, endpoint):
            if cell in occupied:
                overlaps += 1
            occupied.add(cell)

    return {
        "overlaps": overlaps,
        "painted": len(occupied),
        "turn_lengths": turn_lengths,
        "endpoint_rows": endpoint_rows,
        "turn_dirs": turn_dirs,
    }


def _choose_start_col(width: int, src_count: int, rng: random.Random) -> int:
    lo = 1
    hi = width - src_count - 1
    assert lo <= hi
    candidates = list(range(lo, hi + 1))
    center = (lo + hi) / 2.0
    weights = [1.0 / (1.0 + abs(col - center)) for col in candidates]
    return int(rng.choices(candidates, weights=weights, k=1)[0])


def _side_candidates(src_col: int, width: int) -> list[str]:
    sides = []
    if src_col > 0:
        sides.append("left")
    if src_col < width - 1:
        sides.append("right")
    return sides


def _choose_side_preferences(
    source_cols: list[int],
    endpoint_indices: list[int],
    width: int,
    rng: random.Random,
) -> dict[int, str]:
    side_by_index: dict[int, str] = {}
    if len(endpoint_indices) >= 2 and rng.random() < MIXED_TURN_PROB:
        shuffled = endpoint_indices[:]
        rng.shuffle(shuffled)
        requested = ("left", "right")
        for idx, wanted in zip(shuffled, requested):
            if wanted in _side_candidates(source_cols[idx], width):
                side_by_index[idx] = wanted

    for idx in endpoint_indices:
        if idx in side_by_index:
            continue
        side_by_index[idx] = rng.choice(_side_candidates(source_cols[idx], width))
    return side_by_index


def _sample_endpoint_rows(
    height: int,
    endpoint_indices: list[int],
    rng: random.Random,
) -> dict[int, int]:
    rows = list(range(1, height))
    if len(endpoint_indices) <= len(rows) and rng.random() < DISTINCT_ROW_PROB:
        picked = rng.sample(rows, len(endpoint_indices))
    else:
        picked = [rng.choice(rows) for _ in endpoint_indices]
    return {idx: int(row) for idx, row in zip(endpoint_indices, picked)}


def _choose_overlap_pair(
    source_cols: list[int],
    endpoint_indices: list[int],
    rng: random.Random,
) -> tuple[int, int] | None:
    pairs = [(left_idx, right_idx) for right_idx in endpoint_indices for left_idx in range(right_idx)]
    return rng.choice(pairs) if pairs else None


def _sample_endpoint_positions(
    height: int,
    width: int,
    source_cols: list[int],
    source_colors: list[int],
    endpoint_indices: list[int],
    want_overlap: bool,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    endpoint_indices = sorted(endpoint_indices)

    for _ in range(PLACEMENT_ATTEMPTS):
        side_by_index = _choose_side_preferences(source_cols, endpoint_indices, width, rng)
        row_by_index = _sample_endpoint_rows(height, endpoint_indices, rng)
        overlap_pair = _choose_overlap_pair(source_cols, endpoint_indices, rng) if want_overlap else None

        if overlap_pair is not None:
            left_idx, right_idx = overlap_pair
            side_by_index[right_idx] = "left"
            max_row = row_by_index.get(left_idx, height - 1)
            if max_row < 1:
                continue
            row_by_index[right_idx] = rng.randint(1, max_row)

        used_positions: set[tuple[int, int]] = set()
        endpoints_by_color: dict[int, tuple[int, int]] = {}
        ok = True

        for idx in endpoint_indices:
            src_col = source_cols[idx]
            side = side_by_index[idx]
            row = row_by_index[idx]

            if side == "left":
                cols = list(range(0, src_col))
            else:
                cols = list(range(src_col + 1, width))

            if overlap_pair is not None and idx == overlap_pair[1]:
                cols = [c for c in cols if c <= source_cols[overlap_pair[0]]]

            if not cols:
                ok = False
                break

            if len(cols) > 2 and rng.random() < 0.6:
                farthest = cols[0] if side == "left" else cols[-1]
                near = cols[len(cols) // 2]
                cols = [farthest, farthest, near] + cols

            col = int(rng.choice(cols))
            pos = (row, col)
            if pos in used_positions:
                ok = False
                break

            used_positions.add(pos)
            endpoints_by_color[source_colors[idx]] = pos

        if not ok:
            continue

        stats = _paint_stats((height, width), source_cols, source_colors, endpoints_by_color)
        if want_overlap:
            if int(stats["overlaps"]) < 1:
                continue
        else:
            if int(stats["overlaps"]) != 0:
                continue

        if int(stats["overlaps"]) > 3:
            continue
        if int(stats["painted"]) > int(0.62 * height * width):
            continue
        if stats["turn_lengths"] and max(int(v) for v in stats["turn_lengths"]) < 2:
            continue
        if len(endpoint_indices) >= 2 and len(set(int(v) for v in stats["endpoint_rows"])) == 1:
            continue

        dirs = [int(v) for v in stats["turn_dirs"]]
        if len(endpoint_indices) >= 2 and rng.random() < 0.6 and len(set(dirs)) < 2:
            continue
        return endpoints_by_color

    return None


def _build_compact_input(rng: random.Random) -> np.ndarray | None:
    src_count = rng.choice(SOURCE_COUNT_OPTIONS)
    width_min = max(LOGICAL_WIDTH_RANGE[0], src_count + 2)
    width = rng.randint(width_min, LOGICAL_WIDTH_RANGE[1])
    height = rng.randint(*LOGICAL_HEIGHT_RANGE)

    start_col = _choose_start_col(width, src_count, rng)
    source_cols = [start_col + offset for offset in range(src_count)]
    source_colors = random.sample(COLOR_POOL, src_count)

    endpoint_count = rng.randint(1, min(MAX_ENDPOINTS, src_count - 1))
    endpoint_indices = sorted(rng.sample(range(src_count), endpoint_count))
    want_overlap = endpoint_count >= 2 and rng.random() < OVERLAP_PROB
    endpoints_by_color = _sample_endpoint_positions(
        height,
        width,
        source_cols,
        source_colors,
        endpoint_indices,
        want_overlap,
        rng,
    )
    if endpoints_by_color is None:
        return None

    compact = np.full((height, width), BACKGROUND, dtype=int)
    for col, color in zip(source_cols, source_colors):
        compact[0, col] = color
    for color, (row, col) in endpoints_by_color.items():
        compact[row, col] = color
    return compact


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    compact = _build_compact_input(rng)
    if compact is None:
        return None

    sparse = _expand_with_separators(compact)
    expected = _render_compact_paths(compact)

    try:
        actual = main(sparse)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(compact[0]) < 3:
        return None
    if len({int(v) for v in np.unique(compact) if int(v) != BACKGROUND}) != int(np.count_nonzero(compact[0])):
        return None
    return sparse


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
