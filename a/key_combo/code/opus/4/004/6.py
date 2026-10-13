from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GREY = 5
FG_COLORS = (1, 2, 3, 4)
GRID_CELLS = 6
QUAD_CELLS = 3
GEN_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _runs_between_stops(length: int, stops: list[int]) -> list[tuple[int, ...]]:
    stop_set = set(map(int, stops))
    runs: list[tuple[int, ...]] = []
    start = None
    for idx in range(int(length)):
        if idx in stop_set:
            if start is not None:
                runs.append(tuple(range(start, idx)))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append(tuple(range(start, length)))
    return runs


def _crop_sprite(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    box = _bbox(np.asarray(grid) != BG)
    assert box is not None, "expected a non-black partition sprite"
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int), int(r0), int(c0)


def _find_thick_separator(stops: list[int]) -> tuple[int, int]:
    pairs = [(int(a), int(b)) for a, b in zip(stops, stops[1:]) if int(b) == int(a) + 1]
    assert len(pairs) == 1, "expected exactly one width-2 separator"
    return pairs[0]


def _partition_runs(sprite: np.ndarray) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[int], list[int]]:
    height, width = sprite.shape
    sep_rows = [r for r in range(height) if np.all(sprite[r, :] == GREY)]
    sep_cols = [c for c in range(width) if np.all(sprite[:, c] == GREY)]

    assert len(sep_rows) == 6 and len(sep_cols) == 6, "expected a 6x6 partition grid"
    _find_thick_separator(sep_rows)
    _find_thick_separator(sep_cols)

    row_runs = _runs_between_stops(height, sep_rows)
    col_runs = _runs_between_stops(width, sep_cols)
    assert len(row_runs) == GRID_CELLS and len(col_runs) == GRID_CELLS
    assert len({len(run) for run in row_runs}) == 1
    assert len({len(run) for run in col_runs}) == 1

    for rows in row_runs:
        for cols in col_runs:
            patch = sprite[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            colors = set(map(int, np.unique(patch)))
            assert len(colors) == 1 and GREY not in colors and BG not in colors

    return row_runs, col_runs, sep_rows, sep_cols


def _downscale(sprite: np.ndarray, row_runs: list[tuple[int, ...]], col_runs: list[tuple[int, ...]]) -> np.ndarray:
    compact = np.zeros((len(row_runs), len(col_runs)), dtype=int)
    for i, rows in enumerate(row_runs):
        for j, cols in enumerate(col_runs):
            compact[i, j] = int(sprite[rows[0], cols[0]])
    return compact


def _is_horizontal_mirror(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, np.fliplr(right)))


def _connection_color(left: np.ndarray, right: np.ndarray) -> int:
    shared = set(map(int, np.unique(left))) & set(map(int, np.unique(right)))
    assert shared, "mirror quadrants must share at least one color"

    counts = Counter(int(v) for v in np.concatenate([left.ravel(), right.ravel()]) if int(v) in shared)
    best_count = max(counts.values())
    tied = {color for color, count in counts.items() if count == best_count}
    if len(tied) == 1:
        return next(iter(tied))

    # Visible tie-breaker: earliest row-major appearance in the left quadrant, then right.
    for value in np.concatenate([left.ravel(), right.ravel()]):
        color = int(value)
        if color in tied:
            return color
    raise AssertionError("unreachable tie-break")


def main(grid: np.ndarray) -> np.ndarray:
    """Downscale the partition grid, find mirrored horizontal quadrant pairs, and bridge them on the raw divider.

    Input structure:
    - A black background surrounds one grey-partitioned 6x6 array of equal color cells.
    - Thin grey separator lines split adjacent cells, while the central vertical and horizontal separators are width 2.
    - Each cell interior is monochrome and uses one foreground color from blue, red, green, or yellow.

    Rule:
    - Crop to the non-black partition sprite and detect the 6 row-runs and 6 column-runs between the grey separators.
    - Collapse the cells to a 6x6 compact color map and split it into four 3x3 quadrants.
    - For each horizontal quadrant pair (top pair and bottom pair), test whether the right quadrant is the exact left-right mirror of the left quadrant.
    - For every mirrored pair, find its shared foreground color with the unique highest total frequency, then recolor the corresponding three row-bands inside the central width-2 vertical separator with that color.

    Key insight:
    - The mirror relation is only obvious after collapsing the repeated cell blocks to one color per partition; that compact relation then tells you which raw divider bands should be opened into colored bridges.
    """
    g = np.asarray(grid, dtype=int)
    sprite, base_r, base_c = _crop_sprite(g)
    row_runs, col_runs, _sep_rows, sep_cols = _partition_runs(sprite)
    compact = _downscale(sprite, row_runs, col_runs)

    thick_c0, thick_c1 = _find_thick_separator(sep_cols)
    out = g.copy()

    for start_row in (0, QUAD_CELLS):
        left = compact[start_row : start_row + QUAD_CELLS, :QUAD_CELLS]
        right = compact[start_row : start_row + QUAD_CELLS, QUAD_CELLS:]
        if not _is_horizontal_mirror(left, right):
            continue

        color = _connection_color(left, right)
        for local_row in range(QUAD_CELLS):
            assert np.array_equal(left[local_row], right[local_row, ::-1])
            rows = row_runs[start_row + local_row]
            out[
                base_r + rows[0] : base_r + rows[-1] + 1,
                base_c + thick_c0 : base_c + thick_c1 + 1,
            ] = color

    return out


def _dominant_color(quadrant: np.ndarray) -> int:
    counts = Counter(map(int, quadrant.ravel()))
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "generator expects a unique dominant color"
    return int(best[0])


def _boundary_column_obscures_dominant(quadrant: np.ndarray) -> bool:
    dominant = _dominant_color(quadrant)
    boundary = quadrant[:, -1]
    boundary_counts = Counter(map(int, boundary))
    best_count = max(boundary_counts.values())
    leaders = [color for color, count in boundary_counts.items() if count == best_count]
    return dominant not in leaders


def _random_quadrant(rng: np.random.Generator, *, forbidden_dominant: int | None = None) -> np.ndarray:
    options = [color for color in FG_COLORS if color != forbidden_dominant]
    for _ in range(200):
        dominant = int(rng.choice(options))
        dominant_count = int(rng.choice([4, 5]))
        cells = [dominant] * dominant_count
        others = [color for color in FG_COLORS if color != dominant]
        cells.extend(int(rng.choice(others)) for _ in range(9 - dominant_count))
        rng.shuffle(cells)
        quadrant = np.array(cells, dtype=int).reshape(QUAD_CELLS, QUAD_CELLS)

        if len(set(map(int, quadrant.ravel()))) < 2:
            continue
        counts = Counter(map(int, quadrant.ravel()))
        best_count = max(counts.values())
        best = [color for color, count in counts.items() if count == best_count]
        if len(best) != 1 or int(best[0]) != dominant:
            continue
        if not _boundary_column_obscures_dominant(quadrant):
            continue
        return quadrant
    raise RuntimeError("failed to build a non-degenerate mirrored quadrant")


def _near_mirror_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        left = _random_quadrant(rng)
        right = np.fliplr(left).copy()
        row = int(rng.integers(0, QUAD_CELLS))
        col = int(rng.integers(0, QUAD_CELLS))
        current = int(right[row, col])
        choices = [color for color in FG_COLORS if color != current]
        right[row, col] = int(rng.choice(choices))

        if _is_horizontal_mirror(left, right):
            continue

        matched_rows = sum(np.array_equal(left[r], right[r, ::-1]) for r in range(QUAD_CELLS))
        if matched_rows < 1:
            continue
        return left, right
    raise RuntimeError("failed to build a near-mirror counterexample")


def _mirror_pair(
    rng: np.random.Generator, *, forbidden_pattern: np.ndarray | None = None, forbidden_dominant: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    for _ in range(200):
        left = _random_quadrant(rng, forbidden_dominant=forbidden_dominant)
        if forbidden_pattern is not None and np.array_equal(left, forbidden_pattern):
            continue
        dominant = _dominant_color(left)
        return left, np.fliplr(left), dominant
    raise RuntimeError("failed to build a mirror pair")


def _render_sprite(compact: np.ndarray, cell_size: int) -> tuple[np.ndarray, list[tuple[int, ...]], list[tuple[int, ...]], list[int], list[int]]:
    row_runs: list[tuple[int, ...]] = []
    col_runs: list[tuple[int, ...]] = []
    sep_rows: list[int] = []
    sep_cols: list[int] = []

    pos = 0
    for idx in range(GRID_CELLS):
        row_runs.append(tuple(range(pos, pos + cell_size)))
        pos += cell_size
        if idx < GRID_CELLS - 1:
            width = 2 if idx == QUAD_CELLS - 1 else 1
            sep_rows.extend(range(pos, pos + width))
            pos += width
    height = pos

    pos = 0
    for idx in range(GRID_CELLS):
        col_runs.append(tuple(range(pos, pos + cell_size)))
        pos += cell_size
        if idx < GRID_CELLS - 1:
            width = 2 if idx == QUAD_CELLS - 1 else 1
            sep_cols.extend(range(pos, pos + width))
            pos += width
    width = pos

    sprite = np.full((height, width), GREY, dtype=int)
    for i, rows in enumerate(row_runs):
        for j, cols in enumerate(col_runs):
            sprite[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = int(compact[i, j])
    return sprite, row_runs, col_runs, sep_rows, sep_cols


def _paint_expected(
    grid: np.ndarray,
    *,
    top: bool,
    bottom: bool,
    top_color: int | None,
    bottom_color: int | None,
    offset_r: int,
    offset_c: int,
    row_runs: list[tuple[int, ...]],
    sep_cols: list[int],
) -> np.ndarray:
    out = grid.copy()
    thick_c0, thick_c1 = _find_thick_separator(sep_cols)

    if top:
        assert top_color is not None
        for idx in range(QUAD_CELLS):
            rows = row_runs[idx]
            out[offset_r + rows[0] : offset_r + rows[-1] + 1, offset_c + thick_c0 : offset_c + thick_c1 + 1] = top_color

    if bottom:
        assert bottom_color is not None
        for idx in range(QUAD_CELLS, GRID_CELLS):
            rows = row_runs[idx]
            out[offset_r + rows[0] : offset_r + rows[-1] + 1, offset_c + thick_c0 : offset_c + thick_c1 + 1] = bottom_color

    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = _RNG
    cell_size = int(rng.choice([2, 3]))

    mode = int(rng.choice([0, 1, 2], p=[0.35, 0.35, 0.30]))
    top_mirror = mode != 1
    bottom_mirror = mode != 0

    top_color = None
    bottom_color = None

    if top_mirror:
        top_left, top_right, top_color = _mirror_pair(rng)
    else:
        top_left, top_right = _near_mirror_pair(rng)

    if bottom_mirror:
        bottom_left, bottom_right, bottom_color = _mirror_pair(
            rng,
            forbidden_pattern=top_left if top_mirror else None,
            forbidden_dominant=top_color if top_mirror else None,
        )
    else:
        bottom_left, bottom_right = _near_mirror_pair(rng)

    if top_mirror and bottom_mirror:
        assert not np.array_equal(top_left, bottom_left)
        assert top_color != bottom_color

    if not top_mirror:
        assert not _is_horizontal_mirror(top_left, top_right)
    if not bottom_mirror:
        assert not _is_horizontal_mirror(bottom_left, bottom_right)

    compact = np.vstack(
        [
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right]),
        ]
    ).astype(int)

    sprite, row_runs, _col_runs, _sep_rows, sep_cols = _render_sprite(compact, cell_size)

    base_size = sprite.shape[0]
    total_pad_h = int(rng.integers(2, 30 - base_size + 1))
    total_pad_w = int(rng.integers(2, 30 - base_size + 1))
    top_pad = int(rng.integers(0, total_pad_h + 1))
    left_pad = int(rng.integers(0, total_pad_w + 1))
    bottom_pad = total_pad_h - top_pad
    right_pad = total_pad_w - left_pad

    grid = np.full((base_size + top_pad + bottom_pad, base_size + left_pad + right_pad), BG, dtype=int)
    grid[top_pad : top_pad + base_size, left_pad : left_pad + base_size] = sprite

    expected = _paint_expected(
        grid,
        top=top_mirror,
        bottom=bottom_mirror,
        top_color=top_color,
        bottom_color=bottom_color,
        offset_r=top_pad,
        offset_c=left_pad,
        row_runs=row_runs,
        sep_cols=sep_cols,
    )

    meta = {
        "compact": compact,
        "top_mirror": top_mirror,
        "bottom_mirror": bottom_mirror,
        "top_color": top_color,
        "bottom_color": bottom_color,
        "cell_size": cell_size,
        "matched_rows_top": [bool(np.array_equal(top_left[r], top_right[r, ::-1])) for r in range(QUAD_CELLS)],
        "matched_rows_bottom": [bool(np.array_equal(bottom_left[r], bottom_right[r, ::-1])) for r in range(QUAD_CELLS)],
    }
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected, meta = _build_case()

        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        # Semantic checks beyond self-consistency.
        compact = np.asarray(meta["compact"], dtype=int)
        top_left = compact[:QUAD_CELLS, :QUAD_CELLS]
        top_right = compact[:QUAD_CELLS, QUAD_CELLS:]
        bottom_left = compact[QUAD_CELLS:, :QUAD_CELLS]
        bottom_right = compact[QUAD_CELLS:, QUAD_CELLS:]

        assert _is_horizontal_mirror(top_left, top_right) == bool(meta["top_mirror"])
        assert _is_horizontal_mirror(bottom_left, bottom_right) == bool(meta["bottom_mirror"])

        if bool(meta["top_mirror"]):
            assert meta["top_color"] == _connection_color(top_left, top_right)
            assert any(color != int(meta["top_color"]) for color in top_left[:, -1])
        else:
            assert any(bool(x) for x in meta["matched_rows_top"])

        if bool(meta["bottom_mirror"]):
            assert meta["bottom_color"] == _connection_color(bottom_left, bottom_right)
            assert any(color != int(meta["bottom_color"]) for color in bottom_left[:, -1])
        else:
            assert any(bool(x) for x in meta["matched_rows_bottom"])

        return grid

    raise RuntimeError("failed to generate valid input")
