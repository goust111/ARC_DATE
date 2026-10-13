import itertools
from collections import Counter

import numpy as np


BG = 0
WALL = 5
LEGEND_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_info(grid: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    row = np.asarray(grid[-1], dtype=int)
    cols = np.where((row != BG) & (row != WALL))[0]
    assert cols.size >= 3, "expected a bottom legend string"
    assert np.all(np.diff(cols) == 1), "legend must be one contiguous run"
    c0 = int(cols[0])
    legend = row[c0 : int(cols[-1]) + 1].copy()
    assert len(set(map(int, legend))) == legend.size, "legend colors must be unique"
    return legend, c0, {int(color): idx for idx, color in enumerate(legend)}


def _wall_mask(field: np.ndarray) -> np.ndarray:
    comps = _components_of_color(field, WALL)
    assert comps, "expected at least one wall"
    mask = np.zeros_like(field, dtype=bool)
    for comp in comps:
        cols = {c for _, c in comp}
        assert len(cols) == 1, "each wall must be a one-cell-thick vertical bar"
        rows = sorted(r for r, _ in comp)
        assert rows == list(range(rows[0], rows[-1] + 1)), "wall bars must be vertically contiguous"
        for r, c in comp:
            mask[r, c] = True
    return mask


def _fill_ray(
    out: np.ndarray,
    wall_mask: np.ndarray,
    legend: np.ndarray,
    anchor_idx: int,
    r: int,
    c: int,
    direction: int,
    mode: str,
) -> None:
    h, w = out.shape
    if mode == "single_copy":
        steps = range(int(legend.size))
    else:
        steps = range(w)

    for step in steps:
        cc = int(c) + int(direction) * int(step)
        if not (0 <= cc < w):
            break
        if step > 0 and mode != "no_block" and wall_mask[r, cc]:
            break

        if mode == "constant":
            color = int(legend[anchor_idx])
        elif mode == "zero_phase":
            color = int(legend[(cc - int(c)) % int(legend.size)])
        else:
            color = int(legend[(int(anchor_idx) + (cc - int(c))) % int(legend.size)])
        out[r, cc] = color


def _solve(grid: np.ndarray, mode: str = "correct") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 8
    assert np.all(g[-2, :] == BG), "expected a black separator row above the legend"

    legend, _legend_c0, color_to_idx = _legend_info(g)
    field = g[:-2, :]
    wall_mask = _wall_mask(field)

    out = g.copy()
    markers = np.argwhere((field != BG) & (field != WALL))
    assert markers.size > 0, "expected marker pixels"

    seen_rows = set()
    for r, c in markers:
        r = int(r)
        c = int(c)
        color = int(field[r, c])
        assert color in color_to_idx, "marker color must occur in the legend"
        left_wall = c > 0 and wall_mask[r, c - 1]
        right_wall = c + 1 < w and wall_mask[r, c + 1]
        assert left_wall ^ right_wall, "marker must touch exactly one wall side"
        assert r not in seen_rows, "generator guarantees at most one marker per row"
        seen_rows.add(r)
        direction = 1 if left_wall else -1
        _fill_ray(out[:-2, :], wall_mask, legend, color_to_idx[color], r, c, direction, mode)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each wall-adjacent marker away from the wall using the bottom legend as a periodic phase-anchored ray.

    Input structure:
    - The bottom row contains one contiguous horizontal legend string whose colors are all distinct.
    - The row above it is a full black separator, and the playfield above contains black background plus vertical gray wall bars.
    - Colored single-pixel markers sit immediately on the left or right side of a wall, and each marker color appears exactly once in the legend.

    Rule:
    - For each marker, detect which side of the wall it touches and extend only along that row in the direction away from the wall.
    - Fill the ray by repeating the legend periodically, aligned so the marker itself stays at the unique legend index of its color.
    - Stop at the grid edge or just before the next wall; never overwrite wall cells or the bottom legend area.

    Key insight:
    - The marker is a phase anchor rather than a paint seed: wall side chooses the ray direction, the bottom legend supplies the repeating content, and the marker color fixes the alignment inside that cycle.
    """
    return _solve(grid, mode="correct")


def _independent_expected(
    inp: np.ndarray,
    legend: np.ndarray,
    wall_cols: list[int],
    marker_specs: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    field = out[:-2, :]
    wall_mask = np.zeros_like(field, dtype=bool)
    wall_mask[:, wall_cols] = True
    color_to_idx = {int(color): idx for idx, color in enumerate(legend)}
    for row, col, direction in marker_specs:
        anchor = color_to_idx[int(inp[row, col])]
        _fill_ray(field, wall_mask, legend, anchor, int(row), int(col), int(direction), "correct")
    return out


def _shortcut_variants(inp: np.ndarray) -> list[np.ndarray]:
    return [
        _solve(inp, mode="constant"),
        _solve(inp, mode="zero_phase"),
        _solve(inp, mode="single_copy"),
        _solve(inp, mode="no_block"),
    ]


def _sample_wall_columns(rng: np.random.Generator, width: int, legend_len: int, wall_count: int) -> list[int] | None:
    candidates = list(range(3, width - 3))
    combos = []
    for cols in itertools.combinations(candidates, wall_count):
        if cols[0] < legend_len + 1:
            continue
        if any(cols[i + 1] - cols[i] < 4 for i in range(len(cols) - 1)):
            continue
        combos.append(list(cols))
    if not combos:
        return None
    return combos[int(rng.integers(len(combos)))]


def _allocate_markers(
    rng: np.random.Generator,
    field_h: int,
    legend_len: int,
    wall_cols: list[int],
) -> tuple[list[int], list[str], list[int]] | None:
    wall_count = len(wall_cols)
    max_total = min(field_h, legend_len, wall_count + 3)
    if max_total < wall_count:
        return None
    total = int(rng.integers(wall_count, max_total + 1))
    counts = [1] * wall_count
    for _ in range(total - wall_count):
        counts[int(rng.integers(wall_count))] += 1

    sides = ["L"] * wall_count
    if wall_count >= 3:
        for i in range(1, wall_count - 1):
            sides[i] = "L" if bool(rng.integers(0, 2)) else "R"
    sides[-1] = "L"

    rows = rng.choice(field_h, size=total, replace=False).tolist()
    rng.shuffle(rows)
    anchor_indices = rng.choice(legend_len, size=total, replace=False).tolist()
    if anchor_indices[0] == 0:
        swap_idx = next((i for i, idx in enumerate(anchor_indices) if idx != 0), None)
        if swap_idx is None:
            return None
        anchor_indices[0], anchor_indices[swap_idx] = anchor_indices[swap_idx], anchor_indices[0]
    return counts, sides, list(map(int, rows)), list(map(int, anchor_indices))


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    legend_len = int(rng.integers(4, 7))
    width = int(rng.integers(max(16, legend_len + 10), 23))
    wall_count = 2 if width < 19 or rng.random() < 0.55 else 3
    wall_cols = _sample_wall_columns(rng, width, legend_len, wall_count)
    if wall_cols is None:
        return None

    height = int(rng.integers(max(9, wall_count + 5), 17))
    field_h = height - 2
    allocation = _allocate_markers(rng, field_h, legend_len, wall_cols)
    if allocation is None:
        return None
    counts, sides, rows, anchor_indices = allocation

    legend = rng.choice(LEGEND_COLORS, size=legend_len, replace=False).astype(int)
    legend_c0 = int(rng.integers(0, width - legend_len + 1))

    inp = np.zeros((height, width), dtype=int)
    inp[:field_h, wall_cols] = WALL
    inp[-1, legend_c0 : legend_c0 + legend_len] = legend

    marker_specs = []
    cursor = 0
    for wall_idx, wall_col in enumerate(wall_cols):
        side = sides[wall_idx]
        marker_col = wall_col - 1 if side == "L" else wall_col + 1
        direction = -1 if side == "L" else 1
        assert 0 <= marker_col < width
        for _ in range(counts[wall_idx]):
            row = int(rows[cursor])
            color = int(legend[anchor_indices[cursor]])
            inp[row, marker_col] = color
            marker_specs.append((row, marker_col, direction))
            cursor += 1

    expected = _independent_expected(inp, legend, wall_cols, marker_specs)

    if bool(rng.integers(0, 2)):
        inp = np.fliplr(inp).copy()
        expected = np.fliplr(expected).copy()

    return inp, expected


def _is_good_sample(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    legend, _legend_c0, color_to_idx = _legend_info(inp)
    field = inp[:-2, :]
    wall_mask = _wall_mask(field)
    markers = np.argwhere((field != BG) & (field != WALL))
    rays = []
    blocked = False
    wrapped = False
    nonzero_anchor = False
    for r, c in markers:
        r = int(r)
        c = int(c)
        color = int(field[r, c])
        direction = 1 if (c > 0 and wall_mask[r, c - 1]) else -1
        length = 0
        cc = c
        while 0 <= cc < field.shape[1] and not (length > 0 and wall_mask[r, cc]):
            length += 1
            cc += direction
        if 0 <= cc < field.shape[1]:
            blocked = True
        if length > int(legend.size):
            wrapped = True
        if color_to_idx[color] != 0 and length > 1:
            nonzero_anchor = True
        rays.append(length)

    if not blocked or not wrapped or not nonzero_anchor or max(rays) < 3:
        return False

    shortcut_different = sum(not np.array_equal(expected, alt) for alt in _shortcut_variants(inp))
    return shortcut_different >= 3


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt(rng)
        if built is None:
            continue
        inp, expected = built
        if _is_good_sample(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid legend-phase extension sample")
