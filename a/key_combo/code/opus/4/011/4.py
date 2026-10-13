from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
DIVIDER = 5

MARKER_COLORS = (BLUE, RED)
MAX_GENERATION_ATTEMPTS = 400


def _find_divider_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    assert len(candidates) == 1, "expected exactly one full horizontal grey divider"
    divider = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[divider, :] = True
    assert np.array_equal(grid == DIVIDER, mask), "divider color must occur only on the divider row"
    return divider


def _scan_marker(grid: np.ndarray, divider: int, col: int) -> tuple[int, list[int]] | None:
    rows = np.flatnonzero(grid[:divider, col] != BG)
    if rows.size == 0:
        return None

    colors = {int(grid[r, col]) for r in rows.tolist()}
    assert len(colors) == 1, "marker column must be monochromatic"
    color = colors.pop()
    assert color in MARKER_COLORS, "markers above the divider must be blue or red"

    row_list = [int(r) for r in rows.tolist()]
    assert row_list[-1] - row_list[0] + 1 == len(row_list), "marker must be one contiguous vertical segment"
    assert len(row_list) in (1, 2), "marker height must be 1 or 2"
    return color, row_list


def _compress_to_bottom(values: list[int], depth: int) -> np.ndarray:
    out = np.zeros(depth, dtype=int)
    if values:
        out[depth - len(values) :] = values
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one-cell markers toward the divider, erase two-cell markers into cross-divider color injections, then bottom-pack the lower-lane pixels.

    Input structure:
    - The background is black, and exactly one full horizontal grey divider spans the grid.
    - Above the divider, some columns contain a single blue/red marker segment of height 1 or 2.
    - Below the divider, columns may contain scattered green pixels and nothing else.

    Rule:
    - Scan each column independently above the divider and read the marker height.
    - A height-1 marker grows one cell downward toward the divider unless the divider blocks that step immediately.
    - A height-2 marker is erased; then one pixel of that marker color is placed into the first black cell below the divider in the same column.
    - After all injections, compress every column below the divider by moving all non-black pixels to the bottom while preserving their top-to-bottom order.

    Key insight:
    - Marker height is the gating state: one-cell markers act locally above the divider, while two-cell markers disappear and reappear as transferred color in the lower lane before stable bottom compression.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    divider = _find_divider_row(g)
    lower = g[divider + 1 :, :]
    assert set(map(int, np.unique(lower))).issubset({BG, GREEN}), "lower half may only contain green pixels in the input"

    out = g.copy()
    found_marker = False
    for c in range(w):
        marker = _scan_marker(g, divider, c)
        if marker is None:
            continue
        found_marker = True
        color, rows = marker

        if len(rows) == 1:
            dst_r = rows[0] + 1
            if dst_r < divider:
                assert int(g[dst_r, c]) == BG, "single-marker growth target must be background"
                out[dst_r, c] = color
            continue

        for r in rows:
            out[r, c] = BG

        placed = False
        for r in range(divider + 1, h):
            if int(out[r, c]) == BG:
                out[r, c] = color
                placed = True
                break
        assert placed, "double marker needs at least one background cell below the divider"

    assert found_marker, "expected at least one marker above the divider"

    depth = h - divider - 1
    for c in range(w):
        values = [int(v) for v in out[divider + 1 :, c].tolist() if int(v) != BG]
        out[divider + 1 :, c] = _compress_to_bottom(values, depth)

    out[divider, :] = DIVIDER
    return out


def _build_expected(
    shape: tuple[int, int],
    divider: int,
    markers: list[dict[str, int]],
    greens_by_col: dict[int, list[int]],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    out[divider, :] = DIVIDER
    depth = h - divider - 1

    for marker in markers:
        col = marker["col"]
        color = marker["color"]
        bottom = marker["bottom"]
        height = marker["height"]
        if height == 1:
            out[bottom, col] = color
            if bottom + 1 < divider:
                out[bottom + 1, col] = color

    for c in range(w):
        lane = np.zeros(depth, dtype=int)
        for r in greens_by_col.get(c, []):
            lane[r - divider - 1] = GREEN

        doubles = [m for m in markers if m["col"] == c and m["height"] == 2]
        if doubles:
            assert len(doubles) == 1
            color = doubles[0]["color"]
            for idx in range(depth):
                if lane[idx] == BG:
                    lane[idx] = color
                    break
            else:
                raise AssertionError("double marker injection requires a free lower cell")

        values = [int(v) for v in lane.tolist() if int(v) != BG]
        out[divider + 1 :, c] = _compress_to_bottom(values, depth)

    return out


def _nonadjacent_columns(width: int, count: int) -> list[int]:
    base = sorted(random.sample(range(width - count + 1), count))
    return [b + i for i, b in enumerate(base)]


def _sample_marker_layout(width: int, divider: int) -> list[dict[str, int]]:
    max_nonadj = (width + 1) // 2
    total = random.randint(4, min(6, max_nonadj))
    single_count = random.randint(2, min(3, total - 2))
    double_count = total - single_count
    assert single_count >= 2 and double_count >= 2

    cols = _nonadjacent_columns(width, total)
    random.shuffle(cols)
    single_cols = cols[:single_count]
    double_cols = cols[single_count:]

    singles: list[dict[str, int]] = []
    single_colors = [BLUE, RED] + [random.choice(MARKER_COLORS) for _ in range(single_count - 2)]
    random.shuffle(single_colors)

    blocked_col = single_cols[0]
    singles.append({"col": blocked_col, "color": single_colors[0], "height": 1, "bottom": divider - 1})

    growable_bottoms = [divider - 2, divider - 3]
    grow_col = single_cols[1]
    singles.append(
        {
            "col": grow_col,
            "color": single_colors[1],
            "height": 1,
            "bottom": random.choice(growable_bottoms),
        }
    )

    for idx, col in enumerate(single_cols[2:], start=2):
        bottom = random.choice([divider - 1, divider - 2, divider - 3])
        singles.append({"col": col, "color": single_colors[idx], "height": 1, "bottom": bottom})

    doubles: list[dict[str, int]] = []
    double_colors = [BLUE, RED] + [random.choice(MARKER_COLORS) for _ in range(double_count - 2)]
    random.shuffle(double_colors)
    for idx, col in enumerate(double_cols):
        bottom = random.choice([divider - 1, divider - 2, divider - 3])
        doubles.append({"col": col, "color": double_colors[idx], "height": 2, "bottom": bottom})

    markers = singles + doubles
    markers.sort(key=lambda item: item["col"])
    return markers


def _sample_positions(
    depth: int,
    count: int,
    *,
    include_top: bool = False,
    exclude_top: bool = False,
    avoid_bottom_packed: bool = False,
) -> list[int]:
    assert 0 <= count <= min(3, depth)
    if count == 0:
        return []

    positions = set()
    if include_top:
        positions.add(0)
    pool = list(range(depth))
    if exclude_top and 0 in pool:
        pool.remove(0)
    while len(positions) < count:
        positions.add(random.choice(pool))

    if avoid_bottom_packed and sorted(positions) == list(range(depth - count, depth)):
        for candidate in range(depth):
            if candidate in positions:
                continue
            moved = sorted((positions - {max(positions)}) | {candidate})
            if moved != list(range(depth - count, depth)):
                positions = set(moved)
                break

    return sorted(positions)


def _sample_greens(
    shape: tuple[int, int],
    divider: int,
    markers: list[dict[str, int]],
) -> tuple[dict[int, list[int]], dict[str, int]]:
    h, w = shape
    depth = h - divider - 1
    marker_cols = {m["col"] for m in markers}
    double_cols = [m["col"] for m in markers if m["height"] == 2]

    available_nonmarker = [c for c in range(w) if c not in marker_cols]
    green_only_col = random.choice(available_nonmarker)
    top_occupied_double = random.choice(double_cols)
    top_free_choices = [c for c in double_cols if c != top_occupied_double]
    top_free_double = random.choice(top_free_choices)

    greens_by_col: dict[int, list[int]] = {}
    for c in range(w):
        if c == top_occupied_double:
            count = random.randint(1, min(3, depth))
            rel = _sample_positions(depth, count, include_top=True, avoid_bottom_packed=True)
        elif c == top_free_double:
            count = random.randint(1, min(3, depth))
            rel = _sample_positions(depth, count, exclude_top=True, avoid_bottom_packed=True)
        elif c == green_only_col:
            count = random.randint(2, min(3, depth))
            rel = _sample_positions(depth, count, avoid_bottom_packed=True)
        elif c in double_cols:
            count = random.randint(1, min(3, depth))
            rel = _sample_positions(depth, count)
        else:
            count = random.randint(0, min(3, depth))
            rel = _sample_positions(depth, count)
        greens_by_col[c] = [divider + 1 + idx for idx in rel]

    for col in double_cols:
        assert greens_by_col[col], "double-marker columns must contain at least one green pixel"

    return greens_by_col, {
        "green_only_col": green_only_col,
        "top_occupied_double": top_occupied_double,
        "top_free_double": top_free_double,
    }


def _render_input(
    shape: tuple[int, int],
    divider: int,
    markers: list[dict[str, int]],
    greens_by_col: dict[int, list[int]],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[divider, :] = DIVIDER

    for marker in markers:
        col = marker["col"]
        color = marker["color"]
        bottom = marker["bottom"]
        height = marker["height"]
        for offset in range(height):
            grid[bottom - offset, col] = color

    for col, rows in greens_by_col.items():
        for r in rows:
            grid[r, col] = GREEN

    return grid


def _validate_structure(
    grid: np.ndarray,
    markers: list[dict[str, int]],
    greens_by_col: dict[int, list[int]],
    specials: dict[str, int],
) -> None:
    divider = _find_divider_row(grid)
    h, w = grid.shape
    assert max(h, w) <= 30
    assert set(map(int, np.unique(grid))).issubset({BG, BLUE, RED, GREEN, DIVIDER})

    marker_cols = [m["col"] for m in markers]
    assert len(marker_cols) == len(set(marker_cols))
    assert all(b - a > 1 for a, b in zip(sorted(marker_cols), sorted(marker_cols)[1:]))

    singles = [m for m in markers if m["height"] == 1]
    doubles = [m for m in markers if m["height"] == 2]
    assert len(singles) >= 2 and len(doubles) >= 2
    assert any(m["bottom"] == divider - 1 for m in singles), "need a blocked single-marker example"
    assert any(m["bottom"] <= divider - 2 for m in singles), "need a growable single-marker example"
    assert {m["color"] for m in singles} == set(MARKER_COLORS)
    assert {m["color"] for m in doubles} == set(MARKER_COLORS)

    lower_depth = h - divider - 1
    for m in doubles:
        assert greens_by_col[m["col"]], "double-marker column must have green support below"
    assert divider + 1 in greens_by_col[specials["top_occupied_double"]]
    assert divider + 1 not in greens_by_col[specials["top_free_double"]]

    green_only = specials["green_only_col"]
    assert green_only not in marker_cols
    assert len(greens_by_col[green_only]) >= 2

    for c in range(w):
        rel = [r - divider - 1 for r in greens_by_col[c]]
        assert len(rel) <= 3
        assert len(rel) == len(set(rel))
        assert all(0 <= idx < lower_depth for idx in rel)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 17)
    w = random.randint(9, 13)
    divider = random.randint(4, h - 6)

    markers = _sample_marker_layout(w, divider)
    greens_by_col, specials = _sample_greens((h, w), divider, markers)
    grid = _render_input((h, w), divider, markers, greens_by_col)
    expected = _build_expected((h, w), divider, markers, greens_by_col)

    try:
        _validate_structure(grid, markers, greens_by_col, specials)
        actual = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    changed_cols = 0
    for c in range(w):
        if not np.array_equal(grid[:, c], expected[:, c]):
            changed_cols += 1
    if changed_cols < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
