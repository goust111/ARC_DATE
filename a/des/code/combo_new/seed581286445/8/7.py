from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BASELINE = 5
YELLOW = 4
MAROON = 9
PATCH_COLORS = (YELLOW, MAROON)
TOWER_COLORS = (1, 2, 3, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _parse_towers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = g.shape
    assert 4 <= rows <= 30 and 1 <= cols <= 30, "grid must fit ARC bounds"
    assert np.all(g[-1, :] == BASELINE), "expected a grey baseline on the bottom row"

    towers: list[dict[str, int]] = []
    for c in range(cols):
        marker = int(g[0, c])
        height = 0
        color = None
        r = rows - 2
        while r >= 1 and int(g[r, c]) != BACKGROUND:
            cell = int(g[r, c])
            assert cell not in {BASELINE, YELLOW, MAROON}, "tower body must use a non-marker color"
            if color is None:
                color = cell
            else:
                assert cell == color, "each tower must be monochromatic"
            height += 1
            r -= 1

        if height == 0:
            assert marker == BACKGROUND, "top markers must align with towers"
            continue

        assert marker in PATCH_COLORS, "each tower column needs a yellow or maroon marker"
        assert np.all(g[1 : r + 1, c] == BACKGROUND), "tower cells must form one contiguous stack"
        towers.append({"col": int(c), "height": int(height), "color": int(color), "marker": int(marker)})

    assert towers, "expected at least one tower"

    body = g[1:-1, :]
    occupied = int(np.count_nonzero(body != BACKGROUND))
    assert occupied == sum(tower["height"] for tower in towers), "non-background body cells must belong to towers"

    marker_mask = np.isin(g[0, :], PATCH_COLORS)
    assert int(np.count_nonzero(marker_mask)) == len(towers), "expected one top marker per tower"
    assert np.all(g[0, ~marker_mask] == BACKGROUND), "top row may only contain tower markers"

    towers_by_color: dict[int, list[dict[str, int]]] = {}
    for tower in towers:
        towers_by_color.setdefault(tower["color"], []).append(tower)
    assert all(len(group) == 2 for group in towers_by_color.values()), "each tower color must appear exactly twice"

    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Equalize each same-color tower pair by extending the shorter tower with the color named by its top marker.

    Input structure:
    - The background is black and the bottom row is a full grey baseline.
    - Above the baseline sit several one-column monochromatic towers; each tower is a contiguous vertical stack.
    - Every tower color appears exactly twice, and the top row contains one yellow or maroon marker directly above each tower column.

    Rule:
    - For each pair of towers sharing the same body color, compare their heights.
    - Leave the taller tower unchanged, and extend only the shorter tower upward until both towers have equal height.
    - Color the newly added cells with the shorter tower column's top marker color (yellow or maroon); keep all original tower cells, markers, and the baseline unchanged.

    Key insight:
    - The pairing signal comes from the repeated tower body colors, but the fill color for the added patch comes from a separate cue: the marker above the shorter tower's column.
    """
    g = np.asarray(grid, dtype=int)
    towers = _parse_towers(g)

    towers_by_color: dict[int, list[dict[str, int]]] = {}
    for tower in towers:
        towers_by_color.setdefault(tower["color"], []).append(tower)

    out = g.copy()
    baseline_row = g.shape[0] - 1
    for pair in towers_by_color.values():
        a, b = pair
        if a["height"] == b["height"]:
            continue
        short, tall = (a, b) if a["height"] < b["height"] else (b, a)
        fill_top = baseline_row - tall["height"]
        fill_bottom = baseline_row - short["height"]
        assert fill_top >= 1, "patch must remain below the marker row"
        assert np.all(out[fill_top:fill_bottom, short["col"]] == BACKGROUND), "patch area must be empty"
        out[fill_top:fill_bottom, short["col"]] = short["marker"]

    return out


def _sample_columns(cols: int, tower_count: int) -> list[int]:
    base_positions = sorted(random.sample(range(cols - (tower_count - 1)), tower_count))
    return [pos + idx for idx, pos in enumerate(base_positions)]


def _render_case(
    rows: int,
    cols: int,
    assignments: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    grid[-1, :] = BASELINE
    for col, height, color, marker in assignments:
        grid[0, col] = marker
        grid[rows - 1 - height : rows - 1, col] = color
    return grid


def _oracle_accepts(grid: np.ndarray, assignments: list[tuple[int, int, int, int]]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    expected = grid.copy()
    by_color: dict[int, list[tuple[int, int, int]]] = {}
    for col, height, color, marker in assignments:
        by_color.setdefault(color, []).append((col, height, marker))

    baseline_row = grid.shape[0] - 1
    for pair in by_color.values():
        assert len(pair) == 2
        (col0, h0, m0), (col1, h1, m1) = pair
        if h0 < h1:
            expected[baseline_row - h1 : baseline_row - h0, col0] = m0
        elif h1 < h0:
            expected[baseline_row - h0 : baseline_row - h1, col1] = m1

    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    tower_count = pair_count * 2

    min_cols = 2 * tower_count - 1
    cols = random.randint(min_cols, 24)
    rows = random.randint(8, 14)
    max_height = min(8, rows - 2)
    if max_height < 2:
        return None

    columns = _sample_columns(cols, tower_count)
    shuffled_columns = columns[:]
    random.shuffle(shuffled_columns)

    colors = random.sample(TOWER_COLORS, pair_count)
    shorter_markers = [PATCH_COLORS[idx % 2] for idx in range(pair_count)]
    random.shuffle(shorter_markers)

    assignments: list[tuple[int, int, int, int]] = []
    for pair_index, color in enumerate(colors):
        col_a = shuffled_columns[2 * pair_index]
        col_b = shuffled_columns[2 * pair_index + 1]

        short_height = random.randint(1, max_height - 1)
        tall_height = random.randint(short_height + 1, max_height)
        if random.random() < 0.5:
            heights = (short_height, tall_height)
            markers = (
                shorter_markers[pair_index],
                random.choice(PATCH_COLORS),
            )
        else:
            heights = (tall_height, short_height)
            markers = (
                random.choice(PATCH_COLORS),
                shorter_markers[pair_index],
            )

        assignments.append((col_a, heights[0], color, markers[0]))
        assignments.append((col_b, heights[1], color, markers[1]))

    grid = _render_case(rows, cols, assignments)
    if not _oracle_accepts(grid, assignments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
