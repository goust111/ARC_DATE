from __future__ import annotations

import itertools
import random

import numpy as np


FIELD_BG = 0
SEPARATOR_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    assert vals.size > 0
    return int(vals[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != FIELD_BG:
            if np.any(grid[:r] != FIELD_BG) and np.any(grid[r + 1 :] != FIELD_BG):
                candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected one full-width non-background separator row"
    return candidates[0]


def _parse_legend(grid: np.ndarray, separator_row: int, bg: int) -> dict[frozenset[int], int]:
    legend: dict[frozenset[int], int] = {}
    for r in range(separator_row):
        row = grid[r]
        c = 0
        while c < grid.shape[1]:
            if int(row[c]) == bg:
                c += 1
                continue
            c0 = c
            while c < grid.shape[1] and int(row[c]) != bg:
                c += 1
            run = [int(v) for v in row[c0:c]]
            assert len(run) == 3, "each legend entry must be a horizontal triple"
            left, right, result = run
            key = frozenset((left, right))
            assert len(key) == 2, "legend input colors must be different"
            assert key not in legend, "each unordered color pair must appear exactly once"
            legend[key] = int(result)
    return legend


def _extract_markers(
    grid: np.ndarray, separator_row: int, bg: int
) -> tuple[list[tuple[int, int, int]], dict[int, int], dict[int, int]]:
    field = grid[separator_row + 1 :]
    mask = field != bg
    coords = np.argwhere(mask)
    markers: list[tuple[int, int, int]] = []
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    for fr, c in coords:
        fr = int(fr)
        c = int(c)
        local_window = mask[max(0, fr - 1) : fr + 2, max(0, c - 1) : c + 2]
        assert int(local_window.sum()) == 1, "field markers must be isolated single pixels"

        r = separator_row + 1 + fr
        color = int(grid[r, c])
        assert r not in row_colors, "expected at most one marker per field row"
        assert c not in col_colors, "expected at most one marker per field column"
        row_colors[r] = color
        col_colors[c] = color
        markers.append((r, c, color))

    assert markers, "expected at least one field marker"
    return markers, row_colors, col_colors


def _render_field(
    shape: tuple[int, int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
    legend: dict[frozenset[int], int],
    field_row0: int,
    bg: int,
) -> np.ndarray:
    field_h, field_w = map(int, shape)
    out = np.full((field_h, field_w), int(bg), dtype=int)
    for fr in range(field_h):
        r = field_row0 + fr
        row_color = row_colors.get(r)
        for c in range(field_w):
            col_color = col_colors.get(c)
            if row_color is None and col_color is None:
                continue
            if row_color is None:
                out[fr, c] = col_color
            elif col_color is None or row_color == col_color:
                out[fr, c] = row_color
            else:
                out[fr, c] = legend[frozenset((row_color, col_color))]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each field marker into a full row and full column, then recolor mixed-color crossings by legend lookup.

    Input structure:
    - The top band contains one or more horizontal legend triples of the form
      "color A, color B, result color", separated from the field by one
      full-width non-background separator row.
    - Below the separator is a black field containing isolated colored marker
      pixels, with at most one marker in any field row or any field column.

    Rule:
    - Leave the legend band and separator row unchanged.
    - Each field marker paints its entire field row and entire field column in
      the marker's own color.
    - If a row-stroke color and a column-stroke color differ, recolor their
      intersection by the legend entry for that unordered color pair; same-color
      crossings stay that color.

    Key insight:
    - The legend is a binary operation table on unordered color pairs, so the
      final crossing color depends only on which two marker colors meet, not on
      drawing order or on whether the color came from a row or a column stroke.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row, _separator_color = _find_separator_row(g)
    assert separator_row + 1 < g.shape[0], "expected a field below the separator"
    field_bg = _mode_color(g[separator_row + 1 :])

    legend = _parse_legend(g, separator_row, field_bg)
    markers, row_colors, col_colors = _extract_markers(g, separator_row, field_bg)

    used_colors = {color for _r, _c, color in markers}
    expected_pairs = {
        frozenset(pair) for pair in itertools.combinations(sorted(used_colors), 2)
    }
    assert set(legend) == expected_pairs, "legend must define every unordered distinct marker-color pair"

    out = g.copy()
    out[separator_row + 1 :] = _render_field(
        (g.shape[0] - separator_row - 1, g.shape[1]),
        row_colors,
        col_colors,
        legend,
        separator_row + 1,
        field_bg,
    )
    return out


def _sample_spaced_indices(length: int, count: int) -> list[int]:
    assert length >= 2 * count - 1
    bases = sorted(random.sample(range(length - count + 1), count))
    return [base + i for i, base in enumerate(bases)]


def _build_legend(unique_colors: list[int]) -> tuple[list[tuple[int, int, int]], dict[frozenset[int], int]]:
    triplets: list[tuple[int, int, int]] = []
    legend_map: dict[frozenset[int], int] = {}
    for a, b in itertools.combinations(unique_colors, 2):
        result = random.choice([color for color in PALETTE if color not in (a, b)])
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        triplets.append((left, right, result))
        legend_map[frozenset((a, b))] = result
    random.shuffle(triplets)
    return triplets, legend_map


def _build_input_grid(
    field_h: int,
    field_w: int,
    triplets: list[tuple[int, int, int]],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    width = int(field_w)
    options = []
    for gap in (1, 2):
        legend_span = 3 * len(triplets) + gap * max(0, len(triplets) - 1)
        for left_pad in range(3):
            for right_pad in range(3):
                if left_pad + legend_span + right_pad <= width:
                    options.append((gap, left_pad, right_pad))
    assert options, "field width must admit at least one legend layout"
    gap, left_pad, _right_pad = random.choice(options)

    grid = np.full((field_h + 2, width), FIELD_BG, dtype=int)
    grid[1, :] = SEPARATOR_COLOR

    c = left_pad
    for i, (a, b, result) in enumerate(triplets):
        grid[0, c : c + 3] = np.array([a, b, result], dtype=int)
        c += 3
        if i + 1 < len(triplets):
            c += gap

    for fr, c, color in markers:
        grid[2 + fr, c] = int(color)
    return grid


def _semantic_output_from_markers(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    legend_map: dict[frozenset[int], int],
) -> np.ndarray:
    row_colors = {2 + fr: int(color) for fr, _c, color in markers}
    col_colors = {int(c): int(color) for _fr, c, color in markers}
    out = grid.copy()
    out[2:] = _render_field((grid.shape[0] - 2, grid.shape[1]), row_colors, col_colors, legend_map, 2, FIELD_BG)
    return out


def _naive_overlay_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    prefer_vertical: bool,
) -> np.ndarray:
    row_colors = {2 + fr: int(color) for fr, _c, color in markers}
    col_colors = {int(c): int(color) for _fr, c, color in markers}
    out = grid.copy()
    field = np.full((grid.shape[0] - 2, grid.shape[1]), FIELD_BG, dtype=int)
    for fr in range(field.shape[0]):
        r = 2 + fr
        row_color = row_colors.get(r)
        for c in range(field.shape[1]):
            col_color = col_colors.get(c)
            if row_color is None and col_color is None:
                continue
            if row_color is None:
                field[fr, c] = col_color
            elif col_color is None:
                field[fr, c] = row_color
            elif row_color == col_color:
                field[fr, c] = row_color
            else:
                field[fr, c] = col_color if prefer_vertical else row_color
    out[2:] = field
    return out


def _has_extra_same_color_crossing(markers: list[tuple[int, int, int]]) -> bool:
    counts: dict[int, int] = {}
    for _r, _c, color in markers:
        counts[color] = counts.get(color, 0) + 1
    return any(count >= 2 for count in counts.values())


def _generate_candidate() -> np.ndarray | None:
    unique_count = random.choices([2, 3, 4], weights=[3, 4, 3])[0]
    unique_colors = random.sample(PALETTE, unique_count)
    triplets, legend_map = _build_legend(unique_colors)

    duplicate_branch = random.random() < 0.65
    max_markers = min(5, unique_count + 2)
    if duplicate_branch and unique_count < max_markers:
        marker_count = random.randint(unique_count + 1, max_markers)
    else:
        marker_count = random.randint(unique_count, min(max_markers, unique_count + 1))

    min_field_side = max(2 * marker_count - 1, 8)
    legend_min_width = 3 * len(triplets) + max(0, len(triplets) - 1)
    field_w = random.randint(max(min_field_side, legend_min_width), 30)
    field_h = random.randint(min_field_side, min(18, 28))

    rows = _sample_spaced_indices(field_h, marker_count)
    cols = _sample_spaced_indices(field_w, marker_count)
    random.shuffle(rows)
    random.shuffle(cols)

    marker_colors = unique_colors[:]
    while len(marker_colors) < marker_count:
        marker_colors.append(random.choice(unique_colors))
    random.shuffle(marker_colors)

    markers = [(rows[i], cols[i], marker_colors[i]) for i in range(marker_count)]
    grid = _build_input_grid(field_h, field_w, triplets, markers)
    expected = _semantic_output_from_markers(grid, markers, legend_map)

    if np.array_equal(expected, _naive_overlay_output(grid, markers, prefer_vertical=True)):
        return None
    if np.array_equal(expected, _naive_overlay_output(grid, markers, prefer_vertical=False)):
        return None
    if duplicate_branch and not _has_extra_same_color_crossing(markers):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 2")
