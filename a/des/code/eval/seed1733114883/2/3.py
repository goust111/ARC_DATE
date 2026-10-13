from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[frozenset[int], int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid must be large enough to contain a legend and paths"

    candidates: list[tuple[int, int, int, int, dict[frozenset[int], int]]] = []
    for row0 in (0, h - 2):
        active = (g[row0] != bg) & (g[row0 + 1] != bg)
        for c0, c1 in _true_runs(active):
            width = c1 - c0 + 1
            if width < 2 or width % 2 != 0:
                continue

            patch = g[row0 : row0 + 2, c0 : c1 + 1]
            if np.any(patch == bg):
                continue

            mapping: dict[frozenset[int], int] = {}
            valid = True
            for offset in range(0, width, 2):
                top_left = int(patch[0, offset])
                top_right = int(patch[0, offset + 1])
                bottom_left = int(patch[1, offset])
                bottom_right = int(patch[1, offset + 1])

                if top_left == top_right or bottom_left != bottom_right:
                    valid = False
                    break

                key = frozenset((top_left, top_right))
                if len(key) != 2:
                    valid = False
                    break
                if key in mapping and mapping[key] != bottom_left:
                    valid = False
                    break
                mapping[key] = bottom_left

            if valid:
                candidates.append((width // 2, width, row0, c0, mapping))

    assert candidates, "expected exactly one edge legend"
    candidates.sort(reverse=True)
    pair_count, width, row0, c0, mapping = candidates[0]

    best_count = sum(
        cand[0] == pair_count and cand[1] == width for cand in candidates
    )
    assert best_count == 1, "legend detection should be unambiguous"

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[row0 : row0 + 2, c0 : c0 + width] = True
    return mapping, legend_mask


def _collect_paths(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[str, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    path_colors = [
        int(color)
        for color in np.unique(g[(g != bg) & (~legend_mask)])
    ]
    assert path_colors, "expected at least one path color outside the legend"

    paths: list[tuple[str, int, int, int, int]] = []
    for color in path_colors:
        coords = np.argwhere((g == color) & (~legend_mask))
        assert coords.shape[0] >= 2, "each path color must have multiple markers"

        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})

        if len(rows) == 1:
            marker_positions = sorted(int(c) for c in coords[:, 1])
            fixed = rows[0]
            orientation = "h"
        elif len(cols) == 1:
            marker_positions = sorted(int(r) for r in coords[:, 0])
            fixed = cols[0]
            orientation = "v"
        else:
            raise AssertionError("markers of one color must be strictly horizontal or vertical")

        assert len(set(marker_positions)) == len(marker_positions)
        assert any(
            marker_positions[idx + 1] - marker_positions[idx] >= 2
            for idx in range(len(marker_positions) - 1)
        ), "each path must contain at least one hidden gap"

        paths.append((orientation, color, fixed, marker_positions[0], marker_positions[-1]))

    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each same-color marker line, fill the hidden straight path between the
    outermost markers, and replace perpendicular intersections with legend-defined
    mixed colors while keeping the 2xN edge legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    mix_lookup, legend_mask = _extract_legend(g, bg)
    paths = _collect_paths(g, bg, legend_mask)

    coverage: dict[tuple[int, int], list[int]] = {}
    for orientation, color, fixed, start, end in paths:
        if orientation == "h":
            for c in range(start, end + 1):
                if legend_mask[fixed, c]:
                    continue
                coverage.setdefault((fixed, c), []).append(color)
        else:
            for r in range(start, end + 1):
                if legend_mask[r, fixed]:
                    continue
                coverage.setdefault((r, fixed), []).append(color)

    out = g.copy()
    for (r, c), colors in coverage.items():
        unique_colors = sorted(set(colors))
        if len(unique_colors) == 1:
            out[r, c] = unique_colors[0]
            continue

        assert len(unique_colors) == 2, "only pairwise crossings are supported"
        key = frozenset(unique_colors)
        assert key in mix_lookup, "every crossing pair must appear in the legend"
        out[r, c] = mix_lookup[key]

    return out


# --------------- generator ---------------

BG = 0
GRID_MIN = 15
GRID_MAX = 22
LAYOUTS = ((1, 1), (2, 1), (1, 2), (2, 2))
MAX_ATTEMPTS = 400


def _sample_spaced_positions(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    if high < low or high - low + 1 < count:
        return None
    values = list(range(low, high + 1))
    for _ in range(120):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _marker_positions_around_blocked(low: int, high: int, blocked: list[int]) -> list[int] | None:
    blocked = sorted(blocked)
    if not blocked:
        return None

    left_choices = list(range(low, blocked[0]))
    right_choices = list(range(blocked[-1] + 1, high + 1))
    if not left_choices or not right_choices:
        return None

    selected = [random.choice(left_choices), random.choice(right_choices)]
    target_size = 2 + random.randint(0, 2)

    middle_candidates = [p for p in range(selected[0] + 1, selected[1]) if p not in blocked]
    random.shuffle(middle_candidates)
    for pos in middle_candidates:
        if len(selected) >= target_size:
            break
        if all(abs(pos - other) >= 2 for other in selected):
            selected.append(pos)

    selected.sort()
    if any(b - a < 2 for a, b in zip(selected, selected[1:])):
        return None
    return selected


def _draw_legend(
    grid: np.ndarray,
    legend_side: str,
    pairs: list[tuple[int, int, int]],
) -> np.ndarray | None:
    width = 2 * len(pairs)
    if width > grid.shape[1]:
        return None

    row0 = 0 if legend_side == "top" else grid.shape[0] - 2
    start = random.randint(0, grid.shape[1] - width)
    for idx, (color_a, color_b, mix_color) in enumerate(pairs):
        c = start + 2 * idx
        grid[row0, c] = color_a
        grid[row0, c + 1] = color_b
        grid[row0 + 1, c] = mix_color
        grid[row0 + 1, c + 1] = mix_color

    mask = np.zeros_like(grid, dtype=bool)
    mask[row0 : row0 + 2, start : start + width] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    legend_side = random.choice(("top", "bottom"))
    horizontal_count, vertical_count = random.choice(LAYOUTS)
    crossing_count = horizontal_count * vertical_count
    total_colors = horizontal_count + vertical_count + crossing_count

    colors = random.sample(range(1, 10), total_colors)
    horizontal_colors = colors[:horizontal_count]
    vertical_colors = colors[horizontal_count : horizontal_count + vertical_count]
    mix_colors = colors[horizontal_count + vertical_count :]

    usable_top = 3 if legend_side == "top" else 1
    usable_bottom = h - 2 if legend_side == "top" else h - 4
    horizontal_rows = _sample_spaced_positions(horizontal_count, usable_top + 1, usable_bottom - 1, 3)
    vertical_cols = _sample_spaced_positions(vertical_count, 3, w - 4, 3)
    if horizontal_rows is None or vertical_cols is None:
        return None

    grid = np.full((h, w), BG, dtype=int)

    pairs: list[tuple[int, int, int]] = []
    pair_to_mix: dict[frozenset[int], int] = {}
    mix_idx = 0
    for h_color in horizontal_colors:
        for v_color in vertical_colors:
            pair = (h_color, v_color, mix_colors[mix_idx])
            pairs.append(pair)
            pair_to_mix[frozenset((h_color, v_color))] = mix_colors[mix_idx]
            mix_idx += 1
    random.shuffle(pairs)

    legend_mask = _draw_legend(grid, legend_side, pairs)
    if legend_mask is None:
        return None

    horizontal_specs: list[tuple[int, int, list[int]]] = []
    vertical_specs: list[tuple[int, int, list[int]]] = []

    for color, row in zip(horizontal_colors, horizontal_rows, strict=False):
        markers = _marker_positions_around_blocked(1, w - 2, vertical_cols)
        if markers is None:
            return None
        grid[row, markers] = color
        horizontal_specs.append((color, row, markers))

    for color, col in zip(vertical_colors, vertical_cols, strict=False):
        markers = _marker_positions_around_blocked(usable_top, usable_bottom, horizontal_rows)
        if markers is None:
            return None
        grid[markers, col] = color
        vertical_specs.append((color, col, markers))

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[legend_mask], grid[legend_mask]):
        return None

    vertical_by_col = {col: color for color, col, _markers in vertical_specs}
    horizontal_by_row = {row: color for color, row, _markers in horizontal_specs}

    for color, row, markers in horizontal_specs:
        for c in range(markers[0], markers[-1] + 1):
            if c in vertical_by_col:
                expected = pair_to_mix[frozenset((color, vertical_by_col[c]))]
            else:
                expected = color
            if int(out[row, c]) != expected:
                return None

    for color, col, markers in vertical_specs:
        for r in range(markers[0], markers[-1] + 1):
            if r in horizontal_by_row:
                expected = pair_to_mix[frozenset((color, horizontal_by_row[r]))]
            else:
                expected = color
            if int(out[r, col]) != expected:
                return None

    if not any(
        int(out[row, col]) == pair_to_mix[frozenset((h_color, v_color))]
        for h_color, row, _ in horizontal_specs
        for v_color, col, _ in vertical_specs
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-path input")
