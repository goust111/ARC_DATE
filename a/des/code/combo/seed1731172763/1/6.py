from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (RED, GREEN, YELLOW)
QUADRANT_TO_TURNS_CW = (0, 1, 3, 2)  # UL, UR, LL, LR

# Quadrant order is the tie-break order: upper before lower, then left before right.
WINNER_PATTERNS = {
    0: [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 1),
        (2, 0, 2, 1),
        (2, 1, 0, 2),
        (3, 1, 1, 0),
    ],
    1: [
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 2, 1, 2),
        (1, 2, 1, 0),
        (0, 3, 1, 1),
    ],
    2: [
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 2, 2),
        (0, 1, 2, 0),
        (0, 1, 2, 2),
    ],
    3: [
        (0, 0, 0, 1),
        (1, 0, 0, 2),
        (0, 1, 1, 2),
        (1, 1, 0, 2),
        (0, 0, 1, 3),
    ],
}

PANEL_ROWS_RANGE = (2, 4)
PANEL_COLS_RANGE = (2, 4)
PANEL_SIZES = (7, 8, 9, 10)
SHAPE_AREA_RANGE = (4, 8)
MAX_SHAPE_ATTEMPTS = 200
MAX_PANEL_ATTEMPTS = 200
MAX_GRID_ATTEMPTS = 300
MARKER_CLEARANCE = 1


def _separator_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    if not separators:
        return [(0, size)]

    bounds = [-1, *separators, size]
    intervals = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        end = bounds[i + 1]
        assert start < end, "separator lines must leave non-empty panels"
        intervals.append((start, end))

    lengths = {end - start for start, end in intervals}
    assert len(lengths) == 1, "all panels must be the same size"
    return intervals


def _quadrant_index(r: int, c: int, panel_size: int) -> int:
    center = (panel_size - 1) / 2.0
    lower = r > center
    right = c > center
    return (2 if lower else 0) + (1 if right else 0)


def _rotate_positions(
    coords: list[tuple[int, int]],
    panel_size: int,
    turns_cw: int,
) -> list[tuple[int, int]]:
    center = (panel_size - 1) / 2.0
    rotated: list[tuple[int, int]] = []
    for r, c in coords:
        dr = r - center
        dc = c - center
        if turns_cw % 4 == 0:
            nr, nc = center + dr, center + dc
        elif turns_cw % 4 == 1:
            nr, nc = center + dc, center - dr
        elif turns_cw % 4 == 2:
            nr, nc = center - dr, center - dc
        else:
            nr, nc = center - dc, center + dr

        rr = int(round(nr))
        cc = int(round(nc))
        assert abs(nr - rr) < 1e-9 and abs(nc - cc) < 1e-9
        assert 0 <= rr < panel_size and 0 <= cc < panel_size
        rotated.append((rr, cc))
    return rotated


def _marker_color(panel: np.ndarray) -> int:
    marker_coords = np.argwhere((panel != BLACK) & (panel != GREY) & (panel != BLUE))
    assert marker_coords.size > 0, "each panel must contain at least one marker"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in marker_coords:
        by_color.setdefault(int(panel[r, c]), []).append((int(r), int(c)))

    best_color = min(
        by_color,
        key=lambda color: (
            -len(by_color[color]),
            min(by_color[color])[0],
            min(by_color[color])[1],
        ),
    )
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each grey-separated panel, rotate the blue shape toward the marker-majority
    quadrant and recolor it with the most frequent marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    row_separators = [r for r in range(h) if np.all(g[r, :] == GREY)]
    col_separators = [c for c in range(w) if np.all(g[:, c] == GREY)]
    row_intervals = _separator_intervals(row_separators, h)
    col_intervals = _separator_intervals(col_separators, w)
    panel_heights = {r1 - r0 for r0, r1 in row_intervals}
    panel_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert panel_heights == panel_widths, "panels must be square"

    out = np.full_like(g, BLACK)
    out[row_separators, :] = GREY
    out[:, col_separators] = GREY

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            panel_size = panel.shape[0]
            assert panel.shape == (panel_size, panel_size)

            shape_coords = [tuple(map(int, rc)) for rc in np.argwhere(panel == BLUE)]
            assert shape_coords, "each panel must contain a blue shape"

            marker_coords = np.argwhere((panel != BLACK) & (panel != GREY) & (panel != BLUE))
            assert marker_coords.size > 0, "each panel must contain markers"

            quadrant_counts = [0, 0, 0, 0]
            for r, c in marker_coords:
                quadrant_counts[_quadrant_index(int(r), int(c), panel_size)] += 1

            winner = max(range(4), key=lambda idx: (quadrant_counts[idx], -idx))
            color = _marker_color(panel)
            turns_cw = QUADRANT_TO_TURNS_CW[winner]

            panel_out = np.full((panel_size, panel_size), BLACK, dtype=int)
            for rr, cc in _rotate_positions(shape_coords, panel_size, turns_cw):
                panel_out[rr, cc] = color

            out[r0:r1, c0:c1] = panel_out

    return out


def _neighbors4(r: int, c: int, panel_size: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 1 <= nr < panel_size - 1 and 1 <= nc < panel_size - 1:
            yield nr, nc


def _tight_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _is_asymmetric(cells: set[tuple[int, int]]) -> bool:
    mask = _tight_mask(cells)
    keys = {_mask_key(np.rot90(mask, k)) for k in range(4)}
    return len(keys) == 4


def _safe_marker_candidates(shape: set[tuple[int, int]], panel_size: int) -> dict[int, list[tuple[int, int]]]:
    candidates = {0: [], 1: [], 2: [], 3: []}
    center_index = panel_size // 2
    for r in range(panel_size):
        for c in range(panel_size):
            if (r, c) in shape:
                continue
            if panel_size % 2 == 1 and (r == center_index or c == center_index):
                continue

            far_enough = True
            for sr, sc in shape:
                if max(abs(r - sr), abs(c - sc)) <= MARKER_CLEARANCE:
                    far_enough = False
                    break
            if not far_enough:
                continue

            candidates[_quadrant_index(r, c, panel_size)].append((r, c))
    return candidates


def _generate_shape(panel_size: int) -> set[tuple[int, int]] | None:
    center = (panel_size - 1) / 2.0
    upper_left_seeds = [
        (r, c)
        for r in range(1, panel_size - 1)
        for c in range(1, panel_size - 1)
        if r <= center and c <= center
    ]

    for _ in range(MAX_SHAPE_ATTEMPTS):
        area = random.randint(*SHAPE_AREA_RANGE)
        seed = random.choice(upper_left_seeds)
        cells = {seed}

        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, panel_size):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            choices = list(frontier)
            weights = []
            for r, c in choices:
                ul_bias = 3.0 if r <= center and c <= center else 1.0
                weights.append(ul_bias / (1.0 + r + c))
            cells.add(random.choices(choices, weights=weights, k=1)[0])

        if len(cells) != area:
            continue
        if not _is_asymmetric(cells):
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue

        mean_r = sum(rows) / len(rows)
        mean_c = sum(cols) / len(cols)
        if mean_r >= center - 0.25 or mean_c >= center - 0.25:
            continue

        marker_slots = _safe_marker_candidates(cells, panel_size)
        if min(len(marker_slots[idx]) for idx in range(4)) < 3:
            continue
        return cells

    return None


def _balanced_labels(values: tuple[int, ...], count: int) -> list[int]:
    repeated = list(values) * ((count + len(values) - 1) // len(values))
    random.shuffle(repeated)
    return repeated[:count]


def _positive_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _positive_compositions(total - first, parts - 1):
            yield (first, *rest)


def _marker_colors(total: int, dominant_color: int) -> list[int]:
    options: list[list[int]] = []
    other_pool = [color for color in MARKER_COLORS if color != dominant_color]
    for distinct_count in range(1, min(3, total) + 1):
        if distinct_count == 1:
            options.append([dominant_color] * total)
            continue

        for counts in _positive_compositions(total, distinct_count):
            maximum = max(counts)
            if counts.count(maximum) != 1:
                continue

            for others in combinations(other_pool, distinct_count - 1):
                rem_counts = list(counts)
                rem_counts.remove(maximum)
                color_list = [dominant_color] * maximum
                for color, count in zip(others, rem_counts):
                    color_list.extend([color] * count)
                options.append(color_list)

    assert options, "expected at least one valid marker color assignment"
    chosen = random.choice(options).copy()
    random.shuffle(chosen)
    return chosen


def _build_panel(panel_size: int, winner: int, dominant_color: int) -> np.ndarray | None:
    for _ in range(MAX_PANEL_ATTEMPTS):
        shape = _generate_shape(panel_size)
        if shape is None:
            continue

        panel = np.full((panel_size, panel_size), BLACK, dtype=int)
        for r, c in shape:
            panel[r, c] = BLUE

        counts = random.choice(WINNER_PATTERNS[winner])
        total_markers = sum(counts)
        marker_positions: list[tuple[int, int]] = []
        candidates = _safe_marker_candidates(shape, panel_size)
        if any(len(candidates[idx]) < counts[idx] for idx in range(4)):
            continue

        for idx, count in enumerate(counts):
            marker_positions.extend(random.sample(candidates[idx], count))

        random.shuffle(marker_positions)
        colors = _marker_colors(total_markers, dominant_color)
        for (r, c), color in zip(marker_positions, colors):
            panel[r, c] = color
        return panel

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GRID_ATTEMPTS):
        panel_rows = random.randint(*PANEL_ROWS_RANGE)
        panel_cols = random.randint(*PANEL_COLS_RANGE)
        panel_size = random.choice(PANEL_SIZES)

        height = panel_rows * panel_size + (panel_rows - 1)
        width = panel_cols * panel_size + (panel_cols - 1)
        grid = np.full((height, width), BLACK, dtype=int)

        for br in range(1, panel_rows):
            grid[br * panel_size + (br - 1), :] = GREY
        for bc in range(1, panel_cols):
            grid[:, bc * panel_size + (bc - 1)] = GREY

        panel_count = panel_rows * panel_cols
        winners = _balanced_labels((0, 1, 2, 3), panel_count)
        dominant_colors = _balanced_labels(MARKER_COLORS, panel_count)

        ok = True
        index = 0
        for br in range(panel_rows):
            for bc in range(panel_cols):
                panel = _build_panel(panel_size, winners[index], dominant_colors[index])
                index += 1
                if panel is None:
                    ok = False
                    break

                r0 = br * (panel_size + 1)
                c0 = bc * (panel_size + 1)
                grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel
            if not ok:
                break

        if not ok:
            continue

        try:
            main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
