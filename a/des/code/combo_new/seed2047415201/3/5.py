from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = tuple(c for c in range(10) if c not in (BLACK, GREY))
MAX_GENERATION_ATTEMPTS = 400


def _separator_indices(g: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    return [int(c) for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]


def _panel_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected at least one grey separator line"
    bounds = [-1, *sorted(separators), int(length)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        start = int(left + 1)
        end = int(right)
        if start < end:
            intervals.append((start, end))

    sizes = {end - start for start, end in intervals}
    assert intervals and len(sizes) == 1, "panel intervals must be non-empty and equal"
    return intervals


def _first_seen_colors(panel: np.ndarray) -> list[int]:
    seen = {BLACK, GREY}
    colors: list[int] = []
    for value in map(int, panel.ravel()):
        if value not in seen:
            seen.add(value)
            colors.append(value)
    return colors


def _panel_pattern(panel: np.ndarray) -> tuple[frozenset[tuple[int, int, int]], dict[int, tuple[int, int]]]:
    pattern: dict[int, tuple[int, int]] = {}
    for color in _first_seen_colors(panel):
        coords = np.argwhere(panel == int(color))
        assert coords.shape == (4, 2), "each color must occupy exactly one 2x2 block"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert r1 - r0 == 1 and c1 - c0 == 1, "colored block must be 2x2"
        assert bool(np.all(panel[r0 : r0 + 2, c0 : c0 + 2] == int(color)))
        pattern[int(color)] = (r0, c0)

    assert pattern, "each panel must contain colored blocks"
    return frozenset((color, rc[0], rc[1]) for color, rc in pattern.items()), pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair panel color-block arrangements by replacing anomalous panels with the majority layout.

    Input structure:
    - Black is the background and full Grey rows and columns divide the grid into equal square panels.
    - Every panel contains the same set of non-Black, non-Grey colors.
    - Each color appears once per panel as a solid 2x2 block on the Black panel background.

    Rule:
    - Extract each panel's arrangement as the relative top-left position of every colored 2x2 block.
    - Count identical arrangements across all panels and identify the unique majority arrangement.
    - Any panel whose arrangement differs from the majority is cleared to Black inside the panel and redrawn with the majority color positions.
    - Grey separators and already-correct majority panels remain unchanged.

    Key insight:
    - The correct layout is not local to one panel; it is the statistically dominant arrangement across the whole panel grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_intervals = _panel_intervals(_separator_indices(g, axis=0), g.shape[0])
    col_intervals = _panel_intervals(_separator_indices(g, axis=1), g.shape[1])
    panel_heights = {r1 - r0 for r0, r1 in row_intervals}
    panel_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert panel_heights == panel_widths, "panels must be square"

    panel_records: list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int, int]]]] = []
    pattern_maps: dict[frozenset[tuple[int, int, int]], dict[int, tuple[int, int]]] = {}
    color_set: frozenset[int] | None = None

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            signature, mapping = _panel_pattern(panel)
            current_colors = frozenset(mapping)
            if color_set is None:
                color_set = current_colors
            else:
                assert current_colors == color_set, "all panels must use the same colors"
            panel_records.append(((r0, r1, c0, c1), signature))
            pattern_maps.setdefault(signature, mapping)

    counts = Counter(signature for _bbox, signature in panel_records)
    assert counts, "expected at least one panel"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "majority arrangement must be unique"
    majority_signature = ranked[0][0]
    assert ranked[0][1] > len(panel_records) // 2, "expected a strict majority arrangement"
    majority_map = pattern_maps[majority_signature]

    out = g.copy()
    for (r0, r1, c0, c1), signature in panel_records:
        if signature == majority_signature:
            continue
        out[r0:r1, c0:c1] = BLACK
        for color, (dr, dc) in majority_map.items():
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            out[rr : rr + 2, cc : cc + 2] = int(color)

    return out


# --------------- generator ---------------


def _total_size(panel_count: int, panel_size: int) -> int:
    return int(panel_count * panel_size + panel_count - 1)


def _place_positions(panel_size: int, count: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((panel_size, panel_size), dtype=bool)
    candidates = [(r, c) for r in range(panel_size - 1) for c in range(panel_size - 1)]
    random.shuffle(candidates)
    positions: list[tuple[int, int]] = []

    for r, c in candidates:
        if bool(occupied[r : r + 2, c : c + 2].any()):
            continue
        positions.append((int(r), int(c)))
        r0 = max(0, r - 1)
        r1 = min(panel_size, r + 3)
        c0 = max(0, c - 1)
        c1 = min(panel_size, c + 3)
        occupied[r0:r1, c0:c1] = True
        if len(positions) == count:
            return positions
    return None


def _random_pattern(panel_size: int, colors: list[int]) -> dict[int, tuple[int, int]] | None:
    positions = _place_positions(panel_size, len(colors))
    if positions is None:
        return None
    return {int(color): tuple(map(int, pos)) for color, pos in zip(colors, positions, strict=False)}


def _swap_pattern(majority: dict[int, tuple[int, int]], colors: list[int]) -> dict[int, tuple[int, int]]:
    positions = [majority[int(color)] for color in colors]
    for _ in range(40):
        shuffled = positions[:]
        random.shuffle(shuffled)
        candidate = {int(color): tuple(map(int, pos)) for color, pos in zip(colors, shuffled, strict=False)}
        if candidate != majority:
            return candidate

    reversed_positions = list(reversed(positions))
    return {int(color): tuple(map(int, pos)) for color, pos in zip(colors, reversed_positions, strict=False)}


def _different_pattern(
    panel_size: int,
    colors: list[int],
    majority: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]] | None:
    if random.random() < 0.5:
        return _swap_pattern(majority, colors)

    for _ in range(80):
        candidate = _random_pattern(panel_size, colors)
        if candidate is not None and candidate != majority:
            return candidate
    return _swap_pattern(majority, colors)


def _draw_panel(grid: np.ndarray, r0: int, c0: int, pattern: dict[int, tuple[int, int]]) -> None:
    for color, (dr, dc) in pattern.items():
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr : rr + 2, cc : cc + 2] = int(color)


def _render(
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    panel_patterns: dict[tuple[int, int], dict[int, tuple[int, int]]],
) -> np.ndarray:
    height = _total_size(panel_rows, panel_size)
    width = _total_size(panel_cols, panel_size)
    grid = np.full((height, width), BLACK, dtype=int)

    for r in range(panel_size, height, panel_size + 1):
        grid[r, :] = GREY
    for c in range(panel_size, width, panel_size + 1):
        grid[:, c] = GREY

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            _draw_panel(grid, r0, c0, panel_patterns[(pr, pc)])
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = random.choice(
        [
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (2, 4),
            (4, 2),
            (3, 4),
            (4, 3),
            (4, 4),
        ]
    )
    max_panel = min(
        9,
        (30 - (panel_rows - 1)) // panel_rows,
        (30 - (panel_cols - 1)) // panel_cols,
    )
    if max_panel < 6:
        return None

    panel_size = random.randint(6, max_panel)
    color_count = random.randint(3, 4)
    colors = random.sample(COLORS, color_count)

    majority = _random_pattern(panel_size, colors)
    if majority is None:
        return None

    panel_count = panel_rows * panel_cols
    max_bad = min(2, panel_count - panel_count // 2 - 1)
    if max_bad < 1:
        return None
    bad_count = random.randint(1, max_bad)
    bad_panels = set(random.sample([(r, c) for r in range(panel_rows) for c in range(panel_cols)], bad_count))

    panel_patterns: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    expected_patterns: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            key = (pr, pc)
            expected_patterns[key] = majority
            if key in bad_panels:
                anomaly = _different_pattern(panel_size, colors, majority)
                if anomaly is None or anomaly == majority:
                    return None
                panel_patterns[key] = anomaly
            else:
                panel_patterns[key] = majority

    inp = _render(panel_rows, panel_cols, panel_size, panel_patterns)
    expected = _render(panel_rows, panel_cols, panel_size, expected_patterns)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-panel arrangement input")
