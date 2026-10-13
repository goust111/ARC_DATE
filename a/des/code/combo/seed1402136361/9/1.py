from __future__ import annotations

import random

import numpy as np


MODE_ORDER = ("horizontal", "vertical", "rot90")
MODE_RANK = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
MAX_GENERATION_ATTEMPTS = 400
PANEL_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _positive_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), int(length)]
    return [(a + 1, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1]


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = grid.shape
    candidates = []

    for color in map(int, np.unique(grid)):
        row_separators = [r for r in range(height) if np.all(grid[r, :] == color)]
        col_separators = [c for c in range(width) if np.all(grid[:, c] == color)]
        if not row_separators or not col_separators:
            continue

        row_intervals = _positive_intervals(row_separators, height)
        col_intervals = _positive_intervals(col_separators, width)
        if not row_intervals or not col_intervals:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_intervals}
        col_sizes = {c1 - c0 for c0, c1 in col_intervals}
        if len(row_sizes) != 1 or row_sizes != col_sizes:
            continue

        panel_count = len(row_intervals) * len(col_intervals)
        panel_size = next(iter(row_sizes))
        candidates.append(((panel_count, len(row_separators) + len(col_separators), panel_size), color, row_intervals, col_intervals))

    assert candidates, "failed to detect a regular square-panel layout"
    _, separator, row_intervals, col_intervals = max(candidates, key=lambda item: item[0])
    return int(separator), row_intervals, col_intervals


def _orbit(mode: str, size: int, row: int, col: int) -> tuple[tuple[int, int], ...]:
    if mode == "horizontal":
        pts = {(int(row), int(col)), (int(size - 1 - row), int(col))}
    elif mode == "vertical":
        pts = {(int(row), int(col)), (int(row), int(size - 1 - col))}
    else:
        pts = set()
        rr, cc = int(row), int(col)
        for _ in range(4):
            pts.add((rr, cc))
            rr, cc = cc, size - 1 - rr
    return tuple(sorted(pts))


def _complete_panel(panel: np.ndarray, bg: int, mode: str) -> tuple[np.ndarray, int] | None:
    size = int(panel.shape[0])
    out = panel.copy()

    for row, col in np.argwhere(panel != bg):
        color = int(panel[row, col])
        for rr, cc in _orbit(mode, size, int(row), int(col)):
            existing = int(panel[rr, cc])
            if existing != bg and existing != color:
                return None
            out[rr, cc] = color

    additions = int(np.count_nonzero((panel == bg) & (out != bg)))
    return out, additions


def _mode_results(panel: np.ndarray, bg: int) -> list[tuple[int, int, str, np.ndarray]]:
    results = []
    for mode in MODE_ORDER:
        completed = _complete_panel(panel, bg, mode)
        if completed is None:
            continue
        solved, additions = completed
        results.append((int(additions), MODE_RANK[mode], mode, solved))
    results.sort(key=lambda item: (item[0], item[1]))
    return results


def _solve_panel(panel: np.ndarray, bg: int) -> tuple[np.ndarray, str, int, bool]:
    results = _mode_results(panel, bg)
    assert results, "panel has no consistent symmetry completion"
    best_additions = results[0][0]
    winners = [item for item in results if item[0] == best_additions]
    best = results[0]
    return best[3], best[2], best[0], len(winners) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each square panel by choosing the symmetry closure that adds the fewest pixels.

    Input structure:
    - A separator color partitions the canvas into equal square panels.
    - Each panel is mostly background with a few colored seed pixels.

    Rule:
    - For each panel, evaluate horizontal reflection, vertical reflection, and
      90-degree rotational symmetry.
    - Keep the candidate that requires the fewest newly painted pixels and fill
      in the corresponding mirror/rotation partners.
    - Leave the separator lattice unchanged.
    """
    grid = np.asarray(grid, dtype=int)
    separator, row_intervals, col_intervals = _find_layout(grid)
    bg = _get_bg(grid[grid != separator])

    out = grid.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = grid[r0:r1, c0:c1]
            solved, _mode, _additions, _unique = _solve_panel(panel, bg)
            out[r0:r1, c0:c1] = solved
    return out


def _panel_orbits(size: int, mode: str) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    orbits = []
    for row in range(size):
        for col in range(size):
            orbit = _orbit(mode, size, row, col)
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(orbit)
    return orbits


def _pick_distinct_mirror_classes(
    orbits: list[tuple[tuple[int, int], ...]],
    count: int,
    size: int,
    axis: int,
) -> list[tuple[tuple[int, int], ...]] | None:
    shuffled = list(orbits)
    random.shuffle(shuffled)
    chosen = []
    used_classes = set()
    for orbit in shuffled:
        coord = orbit[0][axis]
        mirror_class = min(coord, size - 1 - coord)
        if mirror_class in used_classes:
            continue
        used_classes.add(mirror_class)
        chosen.append(orbit)
        if len(chosen) == count:
            return chosen
    return None


def _target_is_unique(panel: np.ndarray, bg: int, target: str) -> bool:
    if int(np.count_nonzero(panel != bg)) < 2:
        return False
    results = _mode_results(panel, bg)
    if not results:
        return False
    best_additions = results[0][0]
    winners = [mode for additions, _rank, mode, _solved in results if additions == best_additions]
    return best_additions > 0 and winners == [target]


def _make_horizontal_panel(size: int, bg: int, colors: list[int]) -> np.ndarray | None:
    orbits = [orbit for orbit in _panel_orbits(size, "horizontal") if len(orbit) == 2]
    if size % 2 == 1:
        center = size // 2
        orbits = [orbit for orbit in orbits if orbit[0][1] != center]
    if len(orbits) < 2:
        return None

    for _ in range(PANEL_ATTEMPTS):
        total = 2 if len(orbits) == 2 else random.randint(2, min(3, len(orbits)))
        chosen = _pick_distinct_mirror_classes(orbits, total, size, axis=1)
        if chosen is None:
            continue

        partial_indices = {random.randrange(total)}
        if total > 2 and random.random() < 0.5:
            partial_indices.add(random.randrange(total))

        panel = np.full((size, size), bg, dtype=int)
        palette = random.sample(colors, total)
        for idx, (orbit, color) in enumerate(zip(chosen, palette)):
            cells = list(orbit)
            if idx in partial_indices:
                cells = [random.choice(cells)]
            for row, col in cells:
                panel[row, col] = color

        if _target_is_unique(panel, bg, "horizontal"):
            return panel
    return None


def _make_vertical_panel(size: int, bg: int, colors: list[int]) -> np.ndarray | None:
    orbits = [orbit for orbit in _panel_orbits(size, "vertical") if len(orbit) == 2]
    if size % 2 == 1:
        center = size // 2
        orbits = [orbit for orbit in orbits if orbit[0][0] != center]
    if len(orbits) < 2:
        return None

    for _ in range(PANEL_ATTEMPTS):
        total = 2 if len(orbits) == 2 else random.randint(2, min(3, len(orbits)))
        chosen = _pick_distinct_mirror_classes(orbits, total, size, axis=0)
        if chosen is None:
            continue

        partial_indices = {random.randrange(total)}
        if total > 2 and random.random() < 0.5:
            partial_indices.add(random.randrange(total))

        panel = np.full((size, size), bg, dtype=int)
        palette = random.sample(colors, total)
        for idx, (orbit, color) in enumerate(zip(chosen, palette)):
            cells = list(orbit)
            if idx in partial_indices:
                cells = [random.choice(cells)]
            for row, col in cells:
                panel[row, col] = color

        if _target_is_unique(panel, bg, "vertical"):
            return panel
    return None


def _make_rot90_panel(size: int, bg: int, colors: list[int]) -> np.ndarray | None:
    orbits = [orbit for orbit in _panel_orbits(size, "rot90") if len(orbit) == 4]
    if not orbits:
        return None

    for _ in range(PANEL_ATTEMPTS):
        total = 1 if len(orbits) == 1 else random.randint(1, min(2, len(orbits)))
        chosen = random.sample(orbits, total)
        panel = np.full((size, size), bg, dtype=int)
        palette = random.sample(colors, total)
        made_partial = False

        for idx, (orbit, color) in enumerate(zip(chosen, palette)):
            if total == 1 and idx == 0:
                visible = random.choice((2, 3, 3))
            elif idx == 0:
                visible = random.choice((2, 3, 3))
            else:
                visible = random.choice((2, 3, 4))
            made_partial = made_partial or visible < 4
            for row, col in random.sample(list(orbit), visible):
                panel[row, col] = color

        if made_partial and _target_is_unique(panel, bg, "rot90"):
            return panel
    return None


def _make_panel(size: int, bg: int, mode: str, colors: list[int]) -> np.ndarray | None:
    if mode == "horizontal":
        return _make_horizontal_panel(size, bg, colors)
    if mode == "vertical":
        return _make_vertical_panel(size, bg, colors)
    return _make_rot90_panel(size, bg, colors)


def _layout_positions(count: int, panel_size: int, bordered: bool) -> tuple[list[int], list[int], int]:
    starts = []
    separators = [0] if bordered else []
    pos = 1 if bordered else 0

    for idx in range(count):
        starts.append(pos)
        pos += panel_size
        if idx != count - 1 or bordered:
            separators.append(pos)
            pos += 1
    return starts, separators, pos


def _panel_modes(count: int) -> list[str]:
    modes = list(MODE_ORDER)
    while len(modes) < count:
        modes.append(random.choice(MODE_ORDER))
    random.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.randint(2, 3)
    panel_cols = random.randint(2, 3)
    panel_size = random.randint(5, 8)
    row_border = random.random() < 0.5
    col_border = random.random() < 0.5

    bg, separator = random.sample(range(10), 2)
    colors = [color for color in range(10) if color not in {bg, separator}]

    row_starts, row_separators, height = _layout_positions(panel_rows, panel_size, row_border)
    col_starts, col_separators, width = _layout_positions(panel_cols, panel_size, col_border)

    grid = np.full((height, width), bg, dtype=int)
    if row_separators:
        grid[row_separators, :] = separator
    if col_separators:
        grid[:, col_separators] = separator
    expected = grid.copy()

    modes = _panel_modes(panel_rows * panel_cols)
    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            mode = modes[idx]
            idx += 1
            panel = _make_panel(panel_size, bg, mode, colors)
            if panel is None:
                return None
            solved, best_mode, additions, unique = _solve_panel(panel, bg)
            if not unique or best_mode != mode or additions <= 0:
                return None
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel
            expected[r0 : r0 + panel_size, c0 : c0 + panel_size] = solved

    if _get_bg(grid) != bg:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
