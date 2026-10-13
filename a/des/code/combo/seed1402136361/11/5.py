from __future__ import annotations

import numpy as np


def _unique_dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _single_foreground_color(panel: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(panel) if int(color) != int(bg)]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the two panels by pixelwise XOR on foreground occupancy.

    Input structure
    - The grid background is the dominant color.
    - A single full-width separator row splits the grid into two equal panels.
    - Each panel contains background plus one foreground color.

    Rule
    - Compare corresponding cells in the top and bottom panels.
    - Keep only cells whose foreground appears in exactly one panel.
    - Preserve the contributing panel's foreground color.
    - Remove the separator row from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    assert h >= 3 and h % 2 == 1 and w >= 1, "grid must contain two equal panels and one separator row"

    bg = _unique_dominant_color(g)
    panel_h = h // 2
    top = g[:panel_h, :]
    sep_row = g[panel_h, :]
    bottom = g[panel_h + 1 :, :]
    assert top.shape == bottom.shape and panel_h > 0

    sep = int(sep_row[0])
    assert sep != bg and np.all(sep_row == sep), "middle row must be a uniform non-background separator"
    assert not np.any(top == sep) and not np.any(bottom == sep), "separator color must not appear inside panels"

    top_color = _single_foreground_color(top, bg)
    bottom_color = _single_foreground_color(bottom, bg)
    assert top_color != bottom_color, "panel foreground colors must be distinct"

    top_mask = top != bg
    bottom_mask = bottom != bg

    out = np.full(top.shape, bg, dtype=int)
    out[top_mask & ~bottom_mask] = top_color
    out[bottom_mask & ~top_mask] = bottom_color
    return out


# --------------- generator ---------------

MIN_PANEL_H = 5
MAX_PANEL_H = 11
MIN_PANEL_W = 5
MAX_PANEL_W = 13
COLOR_POOL = np.arange(10, dtype=int)
MAX_STATE_COMPONENTS = 3
COMPONENT_START_RETRIES = 24
GENERATION_ATTEMPTS = 300


def _neighbors(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _balanced_case_counts(area: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(4, area // 4, dtype=int)
    extra = area % 4
    if extra:
        for idx in rng.permutation(4)[:extra]:
            counts[int(idx)] += 1
    assert counts.min() > 0
    return counts


def _partition_target(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    min_part = 2 if total >= 2 * parts else 1
    remaining = int(total)
    sizes: list[int] = []
    for i in range(parts - 1):
        remaining_parts = parts - i - 1
        low = min_part
        high = remaining - remaining_parts * min_part
        take = int(rng.integers(low, high + 1))
        sizes.append(take)
        remaining -= take
    sizes.append(remaining)
    rng.shuffle(sizes)
    sizes.sort(reverse=True)
    return sizes


def _grow_component(available: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray | None:
    h, w = available.shape
    free = np.argwhere(available)
    if len(free) < size:
        return None

    for _ in range(COMPONENT_START_RETRIES):
        start_r, start_c = free[int(rng.integers(len(free)))]
        cells = {(int(start_r), int(start_c))}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors(h, w, int(start_r), int(start_c))
            if bool(available[nr, nc])
        }

        while len(cells) < size:
            if not frontier:
                break
            frontier_list = list(frontier)
            pick = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.remove(pick)
            if pick in cells or not bool(available[pick[0], pick[1]]):
                continue

            cells.add(pick)
            for nr, nc in _neighbors(h, w, pick[0], pick[1]):
                if bool(available[nr, nc]) and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) == size:
            coords = np.asarray(sorted(cells), dtype=int)
            assert coords.shape == (size, 2)
            return coords

    return None


def _place_state(layout: np.ndarray, state: int, target: int, rng: np.random.Generator) -> bool:
    max_parts = min(MAX_STATE_COMPONENTS, target)
    parts = int(rng.integers(1, max_parts + 1))
    sizes = _partition_target(target, parts, rng)

    for size in sizes:
        coords = _grow_component(layout < 0, size, rng)
        if coords is None:
            return False
        layout[coords[:, 0], coords[:, 1]] = int(state)
    return True


def _generate_case_layout(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    layout = np.full((h, w), -1, dtype=int)
    counts = _balanced_case_counts(h * w, rng)

    states = [1, 2, 3]
    rng.shuffle(states)
    states.sort(key=lambda state: (int(counts[state]), float(rng.random())), reverse=True)

    for state in states:
        if not _place_state(layout, state, int(counts[state]), rng):
            return None

    layout[layout < 0] = 0
    actual = np.bincount(layout.ravel(), minlength=4)
    if not np.array_equal(actual, counts):
        return None
    return layout


def _render_input(case_map: np.ndarray, bg: int, sep: int, top_color: int, bottom_color: int) -> np.ndarray:
    h, w = case_map.shape
    grid = np.full((2 * h + 1, w), int(bg), dtype=int)

    top = grid[:h, :]
    bottom = grid[h + 1 :, :]
    top[np.isin(case_map, (1, 3))] = int(top_color)
    bottom[np.isin(case_map, (2, 3))] = int(bottom_color)
    grid[h, :] = int(sep)
    return grid


def _expected_output(case_map: np.ndarray, bg: int, top_color: int, bottom_color: int) -> np.ndarray:
    out = np.full(case_map.shape, int(bg), dtype=int)
    out[case_map == 1] = int(top_color)
    out[case_map == 2] = int(bottom_color)
    return out


def _grid_is_valid(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    h, w = grid.shape
    mid = h // 2
    bg = _unique_dominant_color(grid)
    sep = int(grid[mid, 0])
    if np.count_nonzero(grid == sep) != w:
        return False

    top = grid[:mid, :]
    bottom = grid[mid + 1 :, :]
    if not bool(np.any(top == bg)) or not bool(np.any(bottom == bg)):
        return False
    if not bool(np.any(top != bg)) or not bool(np.any(bottom != bg)):
        return False
    if not bool(np.any(expected == bg)):
        return False
    if len(np.unique(expected)) != 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    panel_h = int(rng.integers(MIN_PANEL_H, MAX_PANEL_H + 1))
    panel_w = int(rng.integers(MIN_PANEL_W, MAX_PANEL_W + 1))

    colors = rng.choice(COLOR_POOL, size=4, replace=False)
    bg, sep, top_color, bottom_color = (int(color) for color in colors)

    case_map = _generate_case_layout(panel_h, panel_w, rng)
    if case_map is None:
        return None

    grid = _render_input(case_map, bg, sep, top_color, bottom_color)
    expected = _expected_output(case_map, bg, top_color, bottom_color)

    bg_count = int(np.count_nonzero(grid == bg))
    top_count = int(np.count_nonzero(grid == top_color))
    bottom_count = int(np.count_nonzero(grid == bottom_color))
    sep_count = int(np.count_nonzero(grid == sep))
    if not (bg_count > max(top_count, bottom_count, sep_count)):
        return None

    return grid if _grid_is_valid(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid xor-panels ARC input")
