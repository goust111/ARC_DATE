from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
INPUT_COLORS = [0, 1, 3, 4, 6, 7, 8, 9]
NON_BG_INPUT_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _separator_runs(grid: np.ndarray, sep_color: int = GRAY) -> list[tuple[int, int]]:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == sep_color))]
    assert cols, "expected full-height separator columns"

    runs: list[tuple[int, int]] = []
    start = cols[0]
    prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
        else:
            runs.append((int(start), int(prev)))
            start = col
            prev = col
    runs.append((int(start), int(prev)))
    return runs


def _panel_slices(grid: np.ndarray) -> list[slice]:
    runs = _separator_runs(grid)
    width = int(grid.shape[1])

    panels: list[slice] = []
    left = 0
    for start, end in runs:
        assert start > left, "separator run cannot touch another separator run"
        panels.append(slice(left, start))
        left = end + 1
    assert left < width, "trailing separator run is not allowed"
    panels.append(slice(left, width))

    panel_widths = {sl.stop - sl.start for sl in panels}
    assert 3 <= len(panels) <= 5, "expected 3-5 panels"
    assert len(panel_widths) == 1, "panels must have equal width"
    return panels


def _vote_panels(panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(panels, axis=0)
    count = int(stack.shape[0])
    out = np.full(stack.shape[1:], RED, dtype=int)
    for color in np.unique(stack):
        majority = np.count_nonzero(stack == int(color), axis=0) * 2 > count
        out[majority] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote across same-position pixels from multiple equal-width panels.

    The input uses full-height gray separator columns to split 3-5 panels.
    Each panel is a noisy copy of the same pattern. For every local panel
    coordinate, output the strict majority color, or red if no color wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == GRAY), "expected gray separators"

    panel_slices = _panel_slices(g)
    panels = [g[:, sl].copy() for sl in panel_slices]
    panel_shapes = {panel.shape for panel in panels}
    assert len(panel_shapes) == 1, "all panels must share one shape"
    return _vote_panels(panels)


def _shape_cells(kind: str, height: int, width: int) -> list[tuple[int, int]] | None:
    if kind == "rect":
        if height < 2 or width < 2:
            return None
        oh = random.randint(2, min(4, height))
        ow = random.randint(2, min(4, width))
        top = random.randint(0, height - oh)
        left = random.randint(0, width - ow)
        return [(top + dr, left + dc) for dr in range(oh) for dc in range(ow)]

    if kind == "hline":
        if width < 3:
            return None
        row = random.randint(0, height - 1)
        length = random.randint(3, width)
        left = random.randint(0, width - length)
        return [(row, left + dc) for dc in range(length)]

    if kind == "vline":
        if height < 3:
            return None
        col = random.randint(0, width - 1)
        length = random.randint(3, height)
        top = random.randint(0, height - length)
        return [(top + dr, col) for dr in range(length)]

    if kind == "lshape":
        if height < 2 or width < 2:
            return None
        oh = random.randint(2, min(4, height))
        ow = random.randint(2, min(4, width))
        top = random.randint(0, height - oh)
        left = random.randint(0, width - ow)
        corner = random.choice(["tl", "tr", "bl", "br"])
        cells: set[tuple[int, int]] = set()
        if corner == "tl":
            cells.update((top, left + dc) for dc in range(ow))
            cells.update((top + dr, left) for dr in range(oh))
        elif corner == "tr":
            cells.update((top, left + dc) for dc in range(ow))
            cells.update((top + dr, left + ow - 1) for dr in range(oh))
        elif corner == "bl":
            cells.update((top + oh - 1, left + dc) for dc in range(ow))
            cells.update((top + dr, left) for dr in range(oh))
        else:
            cells.update((top + oh - 1, left + dc) for dc in range(ow))
            cells.update((top + dr, left + ow - 1) for dr in range(oh))
        return sorted(cells)

    if kind == "plus":
        if height < 3 or width < 3:
            return None
        center_r = random.randint(1, height - 2)
        center_c = random.randint(1, width - 2)
        return [
            (center_r, center_c),
            (center_r - 1, center_c),
            (center_r + 1, center_c),
            (center_r, center_c - 1),
            (center_r, center_c + 1),
        ]

    return None


def _build_base_panel(height: int, width: int) -> np.ndarray:
    panel = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    palette = random.sample(NON_BG_INPUT_COLORS, random.randint(2, 4))
    target_objects = random.randint(2, 4)
    kinds = ["rect", "hline", "vline", "lshape", "plus"]

    placed = 0
    for _ in range(200):
        if placed >= target_objects:
            break
        cells = _shape_cells(random.choice(kinds), height, width)
        if not cells:
            continue
        if any(occupied[r, c] for r, c in cells):
            continue
        color = random.choice(palette)
        for r, c in cells:
            occupied[r, c] = True
            panel[r, c] = color
        placed += 1

    if placed == 0:
        row = random.randint(0, height - 1)
        left = random.randint(0, width - 2)
        color = random.choice(palette)
        panel[row, left : left + 2] = color

    assert np.any(panel != BLACK), "base panel must contain signal"
    assert np.any(panel == BLACK), "base panel should keep visible background"
    return panel


def _conflict_colors(panel_count: int, seed_color: int) -> list[int]:
    pool = [seed_color] + [color for color in INPUT_COLORS if color != seed_color]
    if panel_count == 3:
        colors = random.sample(pool, 3)
        random.shuffle(colors)
        return colors
    if panel_count == 4:
        if random.random() < 0.5:
            a = seed_color
            b = random.choice([color for color in INPUT_COLORS if color != a])
            colors = [a, a, b, b]
        else:
            colors = random.sample(pool, 4)
        random.shuffle(colors)
        return colors

    assert panel_count == 5
    if random.random() < 0.5:
        a = seed_color
        others = random.sample([color for color in INPUT_COLORS if color != a], 2)
        colors = [a, a, others[0], others[0], others[1]]
    else:
        colors = random.sample(pool, 5)
    random.shuffle(colors)
    return colors


def _assemble_input(panels: list[np.ndarray], sep_width: int) -> np.ndarray:
    height, width = panels[0].shape
    count = len(panels)
    full_width = count * width + (count - 1) * sep_width
    grid = np.full((height, full_width), BLACK, dtype=int)

    col = 0
    for index, panel in enumerate(panels):
        grid[:, col : col + width] = panel
        col += width
        if index != count - 1:
            grid[:, col : col + sep_width] = GRAY
            col += sep_width
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(3, 5)
    panel_height = random.randint(5, 10)
    panel_width = random.randint(5, 10)
    sep_width = random.randint(1, 2)

    base = _build_base_panel(panel_height, panel_width)
    panels = [base.copy() for _ in range(panel_count)]

    area = panel_height * panel_width
    total_special = random.randint(2, max(2, area // 3))
    conflict_count = total_special // 2
    majority_count = total_special - conflict_count
    if conflict_count == 0:
        conflict_count = 1
        majority_count = max(1, total_special - 1)

    choices = random.sample(range(area), majority_count + conflict_count)
    majority_positions = choices[:majority_count]
    conflict_positions = choices[majority_count:]

    for flat in majority_positions:
        r, c = divmod(flat, panel_width)
        target_color = int(base[r, c])
        majority_size = random.randint(panel_count // 2 + 1, panel_count - 1)
        majority_panels = set(random.sample(range(panel_count), majority_size))
        alternatives = [color for color in INPUT_COLORS if color != target_color]
        changed = False
        for panel_index in range(panel_count):
            if panel_index in majority_panels:
                panels[panel_index][r, c] = target_color
            else:
                new_color = random.choice(alternatives)
                panels[panel_index][r, c] = new_color
                if new_color != target_color:
                    changed = True
        if not changed:
            return None

    for flat in conflict_positions:
        r, c = divmod(flat, panel_width)
        seed_color = int(base[r, c])
        colors = _conflict_colors(panel_count, seed_color)
        for panel_index, color in enumerate(colors):
            panels[panel_index][r, c] = int(color)

    grid = _assemble_input(panels, sep_width)
    if np.any(grid == RED) or np.any(grid == GRAY):
        # Only separator columns should use gray; red is reserved for output.
        internal = np.concatenate(panels, axis=1)
        if np.any(internal == RED) or np.any(internal == GRAY):
            return None

    expected = _vote_panels(panels)
    if not np.any(expected == RED):
        return None

    disagreement_and_majority = False
    for flat in majority_positions:
        r, c = divmod(flat, panel_width)
        vals = [int(panel[r, c]) for panel in panels]
        if len(set(vals)) > 1 and int(expected[r, c]) != RED:
            disagreement_and_majority = True
            break
    if not disagreement_and_majority:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid voting-panels input")
