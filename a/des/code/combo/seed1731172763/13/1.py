from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert cols, "expected at least one full grey separator column"

    runs: list[tuple[int, int]] = []
    start = cols[0]
    prev = cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        runs.append((int(start), int(prev + 1)))
        start = prev = c
    runs.append((int(start), int(prev + 1)))
    return runs


def _panel_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    runs = _separator_runs(grid)
    widths = {end - start for start, end in runs}
    assert len(widths) == 1, "separator bands must have consistent thickness"

    panels: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        assert cursor < start, "separator bands must lie strictly between panels"
        panels.append((int(cursor), int(start)))
        cursor = int(end)
    assert cursor < grid.shape[1], "separator bands cannot touch the right boundary"
    panels.append((int(cursor), int(grid.shape[1])))

    panel_widths = {end - start for start, end in panels}
    assert len(panel_widths) == 1, "all panels must have equal width"
    assert 3 <= len(panels) <= 5, "expected 3-5 panels"
    return panels


def _mode_in_panel_order(values: list[int]) -> int:
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        color = int(value)
        if color not in counts:
            counts[color] = 0
            order.append(color)
        counts[color] += 1

    best = max(counts.values())
    winners = [color for color in order if counts[color] == best]
    assert winners, "mode computation requires at least one value"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the panel patterns, keeping only pixels present in every panel and
    coloring each kept location by the majority panel color at that location.

    Input structure:
    - Background is black.
    - Full grey separator columns split the grid into 3-5 equal-width panels.
    - Each panel contains a noisy variant of the same underlying pattern.

    Rule:
    - Work in panel-local coordinates.
    - Keep a local position only if every panel has a non-black pixel there.
    - At each kept position, choose the most frequent color among the panels.

    Key insight:
    - The reliable signal is the intersection of foreground supports across all
      panels; pixels missing from any panel are noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    slices = _panel_slices(g)
    panels = [g[:, start:end] for start, end in slices]
    height, width = panels[0].shape

    keep = np.ones((height, width), dtype=bool)
    for panel in panels:
        keep &= panel != BG

    out = np.full((height, width), BG, dtype=int)
    for r, c in np.argwhere(keep):
        colors = [int(panel[r, c]) for panel in panels]
        out[r, c] = _mode_in_panel_order(colors)
    return out


def _neighbors8(r: int, c: int, height: int, width: int) -> set[tuple[int, int]]:
    cells = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width:
                cells.add((int(nr), int(nc)))
    return cells


def _normalize(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    normalized = {(int(r - min_r), int(c - min_c)) for r, c in cells}
    height = max(r for r, _c in normalized) + 1
    width = max(c for _r, c in normalized) + 1
    return normalized, int(height), int(width)


def _rotate90(cells: set[tuple[int, int]], height: int, width: int) -> tuple[set[tuple[int, int]], int, int]:
    rotated = {(int(c), int(height - 1 - r)) for r, c in cells}
    return _normalize(rotated)


def _shape_cells(kind: str, panel_h: int, panel_w: int) -> tuple[set[tuple[int, int]], int, int]:
    if kind == "rect":
        box_h = random.randint(1, min(4, panel_h))
        box_w = random.randint(1, min(4, panel_w))
        cells = {(r, c) for r in range(box_h) for c in range(box_w)}
        return cells, int(box_h), int(box_w)

    if kind == "hline":
        length = random.randint(2, min(6, panel_w))
        cells = {(0, c) for c in range(length)}
        return cells, 1, int(length)

    if kind == "vline":
        length = random.randint(2, min(6, panel_h))
        cells = {(r, 0) for r in range(length)}
        return cells, int(length), 1

    if kind == "plus":
        box_h = random.choice([v for v in (3, 5) if v <= panel_h])
        box_w = random.choice([v for v in (3, 5) if v <= panel_w])
        mid_r = box_h // 2
        mid_c = box_w // 2
        cells = {(mid_r, c) for c in range(box_w)} | {(r, mid_c) for r in range(box_h)}
        return cells, int(box_h), int(box_w)

    assert kind == "ell"
    box_h = random.randint(2, min(5, panel_h))
    box_w = random.randint(2, min(5, panel_w))
    cells = {(r, 0) for r in range(box_h)} | {(box_h - 1, c) for c in range(box_w)}
    norm, height, width = _normalize(cells)
    for _ in range(random.randint(0, 3)):
        norm, height, width = _rotate90(norm, height, width)
    return norm, int(height), int(width)


def _place_common_shapes(panel_h: int, panel_w: int) -> set[tuple[int, int]] | None:
    kinds = ["rect", "hline", "vline", "ell"]
    if panel_h >= 3 and panel_w >= 3:
        kinds.append("plus")

    cells: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    shape_count = random.randint(1, 3)

    for _ in range(shape_count):
        placed = False
        for _attempt in range(80):
            kind = random.choice(kinds)
            local_cells, shape_h, shape_w = _shape_cells(kind, panel_h, panel_w)
            if shape_h > panel_h or shape_w > panel_w:
                continue

            top = random.randint(0, panel_h - shape_h)
            left = random.randint(0, panel_w - shape_w)
            absolute = {(int(top + r), int(left + c)) for r, c in local_cells}
            if absolute & blocked:
                continue

            cells |= absolute
            for r, c in absolute:
                blocked |= _neighbors8(r, c, panel_h, panel_w)
            placed = True
            break
        if not placed and not cells:
            return None

    if not cells:
        return None
    density = len(cells) / float(panel_h * panel_w)
    if len(cells) < 3 or density > 0.45:
        return None
    return cells


def _majority_colors(panel_count: int, force_mixed: bool) -> list[int]:
    winner = random.choice(PALETTE)
    min_wins = panel_count // 2 + 1
    max_wins = panel_count - 1 if force_mixed else panel_count
    win_count = random.randint(min_wins, max_wins)

    order = list(range(panel_count))
    random.shuffle(order)
    colors = [winner] * panel_count
    alt_pool = [color for color in PALETTE if color != winner]
    for idx in order[win_count:]:
        colors[idx] = random.choice(alt_pool)
    return [int(color) for color in colors]


def _render_input(
    panel_h: int,
    panel_w: int,
    panel_count: int,
    sep_width: int,
    common_assignments: dict[tuple[int, int], list[int]],
    noise_assignments: dict[tuple[int, int], dict[int, int]],
) -> np.ndarray:
    panels = [np.full((panel_h, panel_w), BG, dtype=int) for _ in range(panel_count)]

    for (r, c), colors in common_assignments.items():
        for panel_idx, color in enumerate(colors):
            panels[panel_idx][r, c] = int(color)

    for (r, c), per_panel in noise_assignments.items():
        for panel_idx, color in per_panel.items():
            panels[panel_idx][r, c] = int(color)

    total_w = panel_count * panel_w + (panel_count - 1) * sep_width
    grid = np.full((panel_h, total_w), BG, dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        grid[:, cursor : cursor + panel_w] = panel
        cursor += panel_w
        if idx != panel_count - 1:
            grid[:, cursor : cursor + sep_width] = SEP
            cursor += sep_width
    return grid


def _expected_output(
    panel_h: int,
    panel_w: int,
    common_assignments: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    out = np.full((panel_h, panel_w), BG, dtype=int)
    for (r, c), colors in common_assignments.items():
        out[r, c] = _mode_in_panel_order(colors)
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(3, 5)
    panel_h = random.randint(5, 10)
    panel_w = random.randint(5, 10)
    sep_width = random.randint(1, 2)

    common_cells = _place_common_shapes(panel_h, panel_w)
    if common_cells is None:
        return None

    mixed_cell = random.choice(tuple(common_cells))
    common_assignments: dict[tuple[int, int], list[int]] = {}
    for cell in common_cells:
        common_assignments[cell] = _majority_colors(panel_count, force_mixed=(cell == mixed_cell))

    available = [(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) not in common_cells]
    if not available:
        return None
    random.shuffle(available)

    noise_slots = random.randint(1, min(len(available), max(2, (panel_h * panel_w) // 6)))
    noise_assignments: dict[tuple[int, int], dict[int, int]] = {}
    multi_panel_noise = False
    for cell in available[:noise_slots]:
        subset_size = random.randint(1, panel_count - 1)
        subset = random.sample(range(panel_count), subset_size)
        if subset_size > 1:
            multi_panel_noise = True
        noise_assignments[cell] = {int(idx): int(random.choice(PALETTE)) for idx in subset}

    if not multi_panel_noise and noise_assignments:
        cell = next(iter(noise_assignments))
        subset = random.sample(range(panel_count), min(2, panel_count - 1))
        noise_assignments[cell] = {int(idx): int(random.choice(PALETTE)) for idx in subset}

    inp = _render_input(panel_h, panel_w, panel_count, sep_width, common_assignments, noise_assignments)
    expected = _expected_output(panel_h, panel_w, common_assignments)

    if not np.array_equal(main(inp), expected):
        return None

    if not np.any(expected != BG):
        return None

    first_panel = inp[:, :panel_w]
    if np.array_equal(first_panel, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-intersection ARC input")
