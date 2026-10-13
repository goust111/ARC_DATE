import random
from collections import deque
from functools import lru_cache
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)

PANEL_COUNT_OPTIONS = (2, 3)
PANEL_HEIGHT_RANGE = (7, 11)
PANEL_WIDTH = 9
SEPARATOR_WIDTH = 1
MAX_COMPONENTS = 5
MAX_ATTEMPTS = 400

SLOT_WIDTH = 2
SLOT_HEIGHT = 2
SLOT_COLS = (0, 3, 6)


def _count_components4(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            total += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return total


def _separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert sep_cols, "expected at least one full-height grey separator"
    runs = []
    start = sep_cols[0]
    prev = sep_cols[0]
    for col in sep_cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev + 1))
        start = prev = col
    runs.append((start, prev + 1))
    assert runs[0][0] > 0 and runs[-1][1] < w, "panels should appear on both sides of separators"
    return runs


def _panel_bounds(width: int, sep_runs: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    panels = []
    cursor = 0
    sep_widths = []
    for start, end in sep_runs:
        assert start > cursor, "separator cannot consume a whole panel"
        panels.append((cursor, start))
        sep_widths.append(end - start)
        cursor = end
    assert cursor < width, "missing trailing panel"
    panels.append((cursor, width))
    panel_widths = [end - start for start, end in panels]
    assert len(set(panel_widths)) == 1, "all panels must have the same width"
    assert len(set(sep_widths)) == 1, "separator widths should be uniform"
    return panels, sep_widths[0]


def _draw_count_dots(panel: np.ndarray, count: int) -> np.ndarray:
    out = panel.copy()
    assert out.shape[1] >= 2 * count - 1, "panel too narrow for requested count dots"
    for i in range(count):
        out[-1, 2 * i] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort equal-width panels by the number of 4-connected components in each panel's pattern,
    then draw that count as red dots across the last row of every panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == GREY), "expected grey separator columns"

    h, w = g.shape
    sep_runs = _separator_runs(g)
    panel_bounds, sep_w = _panel_bounds(w, sep_runs)

    panels: list[np.ndarray] = []
    counts: list[int] = []
    for start, end in panel_bounds:
        panel = g[:, start:end].copy()
        count = _count_components4(panel != BLACK)
        assert 1 <= count <= MAX_COMPONENTS, "panel component count out of range"
        panels.append(panel)
        counts.append(count)

    order = sorted(range(len(panels)), key=lambda idx: counts[idx])
    panel_w = panel_bounds[0][1] - panel_bounds[0][0]
    out = np.full((h, w), BLACK, dtype=int)

    cursor = 0
    for rank, idx in enumerate(order):
        decorated = _draw_count_dots(panels[idx], counts[idx])
        out[:, cursor : cursor + panel_w] = decorated
        cursor += panel_w
        if rank < len(order) - 1:
            out[:, cursor : cursor + sep_w] = GREY
            cursor += sep_w

    assert cursor == w
    return out


@lru_cache(maxsize=None)
def _connected_slot_masks(height: int, width: int) -> tuple[np.ndarray, ...]:
    masks = []
    for area in range(1, height * width + 1):
        for cells in combinations(range(height * width), area):
            mask = np.zeros((height, width), dtype=bool)
            for cell in cells:
                mask[cell // width, cell % width] = True
            if _count_components4(mask) == 1:
                masks.append(mask)
    return tuple(masks)


def _choose_counts(panel_count: int) -> list[int]:
    if random.random() < 0.5:
        shared = random.randint(1, MAX_COMPONENTS)
        counts = [shared, shared]
        while len(counts) < panel_count:
            counts.append(random.randint(1, MAX_COMPONENTS))
        random.shuffle(counts)
    else:
        counts = random.sample(range(1, MAX_COMPONENTS + 1), k=panel_count)
        if random.random() < 0.75:
            while counts == sorted(counts):
                random.shuffle(counts)
    return counts


def _slot_rows(height: int) -> tuple[int, int]:
    usable_h = height - 1
    top_max = max(0, usable_h - 5)
    top_start = random.randint(0, top_max)
    bottom_min = top_start + 3
    bottom_max = usable_h - SLOT_HEIGHT
    assert bottom_min <= bottom_max
    bottom_start = random.randint(bottom_min, bottom_max)
    return top_start, bottom_start


def _build_panel(height: int, color: int, component_count: int) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH), BLACK, dtype=int)
    slot_rows = _slot_rows(height)
    slots = [(r0, c0) for r0 in slot_rows for c0 in SLOT_COLS]
    chosen_slots = random.sample(slots, k=component_count)
    masks = _connected_slot_masks(SLOT_HEIGHT, SLOT_WIDTH)

    for r0, c0 in chosen_slots:
        mask = random.choice(masks)
        panel[r0 : r0 + SLOT_HEIGHT, c0 : c0 + SLOT_WIDTH][mask] = color

    assert np.all(panel[-1, :] == BLACK), "bottom row must stay clear for count dots"
    assert _count_components4(panel != BLACK) == component_count
    return panel


def _build_grid(panels: list[np.ndarray]) -> np.ndarray:
    height = panels[0].shape[0]
    width = len(panels) * PANEL_WIDTH + (len(panels) - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), BLACK, dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        grid[:, cursor : cursor + PANEL_WIDTH] = panel
        cursor += PANEL_WIDTH
        if idx < len(panels) - 1:
            grid[:, cursor : cursor + SEPARATOR_WIDTH] = GREY
            cursor += SEPARATOR_WIDTH
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        panel_count = random.choice(PANEL_COUNT_OPTIONS)
        height = random.randint(*PANEL_HEIGHT_RANGE)
        counts = _choose_counts(panel_count)
        colors = random.sample(PALETTE, k=panel_count)
        panels = [_build_panel(height, color, count) for color, count in zip(colors, counts)]
        grid = _build_grid(panels)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
