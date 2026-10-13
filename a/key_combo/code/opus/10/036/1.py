from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PANEL_COLORS = (1, 2, 3, 4)
SEPARATOR = 5
MARKER = 7

GRID_H = 12
PANEL_W = 4
PANEL_COUNT = 4
INPUT_W = PANEL_COUNT * PANEL_W + (PANEL_COUNT - 1)
SEPARATOR_COLS = (4, 9, 14)

MARKER_COUNTS = (1, 2, 3, 4)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _parse_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    separator_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(separator_cols) == 3, "expected exactly three full-height separator columns"

    panels: list[np.ndarray] = []
    start = 0
    for sep in separator_cols + [w]:
        panels.append(g[:, start:sep].copy())
        start = sep + 1

    assert len(panels) == 4, "expected four panels"
    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1 and next(iter(widths)) > 0, "panels must have equal width"
    return panels


def _front_to_back_order(marker_counts: list[int]) -> list[int]:
    return sorted(range(len(marker_counts)), key=lambda i: (int(marker_counts[i]), int(i)))


def _back_to_front_draw_order(marker_counts: list[int]) -> list[int]:
    return sorted(range(len(marker_counts)), key=lambda i: (-int(marker_counts[i]), -int(i)))


def _overlay_panels_front_to_back(panels: list[np.ndarray], front_order: list[int]) -> np.ndarray:
    h, w = panels[0].shape
    out = np.zeros((h, w), dtype=int)
    filled = np.zeros((h, w), dtype=bool)
    for idx in front_order:
        panel = panels[int(idx)]
        mask = (panel != BG) & (~filled)
        out[mask] = panel[mask]
        filled |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each panel's orange markers, erase them, and overlay the panels by marker-count priority.

    Input structure:
    - The grid is split into four equal-width panels by three full-height grey separator columns.
    - Each panel contains black background, one unique foreground color, and several orange marker pixels on background cells.
    - Corresponding cells across the four panels share the same local panel coordinates.

    Rule:
    - Count the orange marker pixels inside each panel.
    - Replace every orange marker pixel with black background.
    - Overlay the cleaned panels on one shared canvas, treating black as transparent.
      The panel with more markers is farther back; the panel with fewer markers is farther front and overwrites lower layers.
      If counts tie, the left panel has higher priority.

    Key insight:
    - The separators only partition already-aligned layers; the hidden information is the layer priority encoded by each panel's marker count.
    """
    panels = _parse_panels(np.asarray(grid, dtype=int))

    marker_counts: list[int] = []
    cleaned: list[np.ndarray] = []
    foreground_colors: list[int] = []
    for panel in panels:
        marker_counts.append(int(np.count_nonzero(panel == MARKER)))
        panel_clean = panel.copy()
        panel_clean[panel_clean == MARKER] = BG
        colors = sorted(set(map(int, np.unique(panel_clean))) - {BG})
        assert len(colors) == 1, "each panel must contain exactly one foreground color after marker erasure"
        cleaned.append(panel_clean)
        foreground_colors.append(colors[0])

    assert len(set(foreground_colors)) == 4, "panel colors must be distinct"

    draw_order = _back_to_front_draw_order(marker_counts)
    out = np.zeros_like(cleaned[0])
    for idx in draw_order:
        panel = cleaned[int(idx)]
        mask = panel != BG
        out[mask] = panel[mask]
    return out


def _shortest_path(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cell = q.popleft()
        if cell in goals:
            path = []
            cur: tuple[int, int] | None = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        r, c = cell
        nbrs = list(_neighbors4(r, c, h, w))
        random.shuffle(nbrs)
        for nxt in nbrs:
            if nxt in parent or (nxt in blocked and nxt not in goals):
                continue
            parent[nxt] = cell
            q.append(nxt)
    return None


def _is_connected(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nxt in _neighbors4(r, c, h, w):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _shared_pair() -> set[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randint(3, 8)
        c = random.randint(0, 2)
        return {(r, c), (r, c + 1)}
    r = random.randint(3, 7)
    c = random.randint(1, 2)
    return {(r, c), (r + 1, c)}


def _exclusive_candidates(shared: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r in range(GRID_H):
        for c in range(PANEL_W):
            cell = (r, c)
            if cell in shared:
                continue
            dist = min(abs(r - sr) + abs(c - sc) for sr, sc in shared)
            if 1 <= dist <= 4:
                out.append(cell)
    return out


def _grow_blob(
    shared: set[tuple[int, int]],
    exclusive: tuple[int, int],
    reserved_exclusives: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    blocked = set(reserved_exclusives) - {exclusive}
    path = _shortest_path(exclusive, shared, blocked, GRID_H, PANEL_W)
    if path is None:
        return None

    cells = set(path) | set(shared)
    if len(cells) > 8:
        return None

    target = random.randint(max(3, len(cells)), 8)
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nxt in _neighbors4(r, c, GRID_H, PANEL_W):
                if nxt not in cells and nxt not in blocked:
                    frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    if not (3 <= len(cells) <= 8):
        return None
    if not _is_connected(cells, GRID_H, PANEL_W):
        return None
    return cells


def _cells_to_panel(
    cells: set[tuple[int, int]],
    color: int,
    marker_count: int,
) -> np.ndarray:
    panel = np.zeros((GRID_H, PANEL_W), dtype=int)
    for r, c in cells:
        panel[r, c] = int(color)

    bg_cells = [(r, c) for r in range(GRID_H) for c in range(PANEL_W) if panel[r, c] == BG]
    assert len(bg_cells) >= marker_count
    for r, c in random.sample(bg_cells, marker_count):
        panel[r, c] = MARKER
    return panel


def _compose_input(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_H, INPUT_W), SEPARATOR, dtype=int)
    for i, panel in enumerate(panels):
        c0 = i * (PANEL_W + 1)
        grid[:, c0 : c0 + PANEL_W] = panel
    return grid


def _expected_output(panel_shapes: list[np.ndarray], marker_counts: list[int]) -> np.ndarray:
    front_order = _front_to_back_order(marker_counts)
    return _overlay_panels_front_to_back(panel_shapes, front_order)


def _validate_semantics(
    panels: list[np.ndarray],
    panel_shapes: list[np.ndarray],
    marker_counts: list[int],
    panel_colors: list[int],
    shared: set[tuple[int, int]],
    exclusives: list[tuple[int, int]],
) -> bool:
    if sorted(marker_counts) != list(MARKER_COUNTS):
        return False
    if sorted(panel_colors) != list(PANEL_COLORS):
        return False

    overlap_counts = np.sum(np.stack([(panel != BG) for panel in panel_shapes], axis=0), axis=0)
    overlap_cells = int(np.count_nonzero(overlap_counts >= 2))
    if overlap_cells < len(shared) or overlap_cells > 10:
        return False
    if int(overlap_counts.max()) < 2:
        return False

    front_order = _front_to_back_order(marker_counts)
    actual = _overlay_panels_front_to_back(panel_shapes, front_order)

    if len(set(map(int, np.unique(actual))) - {BG}) != 4:
        return False

    for idx, exclusive in enumerate(exclusives):
        if panel_shapes[idx][exclusive] == BG:
            return False
        for jdx, shape in enumerate(panel_shapes):
            if idx != jdx and shape[exclusive] != BG:
                return False

    visible_counts = [int(np.count_nonzero(actual == color)) for color in panel_colors]
    if min(visible_counts) < 1:
        return False

    no_erase = _overlay_panels_front_to_back(panels, front_order)
    if np.array_equal(no_erase, actual):
        return False

    wrong_orders = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [2, 3, 0, 1],
        [1, 2, 3, 0],
        sorted(range(4), key=lambda i: panel_colors[i]),
    ]
    if any(np.array_equal(_overlay_panels_front_to_back(panel_shapes, order), actual) for order in wrong_orders):
        return False

    if not all(np.count_nonzero(panel == MARKER) == marker_counts[i] for i, panel in enumerate(panels)):
        return False

    for panel, shape, color in zip(panels, panel_shapes, panel_colors):
        cleaned = panel.copy()
        cleaned[cleaned == MARKER] = BG
        if not np.array_equal(cleaned, shape):
            return False
        if set(map(int, np.unique(shape))) - {BG} != {int(color)}:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shared = _shared_pair()
    candidates = _exclusive_candidates(shared)
    if len(candidates) < 4:
        return None

    exclusives = random.sample(candidates, 4)
    panel_colors = random.sample(list(PANEL_COLORS), 4)
    marker_counts = random.sample(list(MARKER_COUNTS), 4)

    panel_shapes: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    reserved = set(exclusives)

    for exclusive, color, marker_count in zip(exclusives, panel_colors, marker_counts):
        cells = _grow_blob(shared, exclusive, reserved)
        if cells is None:
            return None
        shape = np.zeros((GRID_H, PANEL_W), dtype=int)
        for r, c in cells:
            shape[r, c] = int(color)
        panel_shapes.append(shape)
        panels.append(_cells_to_panel(cells, color, marker_count))

    if not _validate_semantics(panels, panel_shapes, marker_counts, panel_colors, shared, exclusives):
        return None

    inp = _compose_input(panels)
    expected = _expected_output(panel_shapes, marker_counts)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
