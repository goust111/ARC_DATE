import random

import numpy as np


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    separator_cols = np.zeros(w, dtype=bool)
    for c in range(w):
        col = g[:, c]
        separator_cols[c] = int(col[0]) != 0 and np.all(col == col[0])

    panels = []
    c = 0
    while c < w:
        if separator_cols[c]:
            c += 1
            continue
        start = c
        while c < w and not separator_cols[c]:
            c += 1
        panels.append(g[:, start:c].copy())

    widths = {panel.shape[1] for panel in panels}
    assert 3 <= len(panels) <= 4, "expected 3 or 4 panels"
    assert len(widths) == 1, "all panels must have the same width"
    assert all(panel.shape[0] == h for panel in panels)
    return panels


def _count_holes(panel: np.ndarray) -> int:
    g = np.asarray(panel, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if g[r, c] != 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1

    return holes


def _overlay_panels(panels: list[np.ndarray], order: list[int] | None = None) -> np.ndarray:
    if order is None:
        hole_counts = [_count_holes(panel) for panel in panels]
        order = sorted(range(len(panels)), key=lambda idx: -hole_counts[idx])

    out = np.zeros_like(panels[0])
    for idx in order:
        panel = panels[idx]
        visible = (panel != 0) & (out == 0)
        out[visible] = panel[visible]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the aligned panels using hole count as the front-to-back priority.

    Input structure:
    - The grid is split by full-height separator columns into 3 or 4 same-sized panels.
    - Each panel contains background 0 plus exactly one foreground color, and panel colors are distinct.
    - The foreground shape in each panel has a different number of enclosed background holes.

    Rule:
    - Split the grid into panels by removing the separator columns.
    - Count, for each panel, how many 4-connected background regions are fully enclosed and do not touch the panel border.
    - Align the panels and transparently overlay them, treating 0 as empty, with larger hole counts placed in front of smaller hole counts.

    Key insight:
    - The hidden layer order is determined by a topological property of each panel shape: how many enclosed holes it contains.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    return _overlay_panels(panels)


def _distribute_extra(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _build_vertical_chambers(chambers: int, width: int, extra_rows: int, closed: bool) -> np.ndarray:
    chamber_heights = [1 + extra for extra in _distribute_extra(extra_rows, chambers)]
    height = 2 + sum(chamber_heights) + chambers - 1

    sprite = np.zeros((height, width), dtype=int)
    sprite[0, :] = 1
    sprite[-1, :] = 1
    sprite[:, 0] = 1
    if closed:
        sprite[:, -1] = 1

    row = 1
    for i, chamber_height in enumerate(chamber_heights):
        row += chamber_height
        if i < chambers - 1:
            sprite[row, :] = 1
            row += 1

    return sprite


def _build_horizontal_chambers(chambers: int, height: int, extra_cols: int, closed: bool) -> np.ndarray:
    return _build_vertical_chambers(chambers, height, extra_cols, closed).T


def _shape_mask(hole_count: int, panel_h: int, panel_w: int, max_holes: int) -> np.ndarray | None:
    closed = hole_count > 0
    chambers = hole_count if closed else random.randint(1, max(1, max_holes))

    choices: list[tuple[str, int]] = []

    min_vertical_h = 2 + chambers + chambers - 1
    max_extra_rows = panel_h - 1 - min_vertical_h
    if panel_w >= 3 and max_extra_rows >= 0:
        choices.append(("vertical", max_extra_rows))

    min_horizontal_w = 2 + chambers + chambers - 1
    max_extra_cols = panel_w - min_horizontal_w
    if panel_h >= 3 and max_extra_cols >= 0:
        choices.append(("horizontal", max_extra_cols))

    if not choices:
        return None

    orientation, max_extra = random.choice(choices)
    if orientation == "vertical":
        width = random.randint(3, panel_w)
        extra_rows = random.randint(0, max_extra)
        return _build_vertical_chambers(chambers, width, extra_rows, closed)

    height = random.randint(3, panel_h - 1)
    extra_cols = random.randint(0, max_extra)
    return _build_horizontal_chambers(chambers, height, extra_cols, closed)


def _place_mask(panel_h: int, panel_w: int, sprite: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    sprite_h, sprite_w = sprite.shape
    r0 = max(0, min(panel_h - sprite_h, center_r - sprite_h // 2 + random.randint(-1, 1)))
    c0 = max(0, min(panel_w - sprite_w, center_c - sprite_w // 2 + random.randint(-1, 1)))

    panel = np.zeros((panel_h, panel_w), dtype=int)
    panel[r0 : r0 + sprite_h, c0 : c0 + sprite_w] = sprite
    return panel


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, int(color), 0).astype(int)


def _render_input(colored_panels: list[np.ndarray], separator_color: int) -> np.ndarray:
    panel_h, panel_w = colored_panels[0].shape
    panel_count = len(colored_panels)
    total_w = panel_count * panel_w + (panel_count - 1)
    grid = np.full((panel_h, total_w), int(separator_color), dtype=int)

    for i, panel in enumerate(colored_panels):
        c0 = i * (panel_w + 1)
        grid[:, c0 : c0 + panel_w] = panel
    return grid


def _attempt_generation() -> np.ndarray | None:
    panel_count = random.choice([3, 4])
    hole_counts = list(range(panel_count))
    while True:
        random.shuffle(hole_counts)
        if hole_counts != sorted(hole_counts) and hole_counts != sorted(hole_counts, reverse=True):
            break

    max_panel_w = (30 - (panel_count - 1)) // panel_count
    panel_w = random.randint(5, max_panel_w)
    max_holes = panel_count - 1
    min_panel_h = max(6, 2 * max_holes + 2)
    panel_h = random.randint(min_panel_h, min(11, 30))

    center_r = panel_h // 2 + random.randint(-1, 1)
    center_c = panel_w // 2 + random.randint(-1, 1)

    masks = []
    for hole_count in hole_counts:
        panel_mask = None
        for _ in range(60):
            sprite = _shape_mask(hole_count, panel_h, panel_w, max_holes)
            if sprite is None:
                continue
            candidate = _place_mask(panel_h, panel_w, sprite, center_r, center_c)
            if _count_holes(candidate) == hole_count:
                panel_mask = candidate
                break
        if panel_mask is None:
            return None
        masks.append(panel_mask)

    stack = np.stack([mask != 0 for mask in masks], axis=0)
    overlap_pixels = int(np.count_nonzero(stack.sum(axis=0) >= 2))
    union_pixels = int(np.count_nonzero(stack.any(axis=0)))
    panel_area = panel_h * panel_w
    if overlap_pixels < 2 or overlap_pixels > panel_area // 2:
        return None
    if not (panel_area // 6 <= union_pixels <= (2 * panel_area) // 3):
        return None

    palette = random.sample(range(1, 10), panel_count + 1)
    separator_color = palette[0]
    colored_panels = [_colorize(mask, color) for mask, color in zip(masks, palette[1:])]
    expected = _overlay_panels(colored_panels, order=sorted(range(panel_count), key=lambda idx: -hole_counts[idx]))
    visible_colors = set(map(int, np.unique(expected))) - {0}
    if len(visible_colors) != panel_count:
        return None

    grid = _render_input(colored_panels, separator_color)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlay puzzle")
