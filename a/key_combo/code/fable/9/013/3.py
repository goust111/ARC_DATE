from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

GEN_ATTEMPTS = 400
PANEL_COUNT_CHOICES = (3, 4)
PANEL_HEIGHT_RANGE = (6, 9)
PANEL_WIDTH_RANGE = {
    3: (5, 8),
    4: (5, 6),
}
TICK_VALUES = (1, 2, 3, 4, 5)


def _uniform_nonblack_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        values = np.unique(grid[:, c])
        if len(values) == 1 and int(values[0]) != BLACK:
            cols.append(c)
    return cols


def _parse_panels(grid: np.ndarray) -> tuple[int, list[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_cols = _uniform_nonblack_columns(g)
    assert len(separator_cols) >= 2, "expected at least three panels separated by bars"

    separator_color = int(g[0, separator_cols[0]])
    assert all(np.all(g[:, c] == separator_color) for c in separator_cols), "all separator columns must share one color"
    assert all(b - a > 1 for a, b in zip(separator_cols, separator_cols[1:])), "separator columns must be isolated"

    boundaries = [-1] + separator_cols + [g.shape[1]]
    panels: list[dict] = []
    widths: list[int] = []
    for left, right in zip(boundaries, boundaries[1:]):
        c0 = left + 1
        c1 = right
        panel = g[:, c0:c1].copy()
        assert panel.shape[1] > 0, "separator columns must leave non-empty panels"

        values = set(map(int, np.unique(panel)))
        pattern_colors = values - {BLACK, separator_color}
        assert len(pattern_colors) == 1, "each panel must contain exactly one pattern color"
        pattern_color = int(next(iter(pattern_colors)))
        assert values <= {BLACK, separator_color, pattern_color}, "panel contains unexpected colors"

        tick_count = int(np.count_nonzero(panel == separator_color))
        assert tick_count > 0, "each panel needs at least one separator-color tick"

        panels.append(
            {
                "grid": panel,
                "color": pattern_color,
                "tick_count": tick_count,
            }
        )
        widths.append(int(panel.shape[1]))

    assert len(set(widths)) == 1, "all panels must have the same width"
    counts = [panel["tick_count"] for panel in panels]
    assert len(counts) >= 3, "expected at least three panels"
    assert len(counts) == len(set(counts)), "tick counts must be unique"
    return separator_color, separator_cols, panels


def _compose_from_entries(entries: list[dict], separator_color: int) -> np.ndarray:
    out = np.zeros_like(entries[0]["grid"])
    for entry in sorted(entries, key=lambda item: item["tick_count"]):
        patch = entry["grid"].copy()
        patch[patch == separator_color] = BLACK
        mask = patch != BLACK
        out[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay equal-size panels from back to front using separator-color tick counts as the depth order.

    Input structure:
    - Several equal-width pattern panels are arranged horizontally and separated by full-height vertical bars of one shared color.
    - Inside each panel, black is transparent, one non-black color is the panel's pattern color, and some isolated pixels use the separator-bar color.
    - The separator-color pixels inside the panels have distinct counts across panels.

    Rule:
    - Split the grid into panels at the full-height separator columns.
    - Count the separator-color pixels inside each panel; fewer means deeper, more means closer to the front.
    - Erase those separator-color pixels, then overlay the remaining non-black pattern cells from deepest panel to shallowest.

    Key insight:
    - Left-to-right position is a decoy; the true occlusion order is encoded only by how many separator-color ticks are scattered inside each panel.
    """
    separator_color, _separator_cols, panels = _parse_panels(grid)
    return _compose_from_entries(panels, separator_color)


def _place_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _shape_block(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rect_h = random.randint(2, h - 2)
    rect_w = random.randint(2, w)
    r0 = random.randint(1, h - rect_h - 1)
    c0 = random.randint(0, w - rect_w)
    _place_rect(mask, r0, c0, rect_h, rect_w)
    return mask


def _shape_two_blocks(h: int, w: int) -> np.ndarray:
    mask = _shape_block(h, w)
    for _ in range(30):
        rect_h = random.randint(1, h - 2)
        rect_w = random.randint(1, w)
        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(0, w - rect_w)
        block = np.zeros_like(mask)
        _place_rect(block, r0, c0, rect_h, rect_w)
        if np.any(block & mask):
            mask |= block
            return mask
    return mask


def _shape_cross(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    stem_w = random.randint(1, min(2, w))
    stem_h = random.randint(3, h - 2)
    stem_r0 = random.randint(1, h - stem_h - 1)
    stem_c0 = random.randint(0, w - stem_w)
    _place_rect(mask, stem_r0, stem_c0, stem_h, stem_w)

    arm_h = random.randint(1, 2)
    arm_w = random.randint(3, w)
    arm_r0 = random.randint(stem_r0, stem_r0 + stem_h - arm_h)
    arm_c0 = random.randint(0, w - arm_w)
    _place_rect(mask, arm_r0, arm_c0, arm_h, arm_w)
    return mask


def _shape_t(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bar_h = random.randint(1, 2)
    bar_w = random.randint(3, w)
    box_h = random.randint(max(3, bar_h + 2), h - 2)
    box_w = bar_w
    r0 = random.randint(1, h - box_h - 1)
    c0 = random.randint(0, w - box_w)
    _place_rect(mask, r0, c0, bar_h, bar_w)

    stem_h = box_h - bar_h
    stem_w = random.randint(1, min(2, bar_w))
    stem_c0 = c0 + random.randint(0, bar_w - stem_w)
    _place_rect(mask, r0 + bar_h, stem_c0, stem_h, stem_w)
    return mask


def _shape_l(h: int, w: int) -> np.ndarray:
    local_h = random.randint(3, h - 2)
    local_w = random.randint(3, w)
    r0 = random.randint(1, h - local_h - 1)
    c0 = random.randint(0, w - local_w)
    thickness = random.randint(1, min(2, local_h - 1, local_w - 1))

    local = np.zeros((local_h, local_w), dtype=bool)
    local[:, :thickness] = True
    local[-thickness:, :] = True
    if random.random() < 0.5:
        local = np.fliplr(local)
    if random.random() < 0.5:
        local = np.flipud(local)

    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + local_h, c0 : c0 + local_w] = local
    return mask


def _make_mask(h: int, w: int) -> np.ndarray | None:
    shape_fns = (_shape_block, _shape_two_blocks, _shape_cross, _shape_t, _shape_l)
    min_area = max(5, (h * w) // 5)
    max_area = max(min_area + 1, (h * w * 3) // 5)
    for _ in range(80):
        mask = random.choice(shape_fns)(h, w)
        area = int(np.count_nonzero(mask))
        if not (min_area <= area <= max_area):
            continue
        if np.any(mask[0]) or np.any(mask[-1]):
            continue
        if any(np.all(mask[:, c]) for c in range(w)):
            continue
        if np.all(mask) or not np.any(mask):
            continue
        return mask
    return None


def _choose_tick_positions(mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    candidates = [(r, c) for r in range(h) for c in range(1, w - 1) if not mask[r, c]]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _compose_from_masks(masks: list[np.ndarray], colors: list[int], order: list[int]) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=int)
    for idx in order:
        out[masks[idx]] = int(colors[idx])
    return out


def _build_grid(panels: list[np.ndarray], separator_color: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        pieces.append(panel)
        if idx != len(panels) - 1:
            pieces.append(np.full((panel.shape[0], 1), separator_color, dtype=int))
    return np.hstack(pieces)


def _visible_contributions(masks: list[np.ndarray], order: list[int]) -> dict[int, int]:
    above = np.zeros_like(masks[0], dtype=bool)
    visible: dict[int, int] = {}
    for idx in reversed(order):
        here = masks[idx] & ~above
        visible[idx] = int(np.count_nonzero(here))
        above |= masks[idx]
    return visible


def _effective_overlap(masks: list[np.ndarray], order: list[int], rank: int) -> int:
    deeper = order[rank]
    shallower = order[rank + 1]
    blockers = np.zeros_like(masks[0], dtype=bool)
    for idx in order[rank + 2 :]:
        blockers |= masks[idx]
    overlap = masks[deeper] & masks[shallower] & ~blockers
    return int(np.count_nonzero(overlap))


def _ticks_are_isolated(grid: np.ndarray, separator_cols: list[int], separator_color: int) -> bool:
    h, w = grid.shape
    separator_set = set(separator_cols)
    coords = np.argwhere(grid == separator_color)
    for r, c in coords:
        if int(c) in separator_set:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == separator_color:
                    return False
    return True


def _validate_generated(
    grid: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
    counts: list[int],
    order: list[int],
    separator_color: int,
) -> bool:
    if order == list(range(len(order))) or order == list(range(len(order) - 1, -1, -1)):
        return False

    try:
        parsed_separator, separator_cols, panels = _parse_panels(grid)
    except AssertionError:
        return False

    if parsed_separator != separator_color:
        return False
    if [panel["color"] for panel in panels] != colors:
        return False
    if [panel["tick_count"] for panel in panels] != counts:
        return False
    if not _ticks_are_isolated(grid, separator_cols, separator_color):
        return False

    expected = _compose_from_masks(masks, colors, order)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    visible = _visible_contributions(masks, order)
    if any(visible[idx] <= 0 for idx in order):
        return False
    if any(_effective_overlap(masks, order, rank) <= 0 for rank in range(len(order) - 1)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    panel_count = random.choice(PANEL_COUNT_CHOICES)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE[panel_count])

    separator_color = random.choice(PALETTE)
    panel_colors = random.sample([color for color in PALETTE if color != separator_color], panel_count)

    counts_sorted = sorted(random.sample(TICK_VALUES, panel_count))
    order = list(range(panel_count))
    random.shuffle(order)
    counts = [0] * panel_count
    for depth_rank, panel_idx in enumerate(order):
        counts[panel_idx] = counts_sorted[depth_rank]

    masks: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    for color, tick_count in zip(panel_colors, counts):
        mask = _make_mask(panel_h, panel_w)
        if mask is None:
            return None
        ticks = _choose_tick_positions(mask, tick_count)
        if ticks is None:
            return None

        panel = np.zeros((panel_h, panel_w), dtype=int)
        panel[mask] = int(color)
        for r, c in ticks:
            panel[r, c] = separator_color

        masks.append(mask)
        panels.append(panel)

    grid = _build_grid(panels, separator_color)
    if max(grid.shape) > 30:
        return None

    if not _validate_generated(grid, masks, panel_colors, counts, order, separator_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-count overlay input")
