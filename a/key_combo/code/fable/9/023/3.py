from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _parse_panels(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    divider_rows = [
        r
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert divider_rows, "expected separator rows"

    separator = int(g[divider_rows[0], 0])
    assert all(int(g[r, 0]) == separator for r in divider_rows), "separator rows must share one color"

    panel_count = len(divider_rows) + 1
    assert panel_count % 2 == 1 and panel_count >= 3, "panel count must be odd and at least 3"

    panel_h = int(divider_rows[0])
    assert panel_h >= 2, "panels must be non-empty and tall enough for anchor placement"

    for i, row in enumerate(divider_rows):
        assert row == (i + 1) * panel_h + i, "separator rows must split equal-height panels"
        assert np.all(g[row, :] == separator), "separator row must be solid"

    expected_h = panel_count * panel_h + (panel_count - 1)
    assert g.shape[0] == expected_h, "grid height must match equal-height panel layout"

    panels = []
    start = 0
    for i in range(panel_count):
        panel = g[start : start + panel_h, :]
        assert panel.shape[0] == panel_h
        panels.append(panel)
        start += panel_h
        if i < panel_count - 1:
            start += 1

    return g, bg, separator, panels


def _detect_colors(panels: list[np.ndarray], bg: int, separator: int) -> tuple[int, int]:
    colors = sorted(
        {
            int(color)
            for panel in panels
            for color in np.unique(panel)
            if int(color) not in {bg, separator}
        }
    )
    anchor_candidates = []
    for color in colors:
        per_panel = [int(np.sum(panel == color)) for panel in panels]
        if all(count == 1 for count in per_panel):
            anchor_candidates.append(color)

    assert len(anchor_candidates) == 1, "anchor color must be uniquely identifiable"
    anchor = anchor_candidates[0]
    draw_colors = [color for color in colors if color != anchor]
    assert len(draw_colors) == 1, "expected exactly one drawing color"
    return anchor, draw_colors[0]


def _offsets_by_panel(
    panels: list[np.ndarray], anchor_color: int, draw_color: int
) -> tuple[list[tuple[int, int]], list[set[tuple[int, int]]]]:
    anchors: list[tuple[int, int]] = []
    offsets_per_panel: list[set[tuple[int, int]]] = []
    for panel in panels:
        anchor_cells = np.argwhere(panel == anchor_color)
        assert anchor_cells.shape == (1, 2), "each panel must contain exactly one anchor"
        ar, ac = map(int, anchor_cells[0])
        anchors.append((ar, ac))

        draw_cells = np.argwhere(panel == draw_color)
        offsets = {(int(r) - ar, int(c) - ac) for r, c in draw_cells}
        assert (0, 0) not in offsets, "drawing pixels cannot overwrite the anchor"
        offsets_per_panel.append(offsets)

    return anchors, offsets_per_panel


def _render_panel(
    panel_shape: tuple[int, int],
    bg: int,
    anchor_color: int,
    draw_color: int,
    anchor: tuple[int, int],
    offsets: set[tuple[int, int]],
) -> np.ndarray:
    panel = np.full(panel_shape, bg, dtype=int)
    ar, ac = anchor
    panel[ar, ac] = anchor_color
    for dr, dc in offsets:
        rr, cc = ar + dr, ac + dc
        assert 0 <= rr < panel_shape[0] and 0 <= cc < panel_shape[1]
        panel[rr, cc] = draw_color
    return panel


def _stack_panels(panels: list[np.ndarray], bg: int, separator: int) -> np.ndarray:
    panel_h, width = panels[0].shape
    total_h = len(panels) * panel_h + (len(panels) - 1)
    grid = np.full((total_h, width), bg, dtype=int)
    start = 0
    for i, panel in enumerate(panels):
        grid[start : start + panel_h, :] = panel
        start += panel_h
        if i < len(panels) - 1:
            grid[start, :] = separator
            start += 1
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align equal-height separator-split panels by their anchor pixels and majority-vote the shared drawing.

    Input structure:
    - The grid is partitioned by full-width separator rows of one color into an odd number of equal-height panels.
    - Every panel contains exactly one anchor pixel of a fixed color plus zero or more pixels of one drawing color.
    - The drawing pixels are noisy observations of one hidden pattern: true offsets from the anchor may be missing in some panels, and false offsets may appear in a minority of panels.

    Rule:
    - Use the anchor pixel in each panel as the alignment origin.
    - Count drawing-color occurrences at every anchor-relative offset across panels; keep exactly the offsets present in a strict majority of panels.
    - Output one panel-sized grid with no separator rows, placing the anchor at the same coordinate it occupies in the vertically middle panel and redrawing the majority-approved offsets around it.

    Key insight:
    - Neither local overlay nor union/intersection works: the panels must first be registered by anchor, then each relative position must be decided by strict majority, and the final placement comes specifically from the middle panel.
    """
    g, bg, separator, panels = _parse_panels(grid)
    anchor_color, draw_color = _detect_colors(panels, bg, separator)
    anchors, offsets_per_panel = _offsets_by_panel(panels, anchor_color, draw_color)

    counts: Counter[tuple[int, int]] = Counter()
    for offsets in offsets_per_panel:
        counts.update(offsets)

    threshold = len(panels) // 2
    majority_offsets = {offset for offset, count in counts.items() if count > threshold}
    assert majority_offsets, "majority vote must reconstruct at least one drawing pixel"

    middle_anchor = anchors[len(panels) // 2]
    return _render_panel(
        panels[0].shape,
        bg,
        anchor_color,
        draw_color,
        middle_anchor,
        majority_offsets,
    )


MAX_ATTEMPTS = 500
PANEL_COUNT_OPTIONS = (3, 5)
PANEL_HEIGHT_OPTIONS = {3: (6, 7, 8, 9), 5: (4, 5)}
WIDTH_RANGE = (9, 16)
TRUE_PIXEL_RANGE = (4, 8)
FALSE_PIXEL_RANGE = (2, 4)


def _neighbors(offset: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = offset
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_shape(
    rng: np.random.Generator, candidates: set[tuple[int, int]], size: int
) -> set[tuple[int, int]] | None:
    cells = list(candidates)
    start = cells[int(rng.integers(len(cells)))]
    shape = {start}
    while len(shape) < size:
        frontier = [
            nbr
            for cell in shape
            for nbr in _neighbors(cell)
            if nbr in candidates and nbr not in shape
        ]
        if not frontier:
            return None
        pick = frontier[int(rng.integers(len(frontier)))]
        shape.add(pick)
    return shape


def _safe_offsets(
    panel_shape: tuple[int, int], anchors: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    panel_h, width = panel_shape
    max_up = min(anchor[0] for anchor in anchors)
    max_down = min(panel_h - 1 - anchor[0] for anchor in anchors)
    max_left = min(anchor[1] for anchor in anchors)
    max_right = min(width - 1 - anchor[1] for anchor in anchors)
    return {
        (dr, dc)
        for dr in range(-max_up, max_down + 1)
        for dc in range(-max_left, max_right + 1)
        if (dr, dc) != (0, 0)
    }


def _majority_local_output(
    panels: list[np.ndarray], bg: int, anchor_color: int, draw_color: int
) -> np.ndarray:
    anchors, _ = _offsets_by_panel(panels, anchor_color, draw_color)
    panel_h, width = panels[0].shape
    out = np.full((panel_h, width), bg, dtype=int)
    middle_anchor = anchors[len(panels) // 2]
    out[middle_anchor] = anchor_color

    local_counts = np.zeros((panel_h, width), dtype=int)
    for panel in panels:
        local_counts += (panel == draw_color).astype(int)

    out[local_counts > len(panels) // 2] = draw_color
    out[middle_anchor] = anchor_color
    return out


def _choose_anchor_positions(
    rng: np.random.Generator,
    panel_count: int,
    panel_shape: tuple[int, int],
    shape_offsets: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    panel_h, width = panel_shape
    max_up = max(-dr for dr, _ in shape_offsets if dr < 0) if any(dr < 0 for dr, _ in shape_offsets) else 0
    max_down = max(dr for dr, _ in shape_offsets if dr > 0) if any(dr > 0 for dr, _ in shape_offsets) else 0
    max_left = max(-dc for _, dc in shape_offsets if dc < 0) if any(dc < 0 for _, dc in shape_offsets) else 0
    max_right = max(dc for _, dc in shape_offsets if dc > 0) if any(dc > 0 for _, dc in shape_offsets) else 0

    positions = [
        (r, c)
        for r in range(max_up, panel_h - max_down)
        for c in range(max_left, width - max_right)
    ]
    if len(positions) < 3:
        return None

    mid_index = panel_count // 2
    middle_anchor = positions[int(rng.integers(len(positions)))]
    others = [pos for pos in positions if pos != middle_anchor]
    if not others:
        return None

    anchors = []
    for i in range(panel_count):
        if i == mid_index:
            anchors.append(middle_anchor)
        else:
            anchors.append(others[int(rng.integers(len(others)))])

    if len(set(anchors)) < 2:
        return None
    if sum(anchor == middle_anchor for anchor in anchors) != 1:
        return None
    return anchors


def _assign_observations(
    rng: np.random.Generator,
    panel_count: int,
    true_offsets: set[tuple[int, int]],
    false_offsets: set[tuple[int, int]],
) -> list[set[tuple[int, int]]] | None:
    majority = panel_count // 2 + 1
    offsets_per_panel = [set() for _ in range(panel_count)]
    true_list = list(true_offsets)
    false_list = list(false_offsets)
    rng.shuffle(true_list)
    rng.shuffle(false_list)

    true_sizes = []
    for i, offset in enumerate(true_list):
        if i == 0:
            k = majority
        elif i == 1 and panel_count > majority:
            k = panel_count
        else:
            k = int(rng.integers(majority, panel_count + 1))
        chosen = rng.choice(panel_count, size=k, replace=False)
        true_sizes.append(k)
        for idx in chosen:
            offsets_per_panel[int(idx)].add(offset)

    false_sizes = []
    for i, offset in enumerate(false_list):
        if i == 0:
            k = majority - 1
        else:
            k = int(rng.integers(1, majority))
        chosen = rng.choice(panel_count, size=k, replace=False)
        false_sizes.append(k)
        for idx in chosen:
            offsets_per_panel[int(idx)].add(offset)

    if min(len(offsets) for offsets in offsets_per_panel) < 3:
        return None
    if len({frozenset(offsets) for offsets in offsets_per_panel}) == 1:
        return None
    if not any(k < panel_count for k in true_sizes):
        return None
    if majority - 1 > 0 and not any(k == majority - 1 for k in false_sizes):
        return None
    return offsets_per_panel


def _truth_counts(offsets_per_panel: list[set[tuple[int, int]]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for offsets in offsets_per_panel:
        counts.update(offsets)
    return counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_count = int(rng.choice(PANEL_COUNT_OPTIONS))
    panel_h = int(rng.choice(PANEL_HEIGHT_OPTIONS[panel_count]))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    panel_shape = (panel_h, width)

    palette = np.arange(10, dtype=int)
    bg, separator, anchor_color, draw_color = map(int, rng.choice(palette, size=4, replace=False))

    max_up = min(2, panel_h - 2)
    max_down = min(2, panel_h - 2)
    max_left = min(3, width - 3)
    max_right = min(3, width - 3)
    if min(max_up, max_down, max_left, max_right) < 1:
        return None

    box = {
        (dr, dc)
        for dr in range(-max_up, max_down + 1)
        for dc in range(-max_left, max_right + 1)
        if (dr, dc) != (0, 0)
    }
    true_size_hi = min(TRUE_PIXEL_RANGE[1], len(box) - FALSE_PIXEL_RANGE[0])
    if true_size_hi < TRUE_PIXEL_RANGE[0]:
        return None
    true_size = int(rng.integers(TRUE_PIXEL_RANGE[0], true_size_hi + 1))
    true_offsets = _connected_shape(rng, box, true_size)
    if true_offsets is None:
        return None

    if len({dr for dr, _ in true_offsets}) < 2 or len({dc for _, dc in true_offsets}) < 2:
        return None

    anchors = _choose_anchor_positions(rng, panel_count, panel_shape, true_offsets)
    if anchors is None:
        return None

    safe = _safe_offsets(panel_shape, anchors)
    if not true_offsets <= safe:
        return None

    false_candidates = list(safe - true_offsets)
    false_size_hi = min(FALSE_PIXEL_RANGE[1], len(false_candidates))
    if false_size_hi < FALSE_PIXEL_RANGE[0]:
        return None
    false_size = int(rng.integers(FALSE_PIXEL_RANGE[0], false_size_hi + 1))
    rng.shuffle(false_candidates)
    false_offsets = set(false_candidates[:false_size])

    offsets_per_panel = _assign_observations(rng, panel_count, true_offsets, false_offsets)
    if offsets_per_panel is None:
        return None

    panels = [
        _render_panel(panel_shape, bg, anchor_color, draw_color, anchor, offsets)
        for anchor, offsets in zip(anchors, offsets_per_panel)
    ]
    inp = _stack_panels(panels, bg, separator)

    expected = _render_panel(
        panel_shape,
        bg,
        anchor_color,
        draw_color,
        anchors[panel_count // 2],
        true_offsets,
    )

    counts = _truth_counts(offsets_per_panel)
    majority = panel_count // 2 + 1
    if {offset for offset, count in counts.items() if count >= majority} != true_offsets:
        return None
    if {offset for offset, count in counts.items() if count == panel_count} == true_offsets:
        return None
    if set(counts) == true_offsets:
        return None

    local_majority = _majority_local_output(panels, bg, anchor_color, draw_color)
    if np.array_equal(local_majority, expected):
        return None

    union_panel = _render_panel(
        panel_shape,
        bg,
        anchor_color,
        draw_color,
        anchors[panel_count // 2],
        set(counts),
    )
    intersection_panel = _render_panel(
        panel_shape,
        bg,
        anchor_color,
        draw_color,
        anchors[panel_count // 2],
        {offset for offset, count in counts.items() if count == panel_count},
    )
    if np.array_equal(union_panel, expected) or np.array_equal(intersection_panel, expected):
        return None

    if anchors[0] != anchors[panel_count // 2]:
        wrong_anchor = _render_panel(
            panel_shape, bg, anchor_color, draw_color, anchors[0], true_offsets
        )
        if np.array_equal(wrong_anchor, expected):
            return None

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
