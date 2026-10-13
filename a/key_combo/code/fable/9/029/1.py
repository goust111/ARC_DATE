from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
TEAL = 8

PANEL_SIZE_RANGE = (6, 11)
MAX_SEGMENTS = 4
GEN_MAX_ATTEMPTS = 500
NONSPECIAL_COLORS = (1, 2, 3, 5, 6, 7, 9)


def _find_separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    candidates = []
    for row in range(height):
        color = int(grid[row, 0])
        if color == BLACK:
            continue
        if np.all(grid[row, :] == color) and np.count_nonzero(grid == color) == width:
            candidates.append(row)
    assert len(candidates) == 1, "expected exactly one unique solid separator row"
    return int(candidates[0])


def _detect_lamp_edge(panel: np.ndarray) -> str:
    edges = []
    if np.all(panel[0, :] == YELLOW):
        edges.append("top")
    if np.all(panel[-1, :] == YELLOW):
        edges.append("bottom")
    if np.all(panel[:, 0] == YELLOW):
        edges.append("left")
    if np.all(panel[:, -1] == YELLOW):
        edges.append("right")
    assert len(edges) == 1, "expected exactly one full yellow lamp bar per panel"

    edge = edges[0]
    expected_count = panel.shape[1] if edge in ("top", "bottom") else panel.shape[0]
    assert np.count_nonzero(panel == YELLOW) == expected_count, "yellow should appear only on the lamp bar"
    return edge


def _panel_shadow_mask(panel: np.ndarray) -> np.ndarray:
    height, width = map(int, panel.shape)
    edge = _detect_lamp_edge(panel)
    blockers = (panel != BLACK) & (panel != YELLOW)
    shadow = np.zeros((height, width), dtype=bool)

    if edge == "top":
        for col in range(width):
            rows = np.flatnonzero(blockers[1:, col])
            if rows.size:
                first = int(rows[0]) + 1
                shadow[first + 1 :, col] = True
    elif edge == "bottom":
        for col in range(width):
            rows = np.flatnonzero(blockers[:-1, col])
            if rows.size:
                first = int(rows[-1])
                shadow[:first, col] = True
    elif edge == "left":
        for row in range(height):
            cols = np.flatnonzero(blockers[row, 1:])
            if cols.size:
                first = int(cols[0]) + 1
                shadow[row, first + 1 :] = True
    else:
        for row in range(height):
            cols = np.flatnonzero(blockers[row, :-1])
            if cols.size:
                first = int(cols[-1])
                shadow[row, :first] = True

    return shadow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the cells that lie in shadow in both lamp-lit panels.

    Input structure:
    - A black background grid is split into two equal panels by a single solid separator row whose color appears nowhere else.
    - Each panel contains exactly one full yellow lamp bar attached to one panel edge.
    - All other non-black cells inside a panel are blocker cells in that panel's own color.

    Rule:
    - Within each panel, every lamp cell emits a straight ray perpendicular to the lamp bar into the panel interior.
    - A ray lights every empty cell it crosses and also lights the first blocker it hits, then stops; cells beyond that first blocker stay in shadow.
    - Remove the separator row, align the two panels by position, and output a black grid of panel size with teal only where both aligned panels are shadowed.

    Key insight:
    - The required shape is not drawn directly in either panel; it is the intersection of two derived shadow masks determined jointly by lamp orientation and first-hit blockers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator_row = _find_separator_row(g)
    top = g[:separator_row, :]
    bottom = g[separator_row + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "expected two nonempty equal-size panels"

    top_shadow = _panel_shadow_mask(top)
    bottom_shadow = _panel_shadow_mask(bottom)

    out = np.full(top.shape, BLACK, dtype=int)
    out[top_shadow & bottom_shadow] = TEAL
    return out


def _sample_segmented_lengths(ray_count: int, max_shadow: int) -> np.ndarray | None:
    segment_count = random.randint(2, min(MAX_SEGMENTS, ray_count))
    cuts = sorted(random.sample(range(1, ray_count), segment_count - 1))
    starts = [0] + cuts
    ends = cuts + [ray_count]

    lengths = np.zeros(ray_count, dtype=int)
    segment_values = []
    has_zero = False
    has_positive = False

    for start, end in zip(starts, ends):
        if not has_zero and not has_positive:
            value = random.choice([0, random.randint(1, max_shadow)])
        elif not has_positive:
            value = random.randint(1, max_shadow)
        elif not has_zero:
            value = 0
        else:
            value = random.choice([0] + list(range(1, max_shadow + 1)))
        lengths[start:end] = value
        segment_values.append(value)
        has_zero |= value == 0
        has_positive |= value > 0

    if not (has_zero and has_positive):
        return None

    positive_values = {value for value in segment_values if value > 0}
    if max_shadow >= 2 and len(positive_values) < 2:
        positive_segments = [idx for idx, value in enumerate(segment_values) if value > 0]
        if positive_segments:
            idx = random.choice(positive_segments)
            start, end = starts[idx], ends[idx]
            alt = random.choice([value for value in range(1, max_shadow + 1) if value != segment_values[idx]])
            lengths[start:end] = alt

    if ray_count >= 5 and random.random() < 0.6:
        idx = random.randrange(ray_count)
        current = int(lengths[idx])
        options = [value for value in range(max_shadow + 1) if value != current and abs(value - current) <= 2]
        if options:
            trial = lengths.copy()
            trial[idx] = random.choice(options)
            if np.any(trial == 0) and np.any(trial > 0):
                lengths = trial

    return lengths


def _render_panel(height: int, width: int, edge: str, blocker_color: int, lengths: np.ndarray) -> np.ndarray:
    panel = np.full((height, width), BLACK, dtype=int)

    if edge == "top":
        panel[0, :] = YELLOW
        for col, shadow_len in enumerate(lengths):
            if shadow_len > 0:
                panel[height - int(shadow_len) - 1, col] = blocker_color
    elif edge == "bottom":
        panel[-1, :] = YELLOW
        for col, shadow_len in enumerate(lengths):
            if shadow_len > 0:
                panel[int(shadow_len), col] = blocker_color
    elif edge == "left":
        panel[:, 0] = YELLOW
        for row, shadow_len in enumerate(lengths):
            if shadow_len > 0:
                panel[row, width - int(shadow_len) - 1] = blocker_color
    else:
        panel[:, -1] = YELLOW
        for row, shadow_len in enumerate(lengths):
            if shadow_len > 0:
                panel[row, int(shadow_len)] = blocker_color

    return panel


def _edge_axis(edge: str) -> str:
    return "vertical" if edge in ("top", "bottom") else "horizontal"


def _panel_semantics_ok(panel: np.ndarray) -> bool:
    edge = _detect_lamp_edge(panel)
    blocker_mask = (panel != BLACK) & (panel != YELLOW)
    blocker_count = int(np.count_nonzero(blocker_mask))
    if blocker_count < 2:
        return False

    if edge in ("top", "bottom"):
        rays = [int(np.count_nonzero(blocker_mask[:, col])) for col in range(panel.shape[1])]
    else:
        rays = [int(np.count_nonzero(blocker_mask[row, :])) for row in range(panel.shape[0])]

    return any(count == 0 for count in rays) and any(count == 1 for count in rays)


def _semantic_categories_ok(top: np.ndarray, bottom: np.ndarray) -> bool:
    top_shadow = _panel_shadow_mask(top)
    bottom_shadow = _panel_shadow_mask(bottom)

    categories = (
        top_shadow & bottom_shadow,
        top_shadow & ~bottom_shadow,
        ~top_shadow & bottom_shadow,
        ~top_shadow & ~bottom_shadow,
    )
    if min(int(np.count_nonzero(mask)) for mask in categories) < 2:
        return False

    expected = np.where(top_shadow & bottom_shadow, TEAL, BLACK)
    if np.array_equal(expected, np.where(top_shadow, TEAL, BLACK)):
        return False
    if np.array_equal(expected, np.where(bottom_shadow, TEAL, BLACK)):
        return False

    return True


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*PANEL_SIZE_RANGE)
    width = random.randint(*PANEL_SIZE_RANGE)

    vertical_edge = random.choice(("top", "bottom"))
    horizontal_edge = random.choice(("left", "right"))
    separator_color, first_blocker_color, second_blocker_color = random.sample(NONSPECIAL_COLORS, 3)

    vertical_lengths = _sample_segmented_lengths(width, height - 2)
    horizontal_lengths = _sample_segmented_lengths(height, width - 2)
    if vertical_lengths is None or horizontal_lengths is None:
        return None

    vertical_panel = _render_panel(height, width, vertical_edge, first_blocker_color, vertical_lengths)
    horizontal_panel = _render_panel(height, width, horizontal_edge, second_blocker_color, horizontal_lengths)

    if random.choice((True, False)):
        top, bottom = vertical_panel, horizontal_panel
    else:
        top, bottom = horizontal_panel, vertical_panel

    if _edge_axis(_detect_lamp_edge(top)) == _edge_axis(_detect_lamp_edge(bottom)):
        return None
    if not _panel_semantics_ok(top) or not _panel_semantics_ok(bottom):
        return None
    if not _semantic_categories_ok(top, bottom):
        return None

    grid = np.vstack([top, np.full((1, width), separator_color, dtype=int), bottom])
    if not np.array_equal(main(grid), np.where(_panel_shadow_mask(top) & _panel_shadow_mask(bottom), TEAL, BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid common-shadow puzzle input")
