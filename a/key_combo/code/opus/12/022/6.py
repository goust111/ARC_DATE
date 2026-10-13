from __future__ import annotations

import random

import numpy as np


SEPARATOR = 2
PANEL_BG = 5
LANE_MAP = {8: 1, 1: 2, 4: 3, 3: 4}
LANE_COLORS = tuple(LANE_MAP)
MARK_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    rows = _separator_rows(grid)
    assert len(rows) == 2, "expected exactly two separator rows"
    assert rows[0] + 1 < rows[1], "separator rows must leave a middle panel"
    panels = [
        grid[: rows[0], :].copy(),
        grid[rows[0] + 1 : rows[1], :].copy(),
        grid[rows[1] + 1 :, :].copy(),
    ]
    heights = {panel.shape[0] for panel in panels}
    widths = {panel.shape[1] for panel in panels}
    assert len(heights) == 1 and len(widths) == 1, "panels must share one shape"
    assert next(iter(heights)) > 0 and next(iter(widths)) > 4
    return panels


def _move_panel(panel: np.ndarray) -> np.ndarray:
    moved = np.array(panel, dtype=int, copy=True)
    h, _w = moved.shape
    for r in range(h):
        color = int(moved[r, 0])
        if color == PANEL_BG:
            continue
        assert color in LANE_MAP, "left-edge pixels must use the coded colors"
        moved[r, 0] = PANEL_BG
        moved[r, LANE_MAP[color]] = color
    return moved


def _panel_mark_count(panel: np.ndarray) -> int:
    return int(np.count_nonzero(panel != PANEL_BG))


def _overlay_panels(panels: list[np.ndarray]) -> np.ndarray:
    counts = [_panel_mark_count(panel) for panel in panels]
    assert len(set(counts)) == len(counts), "panel mark counts must be unique"
    order = sorted(range(len(panels)), key=lambda idx: counts[idx], reverse=True)

    out = np.full_like(panels[0], PANEL_BG)
    for idx in order:
        mask = panels[idx] != PANEL_BG
        write_mask = mask & (out == PANEL_BG)
        out[write_mask] = panels[idx][write_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder each panel's left-edge coded pixels, then overlay the panels with count-priority transparency.

    Input structure:
    - The grid is divided by two full red separator rows into three equal-sized horizontal panels.
    - Each panel has a grey background with sparse colored marks, and its leftmost column contains 1-3 coded edge pixels.
    - Edge-pixel colors come from teal, blue, yellow, and green, while other non-grey marks can appear elsewhere in the panel.

    Rule:
    - Split the grid on the separator rows and process each panel independently.
    - In each panel, erase every non-grey left-edge pixel and redraw it on the same row in its fixed lane:
      teal->col 1, blue->col 2, yellow->col 3, green->col 4; the moved pixel overwrites any existing mark there.
    - After all panels are reordered, transparently overlay them into one grey panel, resolving coordinate conflicts
      by keeping the pixel from the panel with the larger total number of non-grey marks.

    Key insight:
    - The answer is not a direct overlay of the input panels: each panel must first be internally rearranged using
      the color-coded left-edge lanes, and the later merge priority depends on whole-panel density rather than panel order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels = _split_panels(g)
    moved = [_move_panel(panel) for panel in panels]
    return _overlay_panels(moved)


def _choose_edge_plan(
    height: int, count_by_panel: list[int]
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, tuple[int, int]]]]:
    top, mid, low = sorted(range(3), key=lambda idx: count_by_panel[idx], reverse=True)
    rows = random.sample(range(height), 2)
    colors = [random.choice(LANE_COLORS), random.choice(LANE_COLORS)]
    conflicts = [
        (rows[0], colors[0], (top, low)),
        (rows[1], colors[1], (mid, low)),
    ]

    edges: list[list[tuple[int, int]]] = [[] for _ in range(3)]
    used_rows = [set() for _ in range(3)]
    for row, color, participants in conflicts:
        for panel_idx in participants:
            edges[panel_idx].append((row, color))
            used_rows[panel_idx].add(row)

    for panel_idx in range(3):
        max_total = 3
        min_total = max(1, len(edges[panel_idx]))
        total = random.randint(min_total, max_total)
        extra_needed = total - len(edges[panel_idx])
        free_rows = [r for r in range(height) if r not in used_rows[panel_idx]]
        random.shuffle(free_rows)
        for row in free_rows[:extra_needed]:
            edges[panel_idx].append((row, random.choice(LANE_COLORS)))
        edges[panel_idx].sort()
    return edges, conflicts


def _sample_counts(min_required: list[int], max_count: int) -> list[int] | None:
    choices = list(range(max(max(min_required), 4), max_count + 1))
    if len(choices) < 3:
        return None
    picked = random.sample(choices, 3)
    random.shuffle(picked)
    if any(picked[i] < min_required[i] for i in range(3)):
        return None
    return picked


def _fill_transformed_panel(
    height: int,
    width: int,
    target_count: int,
    edges: list[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    panel = np.full((height, width), PANEL_BG, dtype=int)
    reserved_targets: set[tuple[int, int]] = set()
    for row, color in edges:
        coord = (int(row), int(LANE_MAP[color]))
        panel[coord] = int(color)
        reserved_targets.add(coord)

    current = _panel_mark_count(panel)
    needed = target_count - current
    assert needed >= 0

    coords = [(r, c) for r in range(height) for c in range(1, width) if (r, c) not in reserved_targets]
    random.shuffle(coords)
    for r, c in coords[:needed]:
        if random.random() < 0.35:
            color = random.choice(LANE_COLORS)
        else:
            color = random.choice(MARK_COLORS)
        panel[r, c] = color

    assert _panel_mark_count(panel) == target_count
    return panel, reserved_targets


def _reverse_to_input(
    transformed: np.ndarray,
    edges: list[tuple[int, int]],
    overwrite_rows: set[tuple[int, int]],
) -> np.ndarray:
    inp = np.full_like(transformed, PANEL_BG)
    reserved_targets = {(int(row), int(LANE_MAP[color])) for row, color in edges}
    mask = transformed != PANEL_BG
    for r, c in np.argwhere(mask):
        coord = (int(r), int(c))
        if coord not in reserved_targets:
            inp[coord] = transformed[coord]

    for row, color in edges:
        target = (int(row), int(LANE_MAP[color]))
        inp[row, 0] = int(color)
        if target in overwrite_rows:
            options = [col for col in MARK_COLORS if col != color]
            inp[target] = random.choice(options)
        else:
            inp[target] = PANEL_BG
    return inp


def _compose_input(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    grid = np.full((3 * h + 2, w), PANEL_BG, dtype=int)
    grid[:h, :] = panels[0]
    grid[h, :] = SEPARATOR
    grid[h + 1 : 2 * h + 1, :] = panels[1]
    grid[2 * h + 1, :] = SEPARATOR
    grid[2 * h + 2 :, :] = panels[2]
    return grid


def _conflict_cells(panels: list[np.ndarray]) -> set[tuple[int, int]]:
    conflicts: set[tuple[int, int]] = set()
    h, w = panels[0].shape
    for r in range(h):
        for c in range(w):
            hits = sum(panel[r, c] != PANEL_BG for panel in panels)
            if hits >= 2:
                conflicts.add((r, c))
    return conflicts


def _validate_structure(inp: np.ndarray, transformed: list[np.ndarray], conflicts: list[tuple[int, int, tuple[int, int]]]) -> bool:
    sep_rows = _separator_rows(inp)
    if len(sep_rows) != 2:
        return False
    try:
        panels = _split_panels(inp)
    except AssertionError:
        return False

    for panel in panels:
        left_colors = [int(panel[r, 0]) for r in range(panel.shape[0]) if int(panel[r, 0]) != PANEL_BG]
        if not (1 <= len(left_colors) <= 3):
            return False
        if any(color not in LANE_MAP for color in left_colors):
            return False
        if not np.any(panel[:, 1:] != PANEL_BG):
            return False
        if np.any(panel[:, 1:] == SEPARATOR):
            return False

    counts = [_panel_mark_count(panel) for panel in transformed]
    if len(set(counts)) != 3:
        return False

    conflict_cells = _conflict_cells(transformed)
    if len(conflict_cells) < 2:
        return False
    for row, color, _participants in conflicts:
        if (row, LANE_MAP[color]) not in conflict_cells:
            return False
    return True


def _expected_output_from_input(inp: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    panels = _split_panels(inp)
    transformed = [_move_panel(panel) for panel in panels]
    return transformed, _overlay_panels(transformed)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(5, 8)
    width = random.randint(10, 18)
    max_count = min(10, height * (width - 1) - 1)
    min_required = [1, 1, 2]
    count_by_panel = _sample_counts(min_required, max_count)
    if count_by_panel is None:
        return None

    edges_by_panel, lane_conflicts = _choose_edge_plan(height, count_by_panel)
    min_after_edges = [len(edges) + 1 for edges in edges_by_panel]
    if any(count_by_panel[idx] < min_after_edges[idx] for idx in range(3)):
        return None

    transformed_panels: list[np.ndarray] = []
    reserved_by_panel: list[set[tuple[int, int]]] = []
    for idx in range(3):
        panel, reserved = _fill_transformed_panel(height, width, count_by_panel[idx], edges_by_panel[idx])
        transformed_panels.append(panel)
        reserved_by_panel.append(reserved)

    overwrite_budget = 1 if random.random() < 0.7 else 0
    all_targets = [
        (panel_idx, int(row), int(LANE_MAP[color]), int(color))
        for panel_idx, edges in enumerate(edges_by_panel)
        for row, color in edges
    ]
    random.shuffle(all_targets)
    overwrite_targets = set()
    for panel_idx, row, col, color in all_targets:
        if overwrite_budget <= 0:
            break
        if transformed_panels[panel_idx][row, col] != color:
            continue
        overwrite_targets.add((panel_idx, row, col))
        overwrite_budget -= 1

    input_panels = []
    for idx in range(3):
        overwrites = {(row, col) for panel_idx, row, col in overwrite_targets if panel_idx == idx}
        input_panels.append(_reverse_to_input(transformed_panels[idx], edges_by_panel[idx], overwrites))

    inp = _compose_input(input_panels)
    transformed, expected = _expected_output_from_input(inp)

    if not _validate_structure(inp, transformed, lane_conflicts):
        return None
    if [_panel_mark_count(panel) for panel in transformed] != count_by_panel:
        return None
    if not np.array_equal(expected, _overlay_panels(transformed_panels)):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp[: height, :], expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
