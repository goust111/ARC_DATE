from __future__ import annotations

import random

import numpy as np


SEPARATOR_COLOR = 1  # blue
QUERY_COLOR = 9      # maroon
USABLE_COLORS = [color for color in range(10) if color not in {SEPARATOR_COLOR, QUERY_COLOR}]

PANEL_SIZE_RANGE = (4, 5)
PANEL_COUNT_RANGE = (4, 6)
STATIC_SHAPE_COUNT_RANGE = (2, 5)
DYNAMIC_GROUP_COUNT_RANGE = (2, 4)
STATIC_QUERY_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 400

_DYNAMIC_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _column_spans_for_color(g: np.ndarray, color: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    full_cols = np.all(g == int(color), axis=0)
    spans: list[tuple[int, int]] = []
    c = 0
    width = g.shape[1]
    while c < width:
        if full_cols[c]:
            c += 1
            continue
        start = c
        while c < width and not full_cols[c]:
            c += 1
        spans.append((start, c))
    return full_cols, spans


def _extract_panels(g: np.ndarray) -> list[np.ndarray]:
    height, _width = g.shape
    candidates: list[list[tuple[int, int]]] = []
    for color in map(int, np.unique(g)):
        full_cols, spans = _column_spans_for_color(g, color)
        if len(spans) < 2:
            continue

        widths = [end - start for start, end in spans]
        if len(set(widths)) != 1 or widths[0] != height:
            continue

        separated = True
        for (_, left_end), (right_start, _) in zip(spans, spans[1:]):
            if right_start <= left_end or not full_cols[left_end:right_start].all():
                separated = False
                break
        if separated:
            candidates.append(spans)

    assert len(candidates) == 1, "expected one separator color producing equal square panels"
    return [g[:, start:end].copy() for start, end in candidates[0]]


def _find_query_color(panels: list[np.ndarray]) -> int:
    seen_before = {int(v) for panel in panels[:-1] for v in np.unique(panel)}
    last_only = [int(v) for v in np.unique(panels[-1]) if int(v) not in seen_before]
    assert len(last_only) == 1, "expected one special query color in the last panel"
    return last_only[0]


def _predict_next_value(history: list[int]) -> int:
    if len(history) == 1:
        return int(history[0])
    for period in range(1, len(history)):
        if all(int(history[i]) == int(history[i - period]) for i in range(period, len(history))):
            return int(history[-period])
    raise AssertionError("query history does not expose a repeating color cycle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the hidden cells in the final panel of a horizontally concatenated panel sequence.

    Input structure:
    - The input is a row of equal square panels separated by full-height separator columns.
    - Panels depict the same spatial pattern at successive time steps.
    - Most positions stay unchanged; some positions follow a repeating color cycle.
    - In the last panel, query cells are replaced by a special marker color.

    Rule:
    - For each query cell, read the same coordinate across the previous panels.
    - Infer the shortest repeating color cycle visible in that history.
    - Replace the marker with the next color in that cycle.
    - Return only the completed final panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _extract_panels(g)
    assert len(panels) >= 2, "expected multiple panels"

    query_color = _find_query_color(panels)
    history_panels = panels[:-1]
    out = panels[-1].copy()

    query_cells = np.argwhere(out == int(query_color))
    assert query_cells.size > 0, "expected at least one query cell"
    for r, c in query_cells:
        history = [int(panel[int(r), int(c)]) for panel in history_panels]
        out[int(r), int(c)] = _predict_next_value(history)

    return out


def _make_base_panel(size: int, palette: list[int]) -> np.ndarray:
    bg = random.choice(palette)
    panel = np.full((size, size), bg, dtype=int)

    for _ in range(random.randint(*STATIC_SHAPE_COUNT_RANGE)):
        color = random.choice([c for c in palette if c != bg])
        kind = random.choice(("rect", "hline", "vline", "point"))
        if kind == "rect":
            h = random.randint(1, max(1, size // 2))
            w = random.randint(1, max(1, size // 2))
            r0 = random.randint(0, size - h)
            c0 = random.randint(0, size - w)
            panel[r0:r0 + h, c0:c0 + w] = color
        elif kind == "hline":
            r = random.randrange(size)
            w = random.randint(1, size - 1)
            c0 = random.randint(0, size - w)
            panel[r, c0:c0 + w] = color
        elif kind == "vline":
            c = random.randrange(size)
            h = random.randint(1, size - 1)
            r0 = random.randint(0, size - h)
            panel[r0:r0 + h, c] = color
        else:
            r = random.randrange(size)
            c = random.randrange(size)
            panel[r, c] = color

    if len(np.unique(panel)) == 1:
        r = random.randrange(size)
        c = random.randrange(size)
        panel[r, c] = random.choice([color for color in palette if color != int(panel[r, c])])
    return panel


def _place_dynamic_cells(size: int, occupied: np.ndarray) -> list[tuple[int, int]] | None:
    shape = random.choice(_DYNAMIC_SHAPES)
    max_r = size - max(r for r, _c in shape)
    max_c = size - max(c for _r, c in shape)

    placements: list[list[tuple[int, int]]] = []
    for r0 in range(max_r):
        for c0 in range(max_c):
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if all(not occupied[r, c] for r, c in cells):
                placements.append(cells)
    if not placements:
        return None

    cells = random.choice(placements)
    for r, c in cells:
        occupied[r, c] = True
    return cells


def _apply_dynamic_groups(
    base: np.ndarray,
    panel_count: int,
    palette: list[int],
) -> tuple[list[np.ndarray], list[dict], set[tuple[int, int]]] | None:
    history_len = panel_count - 1
    max_period = min(4, history_len - 1)
    feasible_periods = list(range(2, max_period + 1))
    if not feasible_periods:
        return None

    panels = [base.copy() for _ in range(panel_count)]
    occupied = np.zeros(base.shape, dtype=bool)
    groups = []

    periods = [max(feasible_periods)]
    extra_count = random.randint(*DYNAMIC_GROUP_COUNT_RANGE) - 1
    periods.extend(random.choice(feasible_periods) for _ in range(extra_count))
    random.shuffle(periods)

    for period in periods:
        cells = _place_dynamic_cells(base.shape[0], occupied)
        if cells is None:
            return None
        cycle = random.sample(palette, period)
        phase = random.randrange(period)
        groups.append({"cells": cells, "cycle": cycle, "phase": phase, "period": period})
        for t, panel in enumerate(panels):
            color = int(cycle[(phase + t) % period])
            for r, c in cells:
                panel[r, c] = color

    return panels, groups, {cell for group in groups for cell in group["cells"]}


def _break_uniform_columns(
    panels: list[np.ndarray],
    reserved: set[tuple[int, int]],
    palette: list[int],
) -> bool:
    size = panels[0].shape[0]
    for c in range(size):
        if not any(np.all(panel[:, c] == panel[0, c]) for panel in panels):
            continue

        free_rows = [r for r in range(size) if (r, c) not in reserved]
        fixed = False
        for r in free_rows:
            original = [int(panel[r, c]) for panel in panels]
            for accent in palette:
                for panel in panels:
                    panel[r, c] = int(accent)
                if all(not np.all(panel[:, c] == panel[0, c]) for panel in panels):
                    fixed = True
                    break
                for panel, value in zip(panels, original):
                    panel[r, c] = value
            if fixed:
                break
        if not fixed:
            return False
    return True


def _choose_query_cells(
    panels: list[np.ndarray],
    groups: list[dict],
    dynamic_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    highlight_period = max(int(group["period"]) for group in groups)
    highlight_indices = [i for i, group in enumerate(groups) if int(group["period"]) == highlight_period]
    chosen_groups = {random.choice(highlight_indices)}

    other_indices = [i for i in range(len(groups)) if i not in chosen_groups]
    extra_groups = random.randint(0, min(2, len(other_indices)))
    if extra_groups:
        chosen_groups.update(random.sample(other_indices, extra_groups))

    query_cells: set[tuple[int, int]] = set()
    for idx in chosen_groups:
        cells = list(groups[idx]["cells"])
        query_count = random.randint(1, min(2, len(cells)))
        query_cells.update(random.sample(cells, query_count))

    if random.random() < STATIC_QUERY_PROB:
        size = panels[0].shape[0]
        static_candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) not in dynamic_cells
        ]
        if static_candidates:
            query_cells.add(random.choice(static_candidates))

    return query_cells


def _compose_input(panels: list[np.ndarray], query_cells: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    target = panels[-1].copy()
    masked_last = target.copy()
    for r, c in query_cells:
        masked_last[int(r), int(c)] = QUERY_COLOR

    shown_panels = panels[:-1] + [masked_last]
    size = shown_panels[0].shape[0]
    width = len(shown_panels) * size + (len(shown_panels) - 1)
    grid = np.full((size, width), SEPARATOR_COLOR, dtype=int)

    col = 0
    for i, panel in enumerate(shown_panels):
        grid[:, col:col + size] = panel
        col += size
        if i != len(shown_panels) - 1:
            col += 1

    return grid, target


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*PANEL_SIZE_RANGE)
    panel_count = random.randint(*PANEL_COUNT_RANGE)

    palette_size = random.randint(4, min(6, len(USABLE_COLORS)))
    palette = random.sample(USABLE_COLORS, palette_size)
    base = _make_base_panel(size, palette)

    built = _apply_dynamic_groups(base, panel_count, palette)
    if built is None:
        return None
    panels, groups, dynamic_cells = built

    if not _break_uniform_columns(panels, dynamic_cells, palette):
        return None

    query_cells = _choose_query_cells(panels, groups, dynamic_cells)
    if not query_cells:
        return None

    inp, expected = _compose_input(panels, query_cells)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-sequence input")
