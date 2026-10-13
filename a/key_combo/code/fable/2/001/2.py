from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, components_mask


BG = 0
PALETTE = list(range(1, 10))
SEPARATOR_WIDTH = 1
PANEL_COUNT_RANGE = (3, 5)
PANEL_HEIGHT_RANGE = (7, 10)
PANEL_WIDTH_RANGE = (5, 8)
AREA_VALUE_RANGE = (6, 15)
MAX_GENERATION_ATTEMPTS = 800
MAX_PANEL_ATTEMPTS = 120
MAX_COMPONENT_ATTEMPTS = 80


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _candidate_panels(grid: np.ndarray, sep_color: int) -> list[np.ndarray] | None:
    height, width = map(int, grid.shape)
    sep_cols = [c for c in range(width) if np.all(grid[:, c] == int(sep_color))]
    runs = _group_consecutive(sep_cols)
    if not runs:
        return None

    sep_widths = [c1 - c0 + 1 for c0, c1 in runs]
    if len(set(sep_widths)) != 1:
        return None

    panel_spans: list[tuple[int, int]] = []
    cursor = 0
    for c0, c1 in runs:
        if cursor >= c0:
            return None
        panel_spans.append((cursor, c0))
        cursor = c1 + 1
    if cursor >= width:
        return None
    panel_spans.append((cursor, width))

    panel_widths = [stop - start for start, stop in panel_spans]
    if len(panel_spans) < 2 or min(panel_widths) <= 0 or len(set(panel_widths)) != 1:
        return None

    panels: list[np.ndarray] = []
    seen_colors: set[int] = set()
    for start, stop in panel_spans:
        panel = grid[:, start:stop].copy()
        fg_colors = colors_present(panel, exclude={BG})
        if len(fg_colors) != 1:
            return None
        fg_color = int(fg_colors[0])
        if fg_color == int(sep_color) or fg_color in seen_colors:
            return None
        seen_colors.add(fg_color)
        panels.append(panel)

    assert all(panel.shape == (height, panel_widths[0]) for panel in panels)
    return panels


def _extract_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in colors_present(grid, exclude={BG}):
        panels = _candidate_panels(grid, int(color))
        if panels is not None:
            candidates.append((int(color), panels))
    assert len(candidates) == 1, "expected one valid separator scheme"
    return candidates[0]


def _component_count(panel: np.ndarray) -> int:
    return len(components_mask(panel != BG, connectivity=4))


def _overlay_panels(panels: list[np.ndarray], order: tuple[int, ...] | list[int]) -> np.ndarray:
    out = np.zeros_like(panels[0])
    for idx in order:
        panel = panels[int(idx)]
        fill = (out == BG) & (panel != BG)
        out[fill] = panel[fill]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay equal-size side panels in ascending 4-component-count priority.

    Input structure:
    - The grid is split by full-height separator columns into several equal-size panels.
    - Each panel uses background 0 plus exactly one unique foreground color.
    - Different panels contain different numbers of 4-connected foreground components.

    Rule:
    - Ignore the separator columns and treat each panel's 0 cells as transparent.
    - Count the 4-connected foreground components in every panel.
    - Stack the panels from top to bottom in increasing component-count order and output one panel-sized image using the first non-zero color seen at each cell.

    Key insight:
    - Left-to-right panel position and total pixel count are decoys; only the number of disconnected foreground pieces determines layer priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep_color, panels = _extract_panels(g)
    counts = [_component_count(panel) for panel in panels]
    assert len(set(counts)) == len(counts), "panel component counts must be distinct"
    order = tuple(int(i) for i in np.argsort(np.array(counts), kind="stable"))
    return _overlay_panels(panels, order)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _sample_sizes(count: int, total_area: int) -> list[int]:
    sizes = [1] * int(count)
    for _ in range(int(total_area) - int(count)):
        sizes[random.randrange(count)] += 1
    sizes.sort(reverse=True)
    return sizes


def _choose_seed(blocked: np.ndarray) -> tuple[int, int] | None:
    free = np.argwhere(~blocked)
    if free.size == 0:
        return None
    center_r = (blocked.shape[0] - 1) / 2
    center_c = (blocked.shape[1] - 1) / 2
    coords = [tuple(map(int, rc)) for rc in free]
    weights = [1.0 / (1.0 + abs(r - center_r) + abs(c - center_c)) for r, c in coords]
    return random.choices(coords, weights=weights, k=1)[0]


def _grow_component(
    shape: tuple[int, int],
    blocked: np.ndarray,
    size: int,
    seed: tuple[int, int],
) -> set[tuple[int, int]] | None:
    sr, sc = map(int, seed)
    if blocked[sr, sc]:
        return None

    cells = {(sr, sc)}
    while len(cells) < int(size):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(shape, nr, nc) and not blocked[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None

        candidates = list(frontier)
        weights = []
        for r, c in candidates:
            adj = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            anchor_bias = 1.0 / (1.0 + abs(r - sr) + abs(c - sc))
            weights.append((adj + 1) ** 2 + anchor_bias)
        cells.add(random.choices(candidates, weights=weights, k=1)[0])
    return cells


def _reserve_with_padding(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    height, width = map(int, blocked.shape)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def _build_panel(
    height: int,
    width: int,
    color: int,
    component_count: int,
    total_area: int,
    anchor: tuple[int, int],
) -> np.ndarray | None:
    for _ in range(MAX_PANEL_ATTEMPTS):
        sizes = _sample_sizes(component_count, total_area)
        blocked = np.zeros((height, width), dtype=bool)
        fg_cells: set[tuple[int, int]] = set()

        if sizes:
            anchor_component = _grow_component((height, width), blocked, sizes[0], anchor)
            if anchor_component is None:
                continue
            fg_cells |= anchor_component
            _reserve_with_padding(blocked, anchor_component)

        ok = True
        for size in sizes[1:]:
            placed = False
            for _ in range(MAX_COMPONENT_ATTEMPTS):
                seed = _choose_seed(blocked)
                if seed is None:
                    break
                component = _grow_component((height, width), blocked, size, seed)
                if component is None:
                    continue
                fg_cells |= component
                _reserve_with_padding(blocked, component)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        panel = np.zeros((height, width), dtype=int)
        rows, cols = zip(*fg_cells)
        panel[rows, cols] = int(color)
        if int((panel != BG).sum()) != int(total_area):
            continue
        if _component_count(panel) != int(component_count):
            continue
        return panel
    return None


def _is_monotone(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _choose_layout() -> tuple[int, int, int]:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    feasible_widths = [
        width
        for width in range(PANEL_WIDTH_RANGE[0], PANEL_WIDTH_RANGE[1] + 1)
        if panel_count * width + (panel_count - 1) * SEPARATOR_WIDTH <= 30
    ]
    assert feasible_widths
    return (
        random.randint(*PANEL_HEIGHT_RANGE),
        random.choice(feasible_widths),
        panel_count,
    )


def _choose_area_targets(panel_count: int) -> list[int]:
    lo = max(AREA_VALUE_RANGE[0], panel_count + 1)
    hi = min(AREA_VALUE_RANGE[1], 30)
    values = sorted(random.sample(range(lo, hi + 1), panel_count))
    perm = list(range(panel_count))
    while (
        perm == list(range(panel_count))
        or perm == list(reversed(range(panel_count)))
        or perm[0] in {0, panel_count - 1}
    ):
        random.shuffle(perm)
    return [int(values[i]) for i in perm]


def _choose_input_order(panel_count: int) -> list[int]:
    order = list(range(panel_count))
    while (
        order == list(range(panel_count))
        or order == list(reversed(range(panel_count)))
        or order.index(0) in {0, panel_count - 1}
    ):
        random.shuffle(order)
    return order


def _semantic_checks(grid: np.ndarray) -> bool:
    sep_color, panels = _extract_panels(grid)
    if sep_color == BG:
        return False

    counts = [_component_count(panel) for panel in panels]
    areas = [int((panel != BG).sum()) for panel in panels]
    if len(set(counts)) != len(counts) or len(set(areas)) != len(areas):
        return False
    if _is_monotone(counts):
        return False

    count_order = tuple(int(i) for i in np.argsort(np.array(counts), kind="stable"))
    correct = _overlay_panels(panels, count_order)

    area_by_count = [areas[i] for i in count_order]
    if _is_monotone(area_by_count):
        return False

    input_order = tuple(range(len(panels)))
    reverse_input_order = tuple(reversed(input_order))
    area_asc = tuple(int(i) for i in np.argsort(np.array(areas), kind="stable"))
    area_desc = tuple(int(i) for i in np.argsort(-np.array(areas), kind="stable"))
    for alt in (input_order, reverse_input_order, area_asc, area_desc):
        if alt != count_order and np.array_equal(_overlay_panels(panels, alt), correct):
            return False

    filled = np.zeros_like(correct, dtype=bool)
    visible_layers = 0
    obscured_overlap = False
    for idx in count_order:
        mask = panels[idx] != BG
        if np.any(mask & filled):
            obscured_overlap = True
        newly_visible = mask & ~filled
        if np.any(newly_visible):
            visible_layers += 1
            filled |= newly_visible
    if not obscured_overlap:
        return False
    if visible_layers < min(len(panels), 3):
        return False

    return np.array_equal(main(grid), correct)


def _gen_attempt() -> np.ndarray | None:
    height, width, panel_count = _choose_layout()
    shared_anchor = (
        random.randint(max(1, height // 2 - 1), min(height - 2, height // 2 + 1)),
        random.randint(max(1, width // 2 - 1), min(width - 2, width // 2 + 1)),
    )

    colors = random.sample(PALETTE, panel_count + 1)
    sep_color = int(colors[0])
    panel_colors = [int(c) for c in colors[1:]]
    component_counts = list(range(1, panel_count + 1))
    areas_by_count = _choose_area_targets(panel_count)

    panels_by_count: list[np.ndarray] = []
    for count, area, color in zip(component_counts, areas_by_count, panel_colors):
        panel = _build_panel(height, width, color, count, area, shared_anchor)
        if panel is None:
            return None
        panels_by_count.append(panel)

    input_order = _choose_input_order(panel_count)
    panels = [panels_by_count[idx] for idx in input_order]

    grid_width = panel_count * width + (panel_count - 1) * SEPARATOR_WIDTH
    grid = np.zeros((height, grid_width), dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        grid[:, cursor : cursor + width] = panel
        cursor += width
        if idx != panel_count - 1:
            grid[:, cursor : cursor + SEPARATOR_WIDTH] = sep_color
            cursor += SEPARATOR_WIDTH

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-layer panel puzzle")
