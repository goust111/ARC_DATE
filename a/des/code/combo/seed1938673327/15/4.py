from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


OUTPUT_BG = 0
SEPARATOR_COLOR = 5
PANEL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

PANEL_COUNT_RANGE = (3, 7)
PANEL_HEIGHT_RANGE = (3, 8)
PANEL_WIDTH_RANGE = (3, 8)
SEPARATOR_WIDTH_RANGE = (1, 2)
BASE_PALETTE_RANGE = (2, 5)
OVERLAP_BRANCH_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 300


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    H, W = grid.shape
    candidates: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []

    for color in sorted(set(map(int, grid.reshape(-1)))):
        separator_mask = np.all(grid == color, axis=0)
        separator_runs = _runs(separator_mask)
        if not separator_runs:
            continue
        if separator_runs[0][0] == 0 or separator_runs[-1][1] == W:
            continue

        panel_spans: list[tuple[int, int]] = []
        sep_widths = []
        prev = 0
        ok = True
        for start, end in separator_runs:
            if start <= prev:
                ok = False
                break
            panel_spans.append((prev, start))
            sep_widths.append(end - start)
            prev = end
        if not ok or prev >= W:
            continue
        panel_spans.append((prev, W))

        widths = [end - start for start, end in panel_spans]
        if len(panel_spans) < 2 or min(widths) <= 0:
            continue
        if len(set(widths)) != 1 or len(set(sep_widths)) != 1:
            continue

        candidates.append((len(panel_spans), widths[0], -sep_widths[0], color, panel_spans))

    assert candidates, "expected uniform separator columns splitting equal panels"
    best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    panels = [grid[:, c0:c1].copy() for c0, c1 in best[4]]
    assert len({panel.shape for panel in panels}) == 1, "all panels must share one shape"
    assert panels[0].shape[0] == H, "panels must span the full height"
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each panel's pixel that disagrees with the majority color at its local coordinate.

    Input structure:
    - The grid is a horizontal sequence of equal-size panels.
    - Uniform separator columns split the panels.
    - Panels share one base pattern, with exactly one anomalous pixel per panel.

    Rule:
    - Compare each local coordinate across all panels.
    - When one color has a strict majority at that coordinate, every panel showing
      a different color contributes an anomaly there.
    - Return one panel-sized grid with black background and all anomaly pixels drawn
      in left-to-right panel order, so later panels overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _extract_panels(g)
    panel_count = len(panels)
    assert panel_count >= 3, "majority comparison needs at least three panels"

    panel_h, panel_w = panels[0].shape
    out = np.full((panel_h, panel_w), OUTPUT_BG, dtype=int)
    anomaly_counts = [0] * panel_count

    for r in range(panel_h):
        for c in range(panel_w):
            values = [int(panel[r, c]) for panel in panels]
            counts = Counter(values)
            if len(counts) == 1:
                continue

            majority_color, majority_count = counts.most_common(1)[0]
            assert majority_count > panel_count // 2, "varying positions need a strict majority"
            assert list(counts.values()).count(majority_count) == 1, "majority must be unique"

            for idx, value in enumerate(values):
                if value != majority_color:
                    out[r, c] = value
                    anomaly_counts[idx] += 1

    assert all(count == 1 for count in anomaly_counts), "each panel must contribute one anomaly"
    return out


def _make_base_panel(height: int, width: int, palette: list[int]) -> np.ndarray:
    panel = np.array(random.choices(palette, k=height * width), dtype=int).reshape(height, width)

    for c in range(width):
        if len(set(map(int, panel[:, c]))) == 1:
            r = random.randrange(height)
            choices = [color for color in palette if color != int(panel[r, c])]
            panel[r, c] = random.choice(choices)

    if len(set(map(int, panel.reshape(-1)))) == 1:
        panel[0, 0] = palette[0]
        panel[0, 1] = palette[-1]
    return panel


def _assign_anomaly_positions(height: int, width: int, panel_count: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(height) for c in range(width)]
    max_shared = (panel_count - 1) // 2

    if max_shared >= 2 and random.random() < OVERLAP_BRANCH_PROBABILITY:
        shared_coord = random.choice(coords)
        shared_count = random.randint(2, max_shared)
        shared_panels = set(random.sample(range(panel_count), shared_count))

        positions: list[tuple[int, int]] = [(-1, -1)] * panel_count
        for idx in shared_panels:
            positions[idx] = shared_coord

        remaining_panels = [idx for idx in range(panel_count) if idx not in shared_panels]
        remaining_coords = [coord for coord in coords if coord != shared_coord]
        for idx, coord in zip(remaining_panels, random.sample(remaining_coords, len(remaining_panels))):
            positions[idx] = coord
        return positions

    return random.sample(coords, panel_count)


def _build_panels(
    base_panel: np.ndarray, positions: list[tuple[int, int]]
) -> tuple[list[np.ndarray], np.ndarray]:
    panel_count = len(positions)
    panels = [base_panel.copy() for _ in range(panel_count)]
    out = np.full(base_panel.shape, OUTPUT_BG, dtype=int)

    by_coord: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, coord in enumerate(positions):
        by_coord[coord].append(idx)

    chosen_colors = [None] * panel_count
    for coord, panel_ids in by_coord.items():
        base_color = int(base_panel[coord])
        choices = [color for color in PANEL_COLORS if color != base_color]
        colors = [random.choice(choices) for _ in panel_ids]
        if len(panel_ids) >= 2 and len(set(colors)) == 1:
            alternatives = [color for color in choices if color != colors[-1]]
            colors[-1] = random.choice(alternatives)

        for idx, color in zip(panel_ids, colors):
            panels[idx][coord] = color
            chosen_colors[idx] = color

    for idx, coord in enumerate(positions):
        out[coord] = int(chosen_colors[idx])

    return panels, out


def _assemble_grid(panels: list[np.ndarray], separator_width: int) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    panel_count = len(panels)
    grid_w = panel_count * panel_w + (panel_count - 1) * separator_width
    grid = np.full((panel_h, grid_w), SEPARATOR_COLOR, dtype=int)

    col = 0
    for idx, panel in enumerate(panels):
        grid[:, col : col + panel_w] = panel
        col += panel_w
        if idx != panel_count - 1:
            col += separator_width
    return grid


def _attempt_generate() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    separator_width = random.randint(*SEPARATOR_WIDTH_RANGE)

    palette_size = random.randint(*BASE_PALETTE_RANGE)
    palette = random.sample(PANEL_COLORS, palette_size)
    base_panel = _make_base_panel(panel_h, panel_w, palette)
    positions = _assign_anomaly_positions(panel_h, panel_w, panel_count)
    panels, expected = _build_panels(base_panel, positions)
    grid = _assemble_grid(panels, separator_width)

    for panel in panels:
        if any(len(set(map(int, panel[:, c]))) == 1 for c in range(panel.shape[1])):
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-anomaly puzzle")
