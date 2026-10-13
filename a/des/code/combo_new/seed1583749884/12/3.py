from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEPARATOR = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300


def _consecutive_groups(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev + 1))
        start = prev = value
    groups.append((start, prev + 1))
    return groups


def _panel_spans(width: int, sep_groups: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep_start, sep_end in sep_groups:
        if sep_start <= start:
            return None
        spans.append((start, sep_start))
        start = sep_end
    if start >= width:
        return None
    spans.append((start, width))
    if len(spans) not in (3, 4):
        return None
    panel_widths = {end - begin for begin, end in spans}
    if len(panel_widths) != 1:
        return None
    return spans


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], int]:
    candidates: list[tuple[int, list[tuple[int, int]], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        solid_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
        if len(solid_cols) < 2:
            continue
        spans = _panel_spans(grid.shape[1], _consecutive_groups(solid_cols))
        if spans is None:
            continue

        panels = [grid[:, start:end] for start, end in spans]
        counts = [int(np.count_nonzero(panel != BG)) for panel in panels]
        max_count = max(counts)
        if max_count == 0 or counts.count(max_count) != 1:
            continue

        ref_idx = counts.index(max_count)
        reference = panels[ref_idx]
        valid = True
        for idx, panel in enumerate(panels):
            if idx == ref_idx:
                continue
            if counts[idx] >= max_count:
                valid = False
                break
            if np.any((panel != BG) & (panel != reference)):
                valid = False
                break
        if valid:
            candidates.append((color, spans, ref_idx))

    assert len(candidates) == 1, "expected one valid separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore every damaged panel by copying the unique complete pattern across the separator columns.

    Input structure:
    - Black is the background color.
    - Solid orange-like separator columns split the grid into 3 or 4 equal-width panels.
    - Each panel contains the same multicolor pattern on black, but all except one panel have some pattern pixels erased to black.

    Rule:
    - Identify the separator columns and extract the equal-width panels between them.
    - The panel with the most non-black pixels is the complete reference panel.
    - For every other panel, fill each black cell whose matching reference cell is colored, while leaving the reference panel and separator columns unchanged.

    Key insight:
    - The complete panel is not marked directly; it is the only copy whose foreground pixel count is maximal, so the task reduces to template completion by aligned coordinate transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    _sep_color, spans, ref_idx = _find_layout(g)
    panels = [g[:, start:end] for start, end in spans]
    reference = panels[ref_idx]

    out = g.copy()
    for idx, (start, end) in enumerate(spans):
        if idx == ref_idx:
            continue
        panel = out[:, start:end]
        fill_mask = (panel == BG) & (reference != BG)
        panel[fill_mask] = reference[fill_mask]
        out[:, start:end] = panel
    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _grow_connected_mask(height: int, width: int, cell_count: int) -> np.ndarray | None:
    for _ in range(80):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], height, width))
        while len(cells) < cell_count and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], height, width):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != cell_count:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _colorize_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = np.argwhere(mask)
    seed_ids = random.sample(range(len(coords)), len(colors))
    seeds = [tuple(map(int, coords[idx])) for idx in seed_ids]

    patch = np.full(mask.shape, BG, dtype=int)
    for r, c in coords:
        best_idx = min(
            range(len(seeds)),
            key=lambda idx: (abs(int(r) - seeds[idx][0]) + abs(int(c) - seeds[idx][1]), idx),
        )
        patch[int(r), int(c)] = colors[best_idx]
    return patch


def _make_template(panel_h: int, panel_w: int) -> np.ndarray | None:
    if panel_h < 5 or panel_w < 5:
        return None

    bbox_h = random.randint(3, panel_h - 2)
    bbox_w = random.randint(3, panel_w - 2)
    r0 = random.randint(1, panel_h - bbox_h - 1)
    c0 = random.randint(1, panel_w - bbox_w - 1)

    area = bbox_h * bbox_w
    min_cells = max(6, area // 3)
    max_cells = min(area - 1, (2 * area) // 3)
    if min_cells > max_cells:
        return None

    mask = _grow_connected_mask(bbox_h, bbox_w, random.randint(min_cells, max_cells))
    if mask is None:
        return None

    color_count = random.randint(2, min(4, int(mask.sum())))
    patch = _colorize_mask(mask, random.sample(PALETTE, color_count))

    panel = np.full((panel_h, panel_w), BG, dtype=int)
    panel[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = patch
    return panel


def _assemble_panels(panels: list[np.ndarray], sep_width: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if idx:
            pieces.append(np.full((panel.shape[0], sep_width), SEPARATOR, dtype=int))
        pieces.append(panel)
    return np.hstack(pieces)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice([3, 4])
    sep_width = random.choice([1, 2])
    max_panel_w = (30 - (panel_count - 1) * sep_width) // panel_count
    if max_panel_w < 5:
        return None

    panel_h = random.randint(5, 12)
    panel_w = random.randint(5, min(9, max_panel_w))
    template = _make_template(panel_h, panel_w)
    if template is None:
        return None

    fg_coords = [tuple(map(int, pt)) for pt in np.argwhere(template != BG)]
    if len(fg_coords) < 6:
        return None

    reference_index = random.randrange(panel_count)
    damaged_panels: list[np.ndarray] = []
    max_holes = max(1, min(5, len(fg_coords) // 3))
    for idx in range(panel_count):
        panel = template.copy()
        if idx != reference_index:
            hole_count = random.randint(1, max_holes)
            for r, c in random.sample(fg_coords, hole_count):
                panel[r, c] = BG
        damaged_panels.append(panel)

    inp = _assemble_panels(damaged_panels, sep_width)
    expected = _assemble_panels([template.copy() for _ in range(panel_count)], sep_width)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
