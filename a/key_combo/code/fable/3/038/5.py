from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
RED = 2
CYAN = 8
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 9)

PANEL_COUNT_RANGE = (3, 5)
PANEL_SIDE_VALUES = (4, 5, 6, 7)
COMMON_POINT_RANGE = (2, 4)
EXTRA_POINT_RANGE = (1, 3)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _panel_offsets(panel: np.ndarray) -> set[tuple[int, int]]:
    red_coords = np.argwhere(panel == RED)
    assert red_coords.shape == (1, 2), f"expected one red anchor, got {red_coords.shape[0]}"
    anchor_r, anchor_c = map(int, red_coords[0])

    blank = Counter(map(int, panel.ravel())).most_common(1)[0][0]
    assert blank != RED, "panel blank color cannot be the anchor color"

    scatter = np.argwhere((panel != blank) & (panel != RED))
    assert scatter.size > 0, "each panel must contain at least one scatter point"
    return {(int(r) - anchor_r, int(c) - anchor_c) for r, c in scatter}


def main(grid: np.ndarray) -> np.ndarray:
    """Take the anchor-aligned intersection of the point patterns across rectangular panels.

    Input structure:
    - The most common color is a separator that splits the grid into several rectangular panels of different sizes.
    - Each panel is a fully filled rectangle with one red anchor pixel and sparse scatter points in a panel-specific color over a blank panel color.

    Rule:
    - Treat the red anchor inside each panel as the origin and record every scatter point by its relative coordinate.
    - Keep only the relative coordinates that occur in every panel.
    - Output the smallest black grid containing the red origin and those shared coordinates, drawing the shared points in cyan.

    Key insight:
    - The panels must be aligned by their red content anchors rather than by their bounding boxes, and the answer is the intersection after alignment, not the overlay union.
    """
    g = np.asarray(grid, dtype=int)
    separator = _get_bg(g)

    panels = _find_components(g != separator)
    assert len(panels) >= 2, "expected at least two panels"

    point_sets: list[set[tuple[int, int]]] = []
    for coords in panels:
        r0, c0, r1, c1 = _bbox(coords)
        panel = g[r0 : r1 + 1, c0 : c1 + 1]
        assert panel.size == coords.shape[0], "each panel must fill its rectangle"
        assert np.all(panel != separator), "separator cells cannot appear inside a panel"
        point_sets.append(_panel_offsets(panel))

    common = set.intersection(*point_sets)
    points = {(0, 0)} | common
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    origin_r = -min_r
    origin_c = -min_c
    for dr, dc in common:
        out[origin_r + dr, origin_c + dc] = CYAN
    out[origin_r, origin_c] = RED
    return out


def _feasible_offsets(height: int, width: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {
        (dr, dc)
        for dr in range(-ar, height - ar)
        for dc in range(-ac, width - ac)
        if (dr, dc) != (0, 0)
    }


def _choose_common_offsets(
    candidates: set[tuple[int, int]], min_reserved: int
) -> set[tuple[int, int]] | None:
    candidate_list = list(candidates)
    max_common = min(COMMON_POINT_RANGE[1], len(candidate_list) - min_reserved)
    if max_common < COMMON_POINT_RANGE[0]:
        return None

    for _ in range(80):
        count = random.randint(COMMON_POINT_RANGE[0], max_common)
        chosen = set(random.sample(candidate_list, count))
        rows = [0] + [dr for dr, _ in chosen]
        cols = [0] + [dc for _, dc in chosen]
        if max(rows) > min(rows) and max(cols) > min(cols):
            return chosen
    return None


def _build_panel_specs() -> list[dict] | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    size_pool = [(h, w) for h in PANEL_SIDE_VALUES for w in PANEL_SIDE_VALUES]
    sizes = random.sample(size_pool, panel_count)

    anchors: list[tuple[int, int]] = []
    feasible_sets: list[set[tuple[int, int]]] = []
    for height, width in sizes:
        anchors.append((random.randint(1, height - 2), random.randint(1, width - 2)))
        feasible_sets.append(_feasible_offsets(height, width, anchors[-1]))

    if len(set(anchors)) == 1:
        return None

    common_candidates = set.intersection(*feasible_sets)
    if len(common_candidates) < panel_count + COMMON_POINT_RANGE[0]:
        return None

    common = _choose_common_offsets(common_candidates, min_reserved=panel_count)
    if common is None:
        return None

    available = list(common_candidates - common)
    random.shuffle(available)
    if len(available) < panel_count:
        return None

    max_scatter = [min(6, height * width // 3 + 1) for height, width in sizes]
    extras = [set() for _ in range(panel_count)]
    cursor = 0

    for idx in range(panel_count):
        extras[idx].add(available[cursor])
        cursor += 1

    for offset in available[cursor:]:
        eligible = [
            idx
            for idx in range(panel_count)
            if len(common) + len(extras[idx]) < max_scatter[idx]
        ]
        if not eligible or random.random() > 0.35:
            continue
        subset_size = random.randint(1, min(len(eligible), panel_count - 1))
        for idx in random.sample(eligible, subset_size):
            extras[idx].add(offset)

    specs = []
    for idx, ((height, width), anchor) in enumerate(zip(sizes, anchors)):
        offsets = common | extras[idx]
        if len(offsets) < len(common) + EXTRA_POINT_RANGE[0]:
            return None
        if len(offsets) > len(common) + EXTRA_POINT_RANGE[1] + 1:
            return None
        specs.append({"size": (height, width), "anchor": anchor, "offsets": offsets})

    if set.intersection(*(spec["offsets"] for spec in specs)) != common:
        return None

    return specs


def _render_panel(
    size: tuple[int, int], anchor: tuple[int, int], offsets: set[tuple[int, int]], color: int
) -> np.ndarray:
    height, width = size
    ar, ac = anchor
    panel = np.full((height, width), BLACK, dtype=int)
    panel[ar, ac] = RED
    for dr, dc in offsets:
        panel[ar + dr, ac + dc] = color
    return panel


def _split_rows(n_panels: int) -> list[list[int]]:
    order = list(range(n_panels))
    random.shuffle(order)

    if n_panels >= 4:
        split = 2 if n_panels == 4 else random.choice((2, 3))
        return [order[:split], order[split:]]
    if n_panels == 3 and random.random() < 0.55:
        split = random.choice((1, 2))
        return [order[:split], order[split:]]
    return [order]


def _place_panels(panels: list[np.ndarray], separator: int) -> np.ndarray | None:
    rows = _split_rows(len(panels))
    row_gaps = [[random.randint(2, 4) for _ in range(len(row) - 1)] for row in rows]
    band_heights = [max(panels[idx].shape[0] for idx in row) + random.randint(0, 2) for row in rows]
    row_widths = [
        sum(panels[idx].shape[1] for idx in row) + sum(row_gaps[row_idx])
        for row_idx, row in enumerate(rows)
    ]
    gap_y = random.randint(2, 4) if len(rows) == 2 else 0
    total_area = sum(panel.shape[0] * panel.shape[1] for panel in panels)

    for _ in range(120):
        left = random.randint(1, 5)
        right = random.randint(1, 5)
        top = random.randint(1, 5)
        bottom = random.randint(1, 5)
        width = max(row_widths) + left + right
        height = sum(band_heights) + gap_y * (len(rows) - 1) + top + bottom
        if width > 30 or height > 30:
            continue
        if width * height <= 2 * total_area:
            continue

        grid = np.full((height, width), separator, dtype=int)
        y = top
        for row_idx, row in enumerate(rows):
            x = left + random.randint(0, width - left - right - row_widths[row_idx])
            for panel_pos, panel_idx in enumerate(row):
                panel = panels[panel_idx]
                ph, pw = panel.shape
                py = y + random.randint(0, band_heights[row_idx] - ph)
                grid[py : py + ph, x : x + pw] = panel
                if panel_pos < len(row) - 1:
                    x += pw + row_gaps[row_idx][panel_pos]
            y += band_heights[row_idx] + gap_y

        counts = Counter(map(int, grid.ravel()))
        if counts[separator] <= max(count for color, count in counts.items() if color != separator):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        specs = _build_panel_specs()
        if specs is None:
            continue

        chosen_colors = random.sample(INPUT_COLORS, len(specs) + 1)
        separator = chosen_colors[0]
        panels = [
            _render_panel(spec["size"], spec["anchor"], spec["offsets"], color)
            for spec, color in zip(specs, chosen_colors[1:])
        ]

        grid = _place_panels(panels, separator)
        if grid is None:
            continue

        out = main(grid)
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 5")
