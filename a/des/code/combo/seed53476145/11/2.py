from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (9, 16)
PANEL_COUNT_RANGE = (2, 4)
PANEL_WIDTH_MIN = 5
PANEL_WIDTH_MAX = 10
SEPARATOR_WIDTH_CHOICES = (1, 2)
GRID_WIDTH_MAX = 30

OBJECT_COUNT_RANGE = (2, 4)
RECT_HEIGHT_MAX = 6
RECT_WIDTH_MAX = 6
GENERATION_ATTEMPTS = 400
PANEL_ATTEMPTS = 120


def _dominant(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _background_color(grid: np.ndarray) -> int:
    corners = np.array(
        [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]],
        dtype=int,
    )
    corner_colors, corner_counts = np.unique(corners, return_counts=True)
    winners = np.flatnonzero(corner_counts == corner_counts.max())
    if len(winners) == 1:
        return int(corner_colors[int(winners[0])])
    return _dominant(grid)


def _separator_columns(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    cols: list[int] = []
    colors: set[int] = set()
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            cols.append(int(c))
            colors.add(int(col[0]))
    assert cols, "expected at least one full-height separator column"
    assert len(colors) == 1, "separator columns must have a unique color"
    return next(iter(colors)), cols


def _panel_spans(width: int, separator_cols: list[int]) -> list[tuple[int, int]]:
    sep_mask = np.zeros(width, dtype=bool)
    sep_mask[separator_cols] = True

    spans: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if sep_mask[c]:
            c += 1
            continue
        start = c
        while c < width and not sep_mask[c]:
            c += 1
        spans.append((int(start), int(c - 1)))

    assert len(spans) >= 2, "expected multiple panels"
    return spans


def _solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "object mask must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = int(coords.shape[0])
    rect_area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
    assert area == rect_area, "every object must be a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1), area


def _panel_objects(panel: np.ndarray, bg: int, col_offset: int) -> list[dict]:
    objects: list[dict] = []
    for color in np.unique(panel):
        color = int(color)
        if color == bg:
            continue
        r0, c0, r1, c1, area = _solid_rectangle(panel == color)
        objects.append(
            {
                "color": color,
                "area": area,
                "bbox": (r0, c0 + col_offset, r1, c1 + col_offset),
            }
        )

    assert 2 <= len(objects) <= 4, "each panel must contain 2-4 objects"
    areas = [int(obj["area"]) for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be unique within a panel"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-delimited panel, let the largest rectangle absorb every
    smaller rectangle by expanding to the bounding box covering all objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _background_color(g)
    _sep_color, sep_cols = _separator_columns(g, bg)
    spans = _panel_spans(g.shape[1], sep_cols)

    out = g.copy()
    for c0, c1 in spans:
        objects = _panel_objects(g[:, c0 : c1 + 1], bg, c0)
        largest = max(objects, key=lambda obj: int(obj["area"]))
        r0 = min(int(obj["bbox"][0]) for obj in objects)
        c0_all = min(int(obj["bbox"][1]) for obj in objects)
        r1 = max(int(obj["bbox"][2]) for obj in objects)
        c1_all = max(int(obj["bbox"][3]) for obj in objects)
        out[r0 : r1 + 1, c0_all : c1_all + 1] = int(largest["color"])

    return out


def _bounded_partition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if total < parts * lo or total > parts * hi:
        return None

    values = [int(lo)] * parts
    capacities = [int(hi - lo)] * parts
    remaining = int(total - parts * lo)
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        idx = random.choice(choices)
        values[idx] += 1
        capacities[idx] -= 1
        remaining -= 1
    random.shuffle(values)
    return values


def _rectangle_catalog(height: int, width: int) -> dict[int, list[tuple[int, int]]]:
    max_h = min(RECT_HEIGHT_MAX, height - 2)
    max_w = min(RECT_WIDTH_MAX, width)
    areas: dict[int, list[tuple[int, int]]] = {}
    for rect_h in range(1, max_h + 1):
        for rect_w in range(1, max_w + 1):
            area = rect_h * rect_w
            areas.setdefault(int(area), []).append((int(rect_h), int(rect_w)))
    return areas


def _placement_candidates(
    occupied: np.ndarray,
    rect_h: int,
    rect_w: int,
) -> list[tuple[int, int]]:
    height, width = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, height - rect_h):
        for c0 in range(0, width - rect_w + 1):
            rs = max(0, r0 - 1)
            re = min(height, r0 + rect_h + 1)
            cs = max(0, c0 - 1)
            ce = min(width, c0 + rect_w + 1)
            if not occupied[rs:re, cs:ce].any():
                candidates.append((int(r0), int(c0)))
    return candidates


def _panel_template(height: int, width: int, object_count: int) -> list[tuple[int, int, int, int]] | None:
    catalog = _rectangle_catalog(height, width)
    viable_areas = [area for area in catalog if area <= max(4, (height * width) // 4)]
    if len(viable_areas) < object_count:
        viable_areas = list(catalog)
    if len(viable_areas) < object_count:
        return None

    for _ in range(PANEL_ATTEMPTS):
        chosen_areas = random.sample(viable_areas, object_count)
        dims = [random.choice(catalog[area]) for area in chosen_areas]
        dims.sort(key=lambda dim: dim[0] * dim[1], reverse=True)

        occupied = np.zeros((height, width), dtype=bool)
        placements: list[tuple[int, int, int, int]] = []
        for rect_h, rect_w in dims:
            candidates = _placement_candidates(occupied, rect_h, rect_w)
            if not candidates:
                break
            r0, c0 = random.choice(candidates)
            occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            placements.append((int(r0), int(c0), int(rect_h), int(rect_w)))
        if len(placements) == object_count:
            return placements

    return None


def _render_panel(height: int, width: int, object_count: int) -> np.ndarray | None:
    placements = _panel_template(height, width, object_count)
    if placements is None:
        return None

    colors = random.sample(OBJECT_COLORS, object_count)
    panel = np.full((height, width), BG, dtype=int)
    for color, (r0, c0, rect_h, rect_w) in zip(colors, placements):
        panel[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)
    return panel


def _validate_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or g.size == 0:
            return False
        bg = _background_color(g)
        sep_color, sep_cols = _separator_columns(g, bg)
        if bg != BG or sep_color != SEP:
            return False
        spans = _panel_spans(g.shape[1], sep_cols)
        if not (PANEL_COUNT_RANGE[0] <= len(spans) <= PANEL_COUNT_RANGE[1]):
            return False
        for c0, c1 in spans:
            _panel_objects(g[:, c0 : c1 + 1], bg, c0)
        main(g)
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        panel_count = random.randint(*PANEL_COUNT_RANGE)
        object_counts = [random.randint(*OBJECT_COUNT_RANGE) for _ in range(panel_count)]
        height = random.randint(*HEIGHT_RANGE)

        sep_widths = [random.choice(SEPARATOR_WIDTH_CHOICES) for _ in range(panel_count - 1)]
        panel_width_budget = GRID_WIDTH_MAX - sum(sep_widths)
        max_total = min(panel_count * PANEL_WIDTH_MAX, panel_width_budget)
        min_total = panel_count * PANEL_WIDTH_MIN
        if min_total > max_total:
            continue

        total_panel_width = random.randint(min_total, max_total)
        panel_widths = _bounded_partition(
            total_panel_width,
            panel_count,
            PANEL_WIDTH_MIN,
            PANEL_WIDTH_MAX,
        )
        if panel_widths is None:
            continue

        panels: list[np.ndarray] = []
        for width, object_count in zip(panel_widths, object_counts):
            panel = _render_panel(height, width, object_count)
            if panel is None:
                break
            panels.append(panel)
        if len(panels) != panel_count:
            continue

        total_width = sum(panel_widths) + sum(sep_widths)
        grid = np.full((height, total_width), BG, dtype=int)

        col = 0
        for idx, panel in enumerate(panels):
            width = panel.shape[1]
            grid[:, col : col + width] = panel
            col += width
            if idx + 1 < panel_count:
                sep_width = sep_widths[idx]
                grid[:, col : col + sep_width] = SEP
                col += sep_width

        if _validate_input(grid):
            return grid

    raise AssertionError("failed to generate a valid input grid")
