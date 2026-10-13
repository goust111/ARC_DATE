from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _find_corner_marker(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for cells in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox_from_cells(cells)
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if box.size != len(cells) or not bool(np.all(box == GREY)):
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner:
            candidates.append((int(r0), int(c0), int(r1), int(c1)))

    assert len(candidates) == 1, "expected one grey corner marker rectangle"
    return candidates[0]


def _ordered_colors(grid: np.ndarray, marker_bbox: tuple[int, int, int, int], bg: int) -> list[int]:
    r0, c0, r1, c1 = marker_bbox
    marker_mask = np.zeros_like(grid, dtype=bool)
    marker_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    totals: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    flat = grid.reshape(-1)
    flat_marker = marker_mask.reshape(-1)
    for idx, value in enumerate(flat):
        color = int(value)
        if flat_marker[idx] or color == int(bg):
            continue
        totals[color] += 1
        if color not in first_seen:
            first_seen[color] = int(idx)

    assert totals, "expected at least one counted foreground color"
    return [
        color
        for color, _count in sorted(
            totals.items(),
            key=lambda item: (-int(item[1]), int(first_seen[int(item[0])])),
        )
    ]


def _render_nested_rectangles(shape: tuple[int, int], ordered_colors: list[int]) -> np.ndarray:
    out_h, out_w = shape
    max_layers = (min(out_h, out_w) + 1) // 2
    out = np.zeros((out_h, out_w), dtype=int)
    for layer, color in enumerate(ordered_colors[:max_layers]):
        out[layer : out_h - layer, layer : out_w - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate same-color fragment areas, rank colors by total area, and render
    them as nested solid rectangles inside the grey marker's output canvas size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])
    marker = _find_corner_marker(g)
    r0, c0, r1, c1 = marker
    ordered_colors = _ordered_colors(g, marker, bg)
    return _render_nested_rectangles((int(r1 - r0 + 1), int(c1 - c0 + 1)), ordered_colors)


GRID_HEIGHT_RANGE = (20, 28)
GRID_WIDTH_RANGE = (20, 28)
OUTPUT_HEIGHT_RANGE = (5, 9)
OUTPUT_WIDTH_RANGE = (5, 9)
AREA_CHOICES = tuple(range(4, 14))
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _corner_bbox(corner: str, grid_h: int, grid_w: int, out_h: int, out_w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, out_h - 1, out_w - 1
    if corner == "tr":
        return 0, grid_w - out_w, out_h - 1, grid_w - 1
    if corner == "bl":
        return grid_h - out_h, 0, grid_h - 1, out_w - 1
    if corner == "br":
        return grid_h - out_h, grid_w - out_w, grid_h - 1, grid_w - 1
    raise ValueError(f"unknown corner {corner}")


def _split_total(total: int, parts: int, min_part: int = 2) -> list[int]:
    assert parts >= 1 and total >= parts * min_part
    remaining = int(total)
    sizes: list[int] = []
    for i in range(parts):
        left = parts - i - 1
        lo = int(min_part)
        hi = int(remaining - left * min_part)
        size = hi if i == parts - 1 else random.randint(lo, hi)
        sizes.append(int(size))
        remaining -= int(size)
    random.shuffle(sizes)
    return sizes


def _random_connected_shape(area: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        limit = max(4, area)
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if abs(nr) > limit or abs(nc) > limit or (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        h = int(max_r - min_r + 1)
        w = int(max_c - min_c + 1)
        if h > 6 or w > 6:
            continue
        if area >= 5 and (h == 1 or w == 1):
            continue
        if area >= 4 and area == h * w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r - min_r), int(c - min_c)] = True
        return mask

    raise RuntimeError(f"failed to build a connected shape of area {area}")


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    grid_h, grid_w = blocked.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(grid_h, int(r0 + h + pad))
    cc1 = min(grid_w, int(c0 + w + pad))
    blocked[rr0:rr1, cc0:cc1] = True


def _place_components(
    grid_h: int,
    grid_w: int,
    marker_bbox: tuple[int, int, int, int],
    components: list[dict],
) -> list[tuple[int, int]]:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    mr0, mc0, mr1, mc1 = marker_bbox
    _mark_blocked(blocked, int(mr0), int(mc0), int(mr1 - mr0 + 1), int(mc1 - mc0 + 1))

    placements: list[tuple[int, int] | None] = [None] * len(components)
    ordered = sorted(
        enumerate(components),
        key=lambda item: (-int(item[1]["mask"].shape[0] * item[1]["mask"].shape[1]), -int(item[1]["area"])),
    )
    for idx, comp in ordered:
        mask = comp["mask"]
        sh, sw = mask.shape
        row_lo = 1 if grid_h - sh > 1 else 0
        row_hi = grid_h - sh - 1 if grid_h - sh > 1 else grid_h - sh
        col_lo = 1 if grid_w - sw > 1 else 0
        col_hi = grid_w - sw - 1 if grid_w - sw > 1 else grid_w - sw
        if row_lo > row_hi or col_lo > col_hi:
            raise RuntimeError("component does not fit the grid")

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(int(row_lo), int(row_hi))
            c0 = random.randint(int(col_lo), int(col_hi))
            if bool(blocked[r0 : r0 + sh, c0 : c0 + sw].any()):
                continue
            placements[idx] = (int(r0), int(c0))
            _mark_blocked(blocked, int(r0), int(c0), int(sh), int(sw))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all components")

    return [placement for placement in placements if placement is not None]


def _expected_from_totals(out_h: int, out_w: int, color_totals: dict[int, int]) -> np.ndarray:
    ordered = [
        color
        for color, _count in sorted(
            color_totals.items(),
            key=lambda item: -int(item[1]),
        )
    ]
    return _render_nested_rectangles((out_h, out_w), ordered)


def _build_color_plan(max_layers: int) -> tuple[list[int], dict[int, int]]:
    truncated = random.random() < 0.5
    if truncated:
        n_colors = random.randint(max_layers + 1, min(len(PALETTE), max_layers + 3))
    else:
        n_colors = random.randint(2, min(len(PALETTE), max_layers))

    colors = random.sample(PALETTE, n_colors)
    totals = sorted(random.sample(AREA_CHOICES, n_colors), reverse=True)
    return colors, {int(color): int(total) for color, total in zip(colors, totals)}


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    out_h = random.randint(OUTPUT_HEIGHT_RANGE[0], min(OUTPUT_HEIGHT_RANGE[1], grid_h - 8))
    out_w = random.randint(OUTPUT_WIDTH_RANGE[0], min(OUTPUT_WIDTH_RANGE[1], grid_w - 8))
    max_layers = (min(out_h, out_w) + 1) // 2

    corner = random.choice(["tl", "tr", "bl", "br"])
    marker_bbox = _corner_bbox(corner, grid_h, grid_w, out_h, out_w)
    colors, color_totals = _build_color_plan(max_layers)

    split_count = random.randint(1, max(1, min(3, len(colors))))
    split_colors = set(random.sample(colors, split_count))
    component_specs: list[dict] = []
    first_seen: list[int] = []

    for color in colors:
        total = int(color_totals[int(color)])
        max_parts = min(3, total // 2)
        if color in split_colors and max_parts >= 2:
            parts = random.randint(2, max_parts)
        else:
            parts = random.randint(1, max_parts)
        if parts == 1:
            areas = [total]
        else:
            areas = _split_total(total, parts, min_part=2)
        for area in areas:
            component_specs.append({"color": int(color), "area": int(area), "mask": _random_connected_shape(int(area))})
            first_seen.append(int(color))

    if len(set(first_seen)) == len(first_seen):
        return None

    random.shuffle(component_specs)
    try:
        placements = _place_components(grid_h, grid_w, marker_bbox, component_specs)
    except RuntimeError:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    mr0, mc0, mr1, mc1 = marker_bbox
    grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = GREY

    for comp, (r0, c0) in zip(component_specs, placements):
        mask = comp["mask"]
        sh, sw = mask.shape
        patch = grid[r0 : r0 + sh, c0 : c0 + sw]
        patch[mask] = int(comp["color"])

    counts = Counter(map(int, grid.ravel()))
    if counts[BLACK] <= counts[GREY]:
        return None
    if counts[BLACK] <= max(int(v) for k, v in counts.items() if int(k) != BLACK):
        return None

    expected = _expected_from_totals(out_h, out_w, color_totals)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
