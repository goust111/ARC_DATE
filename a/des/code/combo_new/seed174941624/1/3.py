from __future__ import annotations

import random
import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LEGEND_COLOR_COUNT_RANGE = (2, 4)
REGION_EXTRA_RANGE = (0, 2)
REGION_INTERIOR_H_RANGE = (2, 5)
REGION_INTERIOR_W_RANGE = (2, 5)
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_bboxes(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    rings: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == color):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for cc in range(c0, c1 + 1):
            expected.add((r0, cc))
            expected.add((r1, cc))
        for rr in range(r0 + 1, r1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        if set(comp) == expected:
            rings.append((r0, c0, r1, c1))
    return rings


def _find_ring_color_and_bboxes(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    order = [int(colors[i]) for i in np.argsort(-counts)]
    best_color = None
    best_rings: list[tuple[int, int, int, int]] = []
    for color in order:
        rings = _ring_bboxes(grid, color)
        if len(rings) > len(best_rings):
            best_color = color
            best_rings = rings
    assert best_color is not None and len(best_rings) >= 2, "expected multiple rectangular ring components"
    return int(best_color), best_rings


def _parse_legend(interior: np.ndarray, bg: int, ring_color: int) -> dict[int, tuple[int, int]] | None:
    colors = [int(v) for v in np.unique(interior) if int(v) not in (bg, ring_color)]
    if len(colors) < 2:
        return None

    pos_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(interior == color)]
        if len(coords) != 2:
            return None
        if abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1]) != 1:
            return None
        pos_by_color[color] = coords

    width = interior.shape[1]
    ordered_colors = sorted(colors, key=lambda color: min(pos_by_color[color]))
    candidates: list[dict[int, tuple[int, int]]] = []

    for anchor_col in range(width):
        if not all(any(c == anchor_col for _r, c in pos_by_color[color]) for color in ordered_colors):
            continue
        for parity in (0, 1):
            mapping: dict[int, tuple[int, int]] = {}
            ok = True
            for color in ordered_colors:
                coords = pos_by_color[color]
                anchors = [pt for pt in coords if pt[1] == anchor_col and pt[0] % 2 == parity]
                if len(anchors) != 1:
                    ok = False
                    break
                anchor = anchors[0]
                other = coords[0] if coords[1] == anchor else coords[1]
                dr = other[0] - anchor[0]
                dc = other[1] - anchor[1]
                if abs(dr) + abs(dc) != 1:
                    ok = False
                    break
                mapping[color] = (int(dr), int(dc))
            if ok:
                candidates.append(mapping)

    if len(candidates) != 1:
        return None
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the legend's color-direction mapping to grow each seed into a same-color ray whose length equals the blank area inside its enclosing gray rectangle.

    Input structure:
    - The background is the dominant color.
    - One gray rectangular ring near the top-left is a legend; inside it, each legend color appears as a 2-pixel adjacent pair whose relative offset encodes one cardinal direction.
    - Elsewhere, several separate gray rectangular rings each enclose exactly one colored seed pixel on background cells.

    Rule:
    - Identify the legend ring and decode a mapping from seed color to growth direction from the 2-pixel legend pairs.
    - For every non-legend ring, count how many background cells lie in its interior, excluding the seed and the gray border.
    - Starting from the seed, extend a same-color straight line by exactly that many cells in the mapped direction; the line may pass through the gray border and is clipped only by the grid edge.

    Key insight:
    - The direction and the length come from different sources: the legend pair geometry chooses the ray direction, while the number of blank cells inside the seed's own room determines how far that ray grows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)
    ring_color, rings = _find_ring_color_and_bboxes(g, bg)

    legend_candidates: list[tuple[tuple[int, int, int, int], dict[int, tuple[int, int]]]] = []
    for bbox in rings:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mapping = _parse_legend(interior, bg, ring_color)
        if mapping is not None:
            legend_candidates.append((bbox, mapping))

    assert len(legend_candidates) == 1, "expected exactly one legend ring"
    legend_bbox, direction_by_color = legend_candidates[0]

    out = g.copy()
    for bbox in rings:
        if bbox == legend_bbox:
            continue
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != bg) & (interior != ring_color))
        assert seeds.shape[0] == 1, "each non-legend ring must contain exactly one seed"
        sr, sc = map(int, seeds[0])
        seed_r = r0 + 1 + sr
        seed_c = c0 + 1 + sc
        color = int(interior[sr, sc])
        assert color in direction_by_color, "seed color must appear in the legend"

        length = int(np.count_nonzero(interior == bg))
        dr, dc = direction_by_color[color]
        for step in range(1, length + 1):
            nr = seed_r + dr * step
            nc = seed_c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            out[nr, nc] = color

    return out


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mark_padded(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_is_clear(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _distance_to_edge(seed: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = seed
    dr, dc = direction
    if dr == 1:
        return h - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return w - 1 - c
    return c


def _path_cells(seed: tuple[int, int], direction: tuple[int, int], length: int, h: int, w: int) -> list[tuple[int, int]]:
    r, c = seed
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, length + 1):
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < h and 0 <= nc < w):
            break
        cells.append((nr, nc))
    return cells


def _biased_origin(
    h: int,
    w: int,
    outer_h: int,
    outer_w: int,
    direction: tuple[int, int],
    clipped: bool,
) -> tuple[int, int]:
    row_lo, row_hi = 1, h - outer_h - 1
    col_lo, col_hi = 1, w - outer_w - 1
    assert row_lo <= row_hi and col_lo <= col_hi
    mid_r = (row_lo + row_hi) // 2
    mid_c = (col_lo + col_hi) // 2
    dr, dc = direction
    if dr == 1:
        row_lo = mid_r if clipped else row_lo
        row_hi = row_hi if clipped else mid_r
    elif dr == -1:
        row_lo = row_lo if clipped else mid_r
        row_hi = mid_r if clipped else row_hi
    elif dc == 1:
        col_lo = mid_c if clipped else col_lo
        col_hi = col_hi if clipped else mid_c
    else:
        col_lo = col_lo if clipped else mid_c
        col_hi = mid_c if clipped else col_hi
    row_lo = min(row_lo, row_hi)
    col_lo = min(col_lo, col_hi)
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _legend_patch(colors: list[int], directions: list[tuple[int, int]]) -> np.ndarray:
    count = len(colors)
    interior_h = 2 * count + 1
    patch = np.full((interior_h + 2, 5), BACKGROUND, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    for idx, (color, direction) in enumerate(zip(colors, directions)):
        anchor_r = 1 + 2 * idx
        anchor_c = 1
        dr, dc = direction
        patch[1 + anchor_r, 1 + anchor_c] = color
        patch[1 + anchor_r + dr, 1 + anchor_c + dc] = color

    return patch


def _expected_output(grid: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        color = int(region["color"])
        for r, c in region["path"]:
            out[r, c] = color
    return out


def _place_region(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: tuple[int, int],
    clipped: bool,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(PLACE_RETRIES):
        inner_h = random.randint(*REGION_INTERIOR_H_RANGE)
        inner_w = random.randint(*REGION_INTERIOR_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        r0, c0 = _biased_origin(h, w, outer_h, outer_w, direction, clipped)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if not _box_is_clear(reserved, bbox, pad=1):
            continue

        length = inner_h * inner_w - 1
        candidates: list[tuple[tuple[int, int], list[tuple[int, int]], int]] = []
        for sr in range(r0 + 1, r0 + 1 + inner_h):
            for sc in range(c0 + 1, c0 + 1 + inner_w):
                seed = (sr, sc)
                avail = _distance_to_edge(seed, direction, h, w)
                if (avail < length) != clipped:
                    continue
                path = _path_cells(seed, direction, length, h, w)
                ok = True
                for pr, pc in path:
                    inside_box = r0 <= pr <= bbox[2] and c0 <= pc <= bbox[3]
                    if not inside_box and reserved[pr, pc]:
                        ok = False
                        break
                if ok:
                    candidates.append((seed, path, avail))
        if not candidates:
            continue

        seed, path, avail = random.choice(candidates)
        _draw_ring(grid, bbox, GREY)
        grid[seed] = color
        _mark_padded(reserved, bbox, pad=1)
        for pr, pc in path:
            reserved[pr, pc] = True
        return {
            "bbox": bbox,
            "seed": seed,
            "color": color,
            "direction": direction,
            "length": length,
            "path": path,
            "clipped": avail < length,
        }
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    legend_count = random.randint(*LEGEND_COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, legend_count)
    directions = random.sample(DIRS, legend_count)
    legend = _legend_patch(colors, directions)
    legend_h, legend_w = legend.shape
    legend_r0 = random.randint(0, 1)
    legend_c0 = random.randint(0, 1)
    if legend_r0 + legend_h > h or legend_c0 + legend_w > w:
        return None
    grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + legend_w] = legend
    _mark_padded(
        reserved,
        (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1),
        pad=1,
    )

    region_colors = colors[:]
    extra_regions = random.randint(*REGION_EXTRA_RANGE)
    region_colors.extend(random.choice(colors) for _ in range(extra_regions))
    random.shuffle(region_colors)

    clipped_flags = [random.choice((False, True)) for _ in region_colors]
    if len(clipped_flags) >= 2:
        clipped_flags[0] = False
        clipped_flags[1] = True
        random.shuffle(clipped_flags)

    direction_by_color = dict(zip(colors, directions))
    regions: list[dict[str, object]] = []
    for color, clipped in zip(region_colors, clipped_flags):
        region = _place_region(grid, reserved, color, direction_by_color[color], clipped)
        if region is None:
            return None
        regions.append(region)

    expected = _expected_output(grid, regions)
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != BACKGROUND:
        return None
    non_bg_counts = [int(counts[c]) for c in range(10) if c != BACKGROUND]
    if counts[GREY] != max(non_bg_counts):
        return None
    if len(regions) >= 2:
        clip_kinds = {bool(region["clipped"]) for region in regions}
        if clip_kinds != {False, True}:
            return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
