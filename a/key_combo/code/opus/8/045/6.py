from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

TOP_ROWS = 5
SEPARATOR_ROW = 5

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (15, 22)
REGION_COUNT_CHOICES = (3, 4)
MIN_REGION_SIDE = 2
MAX_GENERATION_ATTEMPTS = 500


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_pillar_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:TOP_ROWS, c]
        if (
            int(col[2]) == GREY
            and int(col[0]) in PALETTE
            and int(col[1]) in PALETTE
            and int(col[3]) in PALETTE
            and int(col[4]) in PALETTE
        ):
            cols.append(int(c))
    return cols


def _reversed_top_colors(grid: np.ndarray, pillar_cols: list[int]) -> list[int]:
    return [int(grid[1, c]) for c in pillar_cols]


def _paint_regions(
    grid: np.ndarray,
    region_coords: list[np.ndarray],
    colors: list[int],
    order: list[int],
) -> np.ndarray:
    out = grid.copy()
    for idx, color in zip(order, colors):
        coords = region_coords[int(idx)]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each top coding pillar, decode its top color, and fill lower wall-partitioned regions by ascending area.

    Input structure:
    - The top five rows contain K vertical 5x1 pillars on a black background.
    - Each pillar has a fixed grey center at row 2, two colored pixels above it, and two colored pixels below it.
    - The lower part contains grey one-cell-wide walls partitioning black cells into exactly K rectangular regions with distinct areas.

    Rule:
    - Sort the pillars from left to right and locally reverse each one: swap rows 0/1 and rows 3/4 while keeping the grey center fixed.
    - Read the top pixel of each reversed pillar to obtain a left-to-right color sequence.
    - Find the lower black regions, sort them by area from smallest to largest, and recolor them with that decoded sequence in order.
    - Keep the grey walls unchanged, and keep the top pillars in their reversed state in the output.

    Key insight:
    - The pillar reversal is a decoding step: the intended color sequence is hidden in row 1 until each pillar is flipped, and that sequence only becomes meaningful after matching it to the lower regions by area rather than by position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= TOP_ROWS + 2

    pillar_cols = _find_pillar_columns(g)
    assert 3 <= len(pillar_cols) <= 4

    out = g.copy()
    for c in pillar_cols:
        out[0, c], out[1, c] = g[1, c], g[0, c]
        out[3, c], out[4, c] = g[4, c], g[3, c]
    decoded_colors = [int(out[0, c]) for c in pillar_cols]

    lower = g[SEPARATOR_ROW:, :]
    regions_local = _components_mask(lower == BLACK)
    assert len(regions_local) == len(pillar_cols)
    areas = [int(coords.shape[0]) for coords in regions_local]
    assert len(set(areas)) == len(areas)
    area_order = sorted(range(len(regions_local)), key=lambda i: areas[i])

    regions_global = [coords + np.array([SEPARATOR_ROW, 0], dtype=int) for coords in regions_local]
    for region_idx, color in zip(area_order, decoded_colors):
        coords = regions_global[int(region_idx)]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _sample_spaced_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - prev) >= 2 for prev in chosen):
            chosen.append(int(c))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_pillars(grid: np.ndarray, pillar_cols: list[int]) -> list[int]:
    decoded = random.sample(PALETTE, len(pillar_cols))
    for c, top_after_reverse in zip(pillar_cols, decoded):
        remaining = [color for color in PALETTE if color != top_after_reverse]
        row0 = random.choice(remaining)
        lower_pair = random.sample(PALETTE, 2)
        grid[0, c] = int(row0)
        grid[1, c] = int(top_after_reverse)
        grid[2, c] = GREY
        grid[3, c] = int(lower_pair[0])
        grid[4, c] = int(lower_pair[1])
    return [int(color) for color in decoded]


def _can_split(rect: tuple[int, int, int, int], orientation: str) -> bool:
    _r0, _c0, h, w = rect
    if orientation == "h":
        return h >= 2 * MIN_REGION_SIDE + 1
    return w >= 2 * MIN_REGION_SIDE + 1


def _split_rect(
    rect: tuple[int, int, int, int], orientation: str
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    r0, c0, h, w = rect
    if not _can_split(rect, orientation):
        return None

    if orientation == "h":
        split = random.randint(MIN_REGION_SIDE, h - MIN_REGION_SIDE - 1)
        top = (r0, c0, split, w)
        bottom = (r0 + split + 1, c0, h - split - 1, w)
        return top, bottom

    split = random.randint(MIN_REGION_SIDE, w - MIN_REGION_SIDE - 1)
    left = (r0, c0, h, split)
    right = (r0, c0 + split + 1, h, w - split - 1)
    return left, right


def _generate_rectangles(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, inner_h, inner_w)]
    for _ in range(count - 1):
        candidates: list[tuple[int, str]] = []
        for idx, rect in enumerate(rects):
            for orientation in ("h", "v"):
                if _can_split(rect, orientation):
                    candidates.append((idx, orientation))
        if not candidates:
            return None
        idx, orientation = random.choice(candidates)
        split_pair = _split_rect(rects[idx], orientation)
        if split_pair is None:
            return None
        rects.pop(idx)
        rects.extend(split_pair)
    return rects


def _render_partition(
    height: int, width: int, rects: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid = np.full((height, width), GREY, dtype=int)
    rects_sorted = sorted(rects, key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))
    for r0, c0, h, w in rects_sorted:
        grid[7 + r0 : 7 + r0 + h, 1 + c0 : 1 + c0 + w] = BLACK
    return grid, rects_sorted


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return int(rect[2] * rect[3])


def _region_order_checks(rects: list[tuple[int, int, int, int]]) -> bool:
    areas = [_rect_area(rect) for rect in rects]
    if len(set(areas)) != len(areas):
        return False

    scan_order = sorted(range(len(rects)), key=lambda i: (rects[i][0], rects[i][1]))
    area_order = sorted(range(len(rects)), key=lambda i: areas[i])
    if scan_order == area_order:
        return False

    col_order = sorted(range(len(rects)), key=lambda i: (rects[i][1], rects[i][0]))
    if col_order == area_order:
        return False
    return True


def _reverse_pillars_only(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for c in _find_pillar_columns(inp):
        out[0, c], out[1, c] = inp[1, c], inp[0, c]
        out[3, c], out[4, c] = inp[4, c], inp[3, c]
    return out


def _wrong_outputs_differ(inp: np.ndarray, decoded_colors: list[int]) -> bool:
    correct = main(inp)
    lower_regions = _components_mask(inp[SEPARATOR_ROW:, :] == BLACK)
    if len(lower_regions) != len(decoded_colors):
        return False
    lower_regions = [coords + np.array([SEPARATOR_ROW, 0], dtype=int) for coords in lower_regions]
    areas = [int(coords.shape[0]) for coords in lower_regions]
    area_order = sorted(range(len(lower_regions)), key=lambda i: areas[i])

    reversed_only = _reverse_pillars_only(inp)
    wrong_no_reverse = _paint_regions(
        reversed_only,
        lower_regions,
        [int(inp[0, c]) for c in _find_pillar_columns(inp)],
        area_order,
    )
    if np.array_equal(wrong_no_reverse, correct):
        return False

    bboxes = [_bbox(coords) for coords in lower_regions]
    scan_order = sorted(range(len(lower_regions)), key=lambda i: (bboxes[i][0], bboxes[i][1]))
    if scan_order == area_order:
        return False
    wrong_scan = _paint_regions(reversed_only, lower_regions, decoded_colors, scan_order)
    if np.array_equal(wrong_scan, correct):
        return False
    return True


def _build_example() -> np.ndarray | None:
    k = random.choice(REGION_COUNT_CHOICES)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    inner_h = height - 8
    inner_w = width - 2
    if inner_h < MIN_REGION_SIDE or inner_w < MIN_REGION_SIDE:
        return None

    pillar_cols = _sample_spaced_columns(width, k)
    if pillar_cols is None:
        return None

    rects = _generate_rectangles(inner_h, inner_w, k)
    if rects is None or not _region_order_checks(rects):
        return None

    grid = np.zeros((height, width), dtype=int)
    decoded_colors = _build_pillars(grid, pillar_cols)
    grid[SEPARATOR_ROW, :] = GREY
    lower, rects_sorted = _render_partition(height, width, rects)
    grid[SEPARATOR_ROW:, :] = lower[SEPARATOR_ROW:, :]

    if len(set(decoded_colors)) != k:
        return None
    if len(set(_reversed_top_colors(grid, pillar_cols))) != k:
        return None

    lower_regions = _components_mask(grid[SEPARATOR_ROW:, :] == BLACK)
    if len(lower_regions) != k:
        return None
    if len({int(coords.shape[0]) for coords in lower_regions}) != k:
        return None

    if not _wrong_outputs_differ(grid, decoded_colors):
        return None

    out = main(grid)
    for c in pillar_cols:
        if int(out[0, c]) != int(grid[1, c]):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid local-reverse region-coloring example")
