from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MIN_RECT_SIDE = 6
MAX_ATTEMPTS = 400

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _component_corner_info(coords: np.ndarray) -> tuple[Coord, Coord] | None:
    if coords.shape != (3, 2):
        return None
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None

    local = np.zeros((2, 2), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if int(local.sum()) != 3:
        return None

    missing = np.argwhere(~local)
    if missing.shape != (1, 2):
        return None
    mr, mc = map(int, missing[0])
    corner = (r0 + 1 - mr, c0 + 1 - mc)
    interior_step = (2 * mr - 1, 2 * mc - 1)
    return corner, interior_step


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _find_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    H, W = grid.shape
    for r in range(H - 1, -1, -1):
        colored = np.flatnonzero((grid[r] != BG) & (grid[r] != GREY))
        if len(colored) < 2:
            continue
        if not np.all(np.diff(colored) == 1):
            continue
        left, right = int(colored[0]), int(colored[-1])
        if np.any(grid[r, :left] != BG) or np.any(grid[r, right + 1 :] != BG):
            continue
        return r, [int(x) for x in grid[r, left : right + 1]]
    raise AssertionError("expected one bottom palette row")


def _extract_rectangles(grid: np.ndarray) -> list[BBox]:
    rectangles: list[BBox] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, GREY)):
        infos = []
        for comp in _find_components(grid, color):
            info = _component_corner_info(comp)
            if info is not None:
                infos.append(info)
        if len(infos) != 2:
            continue

        (a_corner, a_step), (b_corner, b_step) = infos
        dr = _sign(b_corner[0] - a_corner[0])
        dc = _sign(b_corner[1] - a_corner[1])
        if dr == 0 or dc == 0:
            continue
        if (dr, dc) != a_step:
            continue
        if (-dr, -dc) != b_step:
            continue

        rectangles.append(
            (
                min(a_corner[0], b_corner[0]),
                min(a_corner[1], b_corner[1]),
                max(a_corner[0], b_corner[0]),
                max(a_corner[1], b_corner[1]),
            )
        )
    return rectangles


def _count_cover(shape: tuple[int, int], rectangles: list[BBox]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def _marker_cells(corner: Coord, step: Coord) -> set[Coord]:
    r, c = map(int, corner)
    dr, dc = map(int, step)
    return {(r, c), (r + dr, c), (r, c + dc)}


def _rect_corner_specs(bbox: BBox, diag: int) -> list[tuple[Coord, Coord]]:
    r0, c0, r1, c1 = bbox
    if diag == 0:
        return [((r0, c0), (1, 1)), ((r1, c1), (-1, -1))]
    return [((r0, c1), (1, -1)), ((r1, c0), (-1, 1))]


def _rect_marker_cells(bbox: BBox, diag: int) -> set[Coord]:
    cells: set[Coord] = set()
    for corner, step in _rect_corner_specs(bbox, diag):
        cells |= _marker_cells(corner, step)
    return cells


def _expand_cells(cells: set[Coord], shape: tuple[int, int], radius: int = 1) -> set[Coord]:
    H, W = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _strict_contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _partial_overlap(a: BBox, b: BBox) -> bool:
    inter_r0 = max(a[0], b[0])
    inter_c0 = max(a[1], b[1])
    inter_r1 = min(a[2], b[2])
    inter_c1 = min(a[3], b[3])
    if inter_r0 > inter_r1 or inter_c0 > inter_c1:
        return False
    return not _strict_contains(a, b) and not _strict_contains(b, a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct same-color opposite-corner rectangles from L markers and recolor each grey probe by the palette color of its coverage depth.

    Input structure:
    - A black background contains several pairs of same-color 2x2 L-corner markers; each pair marks two opposite corners of one unseen axis-aligned rectangle, and each L opens toward that rectangle's interior.
    - Grey single-pixel probes are scattered across the canvas, and a horizontal palette row sits near the bottom-left.

    Rule:
    - Recover each implied rectangle from its two same-color opposite corner markers, using the rectangles only for geometric counting and never drawing their borders.
    - For every grey probe, count how many recovered rectangles cover it; recolor depth d>=1 with the d-th palette color from left to right, and clear depth 0 to black.
    - Keep the corner markers and the palette row unchanged.

    Key insight:
    - The colored corner markers determine geometry but do not determine the output color; only the rectangle coverage count matters, and the palette converts that count into the new probe color.
    """
    g = np.asarray(grid, dtype=int)
    palette_row, palette = _find_palette(g)
    rectangles = _extract_rectangles(g)
    counts = _count_cover(g.shape, rectangles)

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        if int(r) == int(palette_row):
            continue
        depth = int(counts[r, c])
        if depth == 0:
            out[r, c] = BG
        else:
            assert depth <= len(palette), "palette too short for rectangle depth"
            out[r, c] = palette[depth - 1]
    return out


def _sample_nested_chain(depth: int, shape: tuple[int, int], outer_margin: int) -> list[BBox] | None:
    H, W = shape
    max_r1 = H - 3
    max_c1 = W - 2
    available_h = max_r1 - 2 * outer_margin + 1
    available_w = max_c1 - 2 * outer_margin + 1
    outer_min = MIN_RECT_SIDE + 4 * (depth - 1)
    if available_h < outer_min or available_w < outer_min:
        return None

    outer_h = random.randint(outer_min, available_h)
    outer_w = random.randint(outer_min, available_w)
    r0 = random.randint(outer_margin, max_r1 - outer_margin - outer_h + 1)
    c0 = random.randint(outer_margin, max_c1 - outer_margin - outer_w + 1)
    rects: list[BBox] = [(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)]

    for level in range(1, depth):
        prev = rects[-1]
        pr0, pc0, pr1, pc1 = prev
        prev_h = pr1 - pr0 + 1
        prev_w = pc1 - pc0 + 1
        remaining = depth - level - 1
        min_h = MIN_RECT_SIDE + 4 * remaining
        min_w = MIN_RECT_SIDE + 4 * remaining
        if prev_h < min_h + 4 or prev_w < min_w + 4:
            return None

        next_h = random.randint(min_h, prev_h - 4)
        next_w = random.randint(min_w, prev_w - 4)
        total_r = prev_h - next_h
        total_c = prev_w - next_w
        top_inset = random.randint(2, total_r - 2)
        left_inset = random.randint(2, total_c - 2)
        bottom_inset = total_r - top_inset
        right_inset = total_c - left_inset
        rects.append((pr0 + top_inset, pc0 + left_inset, pr1 - bottom_inset, pc1 - right_inset))

    return rects


def _sample_overlap_rect(host: BBox, shape: tuple[int, int]) -> BBox | None:
    H, W = shape
    max_r1 = H - 3
    max_c1 = W - 2
    hr0, hc0, hr1, hc1 = host
    host_h = hr1 - hr0 + 1
    host_w = hc1 - hc0 + 1
    reach_r = max(4, host_h // 2)
    reach_c = max(4, host_w // 2)
    patterns = ["tl", "tr", "bl", "br"]

    for _ in range(120):
        pat = random.choice(patterns)
        if pat == "tl":
            r0_lo, r0_hi = max(1, hr0 - 7), hr0 - 1
            c0_lo, c0_hi = max(1, hc0 - 7), hc0 - 1
            r1_lo, r1_hi = hr0 + 4, min(hr1 - 1, hr0 + reach_r)
            c1_lo, c1_hi = hc0 + 4, min(hc1 - 1, hc0 + reach_c)
            if r0_lo > r0_hi or c0_lo > c0_hi or r1_lo > r1_hi or c1_lo > c1_hi:
                continue
            r0 = random.randint(r0_lo, r0_hi)
            c0 = random.randint(c0_lo, c0_hi)
            r1 = random.randint(r1_lo, r1_hi)
            c1 = random.randint(c1_lo, c1_hi)
        elif pat == "tr":
            r0_lo, r0_hi = max(1, hr0 - 7), hr0 - 1
            c1_lo, c1_hi = hc1 + 1, min(max_c1, hc1 + 7)
            r1_lo, r1_hi = hr0 + 4, min(hr1 - 1, hr0 + reach_r)
            c0_lo, c0_hi = max(hc0 + 1, hc1 - reach_c), hc1 - 4
            if r0_lo > r0_hi or c1_lo > c1_hi or r1_lo > r1_hi or c0_lo > c0_hi:
                continue
            r0 = random.randint(r0_lo, r0_hi)
            c1 = random.randint(c1_lo, c1_hi)
            r1 = random.randint(r1_lo, r1_hi)
            c0 = random.randint(c0_lo, c0_hi)
        elif pat == "bl":
            r1_lo, r1_hi = hr1 + 1, min(max_r1, hr1 + 7)
            c0_lo, c0_hi = max(1, hc0 - 7), hc0 - 1
            r0_lo, r0_hi = max(hr0 + 1, hr1 - reach_r), hr1 - 4
            c1_lo, c1_hi = hc0 + 4, min(hc1 - 1, hc0 + reach_c)
            if r1_lo > r1_hi or c0_lo > c0_hi or r0_lo > r0_hi or c1_lo > c1_hi:
                continue
            r1 = random.randint(r1_lo, r1_hi)
            c0 = random.randint(c0_lo, c0_hi)
            r0 = random.randint(r0_lo, r0_hi)
            c1 = random.randint(c1_lo, c1_hi)
        else:
            r1_lo, r1_hi = hr1 + 1, min(max_r1, hr1 + 7)
            c1_lo, c1_hi = hc1 + 1, min(max_c1, hc1 + 7)
            r0_lo, r0_hi = max(hr0 + 1, hr1 - reach_r), hr1 - 4
            c0_lo, c0_hi = max(hc0 + 1, hc1 - reach_c), hc1 - 4
            if r1_lo > r1_hi or c1_lo > c1_hi or r0_lo > r0_hi or c0_lo > c0_hi:
                continue
            r1 = random.randint(r1_lo, r1_hi)
            c1 = random.randint(c1_lo, c1_hi)
            r0 = random.randint(r0_lo, r0_hi)
            c0 = random.randint(c0_lo, c0_hi)

        bbox = (r0, c0, r1, c1)
        if r1 - r0 + 1 < MIN_RECT_SIDE or c1 - c0 + 1 < MIN_RECT_SIDE:
            continue
        return bbox
    return None


def _pick_spaced(candidates: list[Coord], chosen: list[Coord]) -> Coord | None:
    random.shuffle(candidates)
    for cell in candidates:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            return cell
    return candidates[0] if candidates else None


def _render_input(
    shape: tuple[int, int],
    rect_specs: list[tuple[BBox, int, int]],
    palette: list[int],
    palette_start: int,
    grey_points: list[Coord],
) -> np.ndarray:
    H, W = shape
    grid = np.full((H, W), BG, dtype=int)
    for bbox, diag, color in rect_specs:
        for r, c in _rect_marker_cells(bbox, diag):
            grid[r, c] = int(color)
    for i, color in enumerate(palette):
        grid[H - 1, palette_start + i] = int(color)
    for r, c in grey_points:
        grid[r, c] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    want_overlap = random.random() < 0.7
    base_depth = random.choice([2, 3] if want_overlap else [2, 3, 4])
    outer_margin = 4 if want_overlap else 1
    min_dim = MIN_RECT_SIDE + 4 * (base_depth - 1) + 2 * outer_margin + 2
    H = random.randint(max(18, min_dim), 30)
    W = random.randint(max(18, min_dim), 30)

    chain = _sample_nested_chain(base_depth, (H, W), outer_margin)
    if chain is None:
        return None

    bboxes = list(chain)
    if want_overlap:
        extra = _sample_overlap_rect(chain[0], (H, W))
        if extra is None:
            return None
        if any(_strict_contains(extra, old) or _strict_contains(old, extra) for old in bboxes):
            return None
        if not any(_partial_overlap(extra, old) for old in bboxes):
            return None
        bboxes.append(extra)

    if not any(_strict_contains(bboxes[i], bboxes[i + 1]) for i in range(base_depth - 1)):
        return None
    if want_overlap and not any(_partial_overlap(a, b) for a, b in combinations(bboxes, 2)):
        return None

    diags = [random.randint(0, 1) for _ in bboxes]
    marker_cells: set[Coord] = set()
    blocked: set[Coord] = set()
    for bbox, diag in zip(bboxes, diags):
        cells = _rect_marker_cells(bbox, diag)
        if any(r >= H - 2 for r, _c in cells):
            return None
        if cells & blocked:
            return None
        marker_cells |= cells
        blocked = _expand_cells(marker_cells, (H, W), radius=1)

    counts = _count_cover((H, W), bboxes)
    max_depth = int(counts[: H - 2, :].max())
    if max_depth < 2:
        return None
    if len(bboxes) + max_depth > len(COLOR_POOL):
        return None

    color_order = COLOR_POOL[:]
    random.shuffle(color_order)
    corner_colors = color_order[: len(bboxes)]
    palette = color_order[len(bboxes) : len(bboxes) + max_depth]

    palette_start = random.randint(1, max(1, W - len(palette) - 2))
    palette_cells = {(H - 1, palette_start + i) for i in range(len(palette))}
    unsafe = _expand_cells(marker_cells | palette_cells, (H, W), radius=1)

    background_mask = np.ones((H, W), dtype=bool)
    for r, c in unsafe:
        background_mask[r, c] = False
    background_mask[H - 1, :] = False
    background_mask[H - 2, :] = False

    masks = []
    for bbox in bboxes:
        m = np.zeros((H, W), dtype=bool)
        r0, c0, r1, c1 = bbox
        m[r0 : r1 + 1, c0 : c1 + 1] = True
        masks.append(m)

    by_depth: dict[int, list[Coord]] = {}
    for depth in range(max_depth + 1):
        coords = np.argwhere((counts == depth) & background_mask)
        by_depth[depth] = [tuple(map(int, rc)) for rc in coords]
        if depth == 0 and len(by_depth[depth]) == 0:
            return None
        if depth >= 1 and len(by_depth[depth]) == 0:
            return None

    chosen: list[Coord] = []

    depth0 = _pick_spaced(by_depth[0][:], chosen)
    if depth0 is None:
        return None
    chosen.append(depth0)

    for depth in range(1, max_depth + 1):
        point = _pick_spaced(by_depth[depth][:], chosen)
        if point is None:
            return None
        chosen.append(point)

    if want_overlap:
        extra_mask = masks[-1]
        chain_union = np.zeros((H, W), dtype=bool)
        for m in masks[:-1]:
            chain_union |= m
        extra_only = np.argwhere(extra_mask & ~chain_union & background_mask)
        chain_only = np.argwhere(chain_union & ~extra_mask & (counts == 1) & background_mask)
        overlap_zone = np.argwhere(extra_mask & chain_union & background_mask)

        for group in (extra_only, chain_only, overlap_zone):
            if len(group) == 0:
                continue
            point = _pick_spaced([tuple(map(int, rc)) for rc in group], chosen)
            if point is not None and point not in chosen:
                chosen.append(point)

    target_total = random.randint(max_depth + 2, min(8, max_depth + 5))
    all_candidates = [cell for depth_cells in by_depth.values() for cell in depth_cells]
    while len(chosen) < target_total:
        point = _pick_spaced(all_candidates[:], chosen)
        if point is None or point in chosen:
            break
        chosen.append(point)

    rect_specs = [(bbox, diag, color) for bbox, diag, color in zip(bboxes, diags, corner_colors)]
    grid = _render_input((H, W), rect_specs, palette, palette_start, chosen)

    parsed = sorted(_extract_rectangles(grid))
    if parsed != sorted(bboxes):
        return None

    expected = grid.copy()
    for r, c in chosen:
        depth = int(counts[r, c])
        expected[r, c] = BG if depth == 0 else palette[depth - 1]
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
