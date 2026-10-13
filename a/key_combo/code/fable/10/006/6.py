from __future__ import annotations

import random

import numpy as np


GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (9, 18)
GRID_WIDTH_RANGE = (9, 18)
HOLE_HEIGHT_RANGE = (2, 5)
HOLE_WIDTH_RANGE = (2, 5)
WITNESS_POSITIONS = [
    (1, 1),
    (1, 3),
    (1, 5),
    (1, 7),
    (3, 1),
    (3, 3),
    (3, 5),
    (5, 1),
]


def _rot180(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    return h - 1 - r, w - 1 - c


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"expected color {color} to be present"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_hole_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox_of_color(grid, GRAY)
    hole_mask = grid == GRAY
    assert hole_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "gray cells must form a solid rectangle"
    assert int(hole_mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "gray rectangle must be unique"
    return r0, c0, r1, c1


def _infer_color_map(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    visible_colors = set(map(int, np.unique(grid))) - {GRAY}
    for r in range(h):
        for c in range(w):
            src = int(grid[r, c])
            if src == GRAY:
                continue
            rr, cc = _rot180(grid.shape, r, c)
            dst = int(grid[rr, cc])
            if dst == GRAY:
                continue
            if src in mapping and mapping[src] != dst:
                raise AssertionError("inconsistent visible color mapping")
            mapping[src] = dst
    assert visible_colors <= mapping.keys(), "every visible color must be recoverable from unobscured pairs"
    for color, mate in mapping.items():
        assert mate in mapping, "mapping must be closed on visible colors"
        assert mapping[mate] == color, "mapping must be an involution"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a gray rectangular occlusion using 180-degree symmetry plus a learned color swap table.

    Input structure:
    - A rectangular canvas is fully colored except for one solid gray rectangle.
    - Ignoring the gray rectangle, the full hidden picture is invariant under a
      180-degree rotation followed by a per-instance color involution: some
      colors swap in pairs while others map to themselves.
    - The gray rectangle never overlaps its own 180-degree image, and every
      color relation needed by the involution appears somewhere outside the hole.

    Rule:
    - Detect the unique gray rectangle.
    - Compare each visible cell with its point-symmetric counterpart to infer
      the color involution from the unobscured region.
    - For every gray cell, take the color at its 180-degree counterpart, apply
      the inferred involution once, and write that value into the hole.

    Key insight:
    - The missing region is not restored by copying opposite pixels directly:
      the picture is symmetric only after composing the half-turn with a hidden
      color-swap table that must be inferred from the visible pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _find_hole_bbox(g)
    mapping = _infer_color_map(g)
    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = _rot180(g.shape, r, c)
            opposite = int(g[rr, cc])
            assert opposite != GRAY, "occlusion must not overlap its rotated image"
            out[r, c] = mapping[opposite]
    return out


def _rotated_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _fill_symmetric_rect(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    color_map: dict[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    rr0, cc0, rr1, cc1 = _rotated_bbox(grid.shape, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = color_map[color]


def _choose_color_map() -> tuple[dict[int, int], list[int], list[tuple[int, int]]]:
    n_swaps = random.randint(1, 2)
    n_fixed = random.randint(2, min(4, len(COLOR_POOL) - 2 * n_swaps))
    chosen = random.sample(COLOR_POOL, 2 * n_swaps + n_fixed)
    fixed_colors = chosen[:n_fixed]
    swapped_colors = chosen[n_fixed:]
    pairs = [(swapped_colors[i], swapped_colors[i + 1]) for i in range(0, len(swapped_colors), 2)]
    color_map = {c: c for c in fixed_colors}
    for a, b in pairs:
        color_map[a] = b
        color_map[b] = a
    return color_map, fixed_colors, pairs


def _sample_rect(shape: tuple[int, int], max_h: int, max_w: int) -> tuple[int, int, int, int] | None:
    h, w = shape
    rect_h = random.randint(1, min(max_h, h - 1))
    rect_w = random.randint(1, min(max_w, w - 1))
    for _ in range(80):
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if not _bboxes_overlap(bbox, _rotated_bbox(shape, bbox)):
            return bbox
    return None


def _sample_hole(
    full: np.ndarray,
    color_map: dict[int, int],
) -> tuple[int, int, int, int] | None:
    h, w = full.shape
    fixed = {c for c, mate in color_map.items() if c == mate}
    swapped = {c for c, mate in color_map.items() if c != mate}
    for _ in range(200):
        rect_h = random.randint(HOLE_HEIGHT_RANGE[0], min(HOLE_HEIGHT_RANGE[1], h - 1))
        rect_w = random.randint(HOLE_WIDTH_RANGE[0], min(HOLE_WIDTH_RANGE[1], w - 1))
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        rbbox = _rotated_bbox(full.shape, bbox)
        if _bboxes_overlap(bbox, rbbox):
            continue

        values = full[r0 : bbox[2] + 1, c0 : bbox[3] + 1]
        if len(set(map(int, values.flat))) < 2:
            continue

        has_swap_required = False
        touches_fixed = False
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                rr, cc = _rot180(full.shape, r, c)
                if int(full[r, c]) != int(full[rr, cc]):
                    has_swap_required = True
                if int(full[r, c]) in fixed or int(full[rr, cc]) in fixed:
                    touches_fixed = True
        if not has_swap_required:
            continue
        if not touches_fixed and fixed:
            continue

        inp = full.copy()
        inp[r0 : bbox[2] + 1, c0 : bbox[3] + 1] = GRAY
        try:
            inferred = _infer_color_map(inp)
        except AssertionError:
            continue
        if inferred != color_map:
            continue
        return bbox
    return None


def _build_full_grid() -> tuple[np.ndarray, dict[int, int]]:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    color_map, fixed_colors, swap_pairs = _choose_color_map()
    base = fixed_colors[0]
    full = np.full((h, w), base, dtype=int)

    witness_colors = fixed_colors + [a for pair in swap_pairs for a in pair]
    assert len(witness_colors) <= len(WITNESS_POSITIONS)
    for color, (r, c) in zip(witness_colors, WITNESS_POSITIONS):
        rr, cc = _rot180(full.shape, r, c)
        if (r, c) == (rr, cc):
            continue
        full[r, c] = color
        full[rr, cc] = color_map[color]

    if h % 2 == 1 and w % 2 == 1:
        full[h // 2, w // 2] = random.choice(fixed_colors)

    extra_rects = random.randint(10, 18)
    for _ in range(extra_rects):
        bbox = _sample_rect(full.shape, max_h=min(5, h - 1), max_w=min(5, w - 1))
        if bbox is None:
            continue
        color = random.choice(list(color_map))
        _fill_symmetric_rect(full, bbox, color, color_map)

    stripe_trials = random.randint(2, 4)
    for _ in range(stripe_trials):
        if random.choice([True, False]):
            bbox = _sample_rect(full.shape, max_h=1, max_w=min(7, w - 1))
        else:
            bbox = _sample_rect(full.shape, max_h=min(7, h - 1), max_w=1)
        if bbox is None:
            continue
        color = random.choice(list(color_map))
        _fill_symmetric_rect(full, bbox, color, color_map)

    return full, color_map


def _validate_instance(inp: np.ndarray, out: np.ndarray, color_map: dict[int, int]) -> bool:
    try:
        hole = _find_hole_bbox(inp)
    except AssertionError:
        return False

    if np.any(out == GRAY):
        return False
    if not np.array_equal(main(inp), out):
        return False

    rbbox = _rotated_bbox(inp.shape, hole)
    if _bboxes_overlap(hole, rbbox):
        return False

    try:
        inferred = _infer_color_map(inp)
    except AssertionError:
        return False
    if inferred != color_map:
        return False

    r0, c0, r1, c1 = hole
    direct_copy_works_everywhere = True
    swap_needed = 0
    fixed_seen = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = _rot180(inp.shape, r, c)
            if out[r, c] != inp[rr, cc]:
                direct_copy_works_everywhere = False
            if color_map[int(out[r, c])] != int(out[rr, cc]):
                return False
            if out[r, c] != out[rr, cc]:
                swap_needed += 1
            else:
                fixed_seen += 1
    if direct_copy_works_everywhere:
        return False
    if swap_needed == 0 or fixed_seen == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    full, color_map = _build_full_grid()
    hole = _sample_hole(full, color_map)
    if hole is None:
        return None

    inp = full.copy()
    r0, c0, r1, c1 = hole
    inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY
    if not _validate_instance(inp, full, color_map):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-with-color-involution sample")
