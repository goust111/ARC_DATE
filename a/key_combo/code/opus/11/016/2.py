from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7

PALETTE = np.array([BLUE, GREEN, ORANGE], dtype=int)
TOGGLE = {BLUE: ORANGE, ORANGE: BLUE}
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _pattern(shape: tuple[int, int], offset: int = 0) -> np.ndarray:
    rr, cc = np.indices(shape)
    return PALETTE[(rr + cc + int(offset)) % 3]


def _candidate_offsets(grid: np.ndarray) -> list[int]:
    visible = np.argwhere((grid != BLACK) & (grid != GREY))
    candidates: list[int] = []
    for offset in range(3):
        ok = True
        for r, c in visible:
            if int(grid[r, c]) != int(PALETTE[(int(r) + int(c) + offset) % 3]):
                ok = False
                break
        if ok:
            candidates.append(offset)
    return candidates


def _infer_offset(grid: np.ndarray) -> int:
    candidates = _candidate_offsets(grid)
    assert len(candidates) == 1, "expected a unique diagonal phase"
    return int(candidates[0])


def _interior_masks_after_fill(grid: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for comp in _components(grid != GREY):
        if _touches_border(comp, grid.shape):
            continue
        mask = np.zeros(grid.shape, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        masks.append(mask)
    return masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the diagonal 3-cycle, then swap blue and orange inside every gray-enclosed region.

    Input structure:
    - The grid is a 3-color diagonal periodic pattern using blue, green, and orange, with some cells replaced by black holes.
    - One or two single-pixel-thick gray closed loops are overlaid on top of the pattern.
    - Gray pixels are not part of the periodic pattern, and enclosed regions may contain both missing and intact pattern cells.

    Rule:
    - Infer which diagonal phase of the blue/green/orange cycle is visible in the non-black, non-gray cells.
    - Replace every black hole with its periodic diagonal color while leaving gray loop pixels unchanged.
    - Treat gray pixels as walls; every 4-connected non-gray component that does not touch the grid border is an enclosed interior.
    - Inside each enclosed interior, swap blue with orange and keep green unchanged.

    Key insight:
    - The black cells are missing samples from a global diagonal texture, but the color swap is local and only becomes well-defined after the full texture is restored and the gray loops reveal which cells are interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    offset = _infer_offset(g)
    out = g.copy()
    filled = _pattern(g.shape, offset)
    out[out == BLACK] = filled[out == BLACK]

    for mask in _interior_masks_after_fill(out):
        for color_in, color_out in TOGGLE.items():
            out[mask & (out == color_in)] = color_out

    return out


def _rect_boundary_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, h, w = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c0 + w] = True
    mask[r0 + h - 1, c0 : c0 + w] = True
    mask[r0 : r0 + h, c0] = True
    mask[r0 : r0 + h, c0 + w - 1] = True
    return mask


def _rect_interior_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, h, w = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = True
    return mask


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    row_gap = max(ar0 - br1, br0 - ar1, 0)
    col_gap = max(ac0 - bc1, bc0 - ac1, 0)
    return max(row_gap, col_gap) >= int(gap)


def _choose_rings(shape: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    n_rings = 1 if rng.random() < 0.45 else 2
    touch_index = int(rng.integers(n_rings)) if rng.random() < 0.35 else -1
    rings: list[tuple[int, int, int, int]] = []
    for idx in range(n_rings):
        candidates: list[tuple[int, int, int, int]] = []
        for rh in range(4, min(8, h + 1)):
            for rw in range(4, min(8, w + 1)):
                for r0 in range(h - rh + 1):
                    for c0 in range(w - rw + 1):
                        touches = r0 == 0 or c0 == 0 or r0 + rh == h or c0 + rw == w
                        if idx == touch_index and not touches:
                            continue
                        if idx != touch_index and touches and rng.random() < 0.85:
                            continue
                        rect = (r0, c0, rh, rw)
                        if all(_boxes_separated(rect, prev, gap=2) for prev in rings):
                            candidates.append(rect)
        if not candidates:
            return None
        rings.append(candidates[int(rng.integers(len(candidates)))])
    return rings


def _sample_connected_patch(
    allowed: np.ndarray, rng: np.random.Generator, min_size: int, max_size: int
) -> np.ndarray | None:
    coords = np.argwhere(allowed)
    if coords.size == 0 or max_size < min_size:
        return None
    target = int(rng.integers(min_size, max_size + 1))
    start_r, start_c = coords[int(rng.integers(len(coords)))]
    patch = np.zeros_like(allowed, dtype=bool)
    patch[int(start_r), int(start_c)] = True

    while int(patch.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(patch):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc] and not patch[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        patch[nr, nc] = True
    return patch if int(patch.sum()) >= min_size else None


def _place_holes_in_region(
    region_mask: np.ndarray, rng: np.random.Generator, total_limit: int, patch_count: int
) -> np.ndarray | None:
    region_area = int(region_mask.sum())
    if region_area <= 1 or total_limit <= 0:
        return None
    holes = np.zeros_like(region_mask, dtype=bool)
    remaining = min(total_limit, region_area - 1)
    available = region_mask.copy()
    for patch_index in range(patch_count):
        if remaining <= 0:
            break
        min_left_for_visible = 1
        max_here = remaining - max(0, patch_count - patch_index - 1)
        max_here = min(max_here, max(1, region_area - int(holes.sum()) - min_left_for_visible))
        patch = _sample_connected_patch(available, rng, 1, max_here)
        if patch is None:
            break
        holes |= patch
        available &= ~patch
        remaining = min(total_limit, region_area - 1) - int(holes.sum())
    return holes if int(holes.sum()) >= 1 else None


def _toggle_expected(base: np.ndarray, interior_masks: list[np.ndarray]) -> np.ndarray:
    out = base.copy()
    for mask in interior_masks:
        out[mask & (out == BLUE)] = ORANGE
        out[mask & (out == ORANGE)] = BLUE
    return out


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    shape = (int(rng.integers(10, 16)), int(rng.integers(10, 16)))
    base = _pattern(shape, offset=0)
    rings = _choose_rings(shape, rng)
    if rings is None:
        return None

    boundary_mask = np.zeros(shape, dtype=bool)
    interior_masks: list[np.ndarray] = []
    for rect in rings:
        boundary_mask |= _rect_boundary_mask(shape, rect)
        interior_masks.append(_rect_interior_mask(shape, rect))
    all_interiors = np.zeros(shape, dtype=bool)
    for mask in interior_masks:
        all_interiors |= mask

    holes = np.zeros(shape, dtype=bool)
    for interior in interior_masks:
        area = int(interior.sum())
        patch_count = 1 if area < 8 or rng.random() < 0.7 else 2
        hole_mask = _place_holes_in_region(interior, rng, total_limit=max(1, area // 2), patch_count=patch_count)
        if hole_mask is None:
            return None
        visible_inside = interior & ~hole_mask
        if not np.any(visible_inside):
            return None
        holes |= hole_mask

    outside_region = ~boundary_mask & ~all_interiors
    outside_limit = min(8, max(2, int(outside_region.sum()) // 8))
    outside_holes = _place_holes_in_region(
        outside_region,
        rng,
        total_limit=outside_limit,
        patch_count=1 if rng.random() < 0.35 else 2,
    )
    if outside_holes is None:
        return None
    holes |= outside_holes

    inp = base.copy()
    inp[holes] = BLACK
    inp[boundary_mask] = GREY

    expected = _toggle_expected(base, interior_masks)
    expected[boundary_mask] = GREY

    if _candidate_offsets(inp) != [0]:
        return None
    if len(_interior_masks_after_fill(main(inp))) != len(rings):
        return None
    if not np.any(outside_holes):
        return None

    visible_inside_colors = set(map(int, inp[all_interiors & (inp != BLACK)].flat))
    if not {BLUE, ORANGE}.issubset(visible_inside_colors):
        return None
    if not np.any(all_interiors & (inp != BLACK) & (inp != GREY) & (inp != expected)):
        return None

    for interior in interior_masks:
        if int(np.count_nonzero(interior & (inp == BLACK))) < 1:
            return None
        if int(np.count_nonzero(interior & (inp != BLACK) & (inp != GREY))) < 1:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
