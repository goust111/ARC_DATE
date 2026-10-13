from __future__ import annotations

import random
from functools import lru_cache
from math import isqrt

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_piece(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, (r1 - r0 + 1, c1 - c0 + 1)


def _separator_row(g: np.ndarray) -> int:
    h = g.shape[0]
    candidates = []
    for r in range(1, h - 1):
        if int(g[r, 0]) == BG:
            continue
        if np.all(g[r] == g[r, 0]) and np.any(g[r - 1] != BG) and np.any(g[r + 1 :] != BG):
            candidates.append(r)
    assert len(candidates) == 1, "expected one full-width non-black separator row"
    return candidates[0]


@lru_cache(maxsize=None)
def _can_tile_signature(
    signature: tuple[tuple[tuple[int, int], ...], ...],
    rect_h: int,
    rect_w: int,
) -> bool:
    pieces = list(signature)
    n = len(pieces)
    if n == 0:
        return False

    indexed = []
    for piece in pieces:
        ph = 1 + max(r for r, _ in piece)
        pw = 1 + max(c for _, c in piece)
        if ph > rect_h or pw > rect_w:
            return False
        indexed.append((piece, ph, pw))

    indexed.sort(key=lambda item: len(item[0]), reverse=True)
    placements_by_piece: list[dict[int, list[int]]] = []
    full_mask = (1 << (rect_h * rect_w)) - 1

    for piece, ph, pw in indexed:
        cover_map: dict[int, list[int]] = {}
        for dr in range(rect_h - ph + 1):
            for dc in range(rect_w - pw + 1):
                mask = 0
                covered: list[int] = []
                for r, c in piece:
                    idx = (r + dr) * rect_w + (c + dc)
                    covered.append(idx)
                    mask |= 1 << idx
                for idx in covered:
                    cover_map.setdefault(idx, []).append(mask)
        placements_by_piece.append(cover_map)

    @lru_cache(maxsize=None)
    def dfs(used_mask: int, occupied_mask: int) -> bool:
        if used_mask == (1 << n) - 1:
            return occupied_mask == full_mask

        remaining = full_mask ^ occupied_mask
        first_empty = (remaining & -remaining).bit_length() - 1
        piece_order = sorted(
            (
                i
                for i in range(n)
                if not (used_mask & (1 << i)) and first_empty in placements_by_piece[i]
            ),
            key=lambda i: len(placements_by_piece[i][first_empty]),
        )
        for i in piece_order:
            for placement in placements_by_piece[i][first_empty]:
                if placement & occupied_mask:
                    continue
                if dfs(used_mask | (1 << i), occupied_mask | placement):
                    return True
        return False

    return dfs(0, 0)


def _infer_rectangle_dims(piece_shapes: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]) -> tuple[int, int]:
    area = sum(len(shape) for shape, _dims in piece_shapes)
    signature = tuple(shape for shape, _dims in sorted(piece_shapes))
    candidates: list[tuple[int, int]] = []

    for h in range(1, isqrt(area) + 1):
        if area % h != 0:
            continue
        w = area // h
        for rect_h, rect_w in {(h, w), (w, h)}:
            if _can_tile_signature(signature, rect_h, rect_w):
                candidates.append((rect_h, rect_w))

    candidates = sorted(set(candidates))
    assert len(candidates) == 1, "rectangle dimensions must be uniquely determined"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments, reassemble each group into a solid rectangle, then recolor by area rank.

    The input has a one-cell-high color legend above a full-width separator row.
    Below the separator, each non-black color appears in 2-4 disconnected fragments
    that can be translated (without rotation or reflection) to tile a solid rectangle.
    The output keeps the same canvas size, removes the legend/separator, and paints
    the reassembled rectangles using the legend colors in ascending area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_r = _separator_row(g)
    legend_row = g[sep_r - 1]
    legend_cols = np.flatnonzero(legend_row != BG)
    legend_colors = [int(legend_row[c]) for c in legend_cols]
    assert legend_colors
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be distinct"

    lower = g[sep_r + 1 :, :]
    source_colors = sorted(int(color) for color in np.unique(lower) if int(color) != BG)
    assert len(source_colors) == len(legend_colors), "legend entry count must match object count"

    groups = []
    for color in source_colors:
        comps = _find_components(lower == color)
        assert 1 < len(comps) <= 4, "each color should be split into 2-4 fragments"

        piece_shapes = []
        top_lefts = []
        total_area = 0
        for comp in comps:
            total_area += len(comp)
            r0, c0, _r1, _c1 = _bbox(comp)
            piece_shapes.append(_normalize_piece(comp))
            top_lefts.append((r0 + sep_r + 1, c0))

        rect_h, rect_w = _infer_rectangle_dims(piece_shapes)
        anchor_r = min(r for r, _ in top_lefts)
        anchor_c = min(c for _, c in top_lefts)
        assert anchor_r + rect_h <= h and anchor_c + rect_w <= w
        groups.append(
            {
                "area": total_area,
                "anchor": (anchor_r, anchor_c),
                "dims": (rect_h, rect_w),
            }
        )

    areas = [group["area"] for group in groups]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct"

    out = np.full_like(g, BG)
    for legend_color, group in zip(legend_colors, sorted(groups, key=lambda item: item["area"])):
        r0, c0 = group["anchor"]
        rect_h, rect_w = group["dims"]
        patch = out[r0 : r0 + rect_h, c0 : c0 + rect_w]
        assert np.all(patch == BG), "reassembled rectangles must not overlap"
        patch[:] = int(legend_color)
    return out


def _component_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _partition_rectangle(rect_h: int, rect_w: int, piece_count: int) -> list[set[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    if piece_count >= len(cells):
        return None

    seeds = [(0, 0)]
    remaining = [cell for cell in cells if cell != (0, 0)]
    random.shuffle(remaining)
    seeds.extend(remaining[: piece_count - 1])

    pieces = [set([seed]) for seed in seeds]
    owned = {seed for seed in seeds}

    while len(owned) < len(cells):
        choices = []
        for idx, piece in enumerate(pieces):
            frontier = set()
            for r, c in piece:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < rect_h and 0 <= nc < rect_w and (nr, nc) not in owned:
                        frontier.add((nr, nc))
            for cell in frontier:
                choices.append((idx, cell))
        if not choices:
            return None
        idx, cell = random.choice(choices)
        pieces[idx].add(cell)
        owned.add(cell)

    if any(not _component_connected(piece) for piece in pieces):
        return None
    if min(len(piece) for piece in pieces) < 1:
        return None

    anchor_candidates = []
    for idx, piece in enumerate(pieces):
        rows = [r for r, _ in piece]
        cols = [c for _, c in piece]
        if min(rows) == 0 and min(cols) == 0:
            anchor_candidates.append(idx)
    if anchor_candidates != [0]:
        return None

    return pieces


def _piece_patch(piece: set[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in piece]
    cols = [c for _, c in piece]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in piece:
        patch[r - r0, c - c0] = color
    return patch


def _patch_mask(patch: np.ndarray) -> np.ndarray:
    return patch != BG


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    oh, ow = occupied.shape
    if top < 0 or left < 0 or top + ph > oh or left + pw > ow:
        return False
    mask = _patch_mask(patch)
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        r0 = max(0, rr - 1)
        c0 = max(0, cc - 1)
        r1 = min(oh, rr + 2)
        c1 = min(ow, cc + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = _patch_mask(patch)
    canvas[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]
    occupied[top : top + patch.shape[0], left : left + patch.shape[1]] |= mask


def _random_patch_position(
    occupied: np.ndarray,
    patch: np.ndarray,
    min_top: int,
    min_left: int,
) -> tuple[int, int] | None:
    positions = []
    oh, ow = occupied.shape
    ph, pw = patch.shape
    for top in range(min_top, oh - ph + 1):
        for left in range(min_left, ow - pw + 1):
            if _can_place_patch(occupied, patch, top, left):
                positions.append((top, left))
    if not positions:
        return None
    return random.choice(positions)


def _slot_regions(height: int, width: int) -> list[tuple[int, int, int, int]]:
    lower_top = 2
    mid_r = lower_top + (height - lower_top) // 2
    mid_c = width // 2
    raw = [
        (lower_top, mid_r, 0, mid_c),
        (lower_top, mid_r, mid_c, width),
        (mid_r, height, 0, mid_c),
        (mid_r, height, mid_c, width),
    ]
    regions = []
    for r0, r1, c0, c1 in raw:
        rr0 = r0 + 1
        rr1 = r1 - 1
        cc0 = c0 + 1
        cc1 = c1 - 1
        if rr1 - rr0 >= 6 and cc1 - cc0 >= 6:
            regions.append((rr0, rr1, cc0, cc1))
        else:
            regions.append((r0, r1, c0, c1))
    return regions


def _sample_rectangle(slot_h: int, slot_w: int, used_areas: set[int]) -> tuple[int, int] | None:
    candidates = []
    for rect_h in range(2, min(6, slot_h - 1)):
        for rect_w in range(2, min(6, slot_w - 1)):
            area = rect_h * rect_w
            if area in used_areas:
                continue
            if area >= 5:
                candidates.append((rect_h, rect_w))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_group_in_slot(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    src_color: int,
    used_areas: set[int],
) -> dict[str, tuple[int, int] | int] | None:
    r0, r1, c0, c1 = region
    slot_h = r1 - r0
    slot_w = c1 - c0
    rect_dims = _sample_rectangle(slot_h, slot_w, used_areas)
    if rect_dims is None:
        return None
    rect_h, rect_w = rect_dims
    piece_count = random.randint(2, min(4, rect_h * rect_w - 1))

    pieces = _partition_rectangle(rect_h, rect_w, piece_count)
    if pieces is None:
        return None

    piece_shapes = [_normalize_piece(sorted(piece)) for piece in pieces]
    try:
        inferred_dims = _infer_rectangle_dims(piece_shapes)
    except AssertionError:
        return None
    if inferred_dims != (rect_h, rect_w):
        return None

    anchor_top = 0
    anchor_left = 0

    anchor_patch = _piece_patch(pieces[0], src_color)
    local_canvas = np.full((slot_h, slot_w), BG, dtype=int)
    occupied = np.zeros((slot_h, slot_w), dtype=bool)
    if not _can_place_patch(occupied, anchor_patch, anchor_top, anchor_left):
        return None
    _place_patch(local_canvas, occupied, anchor_patch, anchor_top, anchor_left)

    other_patches = [_piece_patch(piece, src_color) for piece in pieces[1:]]
    random.shuffle(other_patches)
    for patch in other_patches:
        position = _random_patch_position(occupied, patch, anchor_top + 1, anchor_left + 1)
        if position is None:
            return None
        top, left = position
        _place_patch(local_canvas, occupied, patch, top, left)

    grid[r0:r1, c0:c1][local_canvas != BG] = local_canvas[local_canvas != BG]
    return {
        "anchor": (r0, c0),
        "dims": (rect_h, rect_w),
        "area": rect_h * rect_w,
    }


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    regions = _slot_regions(height, width)

    group_count = random.randint(3, 4)
    slot_indices = random.sample(range(4), group_count)
    chosen_regions = [regions[idx] for idx in slot_indices]
    if any((r1 - r0) < 7 or (c1 - c0) < 7 for r0, r1, c0, c1 in chosen_regions):
        return None

    colors = random.sample(PALETTE, 2 * group_count)
    source_colors = colors[:group_count]
    legend_colors = colors[group_count:]

    grid = np.full((height, width), BG, dtype=int)
    legend_start = random.randint(0, width - group_count)
    for offset, color in enumerate(legend_colors):
        grid[0, legend_start + offset] = color
    grid[1, :] = SEPARATOR_COLOR

    used_areas: set[int] = set()
    group_specs = list(zip(chosen_regions, source_colors, legend_colors))
    random.shuffle(group_specs)
    group_infos = []

    for region, src_color, _dst_color in group_specs:
        info = _build_group_in_slot(grid, region, src_color, used_areas)
        if info is None:
            return None
        used_areas.add(int(info["area"]))
        group_infos.append(info)

    expected = np.full((height, width), BG, dtype=int)
    for legend_color, info in zip(legend_colors, sorted(group_infos, key=lambda item: int(item["area"]))):
        r0, c0 = info["anchor"]
        rect_h, rect_w = info["dims"]
        expected[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(legend_color)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
