from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
GRID_MAX = 30
GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
COLOR_COUNT_CHOICES = (2, 3)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 6)
RECT_AREA_RANGE = (4, 18)
FRAGMENT_COUNT_RANGE = (2, 4)
PARTITION_RETRIES = 60
PLACEMENT_RETRIES = 60
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            out.append(np.array(coords, dtype=int))

    return out


def _normalize_piece(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    cells = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(cells)


def _piece_dims(piece: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in piece) + 1, max(c for _, c in piece) + 1


def _canonical_pieces(pieces: list[tuple[tuple[int, int], ...]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(sorted(pieces, key=lambda piece: (-len(piece), _piece_dims(piece), piece)))


@lru_cache(maxsize=None)
def _can_tile_rectangle(
    pieces: tuple[tuple[tuple[int, int], ...], ...],
    rect_h: int,
    rect_w: int,
) -> bool:
    if sum(len(piece) for piece in pieces) != rect_h * rect_w:
        return False

    placements: list[dict[int, tuple[int, ...]]] = []
    for piece in pieces:
        piece_h, piece_w = _piece_dims(piece)
        if piece_h > rect_h or piece_w > rect_w:
            return False

        by_cell: dict[int, list[int]] = {}
        for r0 in range(rect_h - piece_h + 1):
            for c0 in range(rect_w - piece_w + 1):
                mask = 0
                touched: list[int] = []
                for pr, pc in piece:
                    idx = (r0 + pr) * rect_w + (c0 + pc)
                    mask |= 1 << idx
                    touched.append(idx)
                for idx in touched:
                    by_cell.setdefault(idx, []).append(mask)
        if not by_cell:
            return False
        placements.append({idx: tuple(masks) for idx, masks in by_cell.items()})

    full_mask = (1 << (rect_h * rect_w)) - 1
    piece_count = len(pieces)

    @lru_cache(maxsize=None)
    def search(used_mask: int, remaining_mask: int) -> bool:
        if remaining_mask == 0:
            return used_mask == full_mask

        missing = full_mask ^ used_mask
        anchor_bit = missing & -missing
        anchor_idx = anchor_bit.bit_length() - 1

        best_piece = -1
        best_candidates: tuple[int, ...] | None = None

        rest = remaining_mask
        while rest:
            bit = rest & -rest
            piece_idx = bit.bit_length() - 1
            candidates = tuple(
                mask
                for mask in placements[piece_idx].get(anchor_idx, ())
                if mask & used_mask == 0
            )
            if candidates and (best_candidates is None or len(candidates) < len(best_candidates)):
                best_piece = piece_idx
                best_candidates = candidates
                if len(best_candidates) == 1:
                    break
            rest ^= bit

        if best_candidates is None:
            return False

        next_remaining = remaining_mask & ~(1 << best_piece)
        for mask in best_candidates:
            if search(used_mask | mask, next_remaining):
                return True
        return False

    return search(0, (1 << piece_count) - 1)


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(math.isqrt(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _infer_rectangle_shape(pieces: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    canonical = _canonical_pieces(pieces)
    area = sum(len(piece) for piece in canonical)
    max_h = max(_piece_dims(piece)[0] for piece in canonical)
    max_w = max(_piece_dims(piece)[1] for piece in canonical)

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for rect_h, rect_w in _factor_pairs(area):
        if rect_h < max_h or rect_w < max_w:
            continue
        if (rect_h, rect_w) in seen:
            continue
        seen.add((rect_h, rect_w))
        if _can_tile_rectangle(canonical, rect_h, rect_w):
            candidates.append((rect_h, rect_w))

    assert len(candidates) == 1, "each color's fragments must reconstruct a unique rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into its full solid rectangle, then stack the rectangles vertically from largest to smallest.

    Input structure:
    - The background is black.
    - Several non-black colors appear, and each color is split into multiple separated 4-connected fragments.
    - All fragments of the same color are translation-only pieces of one solid rectangle.
    - Different colors correspond to rectangles with different pixel areas.

    Rule:
    - Group fragments by color.
    - For each color, infer the unique rectangle shape that the color's fragments can tile exactly when translated without rotation.
    - Sort the reconstructed rectangles by area from largest to smallest.
    - Output a new black canvas whose width is the largest reconstructed width, then paint the solid rectangles top-to-bottom with one black spacer row between consecutive rectangles and all left edges aligned to the first column.

    Key insight:
    - Same-colored fragments are not independent objects: their complementary shapes jointly determine one hidden rectangle, and only those recovered rectangle areas decide the final stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one foreground color"

    rectangles: list[tuple[int, int, int, int]] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) >= 2, "each color must be split into multiple fragments"
        pieces = [_normalize_piece(comp) for comp in comps]
        rect_h, rect_w = _infer_rectangle_shape(pieces)
        rectangles.append((rect_h * rect_w, color, rect_h, rect_w))

    areas = [area for area, _color, _rect_h, _rect_w in rectangles]
    assert len(areas) == len(set(areas)), "rectangle areas must be distinct"

    rectangles.sort(key=lambda item: -item[0])
    out_h = sum(rect_h for _area, _color, rect_h, _rect_w in rectangles) + len(rectangles) - 1
    out_w = max(rect_w for _area, _color, _rect_h, rect_w in rectangles)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for _area, color, rect_h, rect_w in rectangles:
        out[row : row + rect_h, :rect_w] = color
        row += rect_h + 1

    return out


def _all_cells(rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rect_h) for c in range(rect_w)]


def _frontier(piece: set[tuple[int, int]], unassigned: set[tuple[int, int]], rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in piece:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rect_h and 0 <= nc < rect_w and (nr, nc) in unassigned:
                cells.add((nr, nc))
    return list(cells)


def _random_partition(rect_h: int, rect_w: int, part_count: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = _all_cells(rect_h, rect_w)
    if part_count > len(cells):
        return None

    seeds = random.sample(cells, part_count)
    pieces = [{seed} for seed in seeds]
    unassigned = set(cells) - set(seeds)

    while unassigned:
        options: list[tuple[int, list[tuple[int, int]]]] = []
        min_size: int | None = None
        for idx, piece in enumerate(pieces):
            fringe = _frontier(piece, unassigned, rect_h, rect_w)
            if not fringe:
                continue
            size = len(piece)
            min_size = size if min_size is None else min(min_size, size)
            options.append((idx, fringe))
        if not options:
            return None

        if min_size is not None and random.random() < 0.75:
            options = [(idx, fringe) for idx, fringe in options if len(pieces[idx]) == min_size]

        idx, fringe = random.choice(options)
        cell = random.choice(fringe)
        pieces[idx].add(cell)
        unassigned.remove(cell)

    normalized = []
    areas = []
    for piece in pieces:
        coords = np.array(sorted(piece), dtype=int)
        normalized.append(_normalize_piece(coords))
        areas.append(len(piece))

    if max(areas) == 1:
        return None
    if sum(area == 1 for area in areas) > 1 and random.random() < 0.8:
        return None
    return normalized


def _piece_patch(piece: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    piece_h, piece_w = _piece_dims(piece)
    patch = np.full((piece_h, piece_w), BG, dtype=int)
    for r, c in piece:
        patch[r, c] = color
    return patch


def _render_expected(rectangles: list[tuple[int, int, int]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: -(item[1] * item[2]))
    out_h = sum(rect_h for _color, rect_h, _rect_w in ordered) + len(ordered) - 1
    out_w = max(rect_w for _color, _rect_h, rect_w in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for color, rect_h, rect_w in ordered:
        out[row : row + rect_h, :rect_w] = color
        row += rect_h + 1

    return out


def _sample_rectangles() -> list[tuple[int, int]]:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for rect_h in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1):
        for rect_w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1):
            area = rect_h * rect_w
            if RECT_AREA_RANGE[0] <= area <= RECT_AREA_RANGE[1]:
                by_area.setdefault(area, []).append((rect_h, rect_w))

    color_count = random.choice(COLOR_COUNT_CHOICES)
    picked_areas = random.sample(sorted(by_area), color_count)
    return [random.choice(by_area[area]) for area in picked_areas]


def _sample_piece_sets(
    rectangles: list[tuple[int, int]],
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, int, int]]] | None:
    colors = random.sample(PALETTE, len(rectangles))
    patches: list[tuple[int, np.ndarray]] = []
    specs: list[tuple[int, int, int]] = []

    for color, (rect_h, rect_w) in zip(colors, rectangles):
        max_parts = min(FRAGMENT_COUNT_RANGE[1], rect_h * rect_w - 1)
        if max_parts < FRAGMENT_COUNT_RANGE[0]:
            return None
        part_count = random.randint(FRAGMENT_COUNT_RANGE[0], max_parts)

        chosen: list[tuple[tuple[int, int], ...]] | None = None
        for _ in range(PARTITION_RETRIES):
            pieces = _random_partition(rect_h, rect_w, part_count)
            if pieces is None:
                continue
            try:
                inferred = _infer_rectangle_shape(pieces)
            except AssertionError:
                continue
            if inferred != (rect_h, rect_w):
                continue
            chosen = pieces
            break
        if chosen is None:
            return None

        specs.append((color, rect_h, rect_w))
        for piece in chosen:
            patches.append((color, _piece_patch(piece, color)))

    random.shuffle(patches)
    return patches, specs


def _bbox_is_free(reserved: np.ndarray, r0: int, c0: int, patch_h: int, patch_w: int) -> bool:
    r1 = min(reserved.shape[0], r0 + patch_h + 1)
    c1 = min(reserved.shape[1], c0 + patch_w + 1)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    return not bool(reserved[rr0:r1, cc0:c1].any())


def _scatter_patches(patches: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    min_h = max(patch.shape[0] for _color, patch in patches) + 2
    min_w = max(patch.shape[1] for _color, patch in patches) + 2

    for _ in range(PLACEMENT_RETRIES):
        grid_h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
        grid_w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
        if grid_h > GRID_MAX or grid_w > GRID_MAX:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        reserved = np.zeros((grid_h, grid_w), dtype=bool)
        order = sorted(range(len(patches)), key=lambda idx: patches[idx][1].shape[0] * patches[idx][1].shape[1], reverse=True)

        ok = True
        for idx in order:
            color, patch = patches[idx]
            patch_h, patch_w = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(grid_h - patch_h + 1):
                for c0 in range(grid_w - patch_w + 1):
                    if _bbox_is_free(reserved, r0, c0, patch_h, patch_w):
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break

            r0, c0 = random.choice(candidates)
            mask = patch != BG
            grid[r0 : r0 + patch_h, c0 : c0 + patch_w][mask] = color
            reserved[r0 : r0 + patch_h, c0 : c0 + patch_w] = True

        if ok:
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    rectangles = _sample_rectangles()
    sampled = _sample_piece_sets(rectangles)
    if sampled is None:
        return None
    patches, specs = sampled

    grid = _scatter_patches(patches)
    if grid is None:
        return None

    expected = _render_expected(specs)
    if max(grid.shape) > GRID_MAX or max(expected.shape) > GRID_MAX:
        return None

    try:
        return grid if np.array_equal(main(grid), expected) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
