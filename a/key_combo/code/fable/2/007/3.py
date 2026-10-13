from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


SQUARE = 0
MISS_TL = 1
MISS_BL = 2
MISS_TR = 3

TILE_OFFSETS = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
)
BAR_COLORS = (1, 2, 3, 4)
GEN_COLORS = (0, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 700
GEN_PHASE_COUNT = 6

_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_mask(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    mask = g == fg_colors[0]
    assert mask.any(), "expected foreground pixels"
    return bg, mask


def _build_cover(mask: np.ndarray) -> tuple[list[dict], list[list[int]], int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_to_id = {coord: idx for idx, coord in enumerate(coords)}
    placements: list[dict] = []
    by_cell: list[list[int]] = [[] for _ in coords]
    h, w = mask.shape

    for r in range(h - 1):
        for c in range(w - 1):
            for tile_type, offsets in enumerate(TILE_OFFSETS):
                cell_ids: list[int] = []
                for dr, dc in offsets:
                    rr = r + dr
                    cc = c + dc
                    if not mask[rr, cc]:
                        break
                    cell_ids.append(cell_to_id[(rr, cc)])
                else:
                    bitmask = 0
                    for cell_id in cell_ids:
                        bitmask |= 1 << cell_id
                    placement = {
                        "type": tile_type,
                        "anchor": (r, c),
                        "cells": tuple(cell_ids),
                        "bitmask": bitmask,
                    }
                    index = len(placements)
                    placements.append(placement)
                    for cell_id in cell_ids:
                        by_cell[cell_id].append(index)

    return placements, by_cell, len(coords)


def _solve_unique_cover(mask: np.ndarray, *, max_solutions: int = 2) -> dict:
    placements, by_cell, cell_count = _build_cover(mask)
    assert cell_count > 0, "expected foreground cells"
    assert all(options for options in by_cell), "foreground contains an untileable cell"
    full_mask = (1 << cell_count) - 1
    solutions: list[tuple[int, int, int, int]] = []

    def choose_cell(remaining: int) -> list[int] | None:
        best: list[int] | None = None
        bits = remaining
        while bits:
            bit = bits & -bits
            cell_id = bit.bit_length() - 1
            options = [
                placement_id
                for placement_id in by_cell[cell_id]
                if placements[placement_id]["bitmask"] & remaining == placements[placement_id]["bitmask"]
            ]
            if not options:
                return []
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        return best

    def search(remaining: int, counts: list[int]) -> None:
        if len(solutions) >= max_solutions:
            return
        if remaining == 0:
            solutions.append(tuple(counts))
            return

        options = choose_cell(remaining)
        if not options:
            return
        for placement_id in options:
            placement = placements[placement_id]
            counts[placement["type"]] += 1
            search(remaining & ~placement["bitmask"], counts)
            counts[placement["type"]] -= 1
            if len(solutions) >= max_solutions:
                return

    search(full_mask, [0, 0, 0, 0])
    return {
        "solution_count": len(solutions),
        "counts": solutions[0] if len(solutions) == 1 else None,
        "candidate_count": len(placements),
    }


def _component_areas(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            area = 0
            while queue:
                rr, cc = queue.popleft()
                area += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            areas.append(area)
    return areas


def _cells_for_tile(tile_type: int, anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = anchor
    return tuple((r + dr, c + dc) for dr, dc in TILE_OFFSETS[tile_type])


def _touch_score(occupied: np.ndarray, cells: tuple[tuple[int, int], ...]) -> int:
    h, w = occupied.shape
    score = 0
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                score += 1
    return score


def _expanded_bbox_area(current_bbox: tuple[int, int, int, int] | None, cells: tuple[tuple[int, int], ...]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if current_bbox is None:
        r0 = min(rows)
        c0 = min(cols)
        r1 = max(rows)
        c1 = max(cols)
    else:
        r0, c0, r1, c1 = current_bbox
        r0 = min(r0, min(rows))
        c0 = min(c0, min(cols))
        r1 = max(r1, max(rows))
        c1 = max(c1, max(cols))
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _choose_target_counts(phase: int) -> tuple[int, int, int, int]:
    counts = [1, 1, 1, 1]
    if 1 <= phase <= 4:
        counts[phase - 1] = 0

    allowed = [idx for idx in range(4) if counts[idx] > 0]
    extra_lo, extra_hi = (3, 5) if phase == 5 else (1, 4)
    for _ in range(random.randint(extra_lo, extra_hi)):
        counts[random.choice(allowed)] += 1
    return tuple(counts)


def _candidate_anchors(
    occupied: np.ndarray,
    tile_type: int,
    *,
    prefer_touch: bool,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    current_bbox = _bbox(occupied)
    ranked: list[tuple[int, int, float, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            cells = _cells_for_tile(tile_type, (r, c))
            if any(occupied[rr, cc] for rr, cc in cells):
                continue
            touch = _touch_score(occupied, cells)
            if prefer_touch and current_bbox is not None and touch == 0:
                continue
            area = _expanded_bbox_area(current_bbox, cells)
            ranked.append((-touch, area, random.random(), r, c))

    if not ranked and prefer_touch:
        return _candidate_anchors(occupied, tile_type, prefer_touch=False)

    ranked.sort()
    keep = ranked[: min(len(ranked), 18)]
    return [(r, c) for _, _, _, r, c in keep]


def _build_mask_from_counts(counts: tuple[int, int, int, int], phase: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    total_tiles = int(sum(counts))
    min_side = max(7, 4 + (total_tiles + 1) // 2)
    board_h = random.randint(min_side, min(12, min_side + 3))
    board_w = random.randint(min_side, min(12, min_side + 3))
    occupied = np.zeros((board_h, board_w), dtype=bool)

    order: list[int] = []
    for tile_type, count in enumerate(counts):
        order.extend([tile_type] * count)
    random.shuffle(order)

    cluster_bias = 0.92 if phase == 5 else 0.78
    placed = []
    for tile_type in order:
        anchors = _candidate_anchors(
            occupied,
            tile_type,
            prefer_touch=bool(placed) and random.random() < cluster_bias,
        )
        if not anchors:
            return None
        anchor = random.choice(anchors[: min(len(anchors), 6)])
        cells = _cells_for_tile(tile_type, anchor)
        for r, c in cells:
            occupied[r, c] = True
        placed.append((tile_type, anchor))

    areas = _component_areas(occupied)
    if max(areas, default=0) < (8 if phase == 5 else 6):
        return None

    solved = _solve_unique_cover(occupied)
    if solved["solution_count"] != 1 or solved["counts"] != counts:
        return None

    extra_candidates = solved["candidate_count"] - total_tiles
    if extra_candidates < (2 if phase in (0, 5) else 1):
        return None

    return occupied, counts


def _embed_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    fg_count = int(crop.sum())

    top = random.randint(1, 5)
    bottom = random.randint(1, 5)
    left = random.randint(1, 5)
    right = random.randint(1, 5)
    height = crop.shape[0] + top + bottom
    width = crop.shape[1] + left + right

    while height * width <= 2 * fg_count:
        if height <= width and height < 30:
            if random.random() < 0.5:
                top += 1
            else:
                bottom += 1
            height += 1
        elif width < 30:
            if random.random() < 0.5:
                left += 1
            else:
                right += 1
            width += 1
        else:
            break

    assert height <= 30 and width <= 30
    bg = random.choice(GEN_COLORS)
    fg = random.choice([color for color in GEN_COLORS if color != bg])
    grid = np.full((height, width), bg, dtype=int)
    grid[top : top + crop.shape[0], left : left + crop.shape[1]][crop] = fg
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the single-color foreground into the unique allowed 2x2 tiles and output only the per-tile count bars.

    Input structure:
    - The grid has one dominant background color and exactly one foreground color.
    - The foreground mask is exactly coverable, once and only once, by translated copies of four fixed tiles:
      the solid 2x2 square, the 2x2 L missing its top-left cell, the 2x2 L missing its bottom-left cell,
      and the 2x2 L missing its top-right cell.
    - Different tiles may touch and interlock, so the partition is not directly visible from local background gaps.

    Rule:
    - Recover the unique exact cover of the foreground mask by those four tile classes.
    - Count how many placements of each class appear, in the order:
      square, missing-top-left L, missing-bottom-left L, missing-top-right L.
    - Output a 4-row grid on the input background color whose row lengths are those counts, colored blue, red, green, yellow.

    Key insight:
    - The output depends only on the hidden global exact cover counts, so the solver must reason over the whole foreground mask rather than classify local 2x2 patches greedily.
    """
    g = np.asarray(grid, dtype=int)
    bg, fg_mask = _extract_mask(g)
    solved = _solve_unique_cover(fg_mask)
    assert solved["solution_count"] == 1, "expected a unique exact cover"
    counts = solved["counts"]
    assert counts is not None
    width = max(counts)
    assert width > 0

    out = np.full((4, width), bg, dtype=int)
    for row, (count, color) in enumerate(zip(counts, BAR_COLORS)):
        if count:
            out[row, :count] = color
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    phase = _GEN_CALL_COUNT % GEN_PHASE_COUNT
    _GEN_CALL_COUNT += 1
    counts = _choose_target_counts(phase)

    for _ in range(GEN_ATTEMPTS):
        built = _build_mask_from_counts(counts, phase)
        if built is None:
            continue
        mask, expected_counts = built
        grid = _embed_mask(mask)
        solved = _solve_unique_cover(_extract_mask(grid)[1])
        if solved["solution_count"] == 1 and solved["counts"] == expected_counts:
            return grid

    raise RuntimeError("failed to generate a valid invisible-partition sample")
