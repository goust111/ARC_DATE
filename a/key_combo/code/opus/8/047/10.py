from __future__ import annotations

import random
from typing import Iterable

import numpy as np


SELECTOR_COLOR = 5
BACKGROUND = 0
SELECTOR_SIZE = 7
SPRITE_SIZE = 3
MARKER_POSITIONS = ((1, 1), (1, 5), (5, 1), (5, 5))
QUADRANT_ANCHORS = {
    (1, 1): (0, 0),
    (1, 5): (0, 4),
    (5, 1): (4, 0),
    (5, 5): (4, 4),
}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_RANGE = (18, 24)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    raise ValueError(f"unsupported connectivity: {connectivity}")


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _count_holes(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    holes = 0
    for comp in _components(~padded, connectivity=4):
        rr = comp[:, 0]
        cc = comp[:, 1]
        if np.any((rr == 0) | (rr == padded.shape[0] - 1) | (cc == 0) | (cc == padded.shape[1] - 1)):
            continue
        holes += 1
    return holes


def _is_selector(block: np.ndarray) -> bool:
    if block.shape != (SELECTOR_SIZE, SELECTOR_SIZE):
        return False
    for r in range(SELECTOR_SIZE):
        for c in range(SELECTOR_SIZE):
            if (r, c) in QUADRANT_ANCHORS:
                continue
            if int(block[r, c]) != SELECTOR_COLOR:
                return False
    marker_colors = [int(block[r, c]) for r, c in MARKER_POSITIONS]
    return len(set(marker_colors)) == 4 and all(color not in (BACKGROUND, SELECTOR_COLOR) for color in marker_colors)


def _find_selector(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[tuple[int, int], int]]:
    h, w = grid.shape
    matches: list[tuple[tuple[int, int, int, int], dict[tuple[int, int], int]]] = []
    for r0 in range(h - SELECTOR_SIZE + 1):
        for c0 in range(w - SELECTOR_SIZE + 1):
            block = grid[r0 : r0 + SELECTOR_SIZE, c0 : c0 + SELECTOR_SIZE]
            if not _is_selector(block):
                continue
            markers = {(r, c): int(block[r, c]) for r, c in MARKER_POSITIONS}
            matches.append(((r0, c0, r0 + SELECTOR_SIZE - 1, c0 + SELECTOR_SIZE - 1), markers))
    assert len(matches) == 1
    return matches[0]


def _extract_sprite(grid: np.ndarray, color: int, selector_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = grid == int(color)
    r0, c0, r1, c1 = selector_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask, connectivity=8)
    assert len(comps) == 1
    coords = comps[0]
    br0, bc0, br1, bc1 = _bbox(coords)
    sprite = np.full((br1 - br0 + 1, bc1 - bc0 + 1), BACKGROUND, dtype=int)
    sprite[coords[:, 0] - br0, coords[:, 1] - bc0] = int(color)
    return sprite


def _center_in_quadrant(sprite: np.ndarray) -> np.ndarray:
    h, w = sprite.shape
    assert h <= SPRITE_SIZE and w <= SPRITE_SIZE
    out = np.full((SPRITE_SIZE, SPRITE_SIZE), BACKGROUND, dtype=int)
    r0 = (SPRITE_SIZE - h) // 2
    c0 = (SPRITE_SIZE - w) // 2
    _paste(out, sprite, r0, c0, transparent=BACKGROUND)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy only the selector-matched sprites whose topology contains exactly one enclosed hole into the output quadrants.

    Input structure:
    - The grid has a black background, one 7x7 gray selector block, and four external monochrome 8-connected sprites.
    - The selector block is gray except for one colored marker pixel at the center of each 3x3 quadrant.
    - Each marker color matches exactly one external sprite of the same color.
    - Some matched sprites enclose one black 4-connected hole inside their tight bbox, while others have a different hole count.

    Rule:
    - Locate the 7x7 selector block and read the four marker colors from its quadrant centers.
    - For each marker color, find the unique external sprite with that color and count its enclosed black 4-connected holes.
    - Create a 7x7 black output canvas split into four 3x3 quadrants by a blank middle row and column.
    - Copy a matched sprite into its corresponding output quadrant only when its hole count is exactly one; leave other quadrants black.

    Key insight:
    - Marker color identifies which external sprite belongs to each quadrant, but copying is gated by a second test on the sprite's hole topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    selector_bbox, markers = _find_selector(g)
    out = np.full((SELECTOR_SIZE, SELECTOR_SIZE), BACKGROUND, dtype=int)
    for pos, color in markers.items():
        sprite = _extract_sprite(g, color, selector_bbox)
        if _count_holes(sprite == int(color)) != 1:
            continue
        patch = _center_in_quadrant(sprite)
        anchor_r, anchor_c = QUADRANT_ANCHORS[pos]
        _paste(out, patch, anchor_r, anchor_c, transparent=BACKGROUND)
    return out


def _enumerate_sprite_masks() -> dict[int, list[np.ndarray]]:
    masks = {0: [], 1: []}
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=bool).reshape(
            SPRITE_SIZE, SPRITE_SIZE
        )
        if int(mask.sum()) < 4:
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        holes = _count_holes(mask)
        if holes in masks:
            masks[holes].append(mask)
    assert len(masks[0]) >= 10 and len(masks[1]) >= 10
    return masks


SPRITE_MASKS = _enumerate_sprite_masks()


def _bbox_has_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _make_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BACKGROUND, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _make_selector(marker_colors: Iterable[int]) -> np.ndarray:
    selector = np.full((SELECTOR_SIZE, SELECTOR_SIZE), SELECTOR_COLOR, dtype=int)
    for (r, c), color in zip(MARKER_POSITIONS, marker_colors):
        selector[r, c] = int(color)
    return selector


def _choose_layout(h: int, w: int, count: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    selector_positions = [(r, c) for r in range(h - SELECTOR_SIZE + 1) for c in range(w - SELECTOR_SIZE + 1)]
    random.shuffle(selector_positions)
    for sr, sc in selector_positions:
        selector_bbox = (sr, sc, sr + SELECTOR_SIZE - 1, sc + SELECTOR_SIZE - 1)
        candidates = []
        for r in range(h - SPRITE_SIZE + 1):
            for c in range(w - SPRITE_SIZE + 1):
                bbox = (r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1)
                if _bbox_has_gap(bbox, selector_bbox):
                    candidates.append((r, c))
        random.shuffle(candidates)

        def search(chosen: list[tuple[int, int]], chosen_bboxes: list[tuple[int, int, int, int]], start: int):
            if len(chosen) == count:
                return chosen
            for idx in range(start, len(candidates)):
                r, c = candidates[idx]
                bbox = (r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1)
                if all(_bbox_has_gap(bbox, prev) for prev in chosen_bboxes):
                    found = search(chosen + [(r, c)], chosen_bboxes + [bbox], idx + 1)
                    if found is not None:
                        return found
            return None

        sprite_positions = search([], [], 0)
        if sprite_positions is not None:
            return selector_bbox, sprite_positions
    return None


def _expected_output(assignments: list[tuple[tuple[int, int], int, np.ndarray, int]]) -> np.ndarray:
    out = np.full((SELECTOR_SIZE, SELECTOR_SIZE), BACKGROUND, dtype=int)
    for marker_pos, color, mask, holes in assignments:
        if holes != 1:
            continue
        sprite = _make_sprite(mask, color)
        anchor_r, anchor_c = QUADRANT_ANCHORS[marker_pos]
        _paste(out, sprite, anchor_r, anchor_c, transparent=BACKGROUND)
    return out


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    layout = _choose_layout(h, w, count=4)
    if layout is None:
        return None
    selector_bbox, sprite_positions = layout

    marker_colors = random.sample(COLOR_POOL, 4)
    hole_indices = set(random.sample(range(4), 2))
    masks_with_hole = random.sample(SPRITE_MASKS[1], 2)
    masks_without_hole = random.sample(SPRITE_MASKS[0], 2)

    assignments: list[tuple[tuple[int, int], int, np.ndarray, int]] = []
    one_hole_cursor = 0
    zero_hole_cursor = 0
    for idx, marker_pos in enumerate(MARKER_POSITIONS):
        holes = 1 if idx in hole_indices else 0
        if holes == 1:
            mask = masks_with_hole[one_hole_cursor]
            one_hole_cursor += 1
        else:
            mask = masks_without_hole[zero_hole_cursor]
            zero_hole_cursor += 1
        assignments.append((marker_pos, marker_colors[idx], mask, holes))

    grid = np.full((h, w), BACKGROUND, dtype=int)
    selector = _make_selector(marker_colors)
    sr0, sc0, _, _ = selector_bbox
    _paste(grid, selector, sr0, sc0)
    for (_, color, mask, _), (r, c) in zip(assignments, sprite_positions):
        _paste(grid, _make_sprite(mask, color), r, c, transparent=BACKGROUND)

    expected = _expected_output(assignments)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
