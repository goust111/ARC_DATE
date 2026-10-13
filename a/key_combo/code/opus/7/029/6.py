from __future__ import annotations

import random

import numpy as np


BG = 0
BASE = 1
MARK = 2

RECT_COUNT_RANGE = (3, 4)
RECT_HEIGHT_RANGE = (5, 7)
RECT_WIDTH_OPTIONS = (5, 7, 9)
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 24
PLACEMENT_BORDER = 1
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _complete_vertical_symmetry(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    h, w = out.shape
    assert w % 2 == 1, "selected rectangle must have an integer vertical axis"
    assert set(np.unique(out)).issubset({BASE, MARK})

    axis = w // 2
    for r, c in np.argwhere(out == MARK):
        mirror_c = 2 * axis - int(c)
        assert 0 <= mirror_c < w
        if out[int(r), mirror_c] == BASE:
            out[int(r), mirror_c] = MARK
        else:
            assert out[int(r), mirror_c] == MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the sparsest filled rectangle and complete its red pattern by vertical mirroring.

    Input structure:
    - A black background contains several separated filled rectangles of the same size.
    - Each rectangle uses blue as its base fill and red as its interior marker color.
    - Rectangles are axis-aligned, crop cleanly to their bounding boxes, and do not touch each other.

    Rule:
    - Find every non-black connected rectangle and count its non-blue marker pixels.
    - Select the unique rectangle with the fewest markers.
    - Inside that rectangle, mirror every red marker across the rectangle's vertical center line,
      filling any blue mirror location with red.
    - Return the tight crop of the completed rectangle.

    Key insight:
    The rectangle with the fewest visible red pixels is the incomplete symmetric template, so
    the correct action is to choose the sparsest rectangle first and only then mirror its markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects: list[tuple[int, np.ndarray]] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != BG), "each object must fill its bounding box"
        assert set(np.unique(patch)).issubset({BASE, MARK}), "unexpected rectangle colors"
        objects.append((int(np.count_nonzero(patch != BASE)), patch))

    assert objects, "expected at least one rectangle"
    shapes = {patch.shape for _count, patch in objects}
    assert len(shapes) == 1, "all rectangles must share one size"

    min_count = min(count for count, _patch in objects)
    winners = [patch for count, patch in objects if count == min_count]
    assert len(winners) == 1, "expected a unique sparsest rectangle"
    return _complete_vertical_symmetry(winners[0])


def _interior_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _make_winner_patch(h: int, w: int, rect_count: int) -> tuple[np.ndarray, int, np.ndarray]:
    axis = w // 2
    pair_slots = [(r, c, w - 1 - c) for r in range(1, h - 1) for c in range(1, axis)]
    center_slots = [(r, axis) for r in range(1, h - 1)]
    max_visible = (h - 2) * (w - 2) - rect_count

    for _ in range(200):
        patch = np.full((h, w), BASE, dtype=int)

        missing_count = random.randint(1, min(3, len(pair_slots)))
        missing_pairs = random.sample(pair_slots, missing_count)

        remaining = [slot for slot in pair_slots if slot not in missing_pairs]
        complete_cap = min(2, len(remaining))
        complete_count = random.randint(0, complete_cap) if complete_cap > 0 else 0
        complete_pairs = random.sample(remaining, complete_count) if complete_count else []

        center_cap = min(2, len(center_slots))
        center_count = random.randint(0, center_cap) if center_cap > 0 else 0
        centers = random.sample(center_slots, center_count) if center_count else []

        for r, c_left, c_right in missing_pairs:
            patch[r, random.choice((c_left, c_right))] = MARK
        for r, c_left, c_right in complete_pairs:
            patch[r, c_left] = MARK
            patch[r, c_right] = MARK
        for r, c in centers:
            patch[r, c] = MARK

        visible = int(np.count_nonzero(patch == MARK))
        completed = _complete_vertical_symmetry(patch)
        if 1 <= visible <= max_visible and int(np.count_nonzero(completed == MARK)) > visible:
            return patch, visible, completed

    raise RuntimeError("failed to build a valid winner patch")


def _make_noise_patch(h: int, w: int, marker_count: int) -> np.ndarray:
    patch = np.full((h, w), BASE, dtype=int)
    for r, c in random.sample(_interior_coords(h, w), marker_count):
        patch[r, c] = MARK
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    grid_h, grid_w = occupied.shape
    if r0 < PLACEMENT_BORDER or c0 < PLACEMENT_BORDER:
        return False
    if r0 + h > grid_h - PLACEMENT_BORDER or c0 + w > grid_w - PLACEMENT_BORDER:
        return False

    rr0 = max(0, r0 - PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    rr1 = min(grid_h, r0 + h + PLACEMENT_GAP)
    cc1 = min(grid_w, c0 + w + PLACEMENT_GAP)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patches(patches: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(PLACEMENT_BORDER, grid_h - PLACEMENT_BORDER - h + 1)
            for c0 in range(PLACEMENT_BORDER, grid_w - PLACEMENT_BORDER - w + 1)
            if _can_place(occupied, r0, c0, h, w)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    rect_h = random.randint(*RECT_HEIGHT_RANGE)
    rect_w = random.choice(RECT_WIDTH_OPTIONS)
    interior_area = (rect_h - 2) * (rect_w - 2)

    winner_patch, winner_count, winner_output = _make_winner_patch(rect_h, rect_w, rect_count)

    max_other = min(interior_area - 1, winner_count + 5)
    if max_other - winner_count < rect_count - 1:
        return None

    other_counts = random.sample(range(winner_count + 1, max_other + 1), rect_count - 1)
    patches = [winner_patch] + [_make_noise_patch(rect_h, rect_w, count) for count in other_counts]

    min_h = max(GRID_SIZE_MIN, 2 * rect_h + 3)
    min_w = max(GRID_SIZE_MIN, 2 * rect_w + 3)
    max_h = min(GRID_SIZE_MAX, min_h + 6)
    max_w = min(GRID_SIZE_MAX, min_w + 8)

    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    grid = _place_patches(patches, grid_h, grid_w)
    if grid is None:
        return None

    if not np.array_equal(main(grid), winner_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 6 input")
