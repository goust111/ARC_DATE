from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# Fixed ARC colors named explicitly by the puzzle statement.
RED_ANCHOR = 2
PINK_OVERLAP = 6
TEAL_ANCHOR = 8


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _window_regions(grid: np.ndarray, background: int) -> list[np.ndarray]:
    """Extract the two full non-background rectangular window patches."""
    regions: list[np.ndarray] = []
    for coords in _components(grid != background):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        assert coords.shape[0] == patch.size, "each window must be a solid non-background rectangle"
        assert np.all(patch != background), "window rectangles may not contain outer background"
        regions.append(patch)
    assert len(regions) == 2, "expected exactly two separated windows"
    return regions


def _anchor(patch: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(patch == color)
    assert coords.shape == (1, 2), "each window needs exactly one anchor of each anchor color"
    return int(coords[0, 0]), int(coords[0, 1])


def _rotated_vector(vector: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = vector
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    return dr, dc


def _place_mask(
    mask: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Translate a boolean mask onto a target canvas, cropping at its boundary."""
    out = np.zeros(shape, dtype=bool)
    for r, c in np.argwhere(mask):
        rr, cc = int(r) + top, int(c) + left
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Register the smaller two-anchor window to the larger one, then transparently compose their masks.

    Input structure:
    - The dominant outer background separates exactly two solid rectangular windows of unequal area.
    - Both windows share one bottom color, have one private foreground-mask color, and contain exactly
      one teal anchor and one red anchor; the two anchor vectors differ by a quarter-turn rotation.

    Rule:
    - Rotate the smaller window so its teal-to-red vector equals that of the larger window, then translate
      it so the teal anchors coincide. Return a canvas shaped like the larger window: use pink where both
      foreground masks overlap, preserve either private mask color where only it occurs, and use the shared
      window-bottom color elsewhere. Remove both anchor pixels from the output.

    Key insight:
    - The ordered teal/red anchor pair uniquely fixes both rotation and translation; mask overlap is meaningful
      only in the larger window's coordinate system after that rigid registration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    counts = Counter(map(int, g.ravel()))
    background, background_count = counts.most_common(1)[0]
    assert sum(value == background_count for value in counts.values()) == 1, "outer background must be unique"
    regions = _window_regions(g, int(background))

    areas = [int(region.size) for region in regions]
    assert areas[0] != areas[1], "window areas must differ"
    large_i = int(np.argmax(areas))
    large = regions[large_i]
    small = regions[1 - large_i]

    for patch in (large, small):
        _anchor(patch, TEAL_ANCHOR)
        _anchor(patch, RED_ANCHOR)

    large_colors = set(map(int, np.unique(large))) - {TEAL_ANCHOR, RED_ANCHOR}
    small_colors = set(map(int, np.unique(small))) - {TEAL_ANCHOR, RED_ANCHOR}
    shared = large_colors & small_colors
    assert len(shared) == 1, "the two windows must have one unambiguous shared bottom color"
    bottom = next(iter(shared))
    large_private = large_colors - {bottom}
    small_private = small_colors - {bottom}
    assert len(large_private) == 1 and len(small_private) == 1, "each window needs one private mask color"
    large_color = next(iter(large_private))
    small_color = next(iter(small_private))
    assert large_color != small_color, "the two foreground masks must be visibly distinguishable"

    large_teal = _anchor(large, TEAL_ANCHOR)
    large_red = _anchor(large, RED_ANCHOR)
    small_teal = _anchor(small, TEAL_ANCHOR)
    small_red = _anchor(small, RED_ANCHOR)
    large_vector = (large_red[0] - large_teal[0], large_red[1] - large_teal[1])
    small_vector = (small_red[0] - small_teal[0], small_red[1] - small_teal[1])
    assert large_vector != (0, 0) and small_vector != (0, 0)
    rotations = [turns for turns in range(4) if _rotated_vector(small_vector, turns) == large_vector]
    assert len(rotations) == 1, "the ordered anchor pair must determine one quarter-turn rotation"
    turns = rotations[0]

    rotated_small_mask = np.rot90(small == small_color, turns)
    rotated_teal_mask = np.rot90(small == TEAL_ANCHOR, turns)
    rotated_teal = _anchor(rotated_teal_mask.astype(int), 1)
    top = large_teal[0] - rotated_teal[0]
    left = large_teal[1] - rotated_teal[1]

    large_mask = large == large_color
    small_mask = _place_mask(rotated_small_mask, top, left, large.shape)
    out = np.full(large.shape, bottom, dtype=int)
    out[large_mask & ~small_mask] = large_color
    out[small_mask & ~large_mask] = small_color
    out[large_mask & small_mask] = PINK_OVERLAP
    return out


# ------------------------------ generator ------------------------------

_BACKGROUND = 0
_WINDOW_BOTTOM = 5
_MASK_PALETTE = (1, 3, 4, 7, 9)
_MAX_ATTEMPTS = 500
_GENERATION_COUNT = 0


def _neighbors(cell: tuple[int, int], shape: tuple[int, int]):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            yield rr, cc


def _connected_blob(
    shape: tuple[int, int],
    size: int,
    forbidden: set[tuple[int, int]],
    starts: list[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    eligible_starts = [cell for cell in starts if cell not in forbidden]
    if not eligible_starts:
        return None
    cells = {random.choice(eligible_starts)}
    while len(cells) < size:
        frontier = {
            nxt
            for cell in cells
            for nxt in _neighbors(cell, shape)
            if nxt not in cells and nxt not in forbidden
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _small_blob_with_overlap(
    large_mask: set[tuple[int, int]],
    small_box: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
    size: int,
    overlap_size: int,
) -> set[tuple[int, int]] | None:
    top, left, height, width = small_box
    in_box = {
        (r, c)
        for r in range(top, top + height)
        for c in range(left, left + width)
        if (r, c) not in forbidden
    }
    seeds = list(in_box & large_mask)
    if not seeds:
        return None
    cells = {random.choice(seeds)}

    while len(cells & large_mask) < overlap_size:
        frontier = {
            nxt
            for cell in cells
            for nxt in _neighbors(cell, (top + height, left + width))
            if nxt in in_box and nxt in large_mask and nxt not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    while len(cells) < size:
        frontier = {
            nxt
            for cell in cells
            for nxt in _neighbors(cell, (top + height, left + width))
            if nxt in in_box and nxt not in large_mask and nxt not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _mask_array(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _render_composition(
    large_mask: np.ndarray,
    small_mask: np.ndarray,
    large_color: int,
    small_color: int,
) -> np.ndarray:
    out = np.full(large_mask.shape, _WINDOW_BOTTOM, dtype=int)
    out[large_mask & ~small_mask] = large_color
    out[small_mask & ~large_mask] = small_color
    out[large_mask & small_mask] = PINK_OVERLAP
    return out


def _make_input_canvas(large: np.ndarray, small: np.ndarray) -> np.ndarray:
    """Place the windows in random order with at least two background cells between them."""
    patches = [large, small]
    random.shuffle(patches)
    first, second = patches
    gap = random.randint(2, 4)
    horizontal = bool(random.getrandbits(1))

    if horizontal:
        height = max(first.shape[0], second.shape[0]) + 6 + random.randint(0, 3)
        width = first.shape[1] + second.shape[1] + gap + 4 + random.randint(0, 2)
        slack = width - first.shape[1] - second.shape[1] - gap
        c1 = random.randint(1, slack - 1)
        c2 = c1 + first.shape[1] + gap
        r1 = random.randint(1, height - first.shape[0] - 1)
        r2 = random.randint(1, height - second.shape[0] - 1)
    else:
        height = first.shape[0] + second.shape[0] + gap + 4 + random.randint(0, 2)
        width = max(first.shape[1], second.shape[1]) + 6 + random.randint(0, 3)
        slack = height - first.shape[0] - second.shape[0] - gap
        r1 = random.randint(1, slack - 1)
        r2 = r1 + first.shape[0] + gap
        c1 = random.randint(1, width - first.shape[1] - 1)
        c2 = random.randint(1, width - second.shape[1] - 1)

    assert height <= 30 and width <= 30
    canvas = np.full((height, width), _BACKGROUND, dtype=int)
    canvas[r1 : r1 + first.shape[0], c1 : c1 + first.shape[1]] = first
    canvas[r2 : r2 + second.shape[0], c2 : c2 + second.shape[1]] = second
    return canvas


def _gen_attempt(turns: int) -> np.ndarray | None:
    small_h = random.randint(5, 7)
    small_w = random.randint(5, 7)
    large_h = random.randint(small_h + 2, min(11, small_h + 4))
    large_w = random.randint(small_w + 2, min(11, small_w + 4))
    top = random.randint(1, large_h - small_h - 1)
    left = random.randint(1, large_w - small_w - 1)

    vectors = [
        (dr, dc)
        for dr, dc in ((1, 2), (2, 1), (1, -2), (2, -1), (-1, 2), (-2, 1), (-1, -2), (-2, -1))
    ]
    vector = random.choice(vectors)
    anchor_choices = [
        (r, c)
        for r in range(1, small_h - 1)
        for c in range(1, small_w - 1)
        if 1 <= r + vector[0] < small_h - 1 and 1 <= c + vector[1] < small_w - 1
    ]
    if not anchor_choices:
        return None
    small_teal_aligned = random.choice(anchor_choices)
    small_red_aligned = (
        small_teal_aligned[0] + vector[0],
        small_teal_aligned[1] + vector[1],
    )
    large_teal = (top + small_teal_aligned[0], left + small_teal_aligned[1])
    large_red = (top + small_red_aligned[0], left + small_red_aligned[1])
    forbidden = {large_teal, large_red}

    small_box_cells = [
        (r, c)
        for r in range(top, top + small_h)
        for c in range(left, left + small_w)
    ]
    large_cells = _connected_blob(
        (large_h, large_w), random.randint(9, 14), forbidden, small_box_cells
    )
    if large_cells is None:
        return None
    overlap_size = random.randint(1, 3)
    small_cells_global = _small_blob_with_overlap(
        large_cells,
        (top, left, small_h, small_w),
        forbidden,
        random.randint(7, 11),
        overlap_size,
    )
    if small_cells_global is None:
        return None
    if len(large_cells & small_cells_global) != overlap_size:
        return None
    if len(large_cells - small_cells_global) < 4 or len(small_cells_global - large_cells) < 4:
        return None

    large_mask = _mask_array((large_h, large_w), large_cells)
    small_mask_aligned = np.zeros((small_h, small_w), dtype=bool)
    for r, c in small_cells_global:
        small_mask_aligned[r - top, c - left] = True

    large_color, small_color = random.sample(_MASK_PALETTE, 2)
    large_patch = np.full((large_h, large_w), _WINDOW_BOTTOM, dtype=int)
    large_patch[large_mask] = large_color
    large_patch[large_teal] = TEAL_ANCHOR
    large_patch[large_red] = RED_ANCHOR

    small_aligned_patch = np.full((small_h, small_w), _WINDOW_BOTTOM, dtype=int)
    small_aligned_patch[small_mask_aligned] = small_color
    small_aligned_patch[small_teal_aligned] = TEAL_ANCHOR
    small_aligned_patch[small_red_aligned] = RED_ANCHOR
    small_original_patch = np.rot90(small_aligned_patch, -turns).copy()
    small_original_mask = np.rot90(small_mask_aligned, -turns).copy()

    registered_small = _place_mask(np.rot90(small_original_mask, turns), top, left, large_mask.shape)
    expected = _render_composition(large_mask, registered_small, large_color, small_color)

    # Semantic anti-shortcuts: every composition branch and the translation must affect the result.
    if not (
        np.any(large_mask & registered_small)
        and np.any(large_mask & ~registered_small)
        and np.any(registered_small & ~large_mask)
        and (top, left) != (0, 0)
    ):
        return None
    untranslated = _render_composition(
        large_mask,
        _place_mask(np.rot90(small_original_mask, turns), 0, 0, large_mask.shape),
        large_color,
        small_color,
    )
    if np.array_equal(untranslated, expected):
        return None

    # On rotated cases, explicitly reject examples solvable by aligning only the teal anchors.
    if turns:
        original_teal = _anchor(small_original_patch, TEAL_ANCHOR)
        direct_top = large_teal[0] - original_teal[0]
        direct_left = large_teal[1] - original_teal[1]
        direct_mask = _place_mask(small_original_mask, direct_top, direct_left, large_mask.shape)
        direct_output = _render_composition(large_mask, direct_mask, large_color, small_color)
        if np.array_equal(direct_output, expected):
            return None

    grid = _make_input_canvas(large_patch, small_original_patch)
    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != _BACKGROUND:
        return None
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if TEAL_ANCHOR in actual or RED_ANCHOR in actual:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked examples with guaranteed 0/nonzero rotation coverage per three calls."""
    global _GENERATION_COUNT
    # Every three consecutive calls include one already-aligned case and two genuine rotations.
    turns = 0 if _GENERATION_COUNT % 3 == 0 else random.choice((1, 2, 3))
    _GENERATION_COUNT += 1
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(turns)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-anchor registration input")
