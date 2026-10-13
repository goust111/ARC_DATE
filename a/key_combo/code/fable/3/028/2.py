from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
MASK_SIDE = 3
TILE_SIDE = 3
COARSE_SIDE = MASK_SIDE * TILE_SIDE
GEN_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(grid) if int(c) != BG)


def _decode_coarse(crop: np.ndarray, color: int) -> np.ndarray | None:
    if crop.shape != (COARSE_SIDE, COARSE_SIDE):
        return None
    if not np.all((crop == BG) | (crop == color)):
        return None

    layout = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            tile = crop[
                br * TILE_SIDE : (br + 1) * TILE_SIDE,
                bc * TILE_SIDE : (bc + 1) * TILE_SIDE,
            ]
            if np.all(tile == color):
                layout[br, bc] = True
            elif np.all(tile == BG):
                layout[br, bc] = False
            else:
                return None

    if not layout.any() or layout.all():
        return None
    return layout


def _extract_roles(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    colors = _foreground_colors(grid)
    assert len(colors) == 2, "expected exactly two non-black colors"

    coarse_candidates: list[tuple[int, np.ndarray]] = []
    fine_candidates: list[tuple[int, np.ndarray]] = []

    for color in colors:
        box = _bbox(grid == color)
        assert box is not None, "foreground color must be present"
        r0, c0, r1, c1 = box
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        shape = _bbox_shape(box)

        if shape == (COARSE_SIDE, COARSE_SIDE):
            layout = _decode_coarse(crop, color)
            if layout is not None:
                coarse_candidates.append((color, layout))

        if shape == (MASK_SIDE, MASK_SIDE) and np.all((crop == BG) | (crop == color)):
            fine_mask = crop == color
            if fine_mask.any() and (~fine_mask).any():
                fine_candidates.append((color, fine_mask))

    assert len(coarse_candidates) == 1, "expected one 9x9 block-encoded mask"
    assert len(fine_candidates) == 1, "expected one 3x3 fine stamp"

    color_a, layout = coarse_candidates[0]
    color_b, fine_mask = fine_candidates[0]
    assert color_a != color_b, "coarse mask and fine stamp must use different colors"
    return layout, fine_mask, color_a, color_b


def _render_output(layout: np.ndarray, fine_mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.full((COARSE_SIDE, COARSE_SIDE), BG, dtype=int)
    positive = np.where(fine_mask, color_b, BG)
    negative = np.where(~fine_mask, color_a, BG)

    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            r0, c0 = br * TILE_SIDE, bc * TILE_SIDE
            stamp = positive if bool(layout[br, bc]) else negative
            out[r0 : r0 + TILE_SIDE, c0 : c0 + TILE_SIDE] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a block-enlarged coarse mask and stamp positive or negative copies of a fine mask.

    Input structure:
    - The background is black and there are exactly two non-black colors.
    - One foreground color has a 9x9 bounding box divided into nine uniform
      3x3 tiles, each either solid in that color or black.
    - The other foreground color has a 3x3 bounding box containing the fine
      binary stamp, with black cells as the stamp background.

    Rule:
    - Compress the 9x9 tiled figure into a 3x3 layout mask.
    - Return a 9x9 grid partitioned into the same nine 3x3 blocks.
    - For every on layout cell, place the fine stamp in its original form using
      the fine-stamp color.
    - For every off layout cell, place the figure-ground inverse of the fine
      stamp using the coarse-mask color.

    Key insight:
    - The two colors do not define roles by value or position; the 9x9 scale is
      the layout instruction and the 3x3 scale is the stamp instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"
    assert Counter(map(int, g.ravel())).most_common(1)[0][0] == BG, "black must be the background"

    layout, fine_mask, color_a, color_b = _extract_roles(g)
    return _render_output(layout, fine_mask, color_a, color_b)


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if not cells:
        return False
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    seen = set()
    queue: deque[tuple[int, int]] = deque([next(iter(cells))])
    while queue:
        r, c = queue.popleft()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in deltas:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                queue.append(nxt)
    return seen == cells


def _catalog_masks(
    min_count: int,
    max_count: int,
    *,
    connected: bool = False,
) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for count in range(min_count, max_count + 1):
        for cells in itertools.combinations(range(MASK_SIDE * MASK_SIDE), count):
            mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
            for idx in cells:
                mask[idx // MASK_SIDE, idx % MASK_SIDE] = True
            if _bbox(mask) != (0, 0, MASK_SIDE - 1, MASK_SIDE - 1):
                continue
            if connected and not _is_connected(mask, connectivity=8):
                continue
            masks.append(mask)
    return tuple(masks)


LAYOUT_CATALOG = _catalog_masks(2, 7, connected=False)
FINE_CATALOG = _catalog_masks(3, 6, connected=True)


def _rects_clear(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int,
) -> bool:
    r0, c0, h0, w0 = first
    r1, c1, h1, w1 = second
    return (
        r0 + h0 + gap <= r1
        or r1 + h1 + gap <= r0
        or c0 + w0 + gap <= c1
        or c1 + w1 + gap <= c0
    )


def _choose_locations(rows: int, cols: int, gap: int) -> tuple[int, int, int, int] | None:
    for _ in range(300):
        cr = random.randint(0, rows - COARSE_SIDE)
        cc = random.randint(0, cols - COARSE_SIDE)
        fr = random.randint(0, rows - MASK_SIDE)
        fc = random.randint(0, cols - MASK_SIDE)
        coarse_rect = (cr, cc, COARSE_SIDE, COARSE_SIDE)
        fine_rect = (fr, fc, MASK_SIDE, MASK_SIDE)
        if _rects_clear(coarse_rect, fine_rect, gap):
            return cr, cc, fr, fc
    return None


def _semantic_accepts(
    inp: np.ndarray,
    expected: np.ndarray,
    layout: np.ndarray,
    fine_mask: np.ndarray,
    color_a: int,
    color_b: int,
) -> bool:
    try:
        extracted_layout, extracted_fine, extracted_a, extracted_b = _extract_roles(inp)
        actual = main(inp)
    except Exception:
        return False

    if extracted_a != color_a or extracted_b != color_b:
        return False
    if not np.array_equal(extracted_layout, layout) or not np.array_equal(extracted_fine, fine_mask):
        return False
    if not np.array_equal(actual, expected):
        return False
    if not (np.any(expected == color_a) and np.any(expected == color_b)):
        return False

    positive = np.where(fine_mask, color_b, BG)
    negative = np.where(~fine_mask, color_a, BG)
    if not np.any(positive == color_b) or not np.any(negative == color_a):
        return False
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            block = expected[
                br * TILE_SIDE : (br + 1) * TILE_SIDE,
                bc * TILE_SIDE : (bc + 1) * TILE_SIDE,
            ]
            want = positive if bool(layout[br, bc]) else negative
            if not np.array_equal(block, want):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(LAYOUT_CATALOG).copy()
    fine_mask = random.choice(FINE_CATALOG).copy()
    color_a, color_b = random.sample(range(1, 10), 2)

    rows = random.randint(14, 30)
    cols = random.randint(14, 30)
    gap = random.choice([1, 1, 2])
    loc = _choose_locations(rows, cols, gap)
    if loc is None:
        return None
    cr, cc, fr, fc = loc

    inp = np.full((rows, cols), BG, dtype=int)
    coarse_patch = np.where(
        np.kron(layout, np.ones((TILE_SIDE, TILE_SIDE), dtype=bool)),
        color_a,
        BG,
    )
    fine_patch = np.where(fine_mask, color_b, BG)

    inp[cr : cr + COARSE_SIDE, cc : cc + COARSE_SIDE] = coarse_patch
    inp[fr : fr + MASK_SIDE, fc : fc + MASK_SIDE] = fine_patch

    expected = _render_output(layout, fine_mask, color_a, color_b)
    if not _semantic_accepts(inp, expected, layout, fine_mask, color_a, color_b):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-scale mask input")
