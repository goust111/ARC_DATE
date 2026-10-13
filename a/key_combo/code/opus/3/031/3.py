from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLLISION_COLOR = 2
INPUT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 29)
GRID_WIDTH_RANGE = (18, 29)
MOTIF_HEIGHT_RANGE = (2, 3)
MOTIF_WIDTH_RANGE = (3, 4)
MIN_RIGHT_STEPS = 3
MIN_LEFT_STEPS = 2
MAX_GENERATION_ATTEMPTS = 500

BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_opaque(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst[rr0:rr1, cc0:cc1] = patch[pr0:pr1, pc0:pc1]


def _mark_foreground(mask: np.ndarray, patch: np.ndarray, color: int, r0: int, c0: int) -> None:
    rows, cols = np.nonzero(patch == int(color))
    rr = rows + int(r0)
    cc = cols + int(c0)
    in_bounds = (
        (0 <= rr)
        & (rr < mask.shape[0])
        & (0 <= cc)
        & (cc < mask.shape[1])
    )
    mask[rr[in_bounds], cc[in_bounds]] = True


def _visible_positions(
    grid_shape: tuple[int, int],
    origin_r: int,
    origin_c: int,
    patch_shape: tuple[int, int],
    step_r: int,
    step_c: int,
) -> list[tuple[int, int]]:
    h, w = map(int, patch_shape)
    H, W = map(int, grid_shape)
    positions: list[tuple[int, int]] = []
    r = int(origin_r) + int(step_r)
    c = int(origin_c) + int(step_c)
    while not (r >= H or c >= W or r + h <= 0 or c + w <= 0):
        positions.append((r, c))
        r += int(step_r)
        c += int(step_c)
    return positions


def _invert_binary_patch(patch: np.ndarray, color: int) -> np.ndarray:
    return np.where(patch == BACKGROUND, int(color), BACKGROUND).astype(int)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    colors = sorted(set(map(int, np.unique(g))) - {BACKGROUND})
    assert len(colors) == 1, "expected exactly one non-background color in the input"
    color = int(colors[0])
    assert color != COLLISION_COLOR, "input foreground color must differ from the red collision color"

    box = _bbox(g != BACKGROUND)
    assert box is not None, "expected one visible motif"
    r0, c0, _r1, _c1 = box
    motif = _crop_bbox(g, box)

    assert set(map(int, np.unique(motif))) <= {BACKGROUND, color}
    assert np.any(motif == color) and np.any(motif == BACKGROUND), "motif must mix black and one foreground color"
    assert (motif == color).any(axis=0).all() and (motif == color).any(axis=1).all(), "visible motif must define its full bbox"

    h, w = motif.shape
    inverted = _invert_binary_patch(motif, color)

    right_positions = _visible_positions(g.shape, r0, c0, motif.shape, h, 1)
    left_positions = _visible_positions(g.shape, r0, c0, motif.shape, h, -1)

    out = np.full_like(g, BACKGROUND)
    original_hits = np.zeros(g.shape, dtype=bool)
    inverted_hits = np.zeros(g.shape, dtype=bool)

    _mark_foreground(original_hits, motif, color, r0, c0)
    for rr, cc in right_positions:
        _mark_foreground(original_hits, motif, color, rr, cc)
    for rr, cc in left_positions:
        _mark_foreground(inverted_hits, inverted, color, rr, cc)

    for rr, cc in reversed(right_positions):
        _paste_opaque(out, motif, rr, cc)
    for rr, cc in reversed(left_positions):
        _paste_opaque(out, inverted, rr, cc)
    _paste_opaque(out, motif, r0, c0)

    out[original_hits & inverted_hits] = COLLISION_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a binary motif down two opposing diagonals, using its color inverse on the left branch and painting foreground collisions red.

    Input structure:
    - The background is black.
    - The only non-black object is one small binary motif that uses exactly one non-red foreground color.
    - The motif's visible foreground touches every row and every column of its bbox, so that bbox defines the full motif block.

    Rule:
    - Crop the motif bbox and build its binary color inverse inside the same bbox.
    - Keep the original motif, then stamp original-color copies one motif-height lower and one column to the right per step.
    - Stamp inverted copies one motif-height lower and one column to the left per step; each stamp is opaque, so black cells can overwrite earlier paint.
    - Render the right branch from far to near, then the left branch from far to near, then the original motif.
    - Finally recolor every cell red where the original branch and the inverted branch both placed the foreground color at some point.

    Key insight:
    - The two branches carry complementary versions of the same motif, so red cells emerge exactly where a foreground cell from each branch lands on the same output location.
    """
    return _solve(grid)


def _sample_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(200):
        mask = np.array(
            [[random.random() < random.uniform(0.35, 0.65) for _ in range(w)] for _ in range(h)],
            dtype=bool,
        )
        fg = int(mask.sum())
        if fg == 0 or fg == area:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not (~mask).any(axis=0).all() or not (~mask).any(axis=1).all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _make_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    h = random.randint(*MOTIF_HEIGHT_RANGE)
    w = random.randint(*MOTIF_WIDTH_RANGE)
    color = random.choice(INPUT_COLORS)

    mask = _sample_mask(h, w)
    if mask is None:
        return None

    max_r0 = min(6, H - (MIN_RIGHT_STEPS + 1) * h)
    if max_r0 < 1:
        return None
    max_c0 = min(W - w, max(0, W // 3 + 2))
    if max_c0 < 0:
        return None

    r0 = random.randint(1, max_r0)
    c0 = random.randint(0, max_c0)

    grid = np.full((H, W), BACKGROUND, dtype=int)
    motif = np.where(mask, color, BACKGROUND).astype(int)
    _paste_opaque(grid, motif, r0, c0)

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    right_positions = _visible_positions(grid.shape, r0, c0, motif.shape, h, 1)
    left_positions = _visible_positions(grid.shape, r0, c0, motif.shape, h, -1)
    if len(right_positions) < MIN_RIGHT_STEPS or len(left_positions) < MIN_LEFT_STEPS:
        return None
    if np.count_nonzero(out == COLLISION_COLOR) == 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
