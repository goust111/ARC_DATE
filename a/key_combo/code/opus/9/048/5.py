from __future__ import annotations

import random

import numpy as np
from scipy import ndimage


BLACK = 0
BLUE = 1
RED = 2
MAGENTA = 6
CYAN = 8

HALF_W = 5
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

GRID_H_RANGE = (8, 20)
MAX_GENERATION_ATTEMPTS = 400

_BASE_HOLE = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
    ],
    dtype=int,
)
_BASE_MIXED = np.array(
    [
        [1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0],
    ],
    dtype=int,
)


def _unique_variants(base: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for flipped_lr in (False, True):
        for flipped_ud in (False, True):
            arr = base.copy()
            if flipped_lr:
                arr = np.fliplr(arr)
            if flipped_ud:
                arr = np.flipud(arr)
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
    return variants


HOLE_SEEDS = _unique_variants(_BASE_HOLE)
MIXED_SEEDS = _unique_variants(_BASE_MIXED)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndimage.label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(labels == idx) for idx in range(1, int(count) + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[np.ndarray] = []
    for hole in _components(~sub):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    outline = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _find_marker_side(grid: np.ndarray) -> int:
    markers = np.argwhere(grid == RED)
    assert markers.shape == (2, 2), "expected exactly two red marker pixels"
    assert int(markers[0, 1]) == int(markers[1, 1]), "markers must share one column"
    col = int(markers[0, 1])
    if col < HALF_W:
        assert np.all(markers[:, 1] < HALF_W), "both markers must lie in the left half"
        return 0
    assert np.all(markers[:, 1] >= HALF_W), "both markers must lie in the right half"
    return 1


def _extract_seed_rows(half: np.ndarray) -> tuple[np.ndarray, int]:
    blue_rows = np.flatnonzero(np.any(half == BLUE, axis=1))
    assert blue_rows.size > 0, "selected half must contain blue pixels"
    r0 = int(blue_rows[0])
    r1 = int(blue_rows[-1])
    return half[r0 : r1 + 1].copy(), r0


def _tile_with_phase(shape: tuple[int, int], seed: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    out = np.empty(shape, dtype=int)
    h, w = seed.shape
    r0, c0 = map(int, offset)
    for r in range(shape[0]):
        sr = (int(r) - r0) % h
        for c in range(shape[1]):
            sc = (int(c) - c0) % w
            out[r, c] = int(seed[sr, sc])
    return out


def _decorate_tiled(tiled: np.ndarray) -> np.ndarray:
    out = tiled.copy()
    for coords in _components(tiled == BLUE):
        mask = _component_mask(tiled.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        if holes:
            out[mask] = CYAN
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA
                solid[hole[:, 0], hole[:, 1]] = True
        outline = _outline_mask(solid)
        out[outline & (out == BLACK)] = RED
    return out


def _solve_for_side(grid: np.ndarray, side: int) -> np.ndarray:
    c0 = int(side) * HALF_W
    half = np.asarray(grid[:, c0 : c0 + HALF_W], dtype=int).copy()
    half[half == RED] = BLACK
    seed, r0 = _extract_seed_rows(half)
    tiled = _tile_with_phase(grid.shape, seed, (r0, c0))
    return _decorate_tiled(tiled)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the marked half, crop its short blue band, tile it periodically, then recolor outlined hole-bearing repeats.

    Input structure:
    - The grid is black (0) with width 10, split into two 5-column halves.
    - Exactly two red (2) marker pixels share one column and lie in the same half.
    - Each half contains blue (1) foreground cells; in the marked half they occupy a shorter row band inside the 5-column half.

    Rule:
    - Keep the half indicated by the two aligned red markers, erase those markers to black, and crop that half to the occupied blue row span while keeping all 5 columns.
    - Repeat that cropped 5-column seed vertically and horizontally across the full canvas, preserving the seed's original row band and half position as the tiling phase.
    - For each blue 4-connected component in the tiled result, detect 4-connected background holes inside its bounding box.
    - Components without holes stay blue and receive a red 8-neighborhood outline on background cells only.
    - Components with holes are recolored cyan (8), their holes are filled magenta (6), and the same red outline is drawn around the solid-plus-hole shape.

    Key insight:
    - The chosen half supplies a small periodic seed whose copies can create enclosed holes only after repetition, so both the marker-guided crop and the component-level hole test matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] == 2 * HALF_W, "expected two 5-column halves"
    side = _find_marker_side(g)
    return _solve_for_side(g, side)


def _random_connected_shape() -> np.ndarray | None:
    target = random.randint(3, 8)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if h > 5 or w > HALF_W:
        return None

    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[int(r), int(c)] = BLUE
    if any(_interior_holes(patch == BLUE, np.argwhere(patch == BLUE))):
        return None
    return patch


def _place_patch(half: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    half[top : top + h, left : left + w] = patch


def _paint_other_half(height: int) -> np.ndarray | None:
    for _ in range(80):
        patch = _random_connected_shape()
        if patch is None:
            continue
        h, w = patch.shape
        top = random.randint(0, height - h)
        left = random.randint(0, HALF_W - w)
        half = np.zeros((height, HALF_W), dtype=int)
        _place_patch(half, patch, top, left)
        return half
    return None


def _choose_marker_rows(height: int, seed_top: int, seed_h: int) -> tuple[int, int]:
    above = list(range(0, seed_top))
    below = list(range(seed_top + seed_h, height))
    assert above and below, "need empty rows above and below the seed"
    return random.choice(above), random.choice(below)


def _counts_by_color(grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _valid_generated_example(grid: np.ndarray, expect_mixed: bool) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    counts = _counts_by_color(out)
    if counts.get(CYAN, 0) == 0 or counts.get(MAGENTA, 0) == 0:
        return False
    if expect_mixed and counts.get(BLUE, 0) == 0:
        return False

    chosen = _find_marker_side(grid)
    wrong = _solve_for_side(grid, 1 - chosen)
    if np.array_equal(out, wrong):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    side = random.randint(0, 1)
    mixed = random.random() < 0.55
    seed = random.choice(MIXED_SEEDS if mixed else HOLE_SEEDS).copy()
    seed_h = int(seed.shape[0])
    seed_top = random.randint(1, height - seed_h - 1)
    marker_rows = _choose_marker_rows(height, seed_top, seed_h)

    halves = [np.zeros((height, HALF_W), dtype=int), np.zeros((height, HALF_W), dtype=int)]
    halves[side][seed_top : seed_top + seed_h, :] = seed

    other = _paint_other_half(height)
    if other is None:
        return None
    halves[1 - side][:, :] = other

    marker_col = random.randint(0, HALF_W - 1)
    halves[side][marker_rows[0], marker_col] = RED
    halves[side][marker_rows[1], marker_col] = RED

    grid = np.concatenate(halves, axis=1)
    if np.count_nonzero(grid == RED) != 2:
        return None
    if not _valid_generated_example(grid, expect_mixed=mixed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
