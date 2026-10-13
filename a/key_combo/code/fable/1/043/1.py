from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
ORIENTATION_KS = (0, 1, 2, -1)
PAIR_LIST = tuple(combinations(range(4), 2))
COLOR_POOL = np.arange(1, 10)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((int(nr), int(nc)))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (int(np.random.randint(height)), int(np.random.randint(width)))
    mask[seed] = True
    frontier = set(_neighbors4(seed[0], seed[1], height, width))

    while int(mask.sum()) < area:
        if not frontier:
            cells = np.argwhere(~mask)
            seed = tuple(map(int, cells[int(np.random.randint(len(cells)))]))
            frontier.add(seed)
        cell = tuple(list(frontier)[int(np.random.randint(len(frontier)))])
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if not mask[nb]:
                frontier.add(nb)
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    area = int(len(coords))
    n_colors = int(np.random.randint(2, min(4, area) + 1))
    palette = np.random.choice(COLOR_POOL, size=n_colors, replace=False).astype(int)
    seed_ids = np.random.choice(area, size=n_colors, replace=False)
    seeds = coords[seed_ids]

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        dists = np.abs(seeds[:, 0] - int(r)) + np.abs(seeds[:, 1] - int(c))
        sprite[int(r), int(c)] = int(palette[int(np.argmin(dists))])
    return sprite


def _make_full_sprite(side: int) -> np.ndarray | None:
    min_patch = max(4, side // 2)
    patch_h = int(np.random.randint(min_patch, side + 1))
    patch_w = int(np.random.randint(min_patch, side + 1))
    patch_area = patch_h * patch_w
    area_lo = max(10, int(np.ceil(0.35 * patch_area)))
    area_hi = max(area_lo, int(np.floor(0.65 * patch_area)))
    area = int(np.random.randint(area_lo, area_hi + 1))

    mask = _grow_connected_mask(patch_h, patch_w, area)
    box = _bbox(mask)
    assert box is not None
    if box[2] - box[0] + 1 < 3 or box[3] - box[1] + 1 < 3:
        return None

    patch = _colorize_mask(mask)
    full = np.full((side, side), BG, dtype=int)
    top = int(np.random.randint(0, side - patch_h + 1))
    left = int(np.random.randint(0, side - patch_w + 1))
    full[top : top + patch_h, left : left + patch_w] = patch

    if any(np.array_equal(full, np.rot90(full, k)) for k in (1, 2, 3)):
        return None
    if np.array_equal(full, np.fliplr(full)) or np.array_equal(full, np.flipud(full)):
        return None
    return full


def _assign_visibility(full_sprite: np.ndarray) -> np.ndarray | None:
    full_mask = full_sprite != BG
    coords = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    area = len(coords)
    if area < len(PAIR_LIST) + 2:
        return None

    visibility = np.zeros((4,) + full_sprite.shape, dtype=bool)
    witness_ids = np.random.choice(area, size=len(PAIR_LIST), replace=False)
    witness_set = set(int(x) for x in witness_ids)

    for idx, pair in enumerate(PAIR_LIST):
        r, c = coords[int(witness_ids[idx])]
        keep_quadrants = set(range(4)) - set(pair)
        for q in keep_quadrants:
            visibility[q, r, c] = True

    weights = np.array([2, 5, 3], dtype=float)
    subset_sizes = np.array([1, 2, 3], dtype=int)
    for idx, (r, c) in enumerate(coords):
        if idx in witness_set:
            continue
        size = int(np.random.choice(subset_sizes, p=weights / weights.sum()))
        subset = np.random.choice(4, size=size, replace=False)
        visibility[subset, r, c] = True

    coverages = [int(np.count_nonzero(visibility[q] & full_mask)) for q in range(4)]
    pair_coverages = [
        int(np.count_nonzero((visibility[i] | visibility[j]) & full_mask))
        for i, j in PAIR_LIST
    ]
    min_single = max(3, int(np.ceil(0.30 * area)))
    max_single = area - 2
    min_pair = max(5, int(np.ceil(0.55 * area)))
    max_pair = area - 2
    if any(not (min_single <= cov <= max_single) for cov in coverages):
        return None
    if any(not (min_pair <= cov <= max_pair) for cov in pair_coverages):
        return None
    if not np.array_equal(np.any(visibility, axis=0), full_mask):
        return None
    return visibility


def _assemble_pinwheel(full_sprite: np.ndarray, visibility: np.ndarray | None = None) -> np.ndarray:
    quadrants = []
    for q, k in enumerate(ORIENTATION_KS):
        if visibility is None:
            aligned = full_sprite
        else:
            aligned = np.full_like(full_sprite, BG)
            mask = visibility[q]
            aligned[mask] = full_sprite[mask]
        quadrants.append(np.rot90(aligned, k))
    top = np.hstack((quadrants[0], quadrants[1]))
    bottom = np.hstack((quadrants[2], quadrants[3]))
    return np.vstack((top, bottom))


def _aligned_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    side = int(grid.shape[0] // 2)
    q0 = grid[:side, :side]
    q1 = np.rot90(grid[:side, side:], -1)
    q2 = np.rot90(grid[side:, :side], 2)
    q3 = np.rot90(grid[side:, side:], 1)
    return [q0.copy(), q1.copy(), q2.copy(), q3.copy()]


def _validate_instance(inp: np.ndarray, full_sprite: np.ndarray) -> bool:
    if inp.ndim != 2 or inp.shape[0] != inp.shape[1] or inp.shape[0] % 2 != 0:
        return False
    if inp.shape[0] != full_sprite.shape[0] * 2:
        return False
    bg = _get_bg(inp)
    if bg != BG:
        return False

    aligned = _aligned_quadrants(inp)
    full_mask = full_sprite != BG
    for patch in aligned:
        mask = patch != BG
        if np.any(mask & ~full_mask):
            return False
        if np.any(mask & (patch != full_sprite)):
            return False

    union = np.zeros_like(full_mask)
    for patch in aligned:
        union |= patch != BG
    if not np.array_equal(union, full_mask):
        return False

    for patch in aligned:
        if np.array_equal(patch != BG, full_mask):
            return False

    for i, j in PAIR_LIST:
        if np.array_equal((aligned[i] != BG) | (aligned[j] != BG), full_mask):
            return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden sprite by undoing the pinwheel rotations and overlaying the four residues.

    Input structure:
    - The input is an even-side square grid with no separator lines.
    - Its four equal quadrants are partial copies of one multicolor sprite arranged as a pinwheel:
      upper-left unchanged, upper-right rotated 90 degrees counterclockwise, lower-left rotated
      180 degrees, and lower-right rotated 90 degrees clockwise.

    Rule:
    - Split the grid into four quadrants and rotate each residue back into the upper-left coordinate system.
    - Overlay all non-background pixels there to recover the complete sprite.
    - Write that completed sprite back into the four quadrants using the same pinwheel rotations.

    Key insight:
    - No quadrant contains the whole sprite; the full image appears only after inverse-rotating the
      four residues into one shared frame and taking their union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0, "expected an even-side square grid"

    bg = _get_bg(g)
    aligned = _aligned_quadrants(g)
    side = height // 2
    sprite = np.full((side, side), bg, dtype=int)

    for patch in aligned:
        mask = patch != bg
        conflict = mask & (sprite != bg) & (sprite != patch)
        assert not np.any(conflict), "rotated residues disagree on color"
        sprite[mask] = patch[mask]

    quadrants = [
        sprite,
        np.rot90(sprite, 1),
        np.rot90(sprite, 2),
        np.rot90(sprite, -1),
    ]
    top = np.hstack((quadrants[0], quadrants[1]))
    bottom = np.hstack((quadrants[2], quadrants[3]))
    return np.vstack((top, bottom))


def _gen_attempt() -> np.ndarray | None:
    side = int(np.random.randint(6, 15))
    full_sprite = _make_full_sprite(side)
    if full_sprite is None:
        return None
    visibility = _assign_visibility(full_sprite)
    if visibility is None:
        return None

    inp = _assemble_pinwheel(full_sprite, visibility)
    expected = _assemble_pinwheel(full_sprite)
    if not _validate_instance(inp, full_sprite):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pinwheel-residue instance")
