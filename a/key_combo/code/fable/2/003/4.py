from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, components_mask, crop_bbox, get_bg, neighbors4, paste


QUADRANT_SLICES = (
    lambda m: (slice(0, m), slice(0, m)),
    lambda m: (slice(0, m), slice(m + 1, 2 * m + 1)),
    lambda m: (slice(m + 1, 2 * m + 1), slice(0, m)),
    lambda m: (slice(m + 1, 2 * m + 1), slice(m + 1, 2 * m + 1)),
)

PALETTE = tuple(range(10))
QUAD_MIN = 7
QUAD_MAX = 14
COMPONENT_COUNT_RANGE = (2, 4)
COMPONENT_BOX_MIN = 2
COMPONENT_BOX_MAX = 5
COMPONENT_AREA_MIN = 3
COMPONENT_AREA_MAX = 8
VISIBLE_COUNT_OPTIONS = np.array([1, 2, 3], dtype=int)
VISIBLE_COUNT_PROBS = np.array([0.35, 0.4, 0.25], dtype=float)
GENERATION_RETRIES = 500


def _find_dividers(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"
    bg = get_bg(g)

    row_candidates = [r for r in range(h) if np.all(g[r, :] == g[r, 0])]
    col_candidates = [c for c in range(w) if np.all(g[:, c] == g[0, c])]

    matches: list[tuple[int, int, int, int]] = []
    for r in row_candidates:
        for c in col_candidates:
            div = int(g[r, 0])
            if div != int(g[0, c]):
                continue
            if div == bg:
                continue
            if r != c or r != h - r - 1 or c != w - c - 1:
                continue
            matches.append((r, c, div, bg))

    assert len(matches) == 1, f"expected one divider cross, found {matches}"
    return matches[0]


def _split_quadrants(grid: np.ndarray, mid: int) -> list[np.ndarray]:
    return [grid[sl_r, sl_c].copy() for sl_r, sl_c in (fn(mid) for fn in QUADRANT_SLICES)]


def _merge_aligned_quadrants(
    quadrants: list[np.ndarray], rotations: tuple[int, int, int, int], bg: int
) -> np.ndarray | None:
    aligned = [np.rot90(quadrants[i], -rotations[i]) for i in range(4)]
    complete = np.full_like(aligned[0], int(bg))
    for part in aligned:
        mask = part != bg
        conflict = mask & (complete != bg) & (complete != part)
        if np.any(conflict):
            return None
        complete[mask] = part[mask]
    return complete


def _render_output(
    source_grid: np.ndarray,
    complete: np.ndarray,
    mid: int,
    rotations: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.asarray(source_grid, dtype=int).copy()
    for idx, rot in enumerate(rotations):
        patch = np.rot90(complete, rot)
        sl_r, sl_c = QUADRANT_SLICES[idx](mid)
        paste(out, patch, sl_r.start, sl_c.start)
    return out


def _candidate_solutions(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    mid_r, mid_c, _divider, bg = _find_dividers(g)
    assert mid_r == mid_c, "quadrants must be square"
    quadrants = _split_quadrants(g, mid_r)

    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
    for rotations in itertools.permutations(range(4)):
        complete = _merge_aligned_quadrants(quadrants, rotations, bg)
        if complete is None:
            continue
        out = _render_output(g, complete, mid_r, rotations)
        candidates.append((tuple(int(x) for x in rotations), complete, out))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a full four-quadrant rotation puzzle from mutually complementary damaged copies.

    Input structure:
    - The grid is split into four equal square quadrants by one full divider row and one full divider column.
    - Each quadrant contains the same pattern in one of the four quarter-turn rotations.
    - Every rotated copy is partially erased by background-colored damage, but every true pattern cell remains visible in at least one quadrant.

    Rule:
    - Infer which quarter-turn rotation belongs to each quadrant by testing which inverse rotations make all visible pixels mutually consistent.
    - Rotate the four damaged quadrants into one common orientation and take the union of their non-background pixels to recover the complete pattern.
    - Rotate that recovered pattern back into each quadrant's original orientation and redraw all four quadrants, keeping the divider cross unchanged.

    Key insight:
    - Missing pixels are not solved locally; they are stored in other quadrants at rotated coordinates, so the only valid reconstruction is the rotation-aligned union of all four partial copies.
    """
    candidates = _candidate_solutions(np.asarray(grid, dtype=int))
    assert candidates, "no rotation-consistent reconstruction found"

    outputs = [candidate[2] for candidate in candidates]
    first = outputs[0]
    assert all(np.array_equal(first, other) for other in outputs[1:]), "ambiguous reconstruction"
    return first


def _grow_component_mask(rng: np.random.Generator, box_h: int, box_w: int, area: int) -> np.ndarray | None:
    occupied = {(int(rng.integers(box_h)), int(rng.integers(box_w)))}
    while len(occupied) < area:
        frontier = []
        for r, c in occupied:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in occupied:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(len(frontier)))]
        occupied.add(pick)

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in occupied:
        mask[r, c] = True
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(int), box).astype(bool)


def _can_place(pattern: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = pattern.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for dr, dc in np.argwhere(patch):
        rr = top + int(dr)
        cc = left + int(dc)
        r0 = max(0, rr - 1)
        c0 = max(0, cc - 1)
        r1 = min(h, rr + 2)
        c1 = min(w, cc + 2)
        if np.any(pattern[r0:r1, c0:c1] != bg):
            return False
    return True


def _has_nontrivial_dihedral_symmetry(pattern: np.ndarray) -> bool:
    for k in (1, 2, 3):
        if np.array_equal(pattern, np.rot90(pattern, k)):
            return True
    mirrored = np.fliplr(pattern)
    for k in range(4):
        if np.array_equal(pattern, np.rot90(mirrored, k)):
            return True
    return False


def _build_base_pattern(rng: np.random.Generator, n: int, bg: int, colors: np.ndarray) -> np.ndarray | None:
    pattern = np.full((n, n), int(bg), dtype=int)
    target_components = len(colors)

    for color in colors:
        placed = False
        for _ in range(120):
            box_h = int(rng.integers(COMPONENT_BOX_MIN, min(COMPONENT_BOX_MAX, n - 1) + 1))
            box_w = int(rng.integers(COMPONENT_BOX_MIN, min(COMPONENT_BOX_MAX, n - 1) + 1))
            max_area = min(COMPONENT_AREA_MAX, box_h * box_w - 1)
            if max_area < COMPONENT_AREA_MIN:
                continue
            area = int(rng.integers(COMPONENT_AREA_MIN, max_area + 1))
            mask = _grow_component_mask(rng, box_h, box_w, area)
            if mask is None:
                continue
            if rng.random() < 0.5:
                mask = np.rot90(mask, int(rng.integers(4)))
            ph, pw = mask.shape
            for _ in range(120):
                top = int(rng.integers(0, n - ph + 1))
                left = int(rng.integers(0, n - pw + 1))
                if not _can_place(pattern, mask, top, left, bg):
                    continue
                patch = np.full(mask.shape, int(bg), dtype=int)
                patch[mask] = int(color)
                paste(pattern, patch, top, left, transparent=bg)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    fg_mask = pattern != bg
    if not np.any(fg_mask):
        return None
    if np.sum(fg_mask) >= (n * n) // 2:
        return None

    parts = components_mask(fg_mask, connectivity=4)
    if len(parts) != target_components:
        return None

    box = bbox(fg_mask)
    assert box is not None
    span_h = int(box[2] - box[0] + 1)
    span_w = int(box[3] - box[1] + 1)
    if span_h < max(4, n // 2) or span_w < max(4, n // 2):
        return None
    if _has_nontrivial_dihedral_symmetry(pattern):
        return None
    return pattern


def _sample_visibility(rng: np.random.Generator, full_mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(full_mask)
    area = int(coords.shape[0])
    visibility = np.zeros((4,) + full_mask.shape, dtype=bool)
    coverage = np.zeros(full_mask.shape, dtype=int)

    for r, c in coords:
        count = int(rng.choice(VISIBLE_COUNT_OPTIONS, p=VISIBLE_COUNT_PROBS))
        chosen = rng.choice(4, size=count, replace=False)
        visibility[chosen, int(r), int(c)] = True
        coverage[int(r), int(c)] = count

    if not np.any(coverage[full_mask] == 1):
        return None
    if not np.any(coverage[full_mask] >= 2):
        return None

    min_visible = max(4, area // 3)
    for q in range(4):
        visible_here = visibility[q] & full_mask
        count = int(np.sum(visible_here))
        if count < min_visible or count >= area:
            return None
        if not np.any(visible_here & (coverage == 1)):
            return None
    return visibility


def _build_input_grid(
    base: np.ndarray,
    visibility: np.ndarray,
    orientations: tuple[int, int, int, int],
    bg: int,
    divider: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n = int(base.shape[0])
    grid = np.full((2 * n + 1, 2 * n + 1), int(bg), dtype=int)
    grid[n, :] = int(divider)
    grid[:, n] = int(divider)

    full_mask = base != bg
    quadrants: list[np.ndarray] = []
    for q in range(4):
        partial = base.copy()
        partial[full_mask & ~visibility[q]] = int(bg)
        rotated = np.rot90(partial, orientations[q])
        quadrants.append(rotated)
        sl_r, sl_c = QUADRANT_SLICES[q](n)
        paste(grid, rotated, sl_r.start, sl_c.start)

    return grid, quadrants


def _aligned_union(quadrants: list[np.ndarray], orientations: tuple[int, int, int, int], bg: int) -> np.ndarray:
    complete = np.full_like(quadrants[0], int(bg))
    for q in range(4):
        aligned = np.rot90(quadrants[q], -orientations[q])
        mask = aligned != bg
        complete[mask] = aligned[mask]
    return complete


def _naive_union(quadrants: list[np.ndarray], bg: int) -> np.ndarray | None:
    merged = np.full_like(quadrants[0], int(bg))
    for quad in quadrants:
        mask = quad != bg
        conflict = mask & (merged != bg) & (merged != quad)
        if np.any(conflict):
            return None
        merged[mask] = quad[mask]
    return merged


def _semantic_checks(
    grid: np.ndarray,
    base: np.ndarray,
    quadrants: list[np.ndarray],
    visibility: np.ndarray,
    orientations: tuple[int, int, int, int],
    bg: int,
    divider: int,
) -> bool:
    candidates = _candidate_solutions(grid)
    if len(candidates) != 4:
        return False
    outputs = [item[2] for item in candidates]
    if not all(np.array_equal(outputs[0], out) for out in outputs[1:]):
        return False

    expected = _render_output(grid, base, base.shape[0], orientations)
    if not np.array_equal(outputs[0], expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    for q in range(4):
        if np.array_equal(quadrants[q], np.rot90(base, orientations[q])):
            return False

    aligned = _aligned_union(quadrants, orientations, bg)
    if not np.array_equal(aligned, base):
        return False

    for q in range(4):
        without_q = np.full_like(base, int(bg))
        for other in range(4):
            if other == q:
                continue
            part = np.rot90(quadrants[other], -orientations[other])
            mask = part != bg
            without_q[mask] = part[mask]
        if np.array_equal(without_q, base):
            return False

    naive = _naive_union(quadrants, bg)
    if naive is not None and any(np.array_equal(naive, np.rot90(base, k)) for k in range(4)):
        return False

    cover = np.sum(visibility, axis=0)
    for q in range(4):
        if not np.any(visibility[q] & (cover == 1)):
            return False

    if divider == bg:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(QUAD_MIN, QUAD_MAX + 1))
    num_components = int(rng.integers(COMPONENT_COUNT_RANGE[0], COMPONENT_COUNT_RANGE[1] + 1))
    chosen = rng.choice(PALETTE, size=num_components + 2, replace=False)
    bg = int(chosen[0])
    divider = int(chosen[1])
    colors = np.asarray(chosen[2:], dtype=int)

    base = _build_base_pattern(rng, n, bg, colors)
    if base is None:
        return None

    visibility = _sample_visibility(rng, base != bg)
    if visibility is None:
        return None

    orientations = tuple(int(x) for x in rng.permutation(4))
    grid, quadrants = _build_input_grid(base, visibility, orientations, bg, divider)

    if not _semantic_checks(grid, base, quadrants, visibility, orientations, bg, divider):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-complement puzzle")
