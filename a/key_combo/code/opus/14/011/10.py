from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
FOREGROUND_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
INPUT_SHAPE = (4, 4)
AREA_RANGE = range(4, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _foreground_color(grid: np.ndarray) -> int:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND)
    assert len(colors) == 1, "expected exactly one foreground color"
    return colors[0]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _detect_convex_corners(grid: np.ndarray, fg: int) -> list[tuple[int, int, str]]:
    h, w = grid.shape
    corners: list[tuple[int, int, str]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(fg):
                continue

            bg_dirs: list[str] = []
            if int(grid[r - 1, c]) == BACKGROUND:
                bg_dirs.append("U")
            if int(grid[r + 1, c]) == BACKGROUND:
                bg_dirs.append("D")
            if int(grid[r, c - 1]) == BACKGROUND:
                bg_dirs.append("L")
            if int(grid[r, c + 1]) == BACKGROUND:
                bg_dirs.append("R")

            if len(bg_dirs) == 2 and any(d in bg_dirs for d in "UD") and any(d in bg_dirs for d in "LR"):
                corners.append((r, c, "".join(sorted(bg_dirs))))
    return corners


def _kronecker_expand_black_mask(grid: np.ndarray, fg: int) -> np.ndarray:
    black_mask = np.asarray(grid == BACKGROUND, dtype=bool)
    return np.where(np.kron(black_mask, black_mask), fg, BACKGROUND).astype(int)


def _stamp_from_corners(
    grid: np.ndarray,
    fg: int,
    corners: list[tuple[int, int, str]] | None = None,
) -> np.ndarray:
    base = np.asarray(grid, dtype=int)
    marked = _detect_convex_corners(base, fg) if corners is None else corners
    out = base.copy()
    h, w = out.shape
    for r, c, _orientation in marked:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        patch = out[r0:r1, c0:c1]
        patch[patch == BACKGROUND] = fg
    return out


def _boundary_pixel_count(grid: np.ndarray, fg: int) -> int:
    h, w = grid.shape
    count = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if any(int(grid[nr, nc]) == BACKGROUND for nr, nc in _neighbors4(r, c)):
                count += 1
    return count


def _stamp_all_boundary_pixels(grid: np.ndarray, fg: int) -> np.ndarray:
    h, w = grid.shape
    out = np.asarray(grid, dtype=int).copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if not any(int(grid[nr, nc]) == BACKGROUND for nr, nc in _neighbors4(r, c)):
                continue
            patch = out[r - 1 : r + 2, c - 1 : c + 2]
            patch[patch == BACKGROUND] = fg
    return out


def _stamp_small_input_then_expand(grid: np.ndarray, fg: int) -> np.ndarray:
    small = np.asarray(grid, dtype=int).copy()
    for r, c, _orientation in _detect_convex_corners(small, fg):
        r0 = max(0, r - 1)
        r1 = min(small.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(small.shape[1], c + 2)
        patch = small[r0:r1, c0:c1]
        patch[patch == BACKGROUND] = fg
    return _kronecker_expand_black_mask(small, fg)


def _iterated_corner_stamping(grid: np.ndarray, fg: int) -> np.ndarray:
    current = np.asarray(grid, dtype=int).copy()
    while True:
        nxt = _stamp_from_corners(current, fg)
        if np.array_equal(current, nxt):
            return current
        current = nxt


def _stamp_neighborhood_overlap(corners: list[tuple[int, int, str]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for r, c, _orientation in corners:
        patch = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
        if seen & patch:
            return True
        seen |= patch
    return False


def _analyze_case(grid: np.ndarray) -> dict[str, object] | None:
    g = np.asarray(grid, dtype=int)
    if g.shape != INPUT_SHAPE:
        return None

    fg = _foreground_color(g)
    fg_mask = g == fg
    fg_area = int(fg_mask.sum())
    if fg_area not in AREA_RANGE or not _is_connected(fg_mask):
        return None

    input_corners = _detect_convex_corners(g, fg)
    if not input_corners:
        return None

    expanded = _kronecker_expand_black_mask(g, fg)
    if max(expanded.shape) > 20:
        return None

    expanded_corners = _detect_convex_corners(expanded, fg)
    if len(expanded_corners) < 2:
        return None

    boundary_count = _boundary_pixel_count(expanded, fg)
    if boundary_count <= len(expanded_corners):
        return None

    stamped = _stamp_from_corners(expanded, fg, expanded_corners)
    if np.array_equal(stamped, expanded):
        return None

    if np.array_equal(stamped, _stamp_all_boundary_pixels(expanded, fg)):
        return None

    if np.array_equal(stamped, _stamp_small_input_then_expand(g, fg)):
        return None

    if np.array_equal(stamped, _iterated_corner_stamping(expanded, fg)):
        return None

    added = int(np.count_nonzero(stamped != expanded))
    if added < 4:
        return None

    orientations = {orientation for _r, _c, orientation in expanded_corners}
    return {
        "mask": fg_mask.copy(),
        "area": fg_area,
        "corner_count": len(expanded_corners),
        "orientation_count": len(orientations),
        "overlap": _stamp_neighborhood_overlap(expanded_corners),
        "added": added,
    }


def _build_valid_case_pool() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for area in AREA_RANGE:
        for cells in itertools.combinations(range(INPUT_SHAPE[0] * INPUT_SHAPE[1]), area):
            mask = np.zeros(INPUT_SHAPE, dtype=bool)
            for idx in cells:
                mask[idx // INPUT_SHAPE[1], idx % INPUT_SHAPE[1]] = True
            if not _is_connected(mask):
                continue

            grid = np.where(mask, 1, BACKGROUND).astype(int)
            info = _analyze_case(grid)
            if info is not None:
                cases.append(info)
    assert cases, "expected at least one valid input mask"
    return cases


VALID_CASES = _build_valid_case_pool()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the black-cell mask by a Kronecker self-product, then stamp only the expanded convex corners.

    Input structure:
    - The input is a small black-background grid with exactly one foreground color.
    - Foreground pixels form one 4-connected blob, while black pixels define the binary mask that will be expanded.
    - The useful geometric cues are the L-shaped corner relationships that appear after the mask is expanded.

    Rule:
    - Build a binary mask where black input cells are 1 and foreground cells are 0.
    - Take the Kronecker self-product of that mask, recoloring 1-cells with the original foreground color and leaving 0-cells black.
    - In the expanded grid, find every foreground pixel whose four orthogonal neighbors contain exactly one vertical black neighbor and one horizontal black neighbor.
    - For each such corner, fill every black cell in its centered 3x3 neighborhood with the foreground color, leaving existing foreground cells unchanged.

    Key insight:
    - The decisive corners are not in the original 4x4 input but in the larger self-similar Kronecker expansion, so expansion must happen before the local 3x3 stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    fg = _foreground_color(g)
    expanded = _kronecker_expand_black_mask(g, fg)
    corners = _detect_convex_corners(expanded, fg)
    out = _stamp_from_corners(expanded, fg, corners)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def generate_input() -> np.ndarray:
    rare_overlap = random.random() < 0.35
    target_overlap = False if rare_overlap else True
    target_area = random.choice(tuple(AREA_RANGE))
    target_orientations = random.choice((1, 2, 3, 4))

    pool = VALID_CASES
    for key, value in (
        ("overlap", target_overlap),
        ("area", target_area),
        ("orientation_count", target_orientations),
    ):
        narrowed = [case for case in pool if case[key] == value]
        if narrowed:
            pool = narrowed

    chosen = random.choice(pool)
    fg = random.choice(FOREGROUND_PALETTE)
    grid = np.where(chosen["mask"], fg, BACKGROUND).astype(int)

    assert _analyze_case(grid) is not None
    return grid
