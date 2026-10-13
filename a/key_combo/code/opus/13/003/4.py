from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


INPUT_SIZE = 4
BACKGROUND = 0
FOREGROUND = 3
RAY_COLOR = 2
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _kronecker_expand(grid: np.ndarray) -> np.ndarray:
    black_mask = np.asarray(grid == BACKGROUND, dtype=bool)
    return np.where(np.kron(black_mask, black_mask), FOREGROUND, BACKGROUND).astype(int)


def _convex_corners(grid: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = grid.shape
    corners: list[tuple[int, int, tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != FOREGROUND:
                continue

            green_neighbors: list[tuple[int, int]] = []
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == FOREGROUND:
                    green_neighbors.append((dr, dc))

            if len(green_neighbors) != 2:
                continue

            vertical = [offset for offset in green_neighbors if offset[1] == 0]
            horizontal = [offset for offset in green_neighbors if offset[0] == 0]
            if len(vertical) != 1 or len(horizontal) != 1:
                continue

            inner_dr = vertical[0][0]
            inner_dc = horizontal[0][1]
            inner_r = r + inner_dr
            inner_c = c + inner_dc
            if not _in_bounds(grid.shape, inner_r, inner_c):
                continue
            if int(grid[inner_r, inner_c]) != BACKGROUND:
                continue

            outward = (-inner_dr, -inner_dc)
            corners.append((r, c, outward))

    return corners


def _cast_rays(expanded: np.ndarray, corners: list[tuple[int, int, tuple[int, int]]]) -> np.ndarray:
    out = expanded.copy()
    for r, c, (dr, dc) in corners:
        rr, cc = r + dr, c + dc
        while _in_bounds(expanded.shape, rr, cc) and int(expanded[rr, cc]) == BACKGROUND:
            out[rr, cc] = RAY_COLOR
            rr += dr
            cc += dc
    return out


def _decode_bits(bits: int) -> np.ndarray:
    flat = np.array([(bits >> idx) & 1 for idx in range(INPUT_SIZE * INPUT_SIZE)], dtype=int)
    return np.where(flat.reshape(INPUT_SIZE, INPUT_SIZE), FOREGROUND, BACKGROUND).astype(int)


def _green_connected(mask: np.ndarray) -> bool:
    start = np.argwhere(mask)
    if start.size == 0:
        return False

    seen = {tuple(map(int, start[0]))}
    stack = [tuple(map(int, start[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < INPUT_SIZE and 0 <= cc < INPUT_SIZE and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))

    return len(seen) == int(mask.sum())


def _corners_well_spaced(corners: list[tuple[int, int, tuple[int, int]]], min_gap: int = 2) -> bool:
    for idx, (r0, c0, _) in enumerate(corners):
        for r1, c1, _ in corners[idx + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < min_gap:
                return False
    return True


def _ray_lengths(expanded: np.ndarray, corners: list[tuple[int, int, tuple[int, int]]]) -> list[int]:
    lengths: list[int] = []
    for r, c, (dr, dc) in corners:
        rr, cc = r + dr, c + dc
        length = 0
        while _in_bounds(expanded.shape, rr, cc) and int(expanded[rr, cc]) == BACKGROUND:
            length += 1
            rr += dr
            cc += dc
        lengths.append(length)
    return lengths


@lru_cache(maxsize=1)
def _candidate_pool() -> tuple[tuple[int, int, int, int, int], ...]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for bits in range(1 << (INPUT_SIZE * INPUT_SIZE)):
        grid = _decode_bits(bits)
        green_count = int(np.count_nonzero(grid == FOREGROUND))
        if not 3 <= green_count <= 6:
            continue
        if not _green_connected(grid == FOREGROUND):
            continue

        expanded = _kronecker_expand(grid)
        corners = _convex_corners(expanded)
        if len(corners) < 4:
            continue
        if not _corners_well_spaced(corners):
            continue

        ray_lengths = _ray_lengths(expanded, corners)
        if max(ray_lengths, default=0) < 2:
            continue

        direction_count = len({direction for _, _, direction in corners})
        total_ray_length = int(sum(ray_lengths))
        candidates.append((bits, green_count, direction_count, len(corners), total_ray_length))

    assert candidates, "expected at least one valid 4x4 input pattern"
    return tuple(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 4x4 black-cell mask into a self-similar green pattern, then fire red diagonal rays from its convex corners.

    Input structure:
    - The input is a 4x4 grid using only black(0) and green(3).
    - Green pixels form one connected foreground pattern; black pixels define the mask used for expansion.
    - The output is a 16x16 grid obtained by replacing each black input cell with a 4x4 copy of the black-cell mask and each green input cell with an all-black 4x4 block.

    Rule:
    - Build the input's black-cell mask M where M[r,c]=1 exactly on black cells.
    - Form the expanded 16x16 grid as the Kronecker self-product of M, recoloring 1-cells green(3) and 0-cells black(0).
    - In that expanded grid, mark every green pixel that has exactly two perpendicular green 4-neighbors and a black diagonal cell in the neighbors' shared quadrant as a convex corner.
    - From each convex corner, extend a red(2) ray along the opposite diagonal direction, painting only black cells until the ray would leave the grid or hit green.

    Key insight:
    - The visible red structure is not present in the 4x4 input directly; it emerges only after understanding that the black mask is copied into every black cell to create new convex corners in the self-similar expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_SIZE, INPUT_SIZE), "expected a 4x4 input grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, FOREGROUND}), "expected only black and green input cells"

    expanded = _kronecker_expand(g)
    corners = _convex_corners(expanded)
    return _cast_rays(expanded, corners)


def generate_input() -> np.ndarray:
    pool = _candidate_pool()
    green_count = random.choice((3, 4, 5, 6))
    bucket = [record for record in pool if record[1] == green_count]
    assert bucket, f"no candidate patterns for green-count bucket {green_count}"

    if random.random() < 0.7:
        richer = [record for record in bucket if record[2] >= 2]
        if richer:
            bucket = richer

    if random.random() < 0.5:
        longer = [record for record in bucket if record[4] >= 8]
        if longer:
            bucket = longer

    bits = random.choice(bucket)[0]
    return _decode_bits(bits)
