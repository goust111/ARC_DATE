from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FOREGROUND = 9
SINGLE_OVERLAP_COLOR = 2
TRIPLE_OVERLAP_COLOR = 3
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != FOREGROUND:
                continue
            if all(int(g[r + dr, c + dc]) == FOREGROUND for dr, dc in ORTHO_DIRS):
                centers.append((r, c))
    return centers


def _coverage_counts(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    H, W = map(int, shape)
    cover = np.zeros((H, W), dtype=int)
    for r, c in centers:
        cover[r - 2 : r + 3, c - 2 : c + 3] += 1
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """Detect maroon '+' centers, invert each local 5x5 neighborhood, and recolor newly revealed overlaps by coverage count.

    Input structure:
    - The grid is a black background with a single foreground color: maroon (9).
    - The maroon pixels form 2-4 thin '+' motifs, optionally connected to short tails or extra maroon fragments.
    - Every valid '+' center is at least two cells away from the border so its 5x5 neighborhood stays in-bounds.

    Rule:
    - A '+' center is any maroon pixel whose four orthogonal neighbors are also maroon.
    - For every detected center, take the 5x5 neighborhood centered on it.
    - For each pixel, count how many such 5x5 neighborhoods cover it.
    - If the count k is odd, invert membership inside the maroon mask: maroon becomes black, black becomes foreground.
    - Newly created foreground pixels are recolored by k: k=1 becomes red (2), k=3 becomes green (3).
    - Pixels with even positive coverage stay unchanged, and pixels with zero coverage stay unchanged.

    Key insight:
    - The detected '+' motifs do not just mark themselves; they define overlapping local inversion windows whose odd/even cancellation determines both where new foreground appears and which overlap color it receives.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    centers = _cross_centers(g)
    if not centers:
        return out

    cover = _coverage_counts(g.shape, centers)
    odd = (cover % 2) == 1

    odd_fg = odd & (g == FOREGROUND)
    odd_bg = odd & (g == BACKGROUND)
    out[odd_fg] = BACKGROUND
    out[odd_bg & (cover == 1)] = SINGLE_OVERLAP_COLOR
    out[odd_bg & (cover >= 3)] = TRIPLE_OVERLAP_COLOR
    return out


def _paint_plus(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = FOREGROUND
    for dr, dc in ORTHO_DIRS:
        grid[r + dr, c + dc] = FOREGROUND


def _extend_tail(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], extra_len: int) -> None:
    if extra_len <= 0:
        return
    r, c = center
    dr, dc = direction
    for step in range(2, 2 + extra_len):
        rr = r + dr * step
        cc = c + dc * step
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = FOREGROUND


def _pairwise_neighborhood_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 4


def _sample_isolated_center(
    H: int,
    W: int,
    existing: list[tuple[int, int]],
    *,
    attempts: int = 200,
) -> tuple[int, int] | None:
    for _ in range(attempts):
        cand = (random.randint(2, H - 3), random.randint(2, W - 3))
        if cand in existing:
            continue
        if any(max(abs(cand[0] - r), abs(cand[1] - c)) <= 4 for r, c in existing):
            continue
        return cand
    return None


def _base_centers(H: int, W: int, count: int, triple_mode: bool) -> list[tuple[int, int]] | None:
    centers: list[tuple[int, int]]
    if triple_mode:
        if H < 11 or W < 11 or count < 3:
            return None
        r0 = random.randint(2, H - 7)
        c0 = random.randint(2, W - 7)
        centers = [(r0, c0), (r0, c0 + 4), (r0 + 4, c0)]
    else:
        gap = random.choice([3, 4])
        horizontal = random.random() < 0.5
        if horizontal:
            r0 = random.randint(2, H - 3)
            c0 = random.randint(2, W - 3 - gap)
            centers = [(r0, c0), (r0, c0 + gap)]
        else:
            r0 = random.randint(2, H - 3 - gap)
            c0 = random.randint(2, W - 3)
            centers = [(r0, c0), (r0 + gap, c0)]

    while len(centers) < count:
        extra = _sample_isolated_center(H, W, centers)
        if extra is None:
            return None
        centers.append(extra)
    return centers


def _add_tails(grid: np.ndarray, center: tuple[int, int], centers: list[tuple[int, int]]) -> None:
    outward_dirs: list[tuple[int, int]] = []
    inward_dirs: list[tuple[int, int]] = []
    r, c = center
    for dr, dc in ORTHO_DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        toward_other = False
        for rr, cc in centers:
            if (rr, cc) == center:
                continue
            if dr != 0 and cc == c and (rr - r) * dr > 0:
                toward_other = True
            if dc != 0 and rr == r and (cc - c) * dc > 0:
                toward_other = True
        if toward_other:
            inward_dirs.append((dr, dc))
        else:
            outward_dirs.append((dr, dc))

    random.shuffle(outward_dirs)
    random.shuffle(inward_dirs)
    if outward_dirs:
        _extend_tail(grid, center, outward_dirs[0], random.choice([1, 2]))
    if len(outward_dirs) >= 2 and random.random() < 0.45:
        _extend_tail(grid, center, outward_dirs[1], random.choice([0, 1, 2]))
    if inward_dirs and random.random() < 0.2:
        _extend_tail(grid, center, inward_dirs[0], 1)


def _intended_overlap_exists(centers: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            if max(dr, dc) in (3, 4) and min(dr, dc) == 0:
                return True
    return False


def _semantic_checks(grid: np.ndarray, *, triple_mode: bool) -> bool:
    centers = _cross_centers(grid)
    if not 2 <= len(centers) <= 4:
        return False
    if any(r < 2 or r > grid.shape[0] - 3 or c < 2 or c > grid.shape[1] - 3 for r, c in centers):
        return False
    if not _intended_overlap_exists(centers):
        return False

    cover = _coverage_counts(grid.shape, centers)
    out = main(grid)

    has_preserved_fg = bool(np.any((grid == FOREGROUND) & (cover == 0) & (out == FOREGROUND)))
    has_erased_fg = bool(np.any((grid == FOREGROUND) & ((cover % 2) == 1) & (out == BACKGROUND)))
    has_single_new = bool(np.any((grid == BACKGROUND) & (cover == 1) & (out == SINGLE_OVERLAP_COLOR)))
    has_even_cancellation = bool(np.any((cover >= 2) & ((cover % 2) == 0) & (out == grid)))

    if not (has_preserved_fg and has_erased_fg and has_single_new and has_even_cancellation):
        return False

    if triple_mode:
        has_triple_new = bool(np.any((grid == BACKGROUND) & (cover == 3) & (out == TRIPLE_OVERLAP_COLOR)))
        if not has_triple_new:
            return False
    else:
        # Keep pair-mode visually simpler: no accidental triple-overlap branch.
        if int(cover.max()) >= 3:
            return False

    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            if not _pairwise_neighborhood_overlap(a, b):
                continue
            r0 = max(a[0] - 2, b[0] - 2)
            r1 = min(a[0] + 2, b[0] + 2)
            c0 = max(a[1] - 2, b[1] - 2)
            c1 = min(a[1] + 2, b[1] + 2)
            if r0 <= r1 and c0 <= c1:
                overlap = cover[r0 : r1 + 1, c0 : c1 + 1]
                if np.any(overlap == 2):
                    return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(11, 16)
    W = random.randint(11, 16)
    triple_mode = random.random() < 0.6
    count = random.choice([3, 4]) if triple_mode else random.choice([2, 3, 4])
    centers = _base_centers(H, W, count, triple_mode)
    if centers is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    for center in centers:
        _paint_plus(grid, center)
    for center in centers:
        _add_tails(grid, center, centers)

    detected = sorted(_cross_centers(grid))
    if detected != sorted(centers):
        return None
    if not _semantic_checks(grid, triple_mode=triple_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid local-inversion cross puzzle")
