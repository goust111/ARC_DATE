from __future__ import annotations

import numpy as np


BG = 0
FG = 1
CROSS = 2
OVERLAP = 3
RAY = 4
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _find_centers(fg: np.ndarray) -> list[tuple[int, int]]:
    h, w = fg.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not fg[r, c]:
                continue
            if all(fg[r + dr, c + dc] for dr, dc in DIRS4):
                centers.append((r, c))
    return centers


def _cross_mask(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        mask[r, c] = True
        for dr, dc in DIRS4:
            mask[r + dr, c + dc] = True
    return mask


def _analyze(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    fg = np.asarray(grid == FG, dtype=bool)
    centers = _find_centers(fg)
    cross = _cross_mask(fg.shape, centers)
    claims = np.zeros(fg.shape, dtype=int)
    h, w = fg.shape

    for r0, c0 in centers:
        for dr, dc in DIRS4:
            r, c = r0, c0
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w) or not fg[r, c]:
                    break
                if not cross[r, c]:
                    claims[r, c] += 1
    return centers, cross, claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect all embedded '+' centers in the blue foreground and recolor the cross footprints, outward rays, and ray overlaps differently.

    Input structure:
    - A black background grid contains one 4-connected blue foreground shape.
    - Some blue pixels are '+' centers: the pixel itself and its four orthogonal neighbors are blue.
    - The shape is large enough that each true center has straight blue continuation in all four directions.

    Rule:
    - Find every blue '+' center.
    - Mark each center together with its four orthogonal neighbors as cross pixels.
    - From every center, continue straight in the four cardinal directions through the blue shape until the blue run ends; pixels already belonging to any cross stay classified as cross, while other traversed blue pixels are ray pixels.
    - Recolor cross pixels red, single-center ray pixels yellow, multi-center ray overlaps green, keep untouched blue foreground blue, and keep background black.

    Key insight:
    - The local '+' test only identifies launch points; the real transformation requires extending all four straight rays from every detected center and then comparing their combined coverage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    centers, cross, claims = _analyze(g)
    out = np.full_like(g, BG)
    out[g == FG] = FG
    out[cross] = CROSS
    out[(claims == 1) & ~cross] = RAY
    out[(claims >= 2) & ~cross] = OVERLAP
    return out


def _sample_rows(rng: np.random.Generator, h: int, k: int) -> list[int] | None:
    rows = []
    lo, hi = 3, h - 4
    candidates = list(range(lo, hi + 1))
    rng.shuffle(candidates)
    for r in sorted(candidates):
        if all(abs(r - prev) >= 4 for prev in rows):
            rows.append(r)
            if len(rows) == k:
                return sorted(rows)
    return None


def _ray_lengths(fg: np.ndarray, center: tuple[int, int]) -> list[int]:
    r0, c0 = center
    lengths = []
    h, w = fg.shape
    for dr, dc in DIRS4:
        steps = 0
        r, c = r0, c0
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w) or not fg[r, c]:
                break
            steps += 1
        lengths.append(steps)
    return lengths


def _connected4(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _validate_case(grid: np.ndarray, intended_centers: set[tuple[int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BG, FG}:
        return False

    fg = grid == FG
    if not _connected4(fg):
        return False

    centers, cross, claims = _analyze(grid)
    center_set = set(centers)
    if center_set != intended_centers:
        return False
    if not (2 <= len(centers) <= 4):
        return False

    for center in centers:
        if min(_ray_lengths(fg, center)) < 3:
            return False

    out = main(grid)
    if np.count_nonzero(out == CROSS) < 5 * len(centers):
        return False
    if not np.any((claims >= 2) & ~cross):
        return False
    if not np.any((claims == 1) & ~cross):
        return False
    if not np.any((fg) & (~cross) & (claims == 0)):
        return False

    area = int(fg.sum())
    if area < 40 or area > 80:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice(np.array([2, 3, 4]), p=np.array([0.3, 0.35, 0.35])))
    min_h = 11 + 2 * k
    max_h = min(25, min_h + 6)
    min_w = 14 if k <= 3 else 13
    max_w = 20 if k <= 3 else 18
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))

    rows = _sample_rows(rng, h, k)
    if rows is None:
        return None

    c0 = int(rng.integers(3, w - 3))
    top_end = int(rng.integers(0, rows[0] - 2))
    bottom_end = int(rng.integers(rows[-1] + 2, h))

    grid = np.zeros((h, w), dtype=int)
    grid[top_end : bottom_end + 1, c0] = FG

    bar_specs: list[tuple[int, int, int]] = []
    for r in rows:
        left_end = int(rng.integers(0, c0 - 1))
        right_end = int(rng.integers(c0 + 2, w))
        if c0 - left_end < 3 or right_end - c0 < 3:
            return None
        grid[r, left_end : right_end + 1] = FG
        bar_specs.append((r, left_end, right_end))

    used_spur_cols: set[int] = set()
    target_area = int(rng.integers(max(44, 40 + 4 * (k - 2)), min(77, 58 + 6 * k)))
    max_spurs = int(rng.integers(2, 6))

    anchors: list[tuple[int, int]] = []
    for r, left_end, right_end in bar_specs:
        for c in range(left_end, right_end + 1):
            if c != c0:
                anchors.append((r, c))
    rng.shuffle(anchors)

    spur_count = 0
    for r, c in anchors:
        if int((grid == FG).sum()) >= target_area or spur_count >= max_spurs:
            break
        if c in used_spur_cols:
            continue
        directions = []
        for dr in (-1, 1):
            length = 0
            nr = r + dr
            while 0 <= nr < h and grid[nr, c] == BG and length < 3:
                length += 1
                nr += dr
            if length > 0:
                directions.append((dr, length))
        if not directions:
            continue
        dr, max_len = directions[int(rng.integers(0, len(directions)))]
        remaining = target_area - int((grid == FG).sum())
        length = int(rng.integers(1, min(max_len, remaining) + 1))
        for step in range(1, length + 1):
            grid[r + dr * step, c] = FG
        used_spur_cols.add(c)
        spur_count += 1

    if int((grid == FG).sum()) < 40:
        extra_anchors = anchors.copy()
        rng.shuffle(extra_anchors)
        for r, c in extra_anchors:
            if int((grid == FG).sum()) >= 40:
                break
            if c in used_spur_cols:
                continue
            for dr in (-1, 1):
                nr = r + dr
                if 0 <= nr < h and grid[nr, c] == BG:
                    grid[nr, c] = FG
                    used_spur_cols.add(c)
                    break

    intended_centers = {(r, c0) for r in rows}
    if not _validate_case(grid, intended_centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-ray sample")
