from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MAX_ATTEMPTS = 300
MAX_CANVAS = 30


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge scattered same-bbox sprite fragments by aligning their bboxes at the top-left.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps = _components8(g != bg)
    assert comps, "expected at least one foreground sprite"

    patches: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patches.append(((r0, c0), g[r0 : r1 + 1, c0 : c1 + 1]))
    patches.sort(key=lambda item: item[0])

    out_h, out_w = map(int, patches[0][1].shape)
    out = np.full((out_h, out_w), bg, dtype=int)
    for _, patch in patches:
        assert patch.shape == (out_h, out_w), "all sprites must share one bbox size"
        mask = patch != bg
        fill = (out == bg) & mask
        out[fill] = patch[fill]
    return out


def _gen_disjoint_walks(k: int, width: int, height: int, rng: np.random.Generator) -> list[list[int]] | None:
    starts = tuple(sorted(rng.choice(width, size=k, replace=False).tolist()))
    init_left = sum((1 << i) for i, c in enumerate(starts) if c == 0)
    init_right = sum((1 << i) for i, c in enumerate(starts) if c == width - 1)
    path_cache: dict[tuple[int, tuple[int, ...], int, int], tuple[int, ...]] = {}
    memo: dict[tuple[int, tuple[int, ...], int, int], bool] = {}

    def dfs(row: int, cols: tuple[int, ...], left_mask: int, right_mask: int) -> bool:
        key = (row, cols, left_mask, right_mask)
        if key in memo:
            return memo[key]

        if row == height - 1:
            full = (1 << k) - 1
            ok = left_mask == full and right_mask == full
            memo[key] = ok
            return ok

        remaining = height - 1 - row
        for i, c in enumerate(cols):
            if not (left_mask >> i) & 1 and c > remaining:
                memo[key] = False
                return False
            if not (right_mask >> i) & 1 and (width - 1 - c) > remaining:
                memo[key] = False
                return False

        candidates: list[list[int]] = []
        for c in cols:
            moves = [nc for d in (-1, 0, 1) if 0 <= (nc := c + d) < width]
            rng.shuffle(moves)
            candidates.append(moves)

        order = sorted(range(k), key=lambda i: len(candidates[i]))
        next_cols = [0] * k
        used: set[int] = set()

        def backtrack(idx: int) -> bool:
            if idx == k:
                nxt = tuple(next_cols)
                next_left = left_mask
                next_right = right_mask
                for i, x in enumerate(nxt):
                    if x == 0:
                        next_left |= 1 << i
                    if x == width - 1:
                        next_right |= 1 << i
                if dfs(row + 1, nxt, next_left, next_right):
                    path_cache[key] = nxt
                    return True
                return False

            i = order[idx]
            vals = list(candidates[i])
            vals.sort(key=lambda x: (x != 0, x != width - 1, abs(2 * x - (width - 1))))
            for x in vals:
                if x in used:
                    continue
                next_cols[i] = x
                used.add(x)
                if backtrack(idx + 1):
                    return True
                used.remove(x)
            return False

        ok = backtrack(0)
        memo[key] = ok
        return ok

    if not dfs(0, starts, init_left, init_right):
        return None

    cols_per_row = [starts]
    cur = starts
    left_mask = init_left
    right_mask = init_right
    for row in range(height - 1):
        nxt = path_cache[(row, cur, left_mask, right_mask)]
        cols_per_row.append(nxt)
        for i, x in enumerate(nxt):
            if x == 0:
                left_mask |= 1 << i
            if x == width - 1:
                right_mask |= 1 << i
        cur = nxt

    return [[cols_per_row[r][i] for r in range(height)] for i in range(k)]


def _build_patches(rng: np.random.Generator) -> list[np.ndarray] | None:
    k = int(rng.integers(2, 5))
    width = int(rng.integers(max(5, k + 2), 8))
    height = int(rng.integers(2 * width, 2 * width + 4))
    walks = _gen_disjoint_walks(k, width, height, rng)
    if walks is None:
        return None

    palette_size = int(rng.integers(3, 7))
    palette = rng.choice(np.arange(1, 10), size=palette_size, replace=False)
    patches: list[np.ndarray] = []
    used_colors: set[int] = set()

    for cols in walks:
        patch = np.full((height, width), BG, dtype=int)
        seg_count = int(rng.integers(1, 4))
        if seg_count == 1:
            cuts: list[int] = []
        else:
            cuts = sorted(rng.choice(np.arange(1, height), size=seg_count - 1, replace=False).tolist())
        bounds = [0] + cuts + [height]
        seg_colors = rng.choice(palette, size=seg_count, replace=True)

        for color, start, end in zip(seg_colors, bounds[:-1], bounds[1:]):
            for r in range(start, end):
                patch[r, cols[r]] = int(color)
                used_colors.add(int(color))
        patches.append(patch)

    if len(used_colors) < 2:
        return None

    if rng.random() < 0.5:
        patches = [patch.T for patch in patches]
    return patches


def _place_patches(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray | None:
    patch_h, patch_w = map(int, patches[0].shape)
    k = len(patches)
    gap = 2

    layouts: list[tuple[int, int, int, int]] = []
    for rows in range(1, k + 1):
        cols = (k + rows - 1) // rows
        min_h = rows * patch_h + (rows + 1) * gap
        min_w = cols * patch_w + (cols + 1) * gap
        if min_h <= MAX_CANVAS and min_w <= MAX_CANVAS:
            layouts.append((rows, cols, min_h, min_w))
    if not layouts:
        return None

    rows, cols, min_h, min_w = layouts[int(rng.integers(len(layouts)))]
    canvas_h = int(rng.integers(min_h, min(MAX_CANVAS, min_h + 4) + 1))
    canvas_w = int(rng.integers(min_w, min(MAX_CANVAS, min_w + 4) + 1))
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    row_edges = np.linspace(gap, canvas_h - gap, rows + 1, dtype=int)
    col_edges = np.linspace(gap, canvas_w - gap, cols + 1, dtype=int)
    slots: list[tuple[int, int]] = []
    for sr in range(rows):
        for sc in range(cols):
            top_min = int(row_edges[sr])
            top_max = int(row_edges[sr + 1] - patch_h)
            left_min = int(col_edges[sc])
            left_max = int(col_edges[sc + 1] - patch_w)
            if top_max < top_min or left_max < left_min:
                continue
            r0 = int(rng.integers(top_min, top_max + 1))
            c0 = int(rng.integers(left_min, left_max + 1))
            slots.append((r0, c0))
    if len(slots) < k:
        return None

    if len(slots) == k:
        chosen = list(slots)
    else:
        chosen = [slots[i] for i in rng.choice(len(slots), size=k, replace=False)]
    order = rng.permutation(k)

    for patch_idx, pos_idx in enumerate(order):
        r0, c0 = chosen[int(pos_idx)]
        patch = patches[patch_idx]
        grid[r0 : r0 + patch_h, c0 : c0 + patch_w] = np.where(
            patch != BG,
            patch,
            grid[r0 : r0 + patch_h, c0 : c0 + patch_w],
        )

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        patches = _build_patches(rng)
        if patches is None:
            continue

        grid = _place_patches(patches, rng)
        if grid is None:
            continue

        comps = _components8(grid != BG)
        if len(comps) != len(patches):
            continue
        bbox_shapes = {
            (r1 - r0 + 1, c1 - c0 + 1)
            for r0, c0, r1, c1 in (_bbox(comp) for comp in comps)
        }
        if len(bbox_shapes) != 1:
            continue

        merged = main(grid)
        if merged.shape != patches[0].shape:
            continue
        return grid

    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
