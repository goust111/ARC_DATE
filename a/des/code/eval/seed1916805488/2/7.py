import random
from collections import deque

import numpy as np


BG = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            total += 1
    return total


def _mirror_index(axis: str, shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    if axis == "v":
        return r, w - 1 - c
    return h - 1 - r, c


def _symmetry_score(mask: np.ndarray, axis: str) -> int:
    if axis == "v":
        return int(np.sum(mask == mask[:, ::-1]))
    return int(np.sum(mask == mask[::-1, :]))


def _repair_patch(patch: np.ndarray, axis: str, colors: tuple[int, int]) -> np.ndarray:
    out = patch.copy()
    mask = patch != BG
    h, w = patch.shape
    if axis == "v":
        pairs = [(r, c, r, w - 1 - c) for r in range(h) for c in range(w // 2)]
    else:
        pairs = [(r, c, h - 1 - r, c) for r in range(h // 2) for c in range(w)]

    a, b = colors

    def swapped(color: int) -> int:
        return b if color == a else a

    for r1, c1, r2, c2 in pairs:
        p = int(out[r1, c1])
        q = int(out[r2, c2])
        p_fg = p != BG
        q_fg = q != BG
        if not p_fg and not q_fg:
            continue
        if p_fg and q_fg:
            if p != swapped(q):
                out[r1, c1] = swapped(q)
            continue

        if p_fg:
            extra_support = _neighbor_count(mask, r1, c1)
            missing_support = _neighbor_count(mask, r2, c2)
            if missing_support >= max(2, extra_support):
                out[r2, c2] = swapped(p)
            else:
                out[r1, c1] = BG
        else:
            extra_support = _neighbor_count(mask, r2, c2)
            missing_support = _neighbor_count(mask, r1, c1)
            if missing_support >= max(2, extra_support):
                out[r1, c1] = swapped(q)
            else:
                out[r2, c2] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    comps = _components(g != BG)
    assert comps, "expected at least one object"
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(int(x) for x in patch.ravel() if int(x) != BG))
        assert len(colors) == 2, "each object must use exactly two colors"
        mask = patch != BG
        axis = "v" if _symmetry_score(mask, "v") >= _symmetry_score(mask, "h") else "h"
        fixed = _repair_patch(patch, axis, (colors[0], colors[1]))
        out[r0 : r1 + 1, c0 : c1 + 1] = fixed
    return out


def _build_half_mask(height: int, half_width: int) -> np.ndarray:
    mask = np.zeros((height, half_width), dtype=bool)
    seeds = [(r, half_width - 1) for r in range(height)]
    random.shuffle(seeds)
    target = random.randint(max(height + 2, height * half_width // 2), max(height + 3, int(height * half_width * 0.75)))
    start = seeds[0]
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target and frontier:
        rr, cc = random.choice(tuple(frontier))
        frontier.discard((rr, cc))
        nbrs = []
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < half_width and not mask[nr, nc]:
                nbrs.append((nr, nc))
        random.shuffle(nbrs)
        take = random.randint(1, max(1, min(3, len(nbrs))))
        for nr, nc in nbrs[:take]:
            mask[nr, nc] = True
            frontier.add((nr, nc))
        if random.random() < 0.5:
            frontier.add((rr, cc))
    for r in range(height):
        mask[r, half_width - 1] = True
    return mask


def _make_object_patch() -> tuple[np.ndarray, np.ndarray]:
    axis = random.choice(["v", "h"])
    c1, c2 = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 9], 2)
    for _ in range(50):
        if axis == "v":
            height = random.randint(4, 8)
            half_width = random.randint(2, 4)
            left = _build_half_mask(height, half_width)
            full_mask = np.concatenate([left, left[:, ::-1]], axis=1)
            patch = np.full(full_mask.shape, BG, dtype=int)
            for r in range(height):
                for c in range(half_width):
                    if not left[r, c]:
                        continue
                    color = c1 if random.random() < 0.5 else c2
                    patch[r, c] = color
                    patch[r, full_mask.shape[1] - 1 - c] = c2 if color == c1 else c1
        else:
            width = random.randint(4, 8)
            half_height = random.randint(2, 4)
            top = _build_half_mask(width, half_height).T
            full_mask = np.concatenate([top, top[::-1, :]], axis=0)
            patch = np.full(full_mask.shape, BG, dtype=int)
            for r in range(half_height):
                for c in range(width):
                    if not top[r, c]:
                        continue
                    color = c1 if random.random() < 0.5 else c2
                    patch[r, c] = color
                    patch[full_mask.shape[0] - 1 - r, c] = c2 if color == c1 else c1
        colors = {int(x) for x in patch.ravel() if int(x) != BG}
        if len(colors) == 2 and len(_components(patch != BG)) == 1:
            return patch, full_mask
    raise RuntimeError("failed to build connected two-color object")


def _missing_candidates(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    if axis == "v":
        it = [(r, c) for r in range(h) for c in range(w // 2)]
    else:
        it = [(r, c) for r in range(h // 2) for c in range(w)]
    for r, c in it:
        rr, cc = _mirror_index(axis, mask.shape, r, c)
        if not mask[r, c] or not mask[rr, cc]:
            continue
        if _neighbor_count(mask, r, c) >= 2 and _neighbor_count(mask, rr, cc) >= 2:
            cells.append((r, c))
            cells.append((rr, cc))
    return cells


def _extra_candidates(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            rr, cc = _mirror_index(axis, mask.shape, r, c)
            if mask[rr, cc]:
                continue
            support_here = _neighbor_count(mask, r, c)
            support_mirror = _neighbor_count(mask, rr, cc)
            if support_here == 1 and support_mirror == 0:
                cells.append((r, c))
    return cells


def _apply_defects(base: np.ndarray, axis: str) -> np.ndarray | None:
    patch = base.copy()
    mask = patch != BG
    branch = random.choice(["missing", "extra"])
    if branch == "missing":
        candidates = _missing_candidates(mask, axis)
        if not candidates:
            return None
        random.shuffle(candidates)
        used = set()
        count = random.randint(1, min(3, len(candidates)))
        applied = 0
        for r, c in candidates:
            if (r, c) in used:
                continue
            rr, cc = _mirror_index(axis, mask.shape, r, c)
            used.add((r, c))
            used.add((rr, cc))
            patch[r, c] = BG
            mask[r, c] = False
            applied += 1
            if applied == count:
                break
        if applied == 0 or len(_components(patch != BG)) != 1:
            return None
        return patch

    candidates = _extra_candidates(mask, axis)
    if not candidates:
        return None
    random.shuffle(candidates)
    count = random.randint(1, min(3, len(candidates)))
    applied = 0
    colors = sorted(set(int(x) for x in base.ravel() if int(x) != BG))
    for r, c in candidates:
        if mask[r, c]:
            continue
        if _neighbor_count(mask, r, c) != 1:
            continue
        rr, cc = _mirror_index(axis, mask.shape, r, c)
        if _neighbor_count(mask, rr, cc) != 0:
            continue
        color = random.choice(colors)
        patch[r, c] = color
        mask[r, c] = True
        applied += 1
        if applied == count:
            break
    if applied == 0 or len(_components(patch != BG)) != 1:
        return None
    return patch


def _touching(mask: np.ndarray, top: int, left: int, occ: np.ndarray) -> bool:
    h, w = mask.shape
    H, W = occ.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return bool(np.any(occ[r0:r1, c0:c1]))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 24)
    W = random.randint(16, 24)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    target_count = random.randint(2, 5)
    placed = 0
    for _ in range(target_count * 8):
        base, mask = _make_object_patch()
        axis = "v" if base.shape[1] % 2 == 0 and np.array_equal(mask, mask[:, ::-1]) else "h"
        patch = _apply_defects(base, axis)
        if patch is None:
            continue
        h, w = patch.shape
        positions = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
        random.shuffle(positions)
        for r0, c0 in positions:
            if np.any(occupied[r0 : r0 + h, c0 : c0 + w] & (patch != BG)):
                continue
            if _touching(patch != BG, r0, c0, occupied):
                continue
            grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
            occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG
            placed += 1
            break
        if placed == target_count:
            break
    if placed < 2:
        return None
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        colors = {int(x) for x in grid[r0 : r1 + 1, c0 : c1 + 1].ravel() if int(x) != BG}
        if len(colors) != 2:
            return None
    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
