from __future__ import annotations

import random

import numpy as np


BG = 0
TRACK = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        path.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    return path


def _analyze_ring(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    path = _ring_path(r0, c0, r1, c1)
    if len(path) != len(comp) or set(path) != set(comp):
        return None

    track = int(grid[r0, c0])
    corners = ((r0, c0), (r0, c1), (r1, c1), (r1, c0))
    if track == BG or any(int(grid[r, c]) != track for r, c in corners):
        return None
    return path, track


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored marker clockwise along its rectangular track by a number of steps equal
    to the number of markers on that same track.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g != BG):
        ring = _analyze_ring(g, comp)
        if ring is None:
            continue
        path, track = ring
        markers = [(idx, int(g[r, c])) for idx, (r, c) in enumerate(path) if int(g[r, c]) != track]
        if not markers:
            continue

        for r, c in path:
            out[r, c] = track

        shift = len(markers)
        ring_len = len(path)
        for idx, color in markers:
            nr, nc = path[(idx + shift) % ring_len]
            out[nr, nc] = color

    return out


def _sample_marker_indices(path: list[tuple[int, int]], count: int) -> list[int]:
    r0, c0 = path[0]
    r1 = max(r for r, _ in path)
    c1 = max(c for _, c in path)
    corners = {(r0, c0), (r0, c1), (r1, c1), (r1, c0)}
    candidates = [idx for idx, pos in enumerate(path) if pos not in corners]
    ring_len = len(path)

    for _ in range(60):
        chosen = sorted(random.sample(candidates, count))
        if count == 1:
            return chosen
        if all(((b - a) % ring_len) > 1 for a, b in zip(chosen, chosen[1:] + [chosen[0] + ring_len])):
            return chosen
    return sorted(random.sample(candidates, count))


def _make_ring_patch(height: int, width: int, marker_count: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = TRACK
    patch[-1, :] = TRACK
    patch[:, 0] = TRACK
    patch[:, -1] = TRACK

    path = _ring_path(0, 0, height - 1, width - 1)
    marker_indices = _sample_marker_indices(path, marker_count)
    colors = random.sample(MARKER_COLORS, marker_count)
    for idx, color in zip(marker_indices, colors):
        r, c = path[idx]
        patch[r, c] = color
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _reserve(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    used[rr0:rr1, cc0:cc1] = True


def _validate_input(grid: np.ndarray) -> bool:
    comps = _components(grid != BG)
    if not comps:
        return False
    for comp in comps:
        ring = _analyze_ring(grid, comp)
        if ring is None:
            return False
        path, track = ring
        markers = [int(grid[r, c]) for r, c in path if int(grid[r, c]) != track]
        if not markers or len(markers) != len(set(markers)):
            return False
    return True


def _gen_ring_size(ring_count: int, height: int, width: int) -> tuple[int, int]:
    max_h_cap = {1: 11, 2: 9, 3: 8, 4: 7}[ring_count]
    max_w_cap = {1: 14, 2: 11, 3: 10, 4: 9}[ring_count]
    ring_h = random.randint(3, min(max_h_cap, height - 2))
    ring_w = random.randint(3, min(max_w_cap, width - 2))
    return ring_h, ring_w


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(1, 4)
    min_side = 10 + 3 * ring_count
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    grid = np.full((height, width), BG, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    for _ in range(ring_count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            ring_h, ring_w = _gen_ring_size(ring_count, height, width)
            if ring_h > height - 2 or ring_w > width - 2:
                continue
            max_r0 = height - ring_h - 1
            max_c0 = width - ring_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue

            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            if not _can_place(used, r0, c0, ring_h, ring_w):
                continue

            path_len = 2 * (ring_h + ring_w) - 4
            non_corner_slots = path_len - 4
            max_markers = min(6, max(1, non_corner_slots // 2))
            marker_count = random.randint(1, max_markers)
            patch = _make_ring_patch(ring_h, ring_w, marker_count)
            grid[r0 : r0 + ring_h, c0 : c0 + ring_w] = patch
            _reserve(used, r0, c0, ring_h, ring_w)
            placed = True
            break
        if not placed:
            return None

    if not _validate_input(grid):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
