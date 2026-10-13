from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
RNG = np.random.default_rng()
GEN_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(coords) > len(best):
                best = coords

    if not best:
        raise ValueError("expected a non-background frame component")
    return np.array(best, dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_colors(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> set[int]:
    vals: set[int] = set(map(int, grid[top, left : right + 1]))
    if bottom != top:
        vals.update(map(int, grid[bottom, left : right + 1]))
    if bottom - top >= 2:
        vals.update(map(int, grid[top + 1 : bottom, left]))
        if right != left:
            vals.update(map(int, grid[top + 1 : bottom, right]))
    return vals


def _extract_layers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    top, left, bottom, right = bbox
    colors: list[int] = []

    while top <= bottom and left <= right:
        vals = _boundary_colors(grid, top, left, bottom, right) - {BG}
        if not vals:
            break
        if len(vals) != 1:
            raise ValueError("expected each nested boundary to have one color")
        colors.append(vals.pop())
        top += 1
        left += 1
        bottom -= 1
        right -= 1

    if not colors:
        raise ValueError("failed to extract any frame layers")
    return colors


def _paint_frame(grid: np.ndarray, top: int, left: int, layer_colors: list[int] | np.ndarray) -> None:
    layer_colors = list(map(int, layer_colors))
    side = 2 * len(layer_colors)
    bottom = top + side - 1
    right = left + side - 1
    for depth, color in enumerate(layer_colors):
        t = top + depth
        l = left + depth
        b = bottom - depth
        r = right - depth
        grid[t, l : r + 1] = color
        grid[b, l : r + 1] = color
        grid[t : b + 1, l] = color
        grid[t : b + 1, r] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the concentric frame layers whose colors appear among the outside seed pixels,
    then rebuild those layers as a compact nest in original outer-to-inner order.
    """
    g = np.asarray(grid, dtype=int)
    frame_coords = _largest_component(g != BG)
    r0, c0, r1, c1 = _bbox(frame_coords)
    layer_colors = _extract_layers(g, (r0, c0, r1, c1))

    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True
    seed_colors = set(map(int, g[(g != BG) & (~frame_mask)]))

    kept = [color for color in layer_colors if color in seed_colors]
    if not kept:
        raise ValueError("expected at least one selected layer")

    out = np.full((2 * len(kept), 2 * len(kept)), BG, dtype=int)
    _paint_frame(out, 0, 0, kept)
    return out


def _seed_candidates(h: int, w: int, top: int, left: int, side: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    row_lo = top - 1
    row_hi = top + side
    col_lo = left - 1
    col_hi = left + side
    for r in range(h):
        for c in range(w):
            if row_lo <= r <= row_hi and col_lo <= c <= col_hi:
                continue
            candidates.append((r, c))
    RNG.shuffle(candidates)
    return candidates


def _is_isolated(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in placed:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layer_count = int(RNG.integers(3, 6))
    side = 2 * layer_count

    h = int(RNG.integers(side + 6, side + 13))
    w = int(RNG.integers(side + 6, side + 13))
    top = (h - side) // 2
    left = (w - side) // 2

    grid = np.full((h, w), BG, dtype=int)
    layer_colors = RNG.choice(PALETTE, size=layer_count, replace=False)
    _paint_frame(grid, top, left, layer_colors)

    keep_count = int(RNG.integers(1, layer_count + 1))
    keep_ids = RNG.choice(layer_count, size=keep_count, replace=False)
    keep_colors = [int(layer_colors[i]) for i in keep_ids]

    extra_count = int(RNG.integers(0, layer_count + 1))
    seed_colors = keep_colors + list(map(int, RNG.choice(keep_colors, size=extra_count, replace=True)))
    RNG.shuffle(seed_colors)

    candidates = _seed_candidates(h, w, top, left, side)
    seeds: list[tuple[int, int]] = []
    for color in seed_colors:
        placed = False
        for idx, pos in enumerate(candidates):
            if _is_isolated(pos, seeds):
                r, c = pos
                grid[r, c] = color
                seeds.append(pos)
                candidates.pop(idx)
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
