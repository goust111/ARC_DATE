import random
from collections import Counter, deque
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]

GRID_SIDE_RANGE = (14, 28)
TRACK_COUNT_RANGE = (1, 4)
TRACK_SIZE_RANGE = (4, 10)
TRACK_PADDING = 2
MAX_GENERATE_ATTEMPTS = 300

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_order_from_bbox(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _ordered_ring_from_component(comp: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = list(comp)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    expected = _ring_order_from_bbox(r0, c0, r1, c1)
    assert set(cells) == set(expected), "each foreground component must be a 1-pixel rectangular ring"
    return expected


def _shift_ring_colors(ring_colors: list[int], shift: int) -> list[int]:
    out = [GREY] * len(ring_colors)
    for idx, color in enumerate(ring_colors):
        if int(color) != GREY:
            out[(idx + shift) % len(ring_colors)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored pixel clockwise along its rectangular track.

    The puzzle text only becomes deterministic if the loop has a canonical seam.
    This solver uses the top-left corner of each ring bbox as that seam; the ring
    order starts there, goes right across the top edge, and thus fixes clockwise.
    The orange marker's shorter arc to that seam gives the shift amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    out = g.copy()

    for comp in _components(g != bg):
        ring = _ordered_ring_from_component(comp)
        ring_colors = [int(g[r, c]) for r, c in ring]
        orange_positions = [idx for idx, color in enumerate(ring_colors) if color == ORANGE]
        assert len(orange_positions) == 1, "each ring must contain exactly one orange driver"
        orange_idx = int(orange_positions[0])
        shift = min(orange_idx, len(ring) - orange_idx)
        shifted = _shift_ring_colors(ring_colors, shift)
        for (r, c), color in zip(ring, shifted):
            out[r, c] = int(color)

    return out


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _can_place(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    return all(_boxes_separate(candidate, box, TRACK_PADDING) for box in boxes)


def _fill_ring(grid: np.ndarray, ring: list[tuple[int, int]], color: int) -> None:
    for r, c in ring:
        grid[r, c] = int(color)


def _decorate_ring(grid: np.ndarray, ring: list[tuple[int, int]]) -> bool:
    ring_len = len(ring)
    max_shift = (ring_len - 1) // 2
    if max_shift < 1:
        return False

    for _ in range(100):
        shift = random.randint(1, max_shift)
        orange_idx = shift if random.random() < 0.5 else ring_len - shift

        ring_colors = [GREY] * ring_len
        ring_colors[orange_idx] = ORANGE

        max_extra = min(5, max(2, ring_len // 3))
        extra_count = random.randint(2, max_extra)
        extra_positions = random.sample([idx for idx in range(ring_len) if idx != orange_idx], extra_count)

        palette_size = random.randint(2, min(4, len(COLOR_POOL)))
        palette = random.sample(COLOR_POOL, k=palette_size)
        extra_colors = [palette[idx % len(palette)] for idx in range(extra_count)]
        random.shuffle(extra_colors)
        for idx, color in zip(extra_positions, extra_colors):
            ring_colors[idx] = int(color)

        shifted = _shift_ring_colors(ring_colors, shift)
        if shifted == ring_colors:
            continue

        for (r, c), color in zip(ring, ring_colors):
            grid[r, c] = int(color)
        return True

    return False


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        bg = _background_color(grid)
        comps = _components(grid != bg)
        if not comps:
            return False
        for comp in comps:
            ring = _ordered_ring_from_component(comp)
            colors = [int(grid[r, c]) for r, c in ring]
            if sum(color == ORANGE for color in colors) != 1:
                return False
            if sum(color != GREY for color in colors) < 2:
                return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    target_tracks = random.randint(*TRACK_COUNT_RANGE)
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(target_tracks):
        placed = False
        for _ in range(120):
            track_h = random.randint(TRACK_SIZE_RANGE[0], min(TRACK_SIZE_RANGE[1], h - 2))
            track_w = random.randint(TRACK_SIZE_RANGE[0], min(TRACK_SIZE_RANGE[1], w - 2))
            if h - track_h - 1 < 1 or w - track_w - 1 < 1:
                continue

            r0 = random.randint(1, h - track_h - 1)
            c0 = random.randint(1, w - track_w - 1)
            r1 = r0 + track_h - 1
            c1 = c0 + track_w - 1
            box = (r0, c0, r1, c1)
            if not _can_place(boxes, box):
                continue

            ring = _ring_order_from_bbox(r0, c0, r1, c1)
            _fill_ring(grid, ring, GREY)
            if not _decorate_ring(grid, ring):
                _fill_ring(grid, ring, BLACK)
                continue

            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
