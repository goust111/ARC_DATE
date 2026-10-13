from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SEED_COLORS = (RED, GREEN, BLUE, YELLOW)
BORDER_COLORS = (5, 6, 7, 8, 9)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
INNER_H_RANGE = (2, 8)
INNER_W_RANGE = (2, 10)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400

_GEN_COUNTER = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False

    point_set = set(points)
    expected = 2 * (h + w) - 4
    if len(point_set) != expected:
        return False

    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False
    return True


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for pts in _components(grid != BG):
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        if not _is_rect_frame(pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        frame_color = next(iter(colors))
        frames.append((r0, c0, r1, c1, frame_color))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame's interior according to its seed color.

    Input structure:
    - Black background.
    - Several non-touching 1-pixel-thick rectangular frames of distinct colors.
    - Each frame contains exactly one seed pixel in its black interior.
    - Seed colors are red, green, blue, or yellow.

    Rule:
    - red seed: fill the whole interior with the frame color.
    - green seed: fill only the seed row inside the frame.
    - blue seed: fill only the seed column inside the frame.
    - yellow seed: fill both the seed row and the seed column.
    - The seed itself is overwritten by the frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for r0, c0, r1, c1, frame_color in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_points = np.argwhere(inner != BG)
        assert len(seed_points) == 1, "each frame must contain exactly one seed"

        sr, sc = map(int, seed_points[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in SEED_COLORS, "seed color must encode one of the four fill modes"
        assert seed_color != frame_color, "seed color must differ from the frame color"

        abs_r = r0 + 1 + sr
        abs_c = c0 + 1 + sc
        if seed_color == RED:
            out[r0 + 1 : r1, c0 + 1 : c1] = frame_color
        elif seed_color == GREEN:
            out[abs_r, c0 + 1 : c1] = frame_color
        elif seed_color == BLUE:
            out[r0 + 1 : r1, abs_c] = frame_color
        else:
            out[abs_r, c0 + 1 : c1] = frame_color
            out[r0 + 1 : r1, abs_c] = frame_color

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _pick_seed_colors(rng: random.Random, count: int, required_seed: int) -> list[int]:
    if count >= 4:
        seeds = list(SEED_COLORS)
        seeds.extend(rng.choice(SEED_COLORS) for _ in range(count - 4))
    else:
        seeds = rng.sample(SEED_COLORS, count)
        if required_seed not in seeds:
            seeds[rng.randrange(count)] = required_seed
    rng.shuffle(seeds)
    return seeds


def _place_frame(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    inner_h: int,
    inner_w: int,
    frame_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if outer_h + 2 > h or outer_w + 2 > w:
        return False

    top_choices = range(1, h - outer_h)
    left_choices = range(1, w - outer_w)
    if not top_choices or not left_choices:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.choice(tuple(top_choices))
        left = rng.choice(tuple(left_choices))
        bottom = top + outer_h - 1
        right = left + outer_w - 1
        halo = reserved[top - 1 : bottom + 2, left - 1 : right + 2]
        if np.any(halo):
            continue

        _draw_frame(grid, top, left, inner_h, inner_w, frame_color)
        seed_r = rng.randrange(top + 1, bottom)
        seed_c = rng.randrange(left + 1, right)
        grid[seed_r, seed_c] = seed_color
        reserved[top - 1 : bottom + 2, left - 1 : right + 2] = True
        return True

    return False


def _gen_attempt(rng: random.Random, required_seed: int) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    capacity = max(1, ((h - 2) * (w - 2)) // 45)
    max_objects = min(len(BORDER_COLORS), capacity)
    count = rng.randint(1, max_objects)

    frame_colors = rng.sample(BORDER_COLORS, count)
    seed_colors = _pick_seed_colors(rng, count, required_seed)

    size_divisor = 2 + (count >= 3) + (count >= 5)
    max_inner_h = min(INNER_H_RANGE[1], max(INNER_H_RANGE[0], h // size_divisor))
    max_inner_w = min(INNER_W_RANGE[1], max(INNER_W_RANGE[0], w // size_divisor))
    if max_inner_h < INNER_H_RANGE[0] or max_inner_w < INNER_W_RANGE[0]:
        return None

    for frame_color, seed_color in zip(frame_colors, seed_colors):
        if rng.random() < 0.25:
            inner_h = rng.randint(INNER_H_RANGE[0], min(INNER_H_RANGE[1], h - 4))
            inner_w = rng.randint(INNER_W_RANGE[0], min(INNER_W_RANGE[1], w - 4))
        else:
            inner_h = rng.randint(INNER_H_RANGE[0], max_inner_h)
            inner_w = rng.randint(INNER_W_RANGE[0], max_inner_w)
        if not _place_frame(rng, grid, reserved, inner_h, inner_w, frame_color, seed_color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = random.Random()
    required_seed = SEED_COLORS[_GEN_COUNTER % len(SEED_COLORS)]
    _GEN_COUNTER += 1

    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng, required_seed)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
