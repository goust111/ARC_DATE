from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("top", "bottom", "left", "right")
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _find_rectangular_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(g == FRAME_COLOR):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = _perimeter_cells(r0, c0, r1, c1)
        if len(comp) == len(expected) and set(comp) == expected:
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _seed_for_frame(g: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = frame
    seeds: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(g[r, c])
            if v not in (BACKGROUND, FRAME_COLOR):
                seeds.append((r, c, v))
    assert len(seeds) == 1, "each rectangular frame must contain exactly one colored seed pixel"
    return seeds[0]


def _seed_side(g: np.ndarray, frame: tuple[int, int, int, int], r: int, c: int) -> str:
    r0, c0, r1, c1 = frame
    sides: list[str] = []
    if r == r0 + 1 and int(g[r - 1, c]) == FRAME_COLOR:
        sides.append("top")
    if r == r1 - 1 and int(g[r + 1, c]) == FRAME_COLOR:
        sides.append("bottom")
    if c == c0 + 1 and int(g[r, c - 1]) == FRAME_COLOR:
        sides.append("left")
    if c == c1 - 1 and int(g[r, c + 1]) == FRAME_COLOR:
        sides.append("right")
    assert len(sides) == 1, "seed must touch exactly one inner wall of its frame"
    return sides[0]


def _paint_inner_wall(
    out: np.ndarray,
    frame: tuple[int, int, int, int],
    side: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = frame
    if side == "top":
        out[r0 + 1, c0 + 1 : c1] = color
    elif side == "bottom":
        out[r1 - 1, c0 + 1 : c1] = color
    elif side == "left":
        out[r0 + 1 : r1, c0 + 1] = color
    else:
        out[r0 + 1 : r1, c1 - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each interior seed pixel into a colored stroke along the touched inner wall of its gray rectangular frame.

    Input structure:
    - The background is black.
    - One or more complete one-pixel-thick gray rectangular frames enclose mostly black interiors.
    - Each frame interior contains exactly one non-black, non-gray seed pixel, and that seed is 4-adjacent to one inner wall: top, bottom, left, or right.

    Rule:
    - Detect every complete gray rectangular frame and find the seed pixel inside it.
    - Use the seed's 4-neighbor contact with the gray frame to decide which inner wall it touches.
    - Recolor the whole corresponding inside row or inside column, from one inner corner to the other, with the seed color while leaving gray frames and all other pixels unchanged.

    Key insight: The seed color is the paint, but the seed's adjacency to the frame wall determines the stroke orientation and location.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_rectangular_frames(g)
    assert frames, "expected at least one complete gray rectangular frame"

    for frame in frames:
        r, c, color = _seed_for_frame(g, frame)
        side = _seed_side(g, frame, r, c)
        _paint_inner_wall(out, frame, side, color)
    return out


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR


def _seed_position(
    frame: tuple[int, int, int, int],
    side: str,
    rng: random.Random,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    if side in ("top", "bottom"):
        lo = c0 + 2
        hi = c1 - 2
        center = (c0 + c1) // 2
        c = max(lo, min(hi, center + rng.randint(-2, 2)))
        r = r0 + 1 if side == "top" else r1 - 1
        return r, c

    lo = r0 + 2
    hi = r1 - 2
    center = (r0 + r1) // 2
    r = max(lo, min(hi, center + rng.randint(-2, 2)))
    c = c0 + 1 if side == "left" else c1 - 1
    return r, c


def _fits(occupied: np.ndarray, frame: tuple[int, int, int, int]) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = frame
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return not bool(occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any())


def _mark_occupied(occupied: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True


def _random_frame(h: int, w: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    frame_h = rng.randint(5, min(10, h - 2))
    frame_w = rng.randint(5, min(11, w - 2))
    if frame_h > h - 2 or frame_w > w - 2:
        return None
    r0 = rng.randint(1, h - frame_h - 1)
    c0 = rng.randint(1, w - frame_w - 1)
    return r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    n_frames = rng.randint(4, 6)

    sides = list(DIRECTIONS)
    while len(sides) < n_frames:
        sides.append(rng.choice(DIRECTIONS))
    rng.shuffle(sides)
    colors = rng.sample(list(SEED_COLORS), n_frames)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[tuple[tuple[int, int, int, int], str, int]] = []

    for side, color in zip(sides, colors):
        frame = None
        for _ in range(150):
            cand = _random_frame(h, w, rng)
            if cand is not None and _fits(occupied, cand):
                frame = cand
                break
        if frame is None:
            return None

        _draw_frame(grid, frame)
        _draw_frame(expected, frame)
        sr, sc = _seed_position(frame, side, rng)
        grid[sr, sc] = color
        expected[sr, sc] = color
        _paint_inner_wall(expected, frame, side, color)
        _mark_occupied(occupied, frame)
        placed.append((frame, side, color))

    if len({side for _, side, _ in placed}) != len(DIRECTIONS):
        return None
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
