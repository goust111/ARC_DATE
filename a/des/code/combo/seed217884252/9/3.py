from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 100


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))

                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    q.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    q.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    q.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    q.append((rr, cc + 1))

            comps.append(comp)

    return comps


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    pts = np.asarray(points, dtype=int)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())

    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    height = r1 - r0 + 1
    width = c1 - c0 + 1
    expected = 2 * (height + width) - 4
    if len(points) != expected:
        return None

    pset = set(points)
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return None

    return r0, r1, c0, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    frames: list[tuple[int, int, int, int, int, int, int, int]] = []

    for color in (int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components4(g == color)
        for comp in comps:
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue

            r0, r1, c0, c1 = bbox
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if np.count_nonzero(interior != BG) != 1:
                continue

            seeds = np.argwhere((interior != BG) & (interior != color))
            if seeds.shape[0] != 1:
                continue

            sr = r0 + 1 + int(seeds[0, 0])
            sc = c0 + 1 + int(seeds[0, 1])
            seed_color = int(g[sr, sc])
            frames.append((r0, r1, c0, c1, color, sr, sc, seed_color))

    frames.sort(key=lambda item: (item[0], item[2], item[1], item[3]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed inside a rectangular frame into clipped concentric square rings.

    The frame border stays unchanged.  Inside the frame, pixels at even Chebyshev
    distance from the seed take the seed color; odd-distance pixels become the
    frame color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, r1, c0, c1, frame_color, sr, sc, seed_color in _find_frames(g):
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                dist = max(abs(r - sr), abs(c - sc))
                out[r, c] = seed_color if dist % 2 == 0 else frame_color

    return out


def _frame_modes(rng: random.Random, n_frames: int) -> list[str]:
    if n_frames == 1:
        return [rng.choice(["center", "off"])]

    modes = ["center", "off"]
    while len(modes) < n_frames:
        modes.append(rng.choice(["center", "off"]))
    rng.shuffle(modes)
    return modes


def _sample_frame_shape(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    mode: str,
    featured: bool,
) -> tuple[int, int] | None:
    min_h = 7 if featured else 4
    min_w = 7 if featured else 4
    max_h = min(11, grid_h - 2)
    max_w = min(11, grid_w - 2)

    if max_h < min_h or max_w < min_w:
        return None

    if mode == "center":
        hs = [v for v in range(max(min_h, 5), max_h + 1) if v % 2 == 1]
        ws = [v for v in range(max(min_w, 5), max_w + 1) if v % 2 == 1]
        if not hs or not ws:
            return None
        return rng.choice(hs), rng.choice(ws)

    return rng.randint(min_h, max_h), rng.randint(min_w, max_w)


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _pick_seed_position(
    rng: random.Random,
    top: int,
    left: int,
    height: int,
    width: int,
    mode: str,
) -> tuple[int, int] | None:
    inner_rows = list(range(top + 1, top + height - 1))
    inner_cols = list(range(left + 1, left + width - 1))
    candidates = [(r, c) for r in inner_rows for c in inner_cols]
    if not candidates:
        return None

    if mode == "center":
        return top + height // 2, left + width // 2

    if len(candidates) == 1:
        return candidates[0]

    if (height - 2) % 2 == 1 and (width - 2) % 2 == 1:
        center = (top + height // 2, left + width // 2)
        candidates = [p for p in candidates if p != center]

    if not candidates:
        return None

    return rng.choice(candidates)


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    seed_color: int,
    mode: str,
    rng: random.Random,
) -> bool:
    bottom = top + height - 1
    right = left + width - 1

    grid[top, left : right + 1] = FRAME
    grid[bottom, left : right + 1] = FRAME
    grid[top : bottom + 1, left] = FRAME
    grid[top : bottom + 1, right] = FRAME

    seed = _pick_seed_position(rng, top, left, height, width, mode)
    if seed is None:
        return False

    sr, sc = seed
    grid[sr, sc] = seed_color
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(12, 28)
    grid_w = rng.randint(12, 28)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    area = grid_h * grid_w
    max_frames = min(4, max(1, area // 110))
    n_frames = rng.randint(1, max_frames)
    seed_colors = rng.sample(SEED_COLORS, n_frames)
    modes = _frame_modes(rng, n_frames)
    featured_idx = rng.randrange(n_frames)

    occupied = np.zeros_like(grid, dtype=bool)

    for idx, (seed_color, mode) in enumerate(zip(seed_colors, modes)):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            shape = _sample_frame_shape(rng, grid_h, grid_w, mode, idx == featured_idx)
            if shape is None:
                return None

            height, width = shape
            max_top = grid_h - height - 1
            max_left = grid_w - width - 1
            if max_top < 1 or max_left < 1:
                return None

            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            if not _can_place(occupied, top, left, height, width):
                continue

            if not _paint_frame(grid, top, left, height, width, seed_color, mode, rng):
                continue

            _mark_occupied(occupied, top, left, height, width)
            placed = True
            break

        if not placed:
            return None

    if np.count_nonzero(grid != BG) >= int(grid.size * 0.45):
        return None

    detected = _find_frames(grid)
    if len(detected) != n_frames:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid grid")
