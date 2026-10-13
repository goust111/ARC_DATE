import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_DELTA)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1))
    if r1 > r0:
        coords.extend((r1, c) for c in range(c1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gap must sit on a non-corner border cell")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if len(comp) < 7:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            perimeter = _perimeter_coords(r0, c0, r1, c1)
            comp_set = set(comp)
            border_cells = []
            gaps = []
            bad_border = False
            for cell in perimeter:
                value = int(g[cell])
                if value == color:
                    border_cells.append(cell)
                elif value == bg:
                    gaps.append(cell)
                else:
                    bad_border = True
                    break
            if bad_border or len(gaps) != 1:
                continue
            if comp_set != set(border_cells):
                continue

            gap = gaps[0]
            try:
                side = _gap_side(gap, (r0, c0, r1, c1))
            except AssertionError:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior == color):
                continue
            seed_mask = (interior != bg) & (interior != color)
            seed_count = int(seed_mask.sum())
            if seed_count < 1:
                continue
            seed_colors = np.unique(interior[seed_mask])
            if seed_colors.size != 1:
                continue

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "side": side,
                    "delta": SIDE_TO_DELTA[side],
                    "frame_color": color,
                    "seed_color": int(seed_colors[0]),
                    "seed_count": seed_count,
                }
            )
    frames.sort(key=lambda frame: (frame["bbox"], frame["gap"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored outward ray from each broken rectangular frame, using the interior seed count as the ray length.

    Input structure:
    - The background is the most common color.
    - Several one-pixel-thick rectangular frames share one border color.
    - Each frame has exactly one single-cell gap on a non-corner border position.
    - Inside each frame are one or more single-pixel seeds that all share one non-background, non-frame color.

    Rule:
    - Detect every rectangular frame whose perimeter is complete except for one background-colored gap.
    - Count the interior seed pixels for that frame and read their shared color.
    - Starting at the gap cell, draw a one-pixel-wide straight ray outward from the frame for exactly that many cells.
    - Leave the original frame pixels, seed pixels, and all unrelated background cells unchanged.

    Key insight:
    - The seeds stay in place: their count becomes the ray length, their color becomes the ray color, and the gap side determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for frame in _extract_frames(g):
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["delta"]
        color = int(frame["seed_color"])
        length = int(frame["seed_count"])
        for step in range(length):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            assert 0 <= rr < h and 0 <= cc < w, "ray must stay inside the grid"
            out[rr, cc] = color

    return out


# --------------- generator ---------------

PALETTE = list(range(10))
GRID_H_BASE = 10
GRID_W_BASE = 11
GRID_MAX = 30
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
OBJECT_COUNT_WEIGHTS = [1, 3, 3, 1]
INTERIOR_SIDE_CHOICES = [1, 2, 3, 4, 5]
INTERIOR_SIDE_WEIGHTS = [1, 3, 3, 2, 1]
MAX_SEEDS_PER_FRAME = 6
PLACE_RETRIES = 160
GEN_RETRIES = 300


def _sample_side_lengths(rng: random.Random) -> tuple[int, int]:
    h = rng.choices(INTERIOR_SIDE_CHOICES, weights=INTERIOR_SIDE_WEIGHTS, k=1)[0]
    w = rng.choices(INTERIOR_SIDE_CHOICES, weights=INTERIOR_SIDE_WEIGHTS, k=1)[0]
    return h, w


def _build_direction_list(rng: random.Random, count: int) -> list[str]:
    sides = list(SIDES)
    rng.shuffle(sides)
    return sides[:count]


def _rect_guard_cells(
    h: int, w: int, r0: int, c0: int, r1: int, c1: int, pad: int = 1
) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _halo(cells: list[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    guarded = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                guarded.add((rr, cc))
    return guarded


def _place_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    border_color: int,
    seed_color: int,
    side: str,
) -> dict | None:
    h, w = grid.shape
    dr, dc = SIDE_TO_DELTA[side]

    for _ in range(PLACE_RETRIES):
        inner_h, inner_w = _sample_side_lengths(rng)
        frame_h, frame_w = inner_h + 2, inner_w + 2
        if frame_h > h or frame_w > w:
            continue

        seed_cap = min(MAX_SEEDS_PER_FRAME, inner_h * inner_w)
        seed_count = rng.randint(1, seed_cap)

        r0 = rng.randint(0, h - frame_h)
        c0 = rng.randint(0, w - frame_w)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        if side in {"up", "down"}:
            gap = (r0 if side == "up" else r1, rng.randint(c0 + 1, c1 - 1))
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)

        ray_cells = []
        rr, cc = gap
        blocked = False
        for _step in range(seed_count):
            if not (0 <= rr < h and 0 <= cc < w) or reserved[rr, cc]:
                blocked = True
                break
            ray_cells.append((rr, cc))
            rr += dr
            cc += dc
        if blocked:
            continue

        box_guard = _rect_guard_cells(h, w, r0, c0, r1, c1, pad=1)
        ray_guard = _halo(ray_cells, h, w, pad=1)
        guard_cells = box_guard | ray_guard
        if any(reserved[r, c] for r, c in guard_cells):
            continue

        frame_cells = set(_perimeter_coords(r0, c0, r1, c1))
        frame_cells.remove(gap)
        inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if len(inner_cells) < seed_count:
            continue
        seed_cells = rng.sample(inner_cells, seed_count)

        for r, c in frame_cells:
            grid[r, c] = border_color
        for r, c in seed_cells:
            grid[r, c] = seed_color
        for r, c in guard_cells:
            reserved[r, c] = True

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "side": side,
            "seed_color": seed_color,
            "seed_count": seed_count,
            "ray_cells": ray_cells,
        }
    return None


def _frame_signature(frame: dict) -> tuple:
    return (
        tuple(frame["bbox"]),
        tuple(frame["gap"]),
        str(frame["side"]),
        int(frame["seed_color"]),
        int(frame["seed_count"]),
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    min_h = min(GRID_MAX, GRID_H_BASE + 3 * object_count)
    min_w = min(GRID_MAX, GRID_W_BASE + 4 * object_count)
    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)

    bg = rng.choice(PALETTE)
    border_color = rng.choice([c for c in PALETTE if c != bg])
    seed_palette = [c for c in PALETTE if c not in {bg, border_color}]
    seed_colors = rng.sample(seed_palette, object_count)
    directions = _build_direction_list(rng, object_count)

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placed = []

    for seed_color, side in zip(seed_colors, directions):
        meta = _place_frame(grid, reserved, rng, border_color, seed_color, side)
        if meta is None:
            return None
        placed.append(meta)

    if get_bg(grid) != bg:
        return None

    extracted = _extract_frames(grid)
    if len(extracted) != len(placed):
        return None
    if sorted(_frame_signature(frame) for frame in extracted) != sorted(
        _frame_signature(frame) for frame in placed
    ):
        return None

    expected = grid.copy()
    for frame in placed:
        for r, c in frame["ray_cells"]:
            expected[r, c] = int(frame["seed_color"])
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
