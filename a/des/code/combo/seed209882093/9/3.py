import random
from collections import Counter

import numpy as np


BORDER_COLOR = 5
PALETTE = list(range(10))
BG_CHOICES = [color for color in PALETTE if color != BORDER_COLOR]
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
OUTER_H_RANGE = (4, 8)
OUTER_W_RANGE = (4, 8)
MAX_FRAMES = 4
MAX_SEEDS = 5
PLACE_TRIES = 160
GEN_RETRIES = 400

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(component: list[tuple[int, int]]):
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 3 or outer_w < 3:
        return None

    comp_set = set(component)
    perimeter = _perimeter_coords(r0, c0, r1, c1)
    if not comp_set.issubset(perimeter):
        return None
    if len(comp_set) != len(perimeter) - 1:
        return None

    missing = perimeter - comp_set
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    gr, gc = gap
    is_corner = (gr in (r0, r1)) and (gc in (c0, c1))
    if is_corner:
        return None

    if gr == r0:
        direction = DIRS["up"]
    elif gr == r1:
        direction = DIRS["down"]
    elif gc == c0:
        direction = DIRS["left"]
    else:
        direction = DIRS["right"]

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each hollow rectangular frame has one border gap and 1..5 same-colored seeds inside.
    The gap side gives the outward direction, the number of seeds gives the ray length,
    and the seed color gives the ray color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frame_candidates: dict[int, list[dict]] = {}
    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        infos = []
        for comp in _components(g == color):
            info = _frame_info(comp)
            if info is not None:
                infos.append(info)
        if infos:
            frame_candidates[color] = infos

    if not frame_candidates:
        return out

    border_color, frames = max(
        frame_candidates.items(),
        key=lambda item: (len(item[1]), sum((f["bbox"][2] - f["bbox"][0] + 1) * (f["bbox"][3] - f["bbox"][1] + 1) for f in item[1])),
    )

    h, w = g.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != bg) & (interior != border_color)
        seed_count = int(seed_mask.sum())
        if seed_count == 0:
            continue

        seed_colors = {int(v) for v in interior[seed_mask]}
        if len(seed_colors) != 1:
            continue
        seed_color = next(iter(seed_colors))

        gr, gc = frame["gap"]
        dr, dc = frame["direction"]
        rr, cc = gr + dr, gc + dc
        for _ in range(seed_count):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = seed_color
            rr += dr
            cc += dc

    return out


def _middle_offsets(length: int) -> list[int]:
    offsets = list(range(1, length - 1))
    center = (length - 1) / 2.0
    offsets.sort(key=lambda x: (abs(x - center), x))
    keep = max(1, (len(offsets) + 1) // 2)
    return offsets[:keep]


def _ray_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    gr, gc = gap
    dr, dc = direction
    return [(gr + dr * step, gc + dc * step) for step in range(1, length + 1)]


def _keepout_cells(
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    ray_cells: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    keepout = set()
    for rr in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for cc in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            keepout.add((rr, cc))
    for rr, cc in ray_cells:
        for nr in range(max(0, rr - 1), min(h - 1, rr + 1) + 1):
            for nc in range(max(0, cc - 1), min(w - 1, cc + 1) + 1):
                keepout.add((nr, nc))
    return keepout


def _sample_frame(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    bg: int,
):
    h, w = grid.shape
    outer_h = rng.randint(*OUTER_H_RANGE)
    outer_w = rng.randint(*OUTER_W_RANGE)
    inner_area = (outer_h - 2) * (outer_w - 2)
    max_seed_count = min(MAX_SEEDS, inner_area)
    if max_seed_count <= 0:
        return None
    seed_count = rng.randint(1, max_seed_count)
    direction_name = rng.choice(list(DIRS))
    direction = DIRS[direction_name]

    r0_min = 1
    r0_max = h - outer_h - 1
    c0_min = 1
    c0_max = w - outer_w - 1
    if direction_name == "up":
        r0_min = max(r0_min, seed_count)
    elif direction_name == "down":
        r0_max = min(r0_max, h - outer_h - seed_count)
    elif direction_name == "left":
        c0_min = max(c0_min, seed_count)
    else:
        c0_max = min(c0_max, w - outer_w - seed_count)

    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = rng.randint(r0_min, r0_max)
    c0 = rng.randint(c0_min, c0_max)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if direction_name in ("up", "down"):
        gap_offset = rng.choice(_middle_offsets(outer_w))
        gap = (r0, c0 + gap_offset) if direction_name == "up" else (r1, c0 + gap_offset)
    else:
        gap_offset = rng.choice(_middle_offsets(outer_h))
        gap = (r0 + gap_offset, c0) if direction_name == "left" else (r0 + gap_offset, c1)

    ray_cells = _ray_cells(gap, direction, seed_count)
    if not all(0 <= rr < h and 0 <= cc < w for rr, cc in ray_cells):
        return None

    keepout = _keepout_cells(h, w, r0, c0, r1, c1, ray_cells)
    if any(reserved[rr, cc] for rr, cc in keepout):
        return None

    perimeter = _perimeter_coords(r0, c0, r1, c1)
    border_cells = perimeter - {gap}
    color_choices = [color for color in PALETTE if color not in {bg, BORDER_COLOR}]
    seed_color = rng.choice(color_choices)

    interior_cells = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)]
    seed_cells = rng.sample(interior_cells, seed_count)

    for rr, cc in border_cells:
        grid[rr, cc] = BORDER_COLOR
    for rr, cc in seed_cells:
        grid[rr, cc] = seed_color
    for rr, cc in keepout:
        reserved[rr, cc] = True

    return {
        "ray_cells": ray_cells,
        "seed_color": seed_color,
        "seed_count": seed_count,
    }


def _expected_output(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        for rr, cc in frame["ray_cells"]:
            out[rr, cc] = frame["seed_color"]
    return out


def _gen_attempt(rng: random.Random):
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_frames = min(MAX_FRAMES, max(1, (h * w) // 110))
    target_frames = rng.randint(1, max_frames)
    frames: list[dict] = []

    for _ in range(target_frames * PLACE_TRIES):
        if len(frames) == target_frames:
            break
        frame = _sample_frame(rng, grid, reserved, bg)
        if frame is not None:
            frames.append(frame)

    if len(frames) != target_frames:
        return None
    if int((grid == bg).sum()) <= (h * w) // 2:
        return None

    expected = _expected_output(grid, frames)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    changed = actual != grid
    if int(changed.sum()) != sum(frame["seed_count"] for frame in frames):
        return None
    if np.any(grid[changed] != bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
