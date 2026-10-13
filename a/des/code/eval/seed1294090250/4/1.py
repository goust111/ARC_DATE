import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames: list[dict] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue

            local = np.zeros((height, width), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True

            border = _border_mask(height, width)
            if np.any(local & ~border):
                continue

            missing = np.argwhere(border & ~local)
            if missing.shape[0] != 1:
                continue
            mr, mc = map(int, missing[0])
            if (mr in (0, height - 1)) and (mc in (0, width - 1)):
                continue

            if mr == 0:
                side = "up"
            elif mr == height - 1:
                side = "down"
            elif mc == 0:
                side = "left"
            elif mc == width - 1:
                side = "right"
            else:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            if np.any(interior == color):
                continue

            seed_mask = (interior != bg) & (interior != color)
            if not np.any(seed_mask):
                continue
            seed_values = np.unique(interior[seed_mask])
            if seed_values.size != 1:
                continue

            seed_color = int(seed_values[0])
            gap = (int(r0 + mr), int(c0 + mc))
            frames.append(
                {
                    "frame_color": color,
                    "seed_color": seed_color,
                    "seed_count": int(seed_mask.sum()),
                    "gap": gap,
                    "direction": side,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored tentacle outward through each frame's unique 1-pixel gap.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _detect_frames(g):
        dr, dc = DIRECTION_VECTORS[frame["direction"]]
        start_r, start_c = frame["gap"]
        color = int(frame["seed_color"])
        length = int(frame["seed_count"])
        for step in range(length):
            rr = start_r + dr * step
            cc = start_c + dc * step
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color

    return out


GEN_ATTEMPTS = 500
GRID_HEIGHT_RANGE = (14, 25)
GRID_WIDTH_RANGE = (14, 25)
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
INNER_HEIGHT_RANGE = range(2, 7)
INNER_WIDTH_RANGE = range(2, 7)
MAX_SEEDS = 8
PLACEMENT_TRIES = 200


def _expand(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((int(nr), int(nc)))
    return expanded


def _footprint_for_spec(spec: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    dr, dc = DIRECTION_VECTORS[spec["direction"]]
    gap_r, gap_c = spec["gap"]
    for step in range(spec["seed_count"]):
        cells.add((int(gap_r + dr * step), int(gap_c + dc * step)))
    return cells


def _draw_input_from_specs(height: int, width: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        gap = spec["gap"]
        frame_color = int(spec["frame_color"])
        seed_color = int(spec["seed_color"])

        border_cells = set()
        for c in range(c0, c1 + 1):
            border_cells.add((r0, c))
            border_cells.add((r1, c))
        for r in range(r0, r1 + 1):
            border_cells.add((r, c0))
            border_cells.add((r, c1))
        border_cells.remove(gap)
        for r, c in border_cells:
            grid[r, c] = frame_color

        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in spec["seed_cells"]:
            assert (r, c) in interior_cells
            grid[r, c] = seed_color

    return grid


def _draw_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        dr, dc = DIRECTION_VECTORS[spec["direction"]]
        gap_r, gap_c = spec["gap"]
        for step in range(spec["seed_count"]):
            out[gap_r + dr * step, gap_c + dc * step] = int(spec["seed_color"])
    return out


def _sample_spec(
    rng: random.Random,
    height: int,
    width: int,
    used: set[tuple[int, int]],
    direction: str,
    frame_color: int,
    seed_color: int,
) -> dict | None:
    seed_count = rng.randint(1, MAX_SEEDS)
    shapes = [
        (ih, iw)
        for ih in INNER_HEIGHT_RANGE
        for iw in INNER_WIDTH_RANGE
        if ih * iw >= seed_count
    ]
    rng.shuffle(shapes)

    for inner_h, inner_w in shapes:
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        min_r0 = 1
        max_r0 = height - outer_h - 1
        min_c0 = 1
        max_c0 = width - outer_w - 1

        if direction == "up":
            min_r0 = max(min_r0, seed_count - 1)
        elif direction == "down":
            max_r0 = min(max_r0, height - outer_h - seed_count + 1)
        elif direction == "left":
            min_c0 = max(min_c0, seed_count - 1)
        elif direction == "right":
            max_c0 = min(max_c0, width - outer_w - seed_count + 1)

        if min_r0 > max_r0 or min_c0 > max_c0:
            continue

        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(min_r0, max_r0)
            c0 = rng.randint(min_c0, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            if direction in ("up", "down"):
                gap = (r0 if direction == "up" else r1, rng.randint(c0 + 1, c1 - 1))
            else:
                gap = (rng.randint(r0 + 1, r1 - 1), c0 if direction == "left" else c1)

            footprint = _footprint_for_spec(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "direction": direction,
                    "seed_count": seed_count,
                }
            )
            padded = _expand(footprint, height, width)
            if padded & used:
                continue

            interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            seed_cells = rng.sample(interior_cells, seed_count)
            return {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "seed_count": seed_count,
                "frame_color": int(frame_color),
                "seed_color": int(seed_color),
                "seed_cells": [(int(r), int(c)) for r, c in seed_cells],
                "reserved": padded,
            }
    return None


def _direction_histogram(specs: list[dict]) -> Counter:
    return Counter(spec["direction"] for spec in specs)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    bg = rng.randrange(10)
    object_count = rng.choice(OBJECT_COUNT_CHOICES)

    colors = [color for color in range(10) if color != bg]
    if len(colors) < 2 * object_count:
        return None
    picked = rng.sample(colors, 2 * object_count)
    seed_colors = picked[:object_count]
    frame_colors = picked[object_count:]

    specs: list[dict] = []
    used: set[tuple[int, int]] = set()
    directions = list(DIRECTION_VECTORS)
    rng.shuffle(directions)

    for i, (frame_color, seed_color) in enumerate(zip(frame_colors, seed_colors)):
        spec = _sample_spec(
            rng,
            height,
            width,
            used,
            directions[i % len(directions)],
            frame_color,
            seed_color,
        )
        if spec is None:
            return None
        specs.append(spec)
        used |= spec["reserved"]

    histogram = _direction_histogram(specs)
    if object_count >= 3 and len(histogram) < 2:
        return None

    grid = _draw_input_from_specs(height, width, bg, specs)
    expected = _draw_expected(grid, specs)

    if _background_color(grid) != bg:
        return None

    detected = _detect_frames(grid)
    if len(detected) != object_count:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
