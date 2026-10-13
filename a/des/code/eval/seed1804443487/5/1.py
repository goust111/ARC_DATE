from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLORS = tuple(range(1, 10))
OUTER_SIZE_CHOICES = {
    1: (6, 8, 10, 12),
    2: (6, 8, 10),
    3: (6, 8),
    4: (6, 8),
}
SEED_COUNT_CHOICES = (1, 2, 3)
MAX_GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
QUADRANTS = ("tl", "tr", "bl", "br")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            comps.append((color, np.array(coords, dtype=int)))

    return comps


def _frame_from_component(grid: np.ndarray, color: int, coords: np.ndarray) -> dict[str, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    border_cells = 2 * height + 2 * width - 4
    if int(coords.shape[0]) != border_cells:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    if not np.all(patch[border_mask] == color):
        return None

    return {"color": int(color), "r0": r0, "c0": c0, "r1": r1, "c1": c1}


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames = []
    for color, coords in _monochrome_components(grid, bg):
        frame = _frame_from_component(grid, color, coords)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"]))
    return frames


def _quadrant_of_seed(frame: dict[str, int], r: int, c: int) -> str:
    vertical = "t" if 2 * r < frame["r0"] + frame["r1"] else "b"
    horizontal = "l" if 2 * c < frame["c0"] + frame["c1"] else "r"
    return vertical + horizontal


def _segment_limits(frame: dict[str, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    top_end = r0 + (r1 - r0) // 2
    bottom_start = r1 - (r1 - r0) // 2
    left_end = c0 + (c1 - c0) // 2
    right_start = c1 - (c1 - c0) // 2
    return top_end, bottom_start, left_end, right_start


def _paint_quadrant_segment(out: np.ndarray, frame: dict[str, int], quadrant: str, color: int) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    top_end, bottom_start, left_end, right_start = _segment_limits(frame)

    if quadrant == "tl":
        out[r0, c0 : left_end + 1] = color
        out[r0 : top_end + 1, c0] = color
    elif quadrant == "tr":
        out[r0, right_start : c1 + 1] = color
        out[r0 : top_end + 1, c1] = color
    elif quadrant == "bl":
        out[r1, c0 : left_end + 1] = color
        out[bottom_start : r1 + 1, c0] = color
    else:
        out[r1, right_start : c1 + 1] = color
        out[bottom_start : r1 + 1, c1] = color


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere(interior != bg)
        quadrant_colors: dict[str, int] = {}

        for sr, sc in seeds:
            ar = r0 + 1 + int(sr)
            ac = c0 + 1 + int(sc)
            out[ar, ac] = bg
            quadrant_colors[_quadrant_of_seed(frame, ar, ac)] = int(g[ar, ac])

        for quadrant in QUADRANTS:
            if quadrant in quadrant_colors:
                _paint_quadrant_segment(out, frame, quadrant, quadrant_colors[quadrant])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each interior seed color onto the matching quarter-border segment of its enclosing rectangle.

    The seed's quadrant relative to the rectangle center selects which corner arc is recolored:
    tl -> top-left half-edges, tr -> top-right, bl -> bottom-left, br -> bottom-right.
    Seed pixels are erased in the output.
    """
    return _transform(grid)


def _quadrant_ranges(size: int, upper: bool) -> list[int]:
    half = size // 2
    if upper:
        return list(range(1, half))
    return list(range(half, size - 1))


def _choose_seed_positions(height: int, width: int, quadrants: list[str], rng: random.Random) -> list[tuple[str, int, int]]:
    picked: list[tuple[str, int, int]] = []
    for quadrant in quadrants:
        rows = _quadrant_ranges(height, quadrant[0] == "t")
        cols = _quadrant_ranges(width, quadrant[1] == "l")
        if not rows or not cols:
            raise ValueError("quadrant has no available seed cells")
        picked.append((quadrant, rng.choice(rows), rng.choice(cols)))
    return picked


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expanded_slice(start: int, size: int, limit: int, pad: int = 1) -> slice:
    return slice(max(0, start - pad), min(limit, start + size + pad))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 28)
    width = rng.randint(16, 28)
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    rect_count = rng.choices((1, 2, 3, 4), weights=(1, 3, 3, 2), k=1)[0]
    frame_colors = rng.sample(FRAME_COLORS, rect_count)

    for index in range(rect_count):
        size_choices = [s for s in OUTER_SIZE_CHOICES[rect_count] if s <= height - 2 and s <= width - 2]
        if not size_choices:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = rng.choice(size_choices)
            rect_w = rng.choice(size_choices)
            r0 = rng.randint(1, height - rect_h - 1)
            c0 = rng.randint(1, width - rect_w - 1)

            rs = _expanded_slice(r0, rect_h, height)
            cs = _expanded_slice(c0, rect_w, width)
            if occupied[rs, cs].any():
                continue

            frame_color = frame_colors[index]
            _draw_frame(grid, r0, c0, rect_h, rect_w, frame_color)

            seed_count = rng.choice(SEED_COUNT_CHOICES)
            quadrants = rng.sample(list(QUADRANTS), seed_count)
            seed_palette = [color for color in FRAME_COLORS if color != frame_color]
            seed_colors = rng.sample(seed_palette, seed_count)
            for (quadrant, local_r, local_c), seed_color in zip(
                _choose_seed_positions(rect_h, rect_w, quadrants, rng), seed_colors
            ):
                del quadrant
                grid[r0 + local_r, c0 + local_c] = seed_color

            occupied[rs, cs] = True
            placed = True
            break

        if not placed:
            return None

    frames = _find_frames(grid, BG)
    if len(frames) != rect_count:
        return None

    total_quadrants = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        seeds = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] != BG)
        if not (1 <= len(seeds) <= 3):
            return None
        seen = set()
        for sr, sc in seeds:
            ar = r0 + 1 + int(sr)
            ac = c0 + 1 + int(sc)
            quadrant = _quadrant_of_seed(frame, ar, ac)
            if quadrant in seen:
                return None
            seen.add(quadrant)
            total_quadrants.add(quadrant)
        if int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == frame["color"])) != 2 * (r1 - r0 + 1) + 2 * (
            c1 - c0 + 1
        ) - 4:
            return None

    if len(total_quadrants) < 2:
        return None

    out = _transform(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_RETRIES} attempts")
