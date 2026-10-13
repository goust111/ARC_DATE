import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
QUADRANTS = ("tl", "tr", "bl", "br")

GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (12, 28)
INNER_H_RANGE = (2, 6)
INNER_W_RANGE = (2, 6)
MAX_OBJECTS = 5
PLACEMENT_PAD = 1
GEN_RETRIES = 300


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


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows), max(cols)


def _frame_pixels(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _half(index: int, length: int) -> int:
    delta = 2 * index - (length - 1)
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _corner_triplet(frame: dict[str, int | str]) -> list[tuple[int, int]]:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    quadrant = str(frame["quadrant"])
    if quadrant == "tl":
        return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0)]
    if quadrant == "tr":
        return [(r0, c1), (r0, c1 - 1), (r0 + 1, c1)]
    if quadrant == "bl":
        return [(r1, c0), (r1, c0 + 1), (r1 - 1, c0)]
    return [(r1, c1), (r1, c1 - 1), (r1 - 1, c1)]


def _parse_frame(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    expected = _frame_pixels(r0, c0, r1, c1)
    if set(points) != expected:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seeds = np.argwhere(interior != BG)
    if len(seeds) != 1:
        return None

    sr, sc = map(int, seeds[0])
    seed_color = int(interior[sr, sc])
    assert seed_color != color, "seed color must differ from its frame color"

    vertical = _half(sr, interior.shape[0])
    horizontal = _half(sc, interior.shape[1])
    assert vertical != 0 and horizontal != 0, "seed must avoid the interior center axes"

    quadrant = {
        (-1, -1): "tl",
        (-1, 1): "tr",
        (1, -1): "bl",
        (1, 1): "br",
    }[(vertical, horizontal)]
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "seed_color": seed_color,
        "quadrant": quadrant,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, int | str]]:
    frames: list[dict[str, int | str]] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for comp in _components(grid == color):
            frame = _parse_frame(grid, comp, color)
            if frame is not None:
                frames.append(frame)
    assert frames, "expected at least one rectangular frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a frame corner according to the unique seed's interior quadrant.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        seed_color = int(frame["seed_color"])
        for r, c in _corner_triplet(frame):
            out[r, c] = seed_color
    return out


def _candidate_indices(length: int, want_upper: bool) -> list[int]:
    if want_upper:
        return [i for i in range(length) if _half(i, length) < 0]
    return [i for i in range(length) if _half(i, length) > 0]


def _sample_frame_colors(rng: random.Random, count: int) -> list[int]:
    if count <= len(PALETTE) and rng.random() < 0.75:
        return rng.sample(PALETTE, count)
    return [rng.choice(PALETTE) for _ in range(count)]


def _specs_for_attempt(rng: random.Random, count: int) -> list[dict[str, int | str]]:
    frame_colors = _sample_frame_colors(rng, count)
    specs: list[dict[str, int | str]] = []
    for frame_color in frame_colors:
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        quadrant = rng.choice(QUADRANTS)
        seed_color = rng.choice([c for c in PALETTE if c != frame_color])
        specs.append(
            {
                "frame_color": frame_color,
                "seed_color": seed_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "quadrant": quadrant,
            }
        )
    specs.sort(key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]), reverse=True)
    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    rr1 = min(occupied.shape[0], r0 + h + PLACEMENT_PAD)
    cc1 = min(occupied.shape[1], c0 + w + PLACEMENT_PAD)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    rr1 = min(occupied.shape[0], r0 + h + PLACEMENT_PAD)
    cc1 = min(occupied.shape[1], c0 + w + PLACEMENT_PAD)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_spec(
    grid: np.ndarray, spec: dict[str, int | str], r0: int, c0: int, rng: random.Random
) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    frame_color = int(spec["frame_color"])
    seed_color = int(spec["seed_color"])
    quadrant = str(spec["quadrant"])

    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    for r, c in _frame_pixels(r0, c0, r1, c1):
        grid[r, c] = frame_color

    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    row_choices = _candidate_indices(inner_h, want_upper=quadrant[0] == "t")
    col_choices = _candidate_indices(inner_w, want_upper=quadrant[1] == "l")
    sr = r0 + 1 + rng.choice(row_choices)
    sc = c0 + 1 + rng.choice(col_choices)
    grid[sr, sc] = seed_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = h * w
    count_cap = min(MAX_OBJECTS, max(1, area // 140 + 1))
    count = rng.randint(1, count_cap)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs = _specs_for_attempt(rng, count)

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if outer_h > h - 2 or outer_w > w - 2:
            return None

        positions = [(r, c) for r in range(1, h - outer_h) for c in range(1, w - outer_w)]
        rng.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue
            _place_spec(grid, spec, r0, c0, rng)
            _mark_occupied(occupied, r0, c0, outer_h, outer_w)
            placed = True
            break
        if not placed:
            return None

    frames = _find_frames(grid)
    if len(frames) != count:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
