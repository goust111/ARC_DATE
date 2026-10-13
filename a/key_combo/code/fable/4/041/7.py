from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRY_LIMIT = 400
GRID_SIDE_RANGE = (15, 30)
SEED_COUNT_CHOICES = (2, 3, 4)
SEED_COUNT_WEIGHTS = (4, 4, 2)
SEED_SIZE_VALUES = (1, 2, 3, 4)
MIXED_SIZE_PROB = 0.75
SEPARATION_PAD_RANGE = (2, 3)
PLACEMENT_ATTEMPTS = 250


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = int(r)
            c0 = c1 = int(c)
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, int(rr))
                r1 = max(r1, int(rr))
                c0 = min(c0, int(cc))
                c1 = max(c1, int(cc))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            components.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _parse_seed(grid: np.ndarray, bg: int, component: dict[str, object]) -> dict[str, int] | None:
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[index]
    height = int(r1) - int(r0) + 1
    width = int(c1) - int(c0) + 1
    if height != width or height < 3:
        return None
    patch = np.asarray(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)
    if bool(np.any(patch == int(bg))):
        return None

    frame_mask = np.ones_like(patch, dtype=bool)
    frame_mask[1:-1, 1:-1] = False
    frame_vals = np.unique(patch[frame_mask])
    core_vals = np.unique(patch[~frame_mask])
    if len(frame_vals) != 1 or len(core_vals) != 1:
        return None
    frame_color = int(frame_vals[0])
    core_color = int(core_vals[0])
    if frame_color == core_color:
        return None
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "t": int(height - 2),
        "frame_color": int(frame_color),
        "core_color": int(core_color),
    }


def _chebyshev_to_bbox(r: int, c: int, seed: dict[str, int]) -> int:
    if r < seed["r0"]:
        dr = seed["r0"] - int(r)
    elif r > seed["r1"]:
        dr = int(r) - seed["r1"]
    else:
        dr = 0

    if c < seed["c0"]:
        dc = seed["c0"] - int(c)
    elif c > seed["c1"]:
        dc = int(c) - seed["c1"]
    else:
        dc = 0
    return max(int(dr), int(dc))


def _arrival_step(r: int, c: int, seed: dict[str, int]) -> int:
    dist = _chebyshev_to_bbox(int(r), int(c), seed)
    if dist == 0:
        return 0
    return int((dist + seed["t"] - 1) // seed["t"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each framed square seed into alternating-color rings, with faster growth for larger cores.

    Input structure:
    - The background contains several separated seed objects.
    - Each seed is a filled t x t core surrounded by a one-cell-thick square frame.
    - Every seed has its own core color, frame color, and core size t.

    Rule:
    - Keep every original seed pixel unchanged.
    - For any background cell, compute when each seed first reaches it: the k-th step reaches cells whose Chebyshev distance from that seed's outer bbox is at most k * t.
    - A cell is colored by the unique earliest-arriving seed; ties stay background.
    - A seed paints its step-k ring with core color for odd k and frame color for even k.

    Key insight:
    - Territory boundaries are determined by arrival time, not plain distance, so larger cores expand faster and push the seam toward smaller seeds.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    components = _find_components(g != int(bg))
    seeds = []
    for component in components:
        seed = _parse_seed(g, int(bg), component)
        if seed is not None:
            seeds.append(seed)
    assert seeds

    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != int(bg):
                continue
            steps = [_arrival_step(int(r), int(c), seed) for seed in seeds]
            best = min(steps)
            if steps.count(best) != 1:
                continue
            winner = seeds[steps.index(best)]
            out[r, c] = winner["core_color"] if best % 2 == 1 else winner["frame_color"]
    return out


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        int(a[2]) + int(pad) < int(b[0])
        or int(b[2]) + int(pad) < int(a[0])
        or int(a[3]) + int(pad) < int(b[1])
        or int(b[3]) + int(pad) < int(a[1])
    )


def _sample_t_values(count: int) -> list[int]:
    if count == 1:
        return [random.choice(SEED_SIZE_VALUES)]
    if random.random() < MIXED_SIZE_PROB:
        while True:
            values = [random.choice(SEED_SIZE_VALUES) for _ in range(count)]
            if len(set(values)) >= 2:
                return values
    value = random.choice(SEED_SIZE_VALUES)
    return [int(value)] * int(count)


def _place_seed_boxes(h: int, w: int, t_values: list[int], pad: int) -> list[tuple[int, int, int, int]] | None:
    placements: list[tuple[int, int, int, int] | None] = [None] * len(t_values)
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(t_values)), key=lambda idx: t_values[idx], reverse=True)

    for idx in order:
        side = int(t_values[idx] + 2)
        if side > h or side > w:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - side)
            c0 = random.randint(0, w - side)
            bbox = (int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1))
            if all(_bboxes_separated(bbox, other, int(pad)) for other in placed):
                placements[idx] = bbox
                placed.append(bbox)
                break
        else:
            return None

    return [bbox for bbox in placements if bbox is not None]


def _competition_stats(shape: tuple[int, int], seeds: list[dict[str, int]]) -> tuple[int, int, int, bool]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    for seed in seeds:
        occupied[seed["r0"] : seed["r1"] + 1, seed["c0"] : seed["c1"] + 1] = True

    claimed = 0
    seam = 0
    max_step = 0
    has_even_ring = False
    for r in range(h):
        for c in range(w):
            if bool(occupied[r, c]):
                continue
            steps = [_arrival_step(int(r), int(c), seed) for seed in seeds]
            best = min(steps)
            if steps.count(best) == 1:
                claimed += 1
                max_step = max(max_step, int(best))
                if best % 2 == 0:
                    has_even_ring = True
            else:
                seam += 1
    return int(claimed), int(seam), int(max_step), bool(has_even_ring)


def _build_grid(h: int, w: int, seeds: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for seed in seeds:
        grid[seed["r0"] : seed["r1"] + 1, seed["c0"] : seed["c1"] + 1] = seed["frame_color"]
        grid[seed["r0"] + 1 : seed["r1"], seed["c0"] + 1 : seed["c1"]] = seed["core_color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    count = random.choices(SEED_COUNT_CHOICES, weights=SEED_COUNT_WEIGHTS, k=1)[0]
    t_values = _sample_t_values(int(count))
    pad = random.randint(*SEPARATION_PAD_RANGE)
    boxes = _place_seed_boxes(int(h), int(w), t_values, int(pad))
    if boxes is None:
        return None

    colors = random.sample(PALETTE, 2 * int(count))
    seeds: list[dict[str, int]] = []
    for idx, bbox in enumerate(boxes):
        core_color = int(colors[2 * idx])
        frame_color = int(colors[2 * idx + 1])
        r0, c0, r1, c1 = bbox
        seeds.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "t": int(t_values[idx]),
                "core_color": int(core_color),
                "frame_color": int(frame_color),
            }
        )

    claimed, seam, max_step, has_even_ring = _competition_stats((int(h), int(w)), seeds)
    if claimed < (h * w) // 4 or seam == 0 or max_step < 2 or not has_even_ring:
        return None

    grid = _build_grid(int(h), int(w), seeds)
    out = main(grid)
    if not bool(np.any(out != grid)):
        return None
    if bool(np.any((grid != BG) & (out != grid))):
        return None
    if int(np.count_nonzero((grid == BG) & (out == BG))) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
