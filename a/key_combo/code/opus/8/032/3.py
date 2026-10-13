from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


GREY = 5
BLACK = 0

SOURCE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTION_TO_OUTPUT = {
    "up": 1,     # from above, sliding downward
    "right": 2,  # from right, sliding leftward
    "down": 3,   # from below, sliding upward
    "left": 4,   # from left, sliding rightward
}
DIRECTION_TO_STEP = {
    "up": (1, 0),
    "right": (0, -1),
    "down": (-1, 0),
    "left": (0, 1),
}

GRID_MARGIN_MIN = 4
GRID_MARGIN_MAX = 8
MARKER_COUNT_CHOICES = (5, 6)
GEN_RETRIES = 400


def _find_grey_block(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey block"
    r0 = int(grey[:, 0].min())
    c0 = int(grey[:, 1].min())
    r1 = int(grey[:, 0].max())
    c1 = int(grey[:, 1].max())
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "grey object must be 3x3"
    assert len(grey) == 9, "grey block must be filled"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey block must be solid"
    return r0, c0, r1, c1


def _classify_marker(
    r: int, c: int, block_bbox: tuple[int, int, int, int]
) -> tuple[str, int]:
    r0, c0, r1, c1 = block_bbox
    if c0 <= c <= c1 and r < r0:
        return "up", r0 - r
    if r0 <= r <= r1 and c > c1:
        return "right", c - c1
    if c0 <= c <= c1 and r > r1:
        return "down", r - r1
    if r0 <= r <= r1 and c < c0:
        return "left", c0 - c
    raise AssertionError("marker must lie on a cardinal ray of the grey block")


def _ray_key(
    direction: str, r: int, c: int, block_bbox: tuple[int, int, int, int]
) -> tuple[str, int]:
    r0, c0, _r1, _c1 = block_bbox
    if direction in ("up", "down"):
        return direction, c - c0
    return direction, r - r0


def _marker_records(
    grid: np.ndarray, block_bbox: tuple[int, int, int, int]
) -> list[dict[str, int | str | tuple[int, int]]]:
    markers = []
    for rr, cc in np.argwhere((grid != BLACK) & (grid != GREY)):
        r = int(rr)
        c = int(cc)
        direction, distance = _classify_marker(r, c, block_bbox)
        markers.append(
            {
                "r": r,
                "c": c,
                "direction": direction,
                "distance": distance,
                "source_color": int(grid[r, c]),
            }
        )
    assert 4 <= len(markers) <= 6, "expected 4-6 colored pixels"
    distances = [int(m["distance"]) for m in markers]
    assert len(set(distances)) == len(distances), "marker distances must be unique"
    return sorted(markers, key=lambda m: int(m["distance"]))


def _simulate_final_positions(
    markers: list[dict[str, int | str | tuple[int, int]]],
    block_bbox: tuple[int, int, int, int],
    *,
    accumulate: bool,
) -> list[tuple[int, int, str]]:
    r0, c0, r1, c1 = block_bbox
    occupied = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }
    base_mass = set(occupied)
    landed: list[tuple[int, int, str]] = []

    for marker in markers:
        r = int(marker["r"])
        c = int(marker["c"])
        direction = str(marker["direction"])
        dr, dc = DIRECTION_TO_STEP[direction]
        cur_r, cur_c = r, c
        mass = occupied if accumulate else base_mass
        while (cur_r + dr, cur_c + dc) not in mass:
            cur_r += dr
            cur_c += dc
        landed.append((cur_r, cur_c, direction))
        if accumulate:
            occupied.add((cur_r, cur_c))
    return landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull cardinally aligned pixels into a central grey 3x3 block, recoloring each landing cell by its arrival side.

    Input structure:
    - A black background contains exactly one filled grey 3x3 block.
    - Other foreground cells are isolated colored pixels placed on the block's cardinal rays.
    - Each colored pixel lies strictly above, right of, below, or left of the grey block, and no two pixels share the same initial distance to the grey block.

    Rule:
    - Measure every colored pixel's distance to the original grey block edge and process them from nearest to farthest.
    - For each pixel, erase its source cell, slide it straight toward the current accumulated mass until the next step would overlap that mass, then stop on the last free cell.
    - Recolor the stopped cell only by arrival side: from above -> blue(1), from right -> red(2), from below -> green(3), from left -> yellow(4).
    - The original grey 3x3 block keeps its grey color throughout.

    Key insight:
    - The input colors are pure distractors: order comes from distance, final color comes from direction, and earlier arrivals grow the mass that later pixels collide with.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    block_bbox = _find_grey_block(g)
    markers = _marker_records(g, block_bbox)

    out = g.copy()
    for marker in markers:
        out[int(marker["r"]), int(marker["c"])] = BLACK

    r0, c0, r1, c1 = block_bbox
    occupied = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }

    for marker in markers:
        direction = str(marker["direction"])
        dr, dc = DIRECTION_TO_STEP[direction]
        r = int(marker["r"])
        c = int(marker["c"])

        while (r + dr, c + dc) not in occupied:
            r += dr
            c += dc

        out[r, c] = DIRECTION_TO_OUTPUT[direction]
        occupied.add((r, c))

    return out


def _ray_coordinate(
    block_bbox: tuple[int, int, int, int], direction: str, offset: int, distance: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox
    if direction == "up":
        return r0 - distance, c0 + offset
    if direction == "right":
        return r0 + offset, c1 + distance
    if direction == "down":
        return r1 + distance, c0 + offset
    if direction == "left":
        return r0 + offset, c0 - distance
    raise AssertionError("unknown direction")


def _ray_capacity(direction: str, margins: dict[str, int]) -> int:
    return int(margins[direction])


def _build_ray_multiset(
    count: int, margins: dict[str, int]
) -> list[tuple[str, int]]:
    rays = [
        ("up", random.randrange(3)),
        ("right", random.randrange(3)),
        ("down", random.randrange(3)),
        ("left", random.randrange(3)),
    ]
    extras_needed = count - 4
    expandable = [ray for ray in rays if _ray_capacity(ray[0], margins) >= 3]
    if len(expandable) < extras_needed:
        raise ValueError("not enough expandable rays")

    extra_rays = random.sample(expandable, extras_needed)
    rays.extend(extra_rays)
    random.shuffle(rays)
    return rays


def _assign_unique_distances(rays: list[tuple[str, int]], margins: dict[str, int]) -> list[int] | None:
    order = list(range(len(rays)))
    random.shuffle(order)
    order.sort(key=lambda idx: _ray_capacity(rays[idx][0], margins))
    assigned = [0] * len(rays)

    def rec(pos: int, used: set[int]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        direction, _offset = rays[idx]
        candidates = list(range(2, _ray_capacity(direction, margins) + 1))
        random.shuffle(candidates)
        for distance in candidates:
            if distance in used:
                continue
            assigned[idx] = distance
            if rec(pos + 1, used | {distance}):
                return True
        assigned[idx] = 0
        return False

    if not rec(0, set()):
        return None
    return assigned


def _build_input_from_spec(
    block_bbox: tuple[int, int, int, int],
    rays: list[tuple[str, int]],
    distances: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    r0, c0, _r1, _c1 = block_bbox
    grid[r0 : r0 + 3, c0 : c0 + 3] = GREY

    colors = random.sample(list(SOURCE_COLOR_POOL), len(rays))
    for color, (direction, offset), distance in zip(colors, rays, distances):
        r, c = _ray_coordinate(block_bbox, direction, offset, distance)
        grid[r, c] = color
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    block_bbox = _find_grey_block(inp)
    markers = _marker_records(inp, block_bbox)
    out = main(inp)

    directions = [str(marker["direction"]) for marker in markers]
    if set(directions) != {"up", "right", "down", "left"}:
        return False

    source_colors = [int(marker["source_color"]) for marker in markers]
    if len(set(source_colors)) != len(source_colors):
        return False

    if not np.all(out[_find_grey_block(inp)[0] : _find_grey_block(inp)[2] + 1, _find_grey_block(inp)[1] : _find_grey_block(inp)[3] + 1] == GREY):
        return False

    landed_with_accum = _simulate_final_positions(markers, block_bbox, accumulate=True)
    landed_without_accum = _simulate_final_positions(markers, block_bbox, accumulate=False)
    if landed_with_accum == landed_without_accum:
        return False

    out_additions = Counter()
    for r, c, _direction in landed_with_accum:
        out_additions[int(out[r, c])] += 1
    expected_additions = Counter(DIRECTION_TO_OUTPUT[str(marker["direction"])] for marker in markers)
    if out_additions != expected_additions:
        return False

    shuffled = inp.copy()
    marker_positions = [(int(m["r"]), int(m["c"])) for m in markers]
    perm = list(range(len(marker_positions)))
    random.shuffle(perm)
    if perm == list(range(len(marker_positions))):
        perm = perm[1:] + perm[:1]
    shuffled_colors = [source_colors[i] for i in perm]
    for (r, c), color in zip(marker_positions, shuffled_colors):
        shuffled[r, c] = color
    if not np.array_equal(main(shuffled), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    bottom = random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    left = random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    right = random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    h = top + 3 + bottom
    w = left + 3 + right
    if h > 30 or w > 30:
        return None

    count = random.choice(MARKER_COUNT_CHOICES)
    margins = {"up": top, "right": right, "down": bottom, "left": left}
    if sum(cap >= 2 for cap in margins.values()) < 4:
        return None

    block_bbox = (top, left, top + 2, left + 2)
    rays = _build_ray_multiset(count, margins)
    distances = _assign_unique_distances(rays, margins)
    if distances is None:
        return None

    duplicate_groups = Counter(rays)
    if max(duplicate_groups.values()) < 2:
        return None

    inp = _build_input_from_spec(block_bbox, rays, distances, (h, w))

    try:
        _ = main(inp)
    except Exception:
        return None

    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
