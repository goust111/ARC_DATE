import random
from itertools import combinations

import numpy as np


BG = 0
ANCHOR = 1
FRAME = 5
SEED_COLORS = (2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 16
GRID_MAX = 30
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 8
GEN_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 500
MIN_SIDE_CELLS = 2
MIN_CHANGED_PER_OBJECT = 2
ANCHOR_DEPTH_MAX = 6
OBJECT_COUNT_CHOICES = (1, 2, 3)
OBJECT_COUNT_WEIGHTS = (2, 2, 1)

STEP_BY_NAME = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed outward along the hidden rail implied by its blue anchor pair.

    Input structure:
    - The background is black.
    - One or more gray one-pixel-thick rectangular fences appear on the grid.
    - Each fence contains exactly one non-black, non-gray seed pixel.
    - Outside the fences, blue pixels appear in disjoint pairs; each pair lies on one
      horizontal, vertical, or 45-degree diagonal rail line.

    Rule:
    - For each gray fence, find its unique interior seed and compute the fence perimeter
      as the number of gray border pixels.
    - Find the unique blue-anchor rail whose line is collinear with that seed.
    - Starting from the seed, walk along the rail in both directions for exactly that
      many steps, stopping early at the grid edge.
    - Recolor only black cells on that walked path with the seed color; keep every fence
      and every fenced interior cell unchanged.

    Key insight:
    - The blue anchor pairs define invisible rail directions, while the gray fence size
      supplies the propagation length through its perimeter.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    anchors = [tuple(map(int, pt)) for pt in np.argwhere(g == ANCHOR)]
    rail_keys = _rail_keys_from_anchors(anchors)
    protected = set()
    for frame in frames:
        protected.update(_bbox_cells(*frame["bbox"]))

    for frame in frames:
        seed = frame["seed"]
        matching = [key for key in rail_keys if _point_on_line(seed, key)]
        assert len(matching) == 1, "each seed must align with exactly one rail"
        step = _step_for_key(matching[0])
        for sign in (-1, 1):
            for dist in range(1, frame["perimeter"] + 1):
                r = seed[0] + sign * step[0] * dist
                c = seed[1] + sign * step[1] * dist
                if not _in_bounds(g.shape, r, c):
                    break
                if (r, c) in protected:
                    continue
                if out[r, c] == BG:
                    out[r, c] = frame["seed_color"]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    count = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    colors = rng.sample(SEED_COLORS, count)
    directions = rng.sample(tuple(STEP_BY_NAME), count)

    blocked: set[tuple[int, int]] = set()
    seeds: list[tuple[int, int]] = []
    objects = []

    for color, direction_name in zip(colors, directions):
        obj = _place_object(rng, grid.shape, color, direction_name, blocked, seeds)
        if obj is None:
            return None
        _draw_object(grid, obj)
        blocked.update(obj["blocked"])
        seeds.append(obj["seed"])
        objects.append(obj)

    if not _validate_generated_sample(grid, objects):
        return None
    return grid


def _place_object(
    rng: random.Random,
    shape: tuple[int, int],
    color: int,
    direction_name: str,
    blocked: set[tuple[int, int]],
    existing_seeds: list[tuple[int, int]],
):
    height, width = shape
    step = STEP_BY_NAME[direction_name]

    for _ in range(PLACEMENT_ATTEMPTS):
        frame_h = rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX)
        frame_w = rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX)
        if frame_h + 2 > height or frame_w + 2 > width:
            continue

        max_r0 = height - frame_h - 1
        max_c0 = width - frame_w - 1
        if max_r0 < 1 or max_c0 < 1:
            continue

        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        bbox = (r0, c0, r1, c1)

        sr = rng.randint(r0 + 1, r1 - 1)
        sc = rng.randint(c0 + 1, c1 - 1)
        seed = (sr, sc)
        line_key = _key_from_seed(direction_name, seed)

        if any(_point_on_line(other_seed, line_key) for other_seed in existing_seeds):
            continue

        side_neg = _outside_cells_on_side(seed, step, bbox, shape, sign=-1)
        side_pos = _outside_cells_on_side(seed, step, bbox, shape, sign=1)
        if len(side_neg) < MIN_SIDE_CELLS or len(side_pos) < MIN_SIDE_CELLS:
            continue

        perimeter = 2 * (frame_h + frame_w) - 4
        track = _track_positions(seed, step, perimeter, bbox, shape)
        if len(track) < 2 * MIN_CHANGED_PER_OBJECT:
            continue

        neg_pool = side_neg[: min(len(side_neg), ANCHOR_DEPTH_MAX)]
        pos_pool = side_pos[: min(len(side_pos), ANCHOR_DEPTH_MAX)]
        anchor_neg = rng.choice(neg_pool)
        anchor_pos = rng.choice(pos_pool)
        anchors = (anchor_neg, anchor_pos)

        visible_track = {pos for pos in track if pos not in anchors}
        if len(visible_track) < MIN_CHANGED_PER_OBJECT:
            continue

        anchor_bridge = set(side_neg[: side_neg.index(anchor_neg) + 1])
        anchor_bridge.update(side_pos[: side_pos.index(anchor_pos) + 1])

        bbox_cells = set(_bbox_cells(*bbox))
        margin_cells = _expanded_bbox_cells(bbox, shape, pad=1)
        local_blocked = bbox_cells | margin_cells | track | anchor_bridge | set(anchors)
        if local_blocked & blocked:
            continue

        return {
            "bbox": bbox,
            "bbox_cells": bbox_cells,
            "seed": seed,
            "seed_color": color,
            "perimeter": perimeter,
            "direction_name": direction_name,
            "line_key": line_key,
            "anchors": anchors,
            "track": track,
            "blocked": local_blocked,
        }

    return None


def _draw_object(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    for r, c in _border_cells(r0, c0, r1, c1):
        grid[r, c] = FRAME
    sr, sc = obj["seed"]
    grid[sr, sc] = obj["seed_color"]
    for r, c in obj["anchors"]:
        grid[r, c] = ANCHOR


def _validate_generated_sample(grid: np.ndarray, objects: list[dict]) -> bool:
    anchors = [anchor for obj in objects for anchor in obj["anchors"]]
    pair_keys = []
    valid_pair_count = 0
    for a, b in combinations(anchors, 2):
        key = _line_key(a, b)
        if key is not None:
            valid_pair_count += 1
            pair_keys.append(key)
    pair_keys = list(dict.fromkeys(pair_keys))
    expected_keys = {obj["line_key"] for obj in objects}
    if valid_pair_count != len(objects):
        return False
    if len(pair_keys) != len(objects) or set(pair_keys) != expected_keys:
        return False

    for obj in objects:
        if sum(_point_on_line(obj["seed"], key) for key in pair_keys) != 1:
            return False

    expected = grid.copy()
    for obj in objects:
        for r, c in obj["track"]:
            if expected[r, c] == BG:
                expected[r, c] = obj["seed_color"]

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    changed = np.count_nonzero(actual != grid)
    return changed >= len(objects) * MIN_CHANGED_PER_OBJECT


def _find_frames(grid: np.ndarray) -> list[dict]:
    frame_cells = {tuple(map(int, pt)) for pt in np.argwhere(grid == FRAME)}
    visited: set[tuple[int, int]] = set()
    frames = []

    for start in sorted(frame_cells):
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            cell = stack.pop()
            if cell in visited:
                continue
            visited.add(cell)
            comp.add(cell)
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in frame_cells and (nr, nc) not in visited:
                    stack.append((nr, nc))

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        expected_border = set(_border_cells(*bbox))
        assert comp == expected_border, "gray component must be a rectangular border"

        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence must have an interior"
        interior = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(grid[r, c])
                if value != BG:
                    interior.append((r, c, value))
        assert len(interior) == 1, "each fence must contain exactly one seed"
        seed_r, seed_c, seed_color = interior[0]
        assert seed_color not in (BG, ANCHOR, FRAME), "seed color must be distinct"
        frames.append(
            {
                "bbox": bbox,
                "seed": (seed_r, seed_c),
                "seed_color": seed_color,
                "perimeter": len(expected_border),
            }
        )

    assert frames, "expected at least one fenced seed"
    return frames


def _rail_keys_from_anchors(anchors: list[tuple[int, int]]) -> list[tuple[str, int]]:
    keys = []
    for a, b in combinations(anchors, 2):
        key = _line_key(a, b)
        if key is not None and key not in keys:
            keys.append(key)
    assert keys, "expected at least one anchor pair"
    return keys


def _line_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, int] | None:
    if a == b:
        return None
    if a[0] == b[0]:
        return ("h", a[0])
    if a[1] == b[1]:
        return ("v", a[1])
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if abs(dr) != abs(dc):
        return None
    if dr * dc > 0:
        return ("d1", a[0] - a[1])
    return ("d2", a[0] + a[1])


def _key_from_seed(direction_name: str, seed: tuple[int, int]) -> tuple[str, int]:
    r, c = seed
    if direction_name == "h":
        return ("h", r)
    if direction_name == "v":
        return ("v", c)
    if direction_name == "d1":
        return ("d1", r - c)
    return ("d2", r + c)


def _step_for_key(key: tuple[str, int]) -> tuple[int, int]:
    return STEP_BY_NAME[key[0]]


def _point_on_line(point: tuple[int, int], key: tuple[str, int]) -> bool:
    r, c = point
    name, value = key
    if name == "h":
        return r == value
    if name == "v":
        return c == value
    if name == "d1":
        return r - c == value
    return r + c == value


def _track_positions(
    seed: tuple[int, int],
    step: tuple[int, int],
    length: int,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    positions = set()
    for sign in (-1, 1):
        for dist in range(1, length + 1):
            r = seed[0] + sign * step[0] * dist
            c = seed[1] + sign * step[1] * dist
            if not _in_bounds(shape, r, c):
                break
            if not _inside_bbox((r, c), bbox):
                positions.add((r, c))
    return positions


def _outside_cells_on_side(
    seed: tuple[int, int],
    step: tuple[int, int],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    sign: int,
) -> list[tuple[int, int]]:
    cells = []
    dist = 1
    while True:
        r = seed[0] + sign * step[0] * dist
        c = seed[1] + sign * step[1] * dist
        if not _in_bounds(shape, r, c):
            break
        if not _inside_bbox((r, c), bbox):
            cells.append((r, c))
        dist += 1
    return cells


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_cells(r0: int, c0: int, r1: int, c1: int):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield (r, c)


def _border_cells(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _expanded_bbox_cells(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad: int,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(shape[0] - 1, r1 + pad)
    ec1 = min(shape[1] - 1, c1 + pad)
    return set(_bbox_cells(er0, ec0, er1, ec1))
