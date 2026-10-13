import random
from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
ORIENTATIONS = ("up", "down", "left", "right")
DIR_FROM_ORIENTATION = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
AREA_TO_SHAPES = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(1, 4), (2, 2), (4, 1)],
    5: [(1, 5), (5, 1)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
}
CHALLENGE_PAIRS = {
    "up": [((2, 2), (2, 3)), ((2, 2), (2, 4)), ((2, 3), (2, 4)), ((3, 2), (3, 3))],
    "down": [((2, 2), (2, 3)), ((2, 2), (2, 4)), ((2, 3), (2, 4)), ((3, 2), (3, 3))],
    "left": [((2, 2), (3, 2)), ((2, 2), (4, 2)), ((3, 2), (4, 2)), ((2, 3), (3, 3))],
    "right": [((2, 2), (3, 2)), ((2, 2), (4, 2)), ((3, 2), (4, 2)), ((2, 3), (3, 3))],
}
GEN_RETRIES = 500
PLACE_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _detect_container(grid: np.ndarray, comp: dict) -> dict | None:
    wall_color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = set(coords)
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if not coord_set.issubset(perimeter):
        return None

    missing = list(perimeter - coord_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if gap[0] == r0:
        orientation = "up"
    elif gap[0] == r1:
        orientation = "down"
    elif gap[1] == c0:
        orientation = "left"
    else:
        orientation = "right"
    dr, dc = DIR_FROM_ORIENTATION[orientation]

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    seed_locs = np.argwhere(inner != BACKGROUND)
    if seed_locs.shape[0] != 1:
        return None
    seed_r = int(r0 + 1 + seed_locs[0, 0])
    seed_c = int(c0 + 1 + seed_locs[0, 1])
    seed_color = int(grid[seed_r, seed_c])
    if seed_color == wall_color:
        return None

    area = int(inner.shape[0] * inner.shape[1])
    start = (gap[0] + dr, gap[1] + dc)
    if not _in_bounds(grid.shape, start[0], start[1]):
        return None

    channel = []
    rr, cc = start
    while _in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) != BACKGROUND:
            return None
        channel.append((rr, cc))
        rr += dr
        cc += dc
    if len(channel) < area:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "wall_color": wall_color,
        "wall_cells": coord_set,
        "gap": gap,
        "orientation": orientation,
        "direction": (dr, dc),
        "seed": (seed_r, seed_c),
        "seed_color": seed_color,
        "inner_h": int(inner.shape[0]),
        "inner_w": int(inner.shape[1]),
        "area": area,
        "channel": channel,
        "start": start,
    }


def _parse_containers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = []
    accounted: set[tuple[int, int]] = set()
    for comp in _components_by_color(g):
        parsed = _detect_container(g, comp)
        if parsed is None:
            continue
        containers.append(parsed)
        accounted.update(parsed["wall_cells"])
        accounted.add(parsed["seed"])

    fg_cells = set(map(tuple, np.argwhere(g != BACKGROUND)))
    assert containers, "expected at least one leaking container"
    assert accounted == fg_cells, "all foreground cells must belong to container walls or seeds"

    areas = [item["area"] for item in containers]
    assert len(set(areas)) == len(areas), "container capacities must be distinct"
    return sorted(containers, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each broken rectangular container leak its interior capacity outward as a seed-colored trail, then seal the gap.

    Input structure:
    - A black background contains several monochrome rectangular ring containers.
    - Each wall ring is missing exactly one non-corner border cell, and each interior contains exactly one differently colored seed pixel.
    - The straight channel outside each gap is empty all the way to the grid edge.

    Rule:
    - For each container, count the number of cells in its enclosed rectangular interior, including the seed cell.
    - Starting from the cell immediately outside the gap, paint exactly that many cells along the outward normal using the seed color.
    - Fill the missing wall cell with the wall color and erase the entire interior back to black.

    Key insight:
    - The trail length comes from invisible capacity inside the ring, while the trail direction comes only from the unique wall gap and the trail color comes only from the interior seed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _parse_containers(g)

    for item in containers:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BACKGROUND
        gr, gc = item["gap"]
        out[gr, gc] = int(item["wall_color"])

    for item in containers:
        dr, dc = item["direction"]
        rr, cc = item["start"]
        for _ in range(item["area"]):
            assert _in_bounds(out.shape, rr, cc)
            assert int(out[rr, cc]) == BACKGROUND
            out[rr, cc] = int(item["seed_color"])
            rr += dr
            cc += dc

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _build_spec_defs(rng: random.Random, count: int) -> list[dict]:
    challenge_orientation = rng.choice(ORIENTATIONS)
    dims_a, dims_b = rng.choice(CHALLENGE_PAIRS[challenge_orientation])

    defs = [
        {"inner_h": dims_a[0], "inner_w": dims_a[1], "orientation": challenge_orientation},
        {"inner_h": dims_b[0], "inner_w": dims_b[1], "orientation": challenge_orientation},
    ]
    used_areas = {dims_a[0] * dims_a[1], dims_b[0] * dims_b[1]}

    if count >= 3:
        defs.append(
            {
                "inner_h": None,
                "inner_w": None,
                "orientation": rng.choice([ori for ori in ORIENTATIONS if ori != challenge_orientation]),
            }
        )
    if count == 4:
        defs.append({"inner_h": None, "inner_w": None, "orientation": rng.choice(ORIENTATIONS)})

    free_areas = [area for area in AREA_TO_SHAPES if area not in used_areas]
    rng.shuffle(free_areas)
    for item in defs:
        if item["inner_h"] is not None:
            continue
        area = free_areas.pop()
        inner_h, inner_w = rng.choice(AREA_TO_SHAPES[area])
        item["inner_h"] = inner_h
        item["inner_w"] = inner_w

    palette = rng.sample(list(range(1, 10)), 2 * count)
    wall_colors = palette[:count]
    seed_colors = palette[count:]
    rng.shuffle(defs)
    for idx, item in enumerate(defs):
        item["wall_color"] = wall_colors[idx]
        item["seed_color"] = seed_colors[idx]
        item["area"] = int(item["inner_h"] * item["inner_w"])
    return defs


def _channel_for_gap(shape: tuple[int, int], gap: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_FROM_ORIENTATION[orientation]
    rr = gap[0] + dr
    cc = gap[1] + dc
    channel = []
    while 0 <= rr < h and 0 <= cc < w:
        channel.append((rr, cc))
        rr += dr
        cc += dc
    return channel


def _pick_seed(
    rng: random.Random,
    spec: dict,
    interior: list[tuple[int, int]],
    gap: tuple[int, int],
) -> tuple[int, int]:
    shuffled = interior[:]
    rng.shuffle(shuffled)
    if spec["orientation"] in ("up", "down"):
        off_axis = [cell for cell in shuffled if cell[1] != gap[1]]
    else:
        off_axis = [cell for cell in shuffled if cell[0] != gap[0]]
    return off_axis[0] if off_axis else shuffled[0]


def _place_container(
    rng: random.Random,
    grid: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask: np.ndarray,
    spec: dict,
) -> dict | None:
    h, w = grid.shape
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    box_h = inner_h + 2
    box_w = inner_w + 2
    area = int(spec["area"])
    orientation = spec["orientation"]

    if orientation == "up":
        r_lo, r_hi = area, h - box_h - 1
        c_lo, c_hi = 1, w - box_w - 1
    elif orientation == "down":
        r_lo, r_hi = 1, h - box_h - area
        c_lo, c_hi = 1, w - box_w - 1
    elif orientation == "left":
        r_lo, r_hi = 1, h - box_h - 1
        c_lo, c_hi = area, w - box_w - 1
    else:
        r_lo, r_hi = 1, h - box_h - 1
        c_lo, c_hi = 1, w - box_w - area

    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if soft_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        if orientation == "up":
            gap = (r0, c0 + rng.randint(1, inner_w))
        elif orientation == "down":
            gap = (r1, c0 + rng.randint(1, inner_w))
        elif orientation == "left":
            gap = (r0 + rng.randint(1, inner_h), c0)
        else:
            gap = (r0 + rng.randint(1, inner_h), c1)

        channel = _channel_for_gap(grid.shape, gap, orientation)
        if len(channel) < area or any(soft_mask[r, c] for r, c in channel):
            continue

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        wall_cells = sorted(perimeter - {gap})
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        seed = _pick_seed(rng, spec, interior, gap)

        visible_cells = wall_cells + [seed]
        if any(soft_mask[r, c] for r, c in visible_cells):
            continue

        for r, c in wall_cells:
            grid[r, c] = int(spec["wall_color"])
            hard_mask[r, c] = True
        grid[seed] = int(spec["seed_color"])
        hard_mask[seed] = True
        for r, c in channel:
            hard_mask[r, c] = True

        soft_mask[:, :] = _dilate(hard_mask)
        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "orientation": orientation,
            "direction": DIR_FROM_ORIENTATION[orientation],
            "wall_color": int(spec["wall_color"]),
            "seed_color": int(spec["seed_color"]),
            "seed": seed,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "area": area,
            "channel": channel,
            "start": channel[0],
        }
    return None


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BACKGROUND
        gr, gc = spec["gap"]
        out[gr, gc] = int(spec["wall_color"])
    for spec in specs:
        dr, dc = spec["direction"]
        rr, cc = spec["start"]
        for _ in range(spec["area"]):
            assert _in_bounds(out.shape, rr, cc)
            assert int(out[rr, cc]) == BACKGROUND
            out[rr, cc] = int(spec["seed_color"])
            rr += dr
            cc += dc
    return out


def _has_challenge_pair(specs: list[dict]) -> bool:
    for left, right in combinations(specs, 2):
        if left["orientation"] != right["orientation"]:
            continue
        if left["orientation"] in ("up", "down"):
            if left["inner_h"] == right["inner_h"] and left["inner_w"] != right["inner_w"]:
                return True
        else:
            if left["inner_w"] == right["inner_w"] and left["inner_h"] != right["inner_h"]:
                return True
    return False


def _has_off_axis_seed(specs: list[dict]) -> bool:
    for spec in specs:
        sr, sc = spec["seed"]
        gr, gc = spec["gap"]
        if spec["orientation"] in ("up", "down") and sc != gc:
            return True
        if spec["orientation"] in ("left", "right") and sr != gr:
            return True
    return False


def _validate_generated_case(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        parsed = _parse_containers(grid)
        predicted = main(grid)
    except AssertionError:
        return False

    if len(parsed) != len(specs):
        return False
    if len({spec["area"] for spec in specs}) != len(specs):
        return False
    if len({spec["orientation"] for spec in specs}) < 2:
        return False
    if not _has_challenge_pair(specs):
        return False
    if not _has_off_axis_seed(specs):
        return False

    expected = _expected_from_specs(grid, specs)
    if not np.array_equal(predicted, expected):
        return False
    if np.array_equal(predicted, grid):
        return False

    by_color_expected = {spec["wall_color"]: spec for spec in specs}
    by_color_parsed = {spec["wall_color"]: spec for spec in parsed}
    if set(by_color_expected) != set(by_color_parsed):
        return False
    for color, expected_spec in by_color_expected.items():
        parsed_spec = by_color_parsed[color]
        for key in ("bbox", "gap", "orientation", "seed", "seed_color", "inner_h", "inner_w", "area"):
            if parsed_spec[key] != expected_spec[key]:
                return False
    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    h = rng.randint(18, 29)
    w = rng.randint(18, 29)
    count = rng.randint(2, 4)
    spec_defs = _build_spec_defs(rng, count)

    grid = np.zeros((h, w), dtype=int)
    hard_mask = np.zeros((h, w), dtype=bool)
    soft_mask = np.zeros((h, w), dtype=bool)
    placed_specs = []

    for spec in sorted(spec_defs, key=lambda item: item["area"], reverse=True):
        placed = _place_container(rng, grid, hard_mask, soft_mask, spec)
        if placed is None:
            return None
        placed_specs.append(placed)
    return grid, placed_specs


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        grid, specs = attempt
        if _validate_generated_case(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid leaking-container input")
