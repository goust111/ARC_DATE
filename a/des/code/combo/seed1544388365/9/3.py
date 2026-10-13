import random
from collections import Counter, defaultdict, deque
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
AZURE = 8

INPUT_COLORS = (BLUE, RED, GREEN, YELLOW)
MIXABLE_COLORS = (BLUE, RED, GREEN)

DIR_SPECS = {
    "up": {"missing": (0, 1), "delta": (-1, 0)},
    "down": {"missing": (2, 1), "delta": (1, 0)},
    "left": {"missing": (1, 0), "delta": (0, -1)},
    "right": {"missing": (1, 2), "delta": (0, 1)},
}
DIR_NAMES = tuple(DIR_SPECS)
REL_CELLS = {
    name: tuple((r, c) for r in range(3) for c in range(3) if (r, c) != spec["missing"])
    for name, spec in DIR_SPECS.items()
}
MIX_COLORS = {
    tuple(sorted((RED, BLUE))): MAGENTA,
    tuple(sorted((RED, GREEN))): YELLOW,
    tuple(sorted((BLUE, GREEN))): AZURE,
}
OVERLAP_OFFSETS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if not (dr == 0 and dc == 0)]

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 500


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(coords, dtype=int)})
    return comps


def _parse_object(comp: dict) -> dict:
    coords = np.asarray(comp["coords"], dtype=int)
    assert coords.shape[0] == 8
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 2 and int(c1 - c0) == 2

    patch = np.zeros((3, 3), dtype=bool)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = True

    missing_pts = np.argwhere(~patch)
    assert missing_pts.shape == (1, 2)
    missing = tuple(map(int, missing_pts[0]))
    direction = None
    for name, spec in DIR_SPECS.items():
        if missing == spec["missing"]:
            direction = name
            break
    assert direction is not None

    return {
        "color": int(comp["color"]),
        "coords": coords,
        "top": int(r0),
        "left": int(c0),
        "direction": direction,
    }


def _bbox(top: int, left: int) -> tuple[int, int, int, int]:
    return int(top), int(left), int(top + 2), int(left + 2)


def _in_bounds(top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= int(top) <= h - 3 and 0 <= int(left) <= w - 3


def _rect_gap_ok(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _cells_from_top_left(top: int, left: int, direction: str) -> set[tuple[int, int]]:
    return {(int(top + dr), int(left + dc)) for dr, dc in REL_CELLS[direction]}


def _placement_for_final(obj: dict, final_top: int, final_left: int, shape: tuple[int, int]) -> dict | None:
    if not _in_bounds(final_top, final_left, shape):
        return None
    step = int(obj["step"])
    dr, dc = DIR_SPECS[obj["direction"]]["delta"]
    input_top = int(final_top - step * dr)
    input_left = int(final_left - step * dc)
    if not _in_bounds(input_top, input_left, shape):
        return None
    return {
        **obj,
        "input_top": input_top,
        "input_left": input_left,
        "final_top": int(final_top),
        "final_left": int(final_left),
        "input_bbox": _bbox(input_top, input_left),
        "final_bbox": _bbox(final_top, final_left),
        "input_cells": _cells_from_top_left(input_top, input_left, obj["direction"]),
        "final_cells": _cells_from_top_left(final_top, final_left, obj["direction"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 3x3-minus-one object in the direction encoded by its notch.
    The travel distance equals the number of objects with the same color.
    Overlaps of supported color pairs become their mixed color.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)

    objects = [_parse_object(comp) for comp in _find_components(g)]
    color_counts = Counter(int(obj["color"]) for obj in objects)
    arrivals: dict[tuple[int, int], list[int]] = defaultdict(list)

    for obj in objects:
        step = int(color_counts[int(obj["color"])])
        dr, dc = DIR_SPECS[obj["direction"]]["delta"]
        shifted = obj["coords"] + np.array([step * dr, step * dc], dtype=int)
        assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < g.shape[0])
        assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < g.shape[1])
        for r, c in shifted:
            arrivals[(int(r), int(c))].append(int(obj["color"]))

    for (r, c), colors in arrivals.items():
        unique = tuple(sorted(set(colors)))
        assert len(unique) == len(colors)
        assert 1 <= len(unique) <= 2
        if len(unique) == 1:
            out[r, c] = int(unique[0])
        else:
            assert unique in MIX_COLORS
            out[r, c] = int(MIX_COLORS[unique])

    return out


def _choose_colors_and_counts(rng: random.Random, want_overlap: bool) -> tuple[list[int], dict[int, int]] | None:
    for _ in range(200):
        num_colors = rng.randint(2, 4)
        colors = rng.sample(list(INPUT_COLORS), num_colors)
        if want_overlap and len(set(colors) & set(MIXABLE_COLORS)) < 2:
            continue
        counts = {color: rng.choices([1, 2, 3], weights=[4, 4, 3])[0] for color in colors}
        total = sum(counts.values())
        if not (3 <= total <= 7):
            continue
        if not any(count > 1 for count in counts.values()):
            continue
        return colors, counts
    return None


def _build_objects(rng: random.Random, colors: list[int], counts: dict[int, int]) -> list[dict]:
    objects: list[dict] = []
    obj_id = 0
    for color in colors:
        for _ in range(int(counts[color])):
            objects.append(
                {
                    "id": obj_id,
                    "color": int(color),
                    "step": int(counts[color]),
                    "direction": rng.choice(DIR_NAMES),
                }
            )
            obj_id += 1

    if len(objects) >= 2 and len({obj["direction"] for obj in objects}) == 1:
        old = objects[-1]["direction"]
        objects[-1]["direction"] = rng.choice([name for name in DIR_NAMES if name != old])

    rng.shuffle(objects)
    return objects


def _choose_overlap_pair(rng: random.Random, objects: list[dict]) -> tuple[int, int] | None:
    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, obj in enumerate(objects):
        by_color[int(obj["color"])].append(int(idx))

    pair_colors = [pair for pair in combinations(sorted(by_color), 2) if pair in MIX_COLORS]
    if not pair_colors:
        return None

    ca, cb = rng.choice(pair_colors)
    return rng.choice(by_color[ca]), rng.choice(by_color[cb])


def _place_overlap_pair(rng: random.Random, shape: tuple[int, int], obj_a: dict, obj_b: dict) -> list[dict] | None:
    h, w = shape
    offsets = OVERLAP_OFFSETS[:]
    rng.shuffle(offsets)
    for _ in range(PLACE_ATTEMPTS):
        base_top = rng.randint(0, h - 3)
        base_left = rng.randint(0, w - 3)
        dr_off, dc_off = rng.choice(offsets)
        other_top = int(base_top + dr_off)
        other_left = int(base_left + dc_off)
        p1 = _placement_for_final(obj_a, base_top, base_left, shape)
        p2 = _placement_for_final(obj_b, other_top, other_left, shape)
        if p1 is None or p2 is None:
            continue
        if not _rect_gap_ok(p2["input_bbox"], [p1["input_bbox"]], gap=1):
            continue
        overlap = p1["final_cells"] & p2["final_cells"]
        if not (1 <= len(overlap) <= 4):
            continue
        if not (p1["final_cells"] - overlap and p2["final_cells"] - overlap):
            continue
        return [p1, p2]
    return None


def _place_single(
    rng: random.Random,
    shape: tuple[int, int],
    obj: dict,
    input_boxes: list[tuple[int, int, int, int]],
    final_boxes: list[tuple[int, int, int, int]],
    final_cells: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        final_top = rng.randint(0, h - 3)
        final_left = rng.randint(0, w - 3)
        placement = _placement_for_final(obj, final_top, final_left, shape)
        if placement is None:
            continue
        if not _rect_gap_ok(placement["input_bbox"], input_boxes, gap=1):
            continue
        if not _rect_gap_ok(placement["final_bbox"], final_boxes, gap=1):
            continue
        if placement["final_cells"] & final_cells:
            continue
        return placement
    return None


def _draw_object(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in REL_CELLS[direction]:
        grid[int(top + dr), int(left + dc)] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_overlap = bool(rng.getrandbits(1))
    chosen = _choose_colors_and_counts(rng, want_overlap)
    if chosen is None:
        return None
    colors, counts = chosen
    objects = _build_objects(rng, colors, counts)
    total = len(objects)

    side_min = max(12, 4 + 4 * int(np.ceil(np.sqrt(total))))
    h = rng.randint(side_min, side_min + 4)
    w = rng.randint(side_min, side_min + 4)
    shape = (h, w)

    placements: list[dict] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    final_boxes: list[tuple[int, int, int, int]] = []
    final_cells: set[tuple[int, int]] = set()
    blocked_ids: set[int] = set()

    if want_overlap:
        pair = _choose_overlap_pair(rng, objects)
        if pair is None:
            return None
        pair_placements = _place_overlap_pair(rng, shape, objects[pair[0]], objects[pair[1]])
        if pair_placements is None:
            return None
        for placement in pair_placements:
            placements.append(placement)
            input_boxes.append(placement["input_bbox"])
            final_boxes.append(placement["final_bbox"])
            final_cells |= placement["final_cells"]
            blocked_ids.add(int(placement["id"]))

    for obj in objects:
        if int(obj["id"]) in blocked_ids:
            continue
        placement = _place_single(rng, shape, obj, input_boxes, final_boxes, final_cells)
        if placement is None:
            return None
        placements.append(placement)
        input_boxes.append(placement["input_bbox"])
        final_boxes.append(placement["final_bbox"])
        final_cells |= placement["final_cells"]

    if len({placement["direction"] for placement in placements}) < 2:
        return None

    grid = np.full(shape, BG, dtype=int)
    for placement in placements:
        _draw_object(
            grid,
            top=int(placement["input_top"]),
            left=int(placement["input_left"]),
            color=int(placement["color"]),
            direction=placement["direction"],
        )

    nonzero = Counter(int(x) for x in grid.flat if int(x) != BG)
    if set(nonzero) != set(colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
