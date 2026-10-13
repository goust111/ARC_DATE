from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BASE = 5
START_COLOR = 4
PORT_COLORS = (1, 2, 3, 4)
CORNER_COLORS = (6, 7, 8, 9)

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
RING_COUNT_RANGE = (3, 5)
CHAIN_LEN_WEIGHTS = {1: 2, 2: 4, 3: 3, 4: 2}
RING_PAD = 1
MIN_CHAIN_GAP = 4
GEN_RETRIES = 400

DIR_VECS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT = {value: key for key, value in LEFT.items()}

SIDE_POS = {
    "N": (0, 1),
    "E": (1, 2),
    "S": (2, 1),
    "W": (1, 0),
}
CORNER_POS = {
    "NW": (0, 0),
    "NE": (0, 2),
    "SE": (2, 2),
    "SW": (2, 0),
}
OUTSIDE_OFFSET = {name: (dr * 2, dc * 2) for name, (dr, dc) in DIR_VECS.items()}


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _move(cell: tuple[int, int], direction: str, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIR_VECS[direction]
    r, c = cell
    return r + dr * steps, c + dc * steps


def _ring_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ring_outer_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _bbox_cells(center)
    cells.remove(center)
    return cells


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _corner_tuple() -> tuple[int, int, int, int]:
    return tuple(random.sample(CORNER_COLORS, 4))


def _ring_patch(corners: tuple[int, int, int, int], port_dir: str, port_color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    for name, color in zip(("NW", "NE", "SE", "SW"), corners):
        r, c = CORNER_POS[name]
        patch[r, c] = color
    for name in ("N", "E", "S", "W"):
        r, c = SIDE_POS[name]
        patch[r, c] = BASE
    r, c = SIDE_POS[port_dir]
    patch[r, c] = int(port_color)
    return patch


def _rotate_ring_patch(patch: np.ndarray, rotate_sides: bool = True, rotate_corners: bool = True) -> np.ndarray:
    old = np.asarray(patch, dtype=int)
    out = old.copy()

    if rotate_corners:
        out[0, 0] = old[2, 0]
        out[0, 2] = old[0, 0]
        out[2, 2] = old[0, 2]
        out[2, 0] = old[2, 2]

    if rotate_sides:
        out[0, 1] = old[1, 2]
        out[1, 2] = old[2, 1]
        out[2, 1] = old[1, 0]
        out[1, 0] = old[0, 1]

    return out


def _majority_and_port(side_values: dict[str, int]) -> tuple[int, str, int]:
    counts = Counter(side_values.values())
    assert sorted(counts.values()) == [1, 3], "expected exactly one side-center outlier port"
    base = next(color for color, count in counts.items() if count == 3)
    port_color = next(color for color, count in counts.items() if count == 1)
    port_dir = next(name for name, color in side_values.items() if color == port_color)
    return int(base), str(port_dir), int(port_color)


def _patch_side_values(patch: np.ndarray) -> dict[str, int]:
    return {name: int(patch[r, c]) for name, (r, c) in SIDE_POS.items()}


def _find_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict[str, object]] = []
    used_outer: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            outer = patch.copy()
            outer[1, 1] = BG
            if np.count_nonzero(outer == BG) != 1:
                continue
            if not np.all(outer[outer != BG] != BG):
                continue
            side_values = _patch_side_values(patch)
            try:
                base, port_dir, port_color = _majority_and_port(side_values)
            except AssertionError:
                continue

            outer_cells = _ring_outer_cells((r, c))
            if outer_cells & used_outer:
                raise AssertionError("overlapping ring detections are not allowed")
            used_outer |= outer_cells
            rings.append(
                {
                    "center": (r, c),
                    "bbox": (r - 1, c - 1, r + 1, c + 1),
                    "base": base,
                    "port_dir": port_dir,
                    "port_color": port_color,
                }
            )

    assert rings, "expected at least one ring"
    return rings


def _start_info(grid: np.ndarray, rings: list[dict[str, object]]) -> tuple[tuple[int, int], int]:
    ring_cells = set()
    for ring in rings:
        ring_cells |= _ring_outer_cells(ring["center"])

    extras = [(int(r), int(c)) for r, c in np.argwhere((grid != BG)) if (int(r), int(c)) not in ring_cells]
    assert len(extras) == 1, "expected exactly one non-ring start marker"
    start = extras[0]

    start_ring_ids = []
    for idx, ring in enumerate(rings):
        center = ring["center"]
        expected = _move(center, OPPOSITE[ring["port_dir"]], 2)
        if expected == start:
            start_ring_ids.append(idx)
    assert len(start_ring_ids) == 1, "start marker must sit behind exactly one ring port"
    return start, int(start_ring_ids[0])


def _trace_ray(
    out: np.ndarray,
    ring_lookup: dict[tuple[int, int], int],
    activated: set[int],
    center: tuple[int, int],
    direction: str,
    color: int,
) -> int | None:
    cell = _move(center, direction, 2)
    while _in_bounds(out.shape, cell):
        r, c = cell
        if int(out[r, c]) == BG:
            out[r, c] = int(color)
            cell = _move(cell, direction, 1)
            continue
        hit = ring_lookup.get(cell)
        if hit is not None and hit not in activated:
            return int(hit)
        return None
    return None


def _simulate(
    grid: np.ndarray,
    *,
    rotate_sides: bool = True,
    rotate_corners: bool = True,
    cascade: bool = True,
    use_rotated_direction: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rings = _find_rings(g)
    start_cell, start_ring = _start_info(g, rings)

    ring_lookup: dict[tuple[int, int], int] = {}
    for idx, ring in enumerate(rings):
        for cell in _ring_outer_cells(ring["center"]):
            ring_lookup[cell] = idx

    out = g.copy()
    out[start_cell] = BG

    activated: set[int] = set()
    queue = [start_ring]

    while queue:
        ring_id = int(queue.pop(0))
        if ring_id in activated:
            continue
        activated.add(ring_id)

        r, c = rings[ring_id]["center"]
        patch = out[r - 1 : r + 2, c - 1 : c + 2].copy()
        rotated = _rotate_ring_patch(patch, rotate_sides=rotate_sides, rotate_corners=rotate_corners)
        out[r - 1 : r + 2, c - 1 : c + 2] = rotated

        old_dir = str(rings[ring_id]["port_dir"])
        old_color = int(rings[ring_id]["port_color"])
        if rotate_sides and use_rotated_direction:
            direction = LEFT[old_dir]
            color = old_color
        elif rotate_sides and not use_rotated_direction:
            direction = old_dir
            color = old_color
        else:
            direction = old_dir
            color = old_color

        next_ring = _trace_ray(out, ring_lookup, activated, (r, c), direction, color)
        if cascade and next_ring is not None:
            queue.append(int(next_ring))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Activate a chain of 3x3 rings whose moved port directions emit colored relay rays.

    Input structure:
    - The grid has a black background with several 3x3 foreground rings, each with a black center.
    - Every ring has gray side-centers except for one differently colored side-center port, and one
      start marker pixel sits just outside the back side of exactly one ring.
    - The corner pixels form a visible ring-local pattern that rotates when the ring activates.

    Rule:
    - Erase the start marker and activate its adjacent ring.
    - When a ring activates, rotate its four side-centers one step counterclockwise and its four
      corners one step clockwise.
    - Then cast a ray outward from the ring through the port's new side; paint black cells with the
      port color until the ray reaches a non-black cell or the grid boundary.
    - If the first non-black cell hit belongs to an unactivated ring, that ring activates next.
    - Activated rings keep their rotated colors, rays remain visible, and untouched rings stay unchanged.

    Key insight:
    - The cascade path is determined by the port after it moves, so the ring rotation changes the
      network's connectivity before the ray is drawn.
    """
    return _simulate(grid)


def _direction_choices(max_chain: int) -> list[int]:
    lengths = list(range(1, max_chain + 1))
    weights = [CHAIN_LEN_WEIGHTS.get(length, 1) for length in lengths]
    return random.choices(lengths, weights=weights, k=1)


def _corridor_between(center_a: tuple[int, int], direction: str, center_b: tuple[int, int]) -> list[tuple[int, int]]:
    assert direction in DIR_VECS
    assert center_b == _move(center_a, direction, abs(center_b[0] - center_a[0]) + abs(center_b[1] - center_a[1]))
    stop = _move(center_b, OPPOSITE[direction], 1)
    cells: list[tuple[int, int]] = []
    cell = _move(center_a, direction, 2)
    while cell != stop:
        cells.append(cell)
        cell = _move(cell, direction, 1)
    assert cells
    return cells


def _corridor_to_boundary(center: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    cell = _move(center, direction, 2)
    while _in_bounds(shape, cell):
        cells.append(cell)
        cell = _move(cell, direction, 1)
    return cells


def _can_place_ring(
    center: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    bbox = _ring_bbox(center)
    if bbox[0] < 1 or bbox[1] < 1 or bbox[2] >= shape[0] - 1 or bbox[3] >= shape[1] - 1:
        return False
    if any(not _in_bounds(shape, cell) for cell in _bbox_cells(center)):
        return False
    if _bbox_cells(center) & reserved:
        return False
    return all(_boxes_separated(bbox, other, RING_PAD) for other in boxes)


def _build_chain(shape: tuple[int, int], chain_len: int) -> dict[str, object] | None:
    h, w = shape
    for _ in range(200):
        directions = [random.choice(tuple(DIR_VECS))]
        for _idx in range(1, chain_len):
            options = [name for name in DIR_VECS if name != OPPOSITE[directions[-1]]]
            random.shuffle(options)
            directions.append(options[0])

        start_candidates = [
            (r, c)
            for r in range(2, h - 2)
            for c in range(2, w - 2)
            if _in_bounds(shape, _move((r, c), LEFT[directions[0]], 2))
        ]
        random.shuffle(start_candidates)

        for start_center in start_candidates:
            boxes: list[tuple[int, int, int, int]] = []
            ring_cells_reserved: set[tuple[int, int]] = set()
            ray_cells: set[tuple[int, int]] = set()
            centers = [start_center]
            boxes.append(_ring_bbox(start_center))
            ring_cells_reserved |= _bbox_cells(start_center)

            start_marker = _move(start_center, LEFT[directions[0]], 2)
            if start_marker in ring_cells_reserved:
                continue
            occupied = set(ring_cells_reserved)
            occupied.add(start_marker)

            ok = True
            segments: list[list[tuple[int, int]]] = []

            for idx in range(chain_len - 1):
                curr = centers[-1]
                direction = directions[idx]
                dr, dc = DIR_VECS[direction]
                if direction in ("N", "S"):
                    if dr < 0:
                        max_gap = curr[0] - 2
                    else:
                        max_gap = (h - 3) - curr[0]
                else:
                    if dc < 0:
                        max_gap = curr[1] - 2
                    else:
                        max_gap = (w - 3) - curr[1]
                gaps = list(range(MIN_CHAIN_GAP, max_gap + 1))
                random.shuffle(gaps)

                found_next = False
                for gap in gaps:
                    nxt = _move(curr, direction, gap)
                    if not _can_place_ring(nxt, boxes, occupied | ray_cells, shape):
                        continue
                    corridor = _corridor_between(curr, direction, nxt)
                    if any(cell in occupied or cell in ray_cells for cell in corridor):
                        continue
                    centers.append(nxt)
                    boxes.append(_ring_bbox(nxt))
                    ring_cells_reserved |= _bbox_cells(nxt)
                    occupied |= _bbox_cells(nxt)
                    ray_cells |= set(corridor)
                    segments.append(corridor)
                    found_next = True
                    break

                if not found_next:
                    ok = False
                    break

            if not ok:
                continue

            last_direction = directions[-1]
            final_corridor = _corridor_to_boundary(centers[-1], last_direction, shape)
            if not final_corridor:
                continue
            if any(cell in occupied or cell in ray_cells for cell in final_corridor):
                continue
            segments.append(final_corridor)
            ray_cells |= set(final_corridor)

            return {
                "centers": centers,
                "directions": directions,
                "start_marker": start_marker,
                "ray_cells": ray_cells,
                "segments": segments,
                "boxes": boxes,
                "occupied": occupied,
            }

    return None


def _distractor_specs(
    shape: tuple[int, int],
    count: int,
    used_boxes: list[tuple[int, int, int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    specs: list[tuple[int, int]] = []
    boxes = list(used_boxes)
    blocked = set(reserved)
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(2, h - 2)
            for c in range(2, w - 2)
            if _can_place_ring((r, c), boxes, blocked, shape)
        ]
        if not candidates:
            return None
        center = random.choice(candidates)
        specs.append(center)
        boxes.append(_ring_bbox(center))
        blocked |= _bbox_cells(center)
    return specs


def _expected_from_plan(
    grid: np.ndarray,
    rings: list[dict[str, object]],
    start_marker: tuple[int, int],
    active_order: list[int],
    segments: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[start_marker] = BG
    for idx, ring_id in enumerate(active_order):
        center = rings[ring_id]["center"]
        r, c = center
        patch = out[r - 1 : r + 2, c - 1 : c + 2].copy()
        out[r - 1 : r + 2, c - 1 : c + 2] = _rotate_ring_patch(patch)
        color = int(rings[ring_id]["port_color"])
        for rr, cc in segments[idx]:
            out[rr, cc] = color
    return out


def _render_input(shape: tuple[int, int], rings: list[dict[str, object]], start_marker: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for ring in rings:
        r, c = ring["center"]
        patch = _ring_patch(
            corners=ring["corners"],
            port_dir=ring["port_dir"],
            port_color=ring["port_color"],
        )
        grid[r - 1 : r + 2, c - 1 : c + 2] = patch
    grid[start_marker] = START_COLOR
    return grid


def _shortcut_checks(inp: np.ndarray, expected: np.ndarray, chain_len: int) -> bool:
    if np.array_equal(_simulate(inp, rotate_sides=False), expected):
        return False
    if np.array_equal(_simulate(inp, rotate_corners=False), expected):
        return False
    if np.array_equal(_simulate(inp, use_rotated_direction=False), expected):
        return False
    if chain_len > 1 and np.array_equal(_simulate(inp, cascade=False), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    ring_count = random.randint(*RING_COUNT_RANGE)
    max_chain = min(4, ring_count - 1)
    chain_len = _direction_choices(max_chain)[0]

    chain = _build_chain(shape, chain_len)
    if chain is None:
        return None

    distractor_count = ring_count - chain_len
    distractors = _distractor_specs(
        shape,
        distractor_count,
        chain["boxes"],
        chain["occupied"] | chain["ray_cells"] | {chain["start_marker"]},
    )
    if distractors is None:
        return None

    all_centers = list(chain["centers"]) + list(distractors)
    rings: list[dict[str, object]] = []
    for idx, center in enumerate(all_centers):
        if idx < chain_len:
            out_dir = chain["directions"][idx]
            port_dir = RIGHT[out_dir]
        else:
            out_dir = random.choice(tuple(DIR_VECS))
            port_dir = RIGHT[out_dir]
        rings.append(
            {
                "center": center,
                "corners": _corner_tuple(),
                "port_dir": port_dir,
                "port_color": random.choice(PORT_COLORS),
            }
        )

    inp = _render_input(shape, rings, chain["start_marker"])
    expected = _expected_from_plan(
        inp,
        rings,
        chain["start_marker"],
        list(range(chain_len)),
        chain["segments"],
    )

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _shortcut_checks(inp, expected, chain_len):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring relay network input")
