import random
from collections import deque

import numpy as np


BG = 0
MARKER = 9
OBJECT_COLORS = (1, 2, 3, 4)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    marker_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == MARKER)}
    assert len(marker_cells) == 5, "marker must be a 5-cell cross"
    center = None
    for r, c in marker_cells:
        if all((r + dr, c + dc) in marker_cells for dr, dc in N4):
            center = (r, c)
            break
    assert center is not None, "marker center not found"
    expected = {
        center,
        (center[0] - 1, center[1]),
        (center[0] + 1, center[1]),
        (center[0], center[1] - 1),
        (center[0], center[1] + 1),
    }
    assert marker_cells == expected, "marker must be a small cross"
    return center, marker_cells


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARKER) or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _classify_side(cells: set[tuple[int, int]], marker_cells: set[tuple[int, int]]) -> str:
    sides = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in marker_cells:
                continue
            if r == nr - 1:
                sides.add("up")
            elif r == nr + 1:
                sides.add("down")
            elif c == nc - 1:
                sides.add("left")
            elif c == nc + 1:
                sides.add("right")
    assert len(sides) == 1, f"object must touch exactly one marker side, got {sides}"
    return next(iter(sides))


def _push_to_boundary(cells: set[tuple[int, int]], side: str, shape: tuple[int, int]) -> tuple[int, set[tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(cells)
    if side == "up":
        dist = r0
    elif side == "down":
        dist = h - 1 - r1
    elif side == "left":
        dist = c0
    else:
        dist = w - 1 - c1
    dr, dc = SIDE_TO_DIR[side]
    return dist, _translate(cells, dr * dist, dc * dist)


def _analyze(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]], list[dict]]:
    _, marker_cells = _find_marker(grid)
    objects = []
    for comp in _components(grid):
        cells = comp["cells"]
        side = _classify_side(cells, marker_cells)
        dist, final_cells = _push_to_boundary(cells, side, grid.shape)
        top, left, _, _ = _bbox(cells)
        objects.append(
            {
                "color": comp["color"],
                "cells": cells,
                "side": side,
                "distance": dist,
                "final_cells": final_cells,
                "top": top,
                "left": left,
            }
        )
    return _, marker_cells, objects


def _render(shape: tuple[int, int], marker_cells: set[tuple[int, int]], objects: list[dict], order_key) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in marker_cells:
        out[r, c] = MARKER
    for obj in sorted(objects, key=order_key):
        for r, c in obj["final_cells"]:
            out[r, c] = obj["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Push each marker-touching object to the boundary and resolve final overlaps by travel distance priority.

    Input structure:
    - A black background contains one maroon 5-cell cross marker and 3 or 4 non-maroon foreground objects.
    - Each foreground object is a 4-connected shape touching exactly one cardinal side of the marker.
    - Objects use visible spatial contact with the marker, not color ids, to encode their push direction.

    Rule:
    - Identify the maroon cross and every non-background, non-maroon 4-connected object.
    - An object touching the marker from above/below/left/right is pushed straight away from the marker in that direction until its leading edge reaches the grid boundary.
    - Clear the original object cells, keep the marker fixed, then redraw all pushed objects in ascending push-distance order; larger-distance objects are drawn later and overwrite earlier ones. If distances tie, the object with the smaller original top row is drawn first.

    Key insight:
    - The final collision result is not determined by color or by the input layering; it is encoded by the push itself, because farther-traveling objects get the last paint in overlap zones.
    """
    g = np.asarray(grid, dtype=int)
    _, marker_cells, objects = _analyze(g)
    return _render(g.shape, marker_cells, objects, lambda obj: (obj["distance"], obj["top"], obj["left"], obj["color"]))


def _random_shortest_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> set[tuple[int, int]]:
    r, c = start
    er, ec = end
    steps = []
    if er > r:
        steps.extend([(1, 0)] * (er - r))
    else:
        steps.extend([(-1, 0)] * (r - er))
    if ec > c:
        steps.extend([(0, 1)] * (ec - c))
    else:
        steps.extend([(0, -1)] * (c - ec))
    rng.shuffle(steps)
    path = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        path.add((r, c))
    return path


def _grow_shape(
    base: set[tuple[int, int]],
    domain: set[tuple[int, int]],
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    shape = set(base)
    if len(shape) > max_size:
        return None
    target = rng.randint(max(min_size, len(shape)), max_size)
    while len(shape) < target:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in domain and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    if not (min_size <= len(shape) <= max_size):
        return None
    return shape


def _corner(mode: str, h: int, w: int) -> tuple[int, int]:
    if mode == "TL":
        return (0, 0)
    if mode == "TR":
        return (0, w - 1)
    if mode == "BL":
        return (h - 1, 0)
    return (h - 1, w - 1)


def _pair_candidate_domains(
    mode: str,
    h: int,
    w: int,
    mr: int,
    mc: int,
    anchor_a: tuple[int, int],
    anchor_b: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    ra, _ = anchor_a
    _, cb = anchor_b
    if mode == "TL":
        dom_a = {(r, c) for r in range(0, ra + 1) for c in range(0, mc + 1)}
        dom_b = {(r, c) for r in range(0, mr + 1) for c in range(0, cb + 1)}
    elif mode == "TR":
        dom_a = {(r, c) for r in range(0, ra + 1) for c in range(mc, w)}
        dom_b = {(r, c) for r in range(0, mr + 1) for c in range(cb, w)}
    elif mode == "BL":
        dom_a = {(r, c) for r in range(ra, h) for c in range(0, mc + 1)}
        dom_b = {(r, c) for r in range(mr, h) for c in range(0, cb + 1)}
    else:
        dom_a = {(r, c) for r in range(ra, h) for c in range(mc, w)}
        dom_b = {(r, c) for r in range(mr, h) for c in range(cb, w)}
    return dom_a, dom_b


def _pair_shift(mode: str, mr: int, mc: int, anchor_a: tuple[int, int], anchor_b: tuple[int, int]) -> tuple[int, int]:
    ra, _ = anchor_a
    _, cb = anchor_b
    if mode == "TL":
        return mr - 2 - ra, mc - 2 - cb
    if mode == "TR":
        return mr - 2 - ra, cb - (mc + 2)
    if mode == "BL":
        return ra - (mr + 2), mc - 2 - cb
    return ra - (mr + 2), cb - (mc + 2)


def _pair_sides(mode: str) -> tuple[str, str]:
    if mode == "TL":
        return "up", "left"
    if mode == "TR":
        return "up", "right"
    if mode == "BL":
        return "down", "left"
    return "down", "right"


def _grow_toward_target(
    seed: tuple[int, int],
    domain: set[tuple[int, int]],
    target: tuple[int, int],
    size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    shape = {seed}
    while len(shape) < size:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in domain and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = sorted(
            set(frontier),
            key=lambda p: (abs(p[0] - target[0]) + abs(p[1] - target[1]), rng.random()),
        )
        pick_pool = frontier[: min(3, len(frontier))]
        shape.add(rng.choice(pick_pool if rng.random() < 0.8 else frontier))
    return shape


def _make_overlap_pair(mode: str, h: int, w: int, mr: int, mc: int, want_tie: bool, rng: random.Random):
    if want_tie:
        return
    side_a, side_b = _pair_sides(mode)
    marker_cells = _marker_cross(mr, mc)
    if mode == "BL":
        variants = [
            {
                "down": {(mr + 2, mc - 3), (mr + 2, mc), (mr + 3, mc - 3), (mr + 3, mc - 2), (mr + 3, mc - 1), (mr + 3, mc)},
                "left": {(mr, mc - 3), (mr, mc - 2), (mr + 1, mc - 2), (mr + 2, mc - 2)},
            },
            {
                "down": {(mr + 2, mc - 3), (mr + 2, mc), (mr + 3, mc - 3), (mr + 3, mc - 2), (mr + 3, mc - 1), (mr + 3, mc)},
                "left": {(mr, mc - 3), (mr, mc - 2), (mr + 1, mc - 3), (mr + 1, mc - 2), (mr + 2, mc - 2)},
            },
        ]
    elif mode == "BR":
        variants = [
            {
                "down": {(mr + 2, mc), (mr + 2, mc + 3), (mr + 3, mc), (mr + 3, mc + 1), (mr + 3, mc + 2), (mr + 3, mc + 3)},
                "right": {(mr, mc + 2), (mr + 1, mc + 2), (mr + 1, mc + 3), (mr + 2, mc + 2)},
            },
            {
                "down": {(mr + 2, mc), (mr + 2, mc + 3), (mr + 3, mc), (mr + 3, mc + 1), (mr + 3, mc + 2), (mr + 3, mc + 3)},
                "right": {(mr, mc + 2), (mr + 1, mc + 2), (mr + 1, mc + 3), (mr + 2, mc + 2), (mr + 2, mc + 3)},
            },
        ]
    else:
        variants = []
    rng.shuffle(variants)
    for variant in variants:
        if any(not all(0 <= r < h and 0 <= c < w for r, c in cells) for cells in variant.values()):
            continue
        if any(cells & marker_cells for cells in variant.values()):
            continue
        all_cells = set()
        ok = True
        for side, cells in variant.items():
            if all_cells & cells:
                ok = False
                break
            all_cells |= cells
            try:
                if _classify_side(cells, marker_cells) != side:
                    ok = False
                    break
            except AssertionError:
                ok = False
                break
        if not ok:
            continue
        dist_a, final_a = _push_to_boundary(variant[side_a], side_a, (h, w))
        dist_b, final_b = _push_to_boundary(variant[side_b], side_b, (h, w))
        if dist_a == dist_b or not (final_a & final_b):
            continue
        yield variant
        return

    if mode == "TL":
        target = (0, 0)
        tip_a = (mr - 2, mc)
        tip_b = (mr, mc - 2)
        dom_a = {(r, c) for r in range(0, mr - 1) for c in range(0, mc + 1)}
        dom_b = {(r, c) for r in range(0, mr + 1) for c in range(0, mc - 1)}
    elif mode == "TR":
        target = (0, w - 1)
        tip_a = (mr - 2, mc)
        tip_b = (mr, mc + 2)
        dom_a = {(r, c) for r in range(0, mr - 1) for c in range(mc, w)}
        dom_b = {(r, c) for r in range(0, mr + 1) for c in range(mc + 2, w)}
    elif mode == "BL":
        target = (h - 1, 0)
        tip_a = (mr + 2, mc)
        tip_b = (mr, mc - 2)
        dom_a = {(r, c) for r in range(mr + 2, h) for c in range(0, mc + 1)}
        dom_b = {(r, c) for r in range(mr, h) for c in range(0, mc - 1)}
    else:
        target = (h - 1, w - 1)
        tip_a = (mr + 2, mc)
        tip_b = (mr, mc + 2)
        dom_a = {(r, c) for r in range(mr + 2, h) for c in range(mc, w)}
        dom_b = {(r, c) for r in range(mr, h) for c in range(mc + 2, w)}

    for _ in range(4000):
        size_a = rng.randint(3, 6)
        size_b = rng.randint(3, 6)
        inp_a = _grow_toward_target(tip_a, dom_a, target, size_a, rng)
        inp_b = _grow_toward_target(tip_b, dom_b, target, size_b, rng)
        if inp_a is None or inp_b is None:
            continue
        if inp_a & marker_cells or inp_b & marker_cells or inp_a & inp_b:
            continue
        try:
            if _classify_side(inp_a, marker_cells) != side_a:
                continue
            if _classify_side(inp_b, marker_cells) != side_b:
                continue
        except AssertionError:
            continue
        dist_a, final_a = _push_to_boundary(inp_a, side_a, (h, w))
        dist_b, final_b = _push_to_boundary(inp_b, side_b, (h, w))
        if want_tie and dist_a != dist_b:
            continue
        if not want_tie and dist_a == dist_b:
            continue
        if not (final_a & final_b):
            continue
        yield {side_a: inp_a, side_b: inp_b}


def _marker_cross(mr: int, mc: int) -> set[tuple[int, int]]:
    return {
        (mr, mc),
        (mr - 1, mc),
        (mr + 1, mc),
        (mr, mc - 1),
        (mr, mc + 1),
    }


_GENERIC_TEMPLATES = {
    "up": [
        {(0, 0), (-1, 0), (-2, 0)},
        {(0, 0), (-1, 0), (-1, -1)},
        {(0, 0), (-1, 0), (-1, 1), (-2, 1)},
        {(0, 0), (0, -1), (-1, -1), (-2, -1)},
    ],
    "down": [
        {(0, 0), (1, 0), (2, 0)},
        {(0, 0), (1, 0), (1, -1)},
        {(0, 0), (1, 0), (1, 1), (2, 1)},
        {(0, 0), (0, -1), (1, -1), (2, -1)},
    ],
    "left": [
        {(0, 0), (0, -1), (0, -2)},
        {(0, 0), (0, -1), (-1, -1)},
        {(0, 0), (0, -1), (1, -1), (1, -2)},
        {(0, 0), (-1, 0), (-1, -1), (-2, -1)},
    ],
    "right": [
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (0, 1), (-1, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2)},
        {(0, 0), (-1, 0), (-1, 1), (-2, 1)},
    ],
}


def _contact_tip(side: str, mr: int, mc: int) -> tuple[int, int]:
    if side == "up":
        return mr - 2, mc
    if side == "down":
        return mr + 2, mc
    if side == "left":
        return mr, mc - 2
    return mr, mc + 2


def _place_generic_object(
    side: str,
    mr: int,
    mc: int,
    h: int,
    w: int,
    marker_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    tip = _contact_tip(side, mr, mc)
    templates = _GENERIC_TEMPLATES[side][:]
    rng.shuffle(templates)
    for template in templates:
        cells = {(tip[0] + dr, tip[1] + dc) for dr, dc in template}
        if not all(0 <= r < h and 0 <= c < w for r, c in cells):
            continue
        if cells & occupied:
            continue
        if cells & marker_cells:
            continue
        try:
            if _classify_side(cells, marker_cells) != side:
                continue
        except AssertionError:
            continue
        return cells
    return None


def _paint_input(h: int, w: int, mr: int, mc: int, objects_by_side: dict[str, set[tuple[int, int]]], rng: random.Random) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in _marker_cross(mr, mc):
        grid[r, c] = MARKER
    colors = list(OBJECT_COLORS)
    rng.shuffle(colors)
    for color, side in zip(colors, sorted(objects_by_side)):
        for r, c in objects_by_side[side]:
            grid[r, c] = color
    return grid


def _wrong_render(marker_cells: set[tuple[int, int]], objects: list[dict], shape: tuple[int, int], tie_desc: bool = False) -> np.ndarray:
    if tie_desc:
        key = lambda obj: (obj["distance"], -obj["top"], obj["left"], obj["color"])
    else:
        key = lambda obj: (-obj["distance"], obj["top"], obj["left"], obj["color"])
    return _render(shape, marker_cells, objects, key)


def _validate_instance(grid: np.ndarray) -> None:
    center, marker_cells = _find_marker(grid)
    h, w = grid.shape
    mr, mc = center
    assert 0 < mr < h - 1 and 0 < mc < w - 1
    _, _, objects = _analyze(grid)
    assert 3 <= len(objects) <= 4
    sides = [obj["side"] for obj in objects]
    assert len(sides) == len(set(sides)), "each marker side can host at most one object"
    for obj in objects:
        assert 3 <= len(obj["cells"]) <= 6
    key_pairs = {(obj["distance"], obj["top"]) for obj in objects}
    assert len(key_pairs) == len(objects), "avoid ambiguous equal-distance equal-top cases"
    overlap_cells = {}
    for idx, obj in enumerate(objects):
        for cell in obj["final_cells"]:
            overlap_cells.setdefault(cell, []).append(idx)
    contested = [cell for cell, owners in overlap_cells.items() if len(owners) >= 2]
    assert contested, "at least one final overlap is required"
    out = main(grid)
    assert not np.array_equal(out, grid), "output must change"
    wrong_desc = _wrong_render(marker_cells, objects, grid.shape, tie_desc=False)
    tie_exists = any(objects[i]["distance"] == objects[j]["distance"] for i in range(len(objects)) for j in range(i + 1, len(objects)))
    if tie_exists:
        wrong_tie = _wrong_render(marker_cells, objects, grid.shape, tie_desc=True)
        assert not np.array_equal(out, wrong_tie), "tie branch must matter when equal distances exist"
    else:
        assert not np.array_equal(out, wrong_desc), "distance priority must matter"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["BL", "BR"])
    h = rng.randint(10, 12)
    w = rng.randint(10, 12)
    if mode in ("TL", "TR"):
        mr = rng.randint(4, 5)
    else:
        mr = h - rng.randint(4, 5)
    if mode in ("TL", "BL"):
        mc = rng.randint(4, 5)
    else:
        mc = w - 1 - rng.randint(4, 5)
    if not (2 <= mr <= h - 3 and 2 <= mc <= w - 3):
        return None
    marker_cells = _marker_cross(mr, mc)
    want_tie = False
    pair = None
    for candidate in _make_overlap_pair(mode, h, w, mr, mc, want_tie, rng):
        occupied = set(marker_cells)
        ok = True
        for cells in candidate.values():
            if cells & occupied:
                ok = False
                break
            occupied |= cells
        if ok:
            pair = candidate
            break
    if pair is None:
        return None
    remaining_sides = [side for side in ("up", "down", "left", "right") if side not in pair]
    rng.shuffle(remaining_sides)
    target_count = 3 if rng.random() < 0.55 else 4
    use_sides = remaining_sides[: target_count - 2]
    objects_by_side = dict(pair)
    occupied = set(marker_cells)
    for cells in objects_by_side.values():
        occupied |= cells
    for side in use_sides:
        cells = _place_generic_object(side, mr, mc, h, w, marker_cells, occupied, rng)
        if cells is None:
            return None
        objects_by_side[side] = cells
        occupied |= cells
    grid = _paint_input(h, w, mr, mc, objects_by_side, rng)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
