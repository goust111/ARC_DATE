from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components


BG = Color.BLACK
AFTERIMAGE = Color.GREY
BLOCKER = Color.TEAL
RESERVED_COLORS = {BG, AFTERIMAGE, BLOCKER}


def _shape_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(np.asarray(coords, dtype=int))
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _anchor(comp: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return int(r0), int(c0)


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _claim_vectors(objects: list[dict], shadows: list[dict]) -> list[dict]:
    shadow_shapes = [_shape_from_coords(comp["coords"]) for comp in shadows]
    shadow_anchors = [_anchor(comp) for comp in shadows]

    claims = []
    used_shadow_ids: list[int] = []
    for obj in objects:
        obj_shape = _shape_from_coords(obj["coords"])
        obj_anchor = _anchor(obj)
        matching = [i for i, shape in enumerate(shadow_shapes) if shape == obj_shape]
        candidates = []

        for i in matching:
            for j in matching:
                if i == j:
                    continue
                back = shadow_anchors[i]
                mid = shadow_anchors[j]
                v = _sub(mid, back)
                if v != (0, 0) and _add(mid, v) == obj_anchor:
                    candidates.append((i, j, v))

        unique = []
        seen = set()
        for i, j, v in candidates:
            key = (i, j, v)
            if key not in seen:
                seen.add(key)
                unique.append((i, j, v))

        assert len(unique) == 1, "each object must claim exactly one ordered afterimage pair"
        back_id, mid_id, vector = unique[0]
        used_shadow_ids.extend([back_id, mid_id])
        claims.append(
            {
                "color": int(obj["color"]),
                "coords": np.asarray(obj["coords"], dtype=int),
                "vector": np.asarray(vector, dtype=int),
            }
        )

    assert len(set(used_shadow_ids)) == len(used_shadow_ids), "afterimages must be claimed once"
    assert len(used_shadow_ids) == len(shadows), "all afterimages must be claimed"
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Jump each colored object along the vector implied by its two grey afterimages until the next landing is blocked.

    Input structure:
    - The background is black, teal pixels are stationary blockers, and dark grey pixels are afterimages.
    - Each non-black, non-teal, non-grey color forms exactly one 4-connected movable object.
    - Every movable object has exactly two dark grey 4-connected afterimages with the same normalized shape; the two afterimages and the object bbox anchors lie on one equally spaced line with the object at the front.

    Rule:
    - Match each object to its two same-shaped afterimages and read the object-specific vector from back afterimage to middle afterimage to object.
    - Erase every original object and afterimage, keep all teal blockers fixed, and repeatedly translate each object by its vector.
    - Stop an object at its current position exactly when the next whole-vector landing would go out of bounds or overlap a teal blocker.

    Key insight:
    - The vector is not drawn directly; it must be claimed by shape matching, and blockers matter only at whole-step landing cells so a one-cell wall between landings can be jumped over.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H <= 30 and W <= 30, "ARC grids must be at most 30 by 30"
    assert BG in set(map(int, np.unique(g))), "expected black background"

    object_colors = [int(c) for c in np.unique(g) if int(c) not in RESERVED_COLORS]
    assert object_colors, "expected at least one movable object"

    objects: list[dict] = []
    for color in object_colors:
        comps = find_components(
            g,
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        assert len(comps) == 1, "each movable color must form one component"
        objects.append(comps[0])

    shadows = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[AFTERIMAGE],
    )
    assert len(shadows) == 2 * len(objects), "expected two afterimages per object"

    claims = _claim_vectors(objects, shadows)
    blockers = g == BLOCKER
    out = np.full_like(g, BG)
    out[blockers] = BLOCKER

    occupied = np.zeros_like(blockers, dtype=bool)
    for claim in claims:
        coords = claim["coords"].copy()
        vector = claim["vector"]

        while True:
            nxt = coords + vector
            in_bounds = (
                (0 <= nxt[:, 0])
                & (nxt[:, 0] < H)
                & (0 <= nxt[:, 1])
                & (nxt[:, 1] < W)
            )
            if not bool(in_bounds.all()):
                break
            if bool(blockers[nxt[:, 0], nxt[:, 1]].any()):
                break
            coords = nxt

        assert not bool(occupied[coords[:, 0], coords[:, 1]].any()), "final objects must not overlap"
        occupied[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = int(claim["color"])

    return out


# --------------- generator ---------------

MOVING_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]
VECTOR_POOL = [
    (0, 3),
    (2, 2),
    (-4, 0),
    (1, 3),
    (2, -3),
    (-4, 1),
    (3, 3),
    (-4, -2),
    (5, 0),
    (-2, 5),
]

SHAPES = [
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]

GEN_RETRIES = 600
PLACE_RETRIES = 500


def _cells(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + r), int(ac + c)) for r, c in shape}


def _fits(cells: set[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _shape_separated_by_vector(shape: tuple[tuple[int, int], ...], v: tuple[int, int]) -> bool:
    return not _touches4(_cells(shape, (0, 0)), _cells(shape, v))


def _direction_key(v: tuple[int, int]) -> tuple[int, int]:
    dr, dc = v
    g = math.gcd(abs(dr), abs(dc))
    return int(dr // g), int(dc // g)


def _choose_vectors(count: int) -> list[tuple[int, int]] | None:
    crossing = [v for v in VECTOR_POOL if math.gcd(abs(v[0]), abs(v[1])) > 1]
    first = random.choice(crossing)
    chosen = [first]
    dirs = {_direction_key(first)}
    lengths = {first[0] * first[0] + first[1] * first[1]}

    rest = [v for v in VECTOR_POOL if v != first]
    random.shuffle(rest)
    for v in rest:
        dkey = _direction_key(v)
        length = v[0] * v[0] + v[1] * v[1]
        if dkey in dirs or length in lengths:
            continue
        chosen.append(v)
        dirs.add(dkey)
        lengths.add(length)
        if len(chosen) == count:
            return chosen
    return None


def _choose_shapes(vectors: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]] | None:
    remaining = list(SHAPES)
    random.shuffle(remaining)
    shapes = []
    for v in vectors:
        choices = [shape for shape in remaining if _shape_separated_by_vector(shape, v)]
        if not choices:
            return None
        shape = random.choice(choices)
        remaining.remove(shape)
        shapes.append(shape)
    return shapes


def _landings(
    shape: tuple[tuple[int, int], ...],
    start: tuple[int, int],
    v: tuple[int, int],
    steps: int,
) -> list[set[tuple[int, int]]]:
    return [_cells(shape, (start[0] + k * v[0], start[1] + k * v[1])) for k in range(steps + 1)]


def _chain_cells(
    shape: tuple[tuple[int, int], ...],
    start: tuple[int, int],
    v: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = set()
    for k in (-2, -1, 0):
        cells |= _cells(shape, (start[0] + k * v[0], start[1] + k * v[1]))
    return cells


def _max_boundary_steps(
    shape: tuple[tuple[int, int], ...],
    start: tuple[int, int],
    v: tuple[int, int],
    H: int,
    W: int,
) -> int:
    steps = 0
    while True:
        nxt = _cells(shape, (start[0] + (steps + 1) * v[0], start[1] + (steps + 1) * v[1]))
        if not _fits(nxt, H, W):
            return steps
        steps += 1
        if steps > 8:
            return steps


def _intermediate_cells(
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    v: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = v
    g = math.gcd(abs(dr), abs(dc))
    if g <= 1:
        return set()
    step = (dr // g, dc // g)
    here = _cells(shape, anchor)
    there = _cells(shape, (anchor[0] + dr, anchor[1] + dc))
    blocked = here | there
    middle = set()
    for r, c in here:
        for t in range(1, g):
            p = (int(r + t * step[0]), int(c + t * step[1]))
            if p not in blocked:
                middle.add(p)
    return middle


def _candidate_ok(
    chain: set[tuple[int, int]],
    success_path: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    protected_landings: set[tuple[int, int]],
) -> bool:
    if chain & (occupied_input | blockers | protected_landings):
        return False
    if success_path & (occupied_input | blockers | protected_landings):
        return False
    return True


def _place_one(
    spec: dict,
    H: int,
    W: int,
    occupied_input: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    protected_landings: set[tuple[int, int]],
) -> dict | None:
    shape = spec["shape"]
    v = spec["vector"]
    stop_kind = spec["stop_kind"]
    needs_crossing = bool(spec["crossing"])

    anchors = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(anchors)
    for start in anchors[:PLACE_RETRIES]:
        chain = _chain_cells(shape, start, v)
        if not _fits(chain, H, W):
            continue

        if stop_kind == "boundary":
            steps = _max_boundary_steps(shape, start, v, H, W)
            if not (1 <= steps <= 5):
                continue
            success_sets = _landings(shape, start, v, steps)
            success_path = set().union(*success_sets)
            stop_cells = set()
        else:
            steps = random.randint(1, 3)
            all_sets = _landings(shape, start, v, steps + 1)
            if not all(_fits(cells, H, W) for cells in all_sets):
                continue
            success_sets = all_sets[:-1]
            success_path = set().union(*success_sets)
            stop_cells = all_sets[-1]
            if stop_cells & (occupied_input | blockers | protected_landings | success_path):
                continue

        if not _candidate_ok(chain, success_path, occupied_input, blockers, protected_landings):
            continue

        new_blockers: set[tuple[int, int]] = set()
        if stop_kind == "obstacle":
            new_blockers.add(random.choice(sorted(stop_cells)))

        crossed = False
        if needs_crossing:
            middle = _intermediate_cells(shape, start, v)
            blocked_places = chain | success_path | stop_cells | occupied_input | blockers | protected_landings
            middle = {p for p in middle if p not in blocked_places and 0 <= p[0] < H and 0 <= p[1] < W}
            if not middle:
                continue
            sample_count = random.randint(1, min(3, len(middle)))
            new_blockers.update(random.sample(sorted(middle), sample_count))
            crossed = True

        if new_blockers & (occupied_input | protected_landings | success_path | chain):
            continue

        final_anchor = (start[0] + steps * v[0], start[1] + steps * v[1])
        return {
            "shape": shape,
            "color": spec["color"],
            "vector": v,
            "start": start,
            "steps": steps,
            "stop_kind": stop_kind,
            "chain": chain,
            "success_path": success_path,
            "final_cells": _cells(shape, final_anchor),
            "new_blockers": new_blockers,
            "crossed": crossed,
        }

    return None


def _render(records: list[dict], blockers: set[tuple[int, int]], H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    for r, c in blockers:
        inp[r, c] = BLOCKER
        out[r, c] = BLOCKER

    for rec in records:
        shape = rec["shape"]
        sr, sc = rec["start"]
        dr, dc = rec["vector"]
        for anchor, color in (
            ((sr - 2 * dr, sc - 2 * dc), AFTERIMAGE),
            ((sr - dr, sc - dc), AFTERIMAGE),
            ((sr, sc), rec["color"]),
        ):
            for r, c in _cells(shape, anchor):
                inp[r, c] = int(color)

        for r, c in rec["final_cells"]:
            out[r, c] = int(rec["color"])
    return inp, out


def _validate_generated(inp: np.ndarray, out: np.ndarray, records: list[dict]) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(records) < 2:
        return False
    if not any(rec["stop_kind"] == "obstacle" for rec in records):
        return False
    if not any(rec["stop_kind"] == "boundary" for rec in records):
        return False
    if not any(rec["crossed"] for rec in records):
        return False
    if not all(rec["steps"] >= 1 for rec in records):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    outputs = [tuple(sorted(rec["final_cells"])) for rec in records]
    if len(set(outputs)) != len(outputs):
        return False
    vectors = [rec["vector"] for rec in records]
    if len({_direction_key(v) for v in vectors}) != len(vectors):
        return False
    if len({v[0] * v[0] + v[1] * v[1] for v in vectors}) != len(vectors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(22, 30)
    W = random.randint(22, 30)
    count = random.choices([2, 3, 4], weights=[4, 5, 1], k=1)[0]

    vectors = _choose_vectors(count)
    if vectors is None:
        return None
    shapes = _choose_shapes(vectors)
    if shapes is None:
        return None

    colors = random.sample(MOVING_COLORS, count)
    specs = []
    for i in range(count):
        if i == 0:
            stop_kind = "obstacle"
            crossing = True
        elif i == 1:
            stop_kind = "boundary"
            crossing = False
        else:
            stop_kind = random.choice(["obstacle", "boundary"])
            crossing = False
        specs.append(
            {
                "shape": shapes[i],
                "vector": vectors[i],
                "color": int(colors[i]),
                "stop_kind": stop_kind,
                "crossing": crossing,
            }
        )

    occupied_input: set[tuple[int, int]] = set()
    protected_landings: set[tuple[int, int]] = set()
    blockers: set[tuple[int, int]] = set()
    records: list[dict] = []

    for spec in specs:
        rec = _place_one(spec, H, W, occupied_input, blockers, protected_landings)
        if rec is None:
            return None
        occupied_input |= rec["chain"]
        protected_landings |= rec["success_path"]
        blockers |= rec["new_blockers"]
        records.append(rec)

    inp, out = _render(records, blockers, H, W)
    if not _validate_generated(inp, out, records):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid afterimage vector jump input")
