from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_NAMES = tuple(DIRS)
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
GENERATION_ATTEMPTS = 800
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_masks(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if seen[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(coords)
    return comps


def _same_color_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _hole_infos_for_component(
    shape: tuple[int, int],
    coords: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    del shape
    r0, c0, r1, c1 = bbox
    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    obj = np.zeros((local_h, local_w), dtype=bool)
    for r, c in coords:
        obj[r - r0, c - c0] = True

    holes: list[dict] = []
    for comp in _component_masks(~obj):
        touches_border = any(
            lr == 0 or lc == 0 or lr == local_h - 1 or lc == local_w - 1
            for lr, lc in comp
        )
        if touches_border:
            continue
        global_coords = [(r0 + lr, c0 + lc) for lr, lc in comp]
        hb = _bbox_from_coords(global_coords)
        holes.append({"coords": global_coords, "coord_set": set(global_coords), "bbox": hb})
    return holes


def _direction_from_hole(
    obj_bbox: tuple[int, int, int, int], hole_bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = obj_bbox
    hr0, hc0, hr1, hc1 = hole_bbox
    dr2 = (hr0 + hr1) - (r0 + r1)
    dc2 = (hc0 + hc1) - (c0 + c1)
    assert (dr2 == 0) ^ (dc2 == 0), "hole center must be offset along exactly one axis"
    if dr2 < 0:
        return "up"
    if dr2 > 0:
        return "down"
    if dc2 < 0:
        return "left"
    assert dc2 > 0
    return "right"


def _build_objects(g: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for comp in _same_color_components(g, bg):
        holes = _hole_infos_for_component(g.shape, comp["coords"], comp["bbox"])
        assert len(holes) <= 1, "each object may contain at most one enclosed hole"
        hole = None
        if holes:
            hole = holes[0]
            hole["direction"] = _direction_from_hole(comp["bbox"], hole["bbox"])
        obj = dict(comp)
        obj["coord_set"] = set(comp["coords"])
        obj["hole"] = hole
        objects.append(obj)
    return objects


def _find_seed_and_initial(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], int]:
    counts = Counter(map(int, g.ravel()))
    seed_hits: list[tuple[int, tuple[int, int], int]] = []

    for comp in _same_color_components(g, bg):
        if len(comp["coords"]) != 1:
            continue
        color = int(comp["color"])
        if counts[color] != 1:
            continue

        sr, sc = comp["coords"][0]
        erased = g.copy()
        erased[sr, sc] = bg
        objects = _build_objects(erased, bg)
        containing = [
            oid
            for oid, obj in enumerate(objects)
            if obj["hole"] is not None and (sr, sc) in obj["hole"]["coord_set"]
        ]
        if len(containing) == 1:
            seed_hits.append((color, (sr, sc), containing[0]))

    assert len(seed_hits) == 1, "expected one seed pixel inside one object hole"
    return seed_hits[0]


def _ray_start(obj: dict) -> tuple[int, int, str]:
    r0, c0, r1, c1 = obj["bbox"]
    direction = obj["hole"]["direction"]
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, (
        "firing object bbox center must lie on a grid cell"
    )
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if direction == "up":
        return r0 - 1, cc, direction
    if direction == "down":
        return r1 + 1, cc, direction
    if direction == "left":
        return cr, c0 - 1, direction
    return cr, c1 + 1, direction


def _id_map(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    ids = np.full(shape, -1, dtype=int)
    for oid, obj in enumerate(objects):
        for r, c in obj["coords"]:
            ids[r, c] = oid
    return ids


def main(grid: np.ndarray) -> np.ndarray:
    """Relay the seed-colored signal along each ignited object's off-center hole direction.

    Input structure:
    - A black background contains several separated monochrome 4-connected objects.
    - Some objects enclose one background hole; the hole center is offset from the object's bbox center along up, down, left, or right.
    - Exactly one holed object's hole contains a singleton seed color that is not used by any object.
    Rule:
    - Erase the seed pixel to background and fire seed-colored one-cell rays only from the seed's object and later hit holed objects.
    - A ray starts outside the firing object's bbox in the direction from bbox center to hole center, paints background until it reaches another object, and stops before that object's boundary.
    - Each hit object is recolored to the seed color while its hole stays background; if that object has a hole it relays the signal from its own hole direction, otherwise the chain stops.
    Key insight:
    - Holed objects are only aiming devices; the active relay chain is uniquely determined by the one seed and by recomputing the hidden hole direction at every ignited station.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = BLACK

    seed_color, seed_coord, initial_id = _find_seed_and_initial(g, bg)
    base = g.copy()
    base[seed_coord] = bg
    objects = _build_objects(base, bg)
    assert initial_id < len(objects)
    assert objects[initial_id]["hole"] is not None
    assert seed_coord in objects[initial_id]["hole"]["coord_set"]

    ids = _id_map(g.shape, objects)
    out = g.copy()
    out[seed_coord] = bg

    current = initial_id
    hit_once: set[int] = set()
    while True:
        obj = objects[current]
        if obj["hole"] is None:
            break

        r, c, direction = _ray_start(obj)
        dr, dc = DIRS[direction]
        hit_id = -1
        while _in_bounds(g.shape, r, c):
            oid = int(ids[r, c])
            if oid >= 0:
                hit_id = oid
                break
            out[r, c] = seed_color
            r += dr
            c += dc

        if hit_id < 0:
            break
        assert hit_id != current and hit_id not in hit_once, "relay chain must not loop"
        hit_once.add(hit_id)

        for rr, cc in objects[hit_id]["coords"]:
            out[rr, cc] = seed_color

        if objects[hit_id]["hole"] is None:
            break
        current = hit_id

    return out


# --------------- generator ---------------


def _choice(seq):
    return random.choice(tuple(seq))


def _holed_size() -> tuple[int, int]:
    return _choice(((7, 7), (7, 9), (9, 7)))


def _solid_size() -> tuple[int, int]:
    return _choice(((3, 3), (3, 5), (5, 3), (5, 5)))


def _bbox_from_center(cr: int, cc: int, h: int, w: int) -> tuple[int, int, int, int]:
    return cr - h // 2, cc - w // 2, cr + h // 2, cc + w // 2


def _bbox_inside(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _clear_bbox(objects: list[dict], bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    expanded = _expanded_bbox(bbox, pad)
    return all(not _bboxes_overlap(expanded, obj["bbox"]) for obj in objects)


def _hole_cells_for_bbox(bbox: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    hr, hc = cr, cc
    dr, dc = DIRS[direction]
    hr += dr
    hc += dc
    return {(r, c) for r in range(hr - 1, hr + 2) for c in range(hc - 1, hc + 2)}


def _object_cells(obj: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if obj["hole_dir"] is None:
        return cells
    holes = _hole_cells_for_bbox(obj["bbox"], obj["hole_dir"])
    return [(r, c) for r, c in cells if (r, c) not in holes]


def _hole_center_for_bbox(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    dr, dc = DIRS[direction]
    return cr + dr, cc + dc


def _new_object(
    bbox: tuple[int, int, int, int],
    color: int,
    hole_dir: str | None,
    role: str,
) -> dict:
    obj = {"bbox": bbox, "color": color, "hole_dir": hole_dir, "role": role}
    obj["coords"] = _object_cells(obj)
    return obj


def _place_target(
    shape: tuple[int, int],
    objects: list[dict],
    source: dict,
    direction: str,
    color: int,
    holed: bool,
    role: str,
) -> dict | None:
    sr0, sc0, sr1, sc1 = source["bbox"]
    source_cr = (sr0 + sr1) // 2
    source_cc = (sc0 + sc1) // 2

    for _ in range(PLACEMENT_ATTEMPTS):
        h, w = _holed_size() if holed else _solid_size()
        gap = random.randint(2, 5)

        if direction in ("left", "right"):
            delta = random.randint(-min(2, h // 2), min(2, h // 2))
            cr = source_cr + delta
            if direction == "right":
                c0 = sc1 + 1 + gap
                c1 = c0 + w - 1
            else:
                c1 = sc0 - 1 - gap
                c0 = c1 - w + 1
            bbox = (cr - h // 2, c0, cr + h // 2, c1)
        else:
            delta = random.randint(-min(2, w // 2), min(2, w // 2))
            cc = source_cc + delta
            if direction == "down":
                r0 = sr1 + 1 + gap
                r1 = r0 + h - 1
            else:
                r1 = sr0 - 1 - gap
                r0 = r1 - h + 1
            bbox = (r0, cc - w // 2, r1, cc + w // 2)

        if not _bbox_inside(shape, bbox):
            continue
        if not _clear_bbox(objects, bbox, pad=1):
            continue
        hole_dir = _choice(DIR_NAMES) if holed else None
        return _new_object(bbox, color, hole_dir, role)
    return None


def _edge_direction_is_clear(
    shape: tuple[int, int], objects: list[dict], source: dict, direction: str
) -> bool:
    r, c, _direction = _ray_start({"bbox": source["bbox"], "hole": {"direction": direction}})
    dr, dc = DIRS[direction]
    object_cells = {cell for obj in objects for cell in obj["coords"]}
    steps = 0
    while _in_bounds(shape, r, c):
        if (r, c) in object_cells:
            return False
        steps += 1
        r += dr
        c += dc
    return steps >= 3


def _render_input(
    shape: tuple[int, int], objects: list[dict], seed_color: int, seed_object_id: int
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r, c] = int(obj["color"])

    seed_obj = objects[seed_object_id]
    assert seed_obj["hole_dir"] is not None
    sr, sc = _hole_center_for_bbox(seed_obj["bbox"], seed_obj["hole_dir"])
    grid[sr, sc] = seed_color
    return grid


def _plan_id_map(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    ids = np.full(shape, -1, dtype=int)
    for oid, obj in enumerate(objects):
        for r, c in obj["coords"]:
            ids[r, c] = oid
    return ids


def _plan_expected(
    inp: np.ndarray, objects: list[dict], seed_object_id: int, seed_color: int
) -> tuple[np.ndarray, list[int], str]:
    out = inp.copy()
    seed_obj = objects[seed_object_id]
    sr, sc = _hole_center_for_bbox(seed_obj["bbox"], seed_obj["hole_dir"])
    out[sr, sc] = BLACK

    ids = _plan_id_map(inp.shape, objects)
    current = seed_object_id
    hit_order: list[int] = []
    hit_set: set[int] = set()
    end_kind = "edge"
    while True:
        direction = objects[current]["hole_dir"]
        assert direction is not None
        r, c, _ = _ray_start({"bbox": objects[current]["bbox"], "hole": {"direction": direction}})
        dr, dc = DIRS[direction]

        hit_id = -1
        while _in_bounds(inp.shape, r, c):
            oid = int(ids[r, c])
            if oid >= 0:
                hit_id = oid
                break
            out[r, c] = seed_color
            r += dr
            c += dc

        if hit_id < 0:
            end_kind = "edge"
            break
        if hit_id == current or hit_id in hit_set:
            raise ValueError("invalid loop in generated relay chain")
        hit_set.add(hit_id)
        hit_order.append(hit_id)

        for rr, cc in objects[hit_id]["coords"]:
            out[rr, cc] = seed_color

        if objects[hit_id]["hole_dir"] is None:
            end_kind = "solid"
            break
        current = hit_id

    return out, hit_order, end_kind


def _validate_generated(
    inp: np.ndarray,
    objects: list[dict],
    seed_color: int,
    seed_object_id: int,
    intended_chain: list[int],
    intended_end: str,
) -> bool:
    try:
        expected, hit_order, end_kind = _plan_expected(inp, objects, seed_object_id, seed_color)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if hit_order != intended_chain or end_kind != intended_end:
        return False
    if len(hit_order) < 1 or objects[hit_order[0]]["hole_dir"] is None:
        return False

    seed_obj = objects[seed_object_id]
    sr, sc = _hole_center_for_bbox(seed_obj["bbox"], seed_obj["hole_dir"])
    if int(actual[sr, sc]) != BLACK:
        return False
    for oid, obj in enumerate(objects):
        if oid in hit_order:
            continue
        for r, c in obj["coords"]:
            if int(actual[r, c]) != int(inp[r, c]):
                return False

    changed_to_seed = int(np.sum((actual == seed_color) & (inp != seed_color)))
    return changed_to_seed >= 12


def _add_distractors(
    shape: tuple[int, int], objects: list[dict], colors: list[int]
) -> bool:
    random.shuffle(colors)
    required = [True, False]
    if random.random() < 0.65:
        required.append(True)
    if random.random() < 0.45:
        required.append(False)

    for holed in required:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            h, w = _holed_size() if holed else _solid_size()
            cr = random.randint(h // 2 + 1, shape[0] - h // 2 - 2)
            cc = random.randint(w // 2 + 1, shape[1] - w // 2 - 2)
            bbox = _bbox_from_center(cr, cc, h, w)
            if not _bbox_inside(shape, bbox):
                continue
            if not _clear_bbox(objects, bbox, pad=1):
                continue
            hole_dir = _choice(DIR_NAMES) if holed else None
            color = colors.pop() if colors else random.randint(1, 9)
            objects.append(_new_object(bbox, color, hole_dir, "dormant"))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    shape = (h, w)
    palette = list(range(1, 10))
    random.shuffle(palette)
    seed_color = palette.pop()

    relay_count = random.choice((1, 1, 2))
    endpoint_hits_solid = random.random() < 0.68
    chain_station_count = 1 + relay_count
    needed_object_colors = chain_station_count + (1 if endpoint_hits_solid else 0) + 4
    if len(palette) < needed_object_colors:
        return None
    colors = [palette.pop() for _ in range(needed_object_colors)]

    objects: list[dict] = []
    ih, iw = _holed_size()
    initial_cr = random.randint(ih // 2 + 4, h - ih // 2 - 5)
    initial_cc = random.randint(iw // 2 + 4, w - iw // 2 - 5)
    initial_bbox = _bbox_from_center(initial_cr, initial_cc, ih, iw)
    objects.append(_new_object(initial_bbox, colors[0], "up", "chain"))

    incoming_dir: str | None = None
    current_id = 0
    intended_chain: list[int] = []

    for segment in range(chain_station_count):
        source = objects[current_id]
        target_exists = segment < chain_station_count - 1 or endpoint_hits_solid
        target_holed = segment < chain_station_count - 1

        directions = list(DIR_NAMES)
        random.shuffle(directions)
        if incoming_dir is not None and OPPOSITE[incoming_dir] in directions:
            directions.remove(OPPOSITE[incoming_dir])
            directions.append(OPPOSITE[incoming_dir])

        placed_target = None
        chosen_dir = None
        for direction in directions:
            if incoming_dir is not None and direction == OPPOSITE[incoming_dir]:
                continue
            if target_exists:
                color = colors[len(objects)]
                role = "chain" if target_holed else "terminal"
                candidate = _place_target(shape, objects, source, direction, color, target_holed, role)
                if candidate is None:
                    continue
                placed_target = candidate
                chosen_dir = direction
                break
            if _edge_direction_is_clear(shape, objects, source, direction):
                chosen_dir = direction
                break

        if chosen_dir is None:
            return None
        objects[current_id]["hole_dir"] = chosen_dir
        objects[current_id]["coords"] = _object_cells(objects[current_id])

        if not target_exists:
            break

        assert placed_target is not None
        objects.append(placed_target)
        new_id = len(objects) - 1
        intended_chain.append(new_id)
        incoming_dir = chosen_dir
        current_id = new_id
        if not target_holed:
            break

    intended_end = "solid" if endpoint_hits_solid else "edge"
    if endpoint_hits_solid and objects[intended_chain[-1]]["hole_dir"] is not None:
        return None
    if not endpoint_hits_solid and objects[current_id]["hole_dir"] is None:
        return None

    remaining_colors = colors[len(objects) :]
    if not _add_distractors(shape, objects, remaining_colors):
        return None

    inp = _render_input(shape, objects, seed_color, 0)
    counts = np.bincount(inp.ravel(), minlength=10)
    if counts[BLACK] <= max(counts[1:]):
        return None
    if not _validate_generated(inp, objects, seed_color, 0, intended_chain, intended_end):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/3 relay-chain input")
