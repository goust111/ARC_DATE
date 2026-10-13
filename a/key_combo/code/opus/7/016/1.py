from __future__ import annotations

import heapq
import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, find_components, in_bounds

        return Color, find_components, in_bounds
    except Exception:
        utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("shared_arc_utils", utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Color, module.find_components, module.in_bounds


Color, find_components, in_bounds = _load_arc_utils()

BG = Color.BLACK
PATH = Color.GRAY
OBJECT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE]
MAX_GENERATION_ATTEMPTS = 300

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
OUTWARD = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}

SYMMETRIC_TEMPLATES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
]

ASYMMETRIC_TEMPLATES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
]


def _normalize_template(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _flip_horizontal(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    width = max(c for _, c in coords) + 1
    return _normalize_template(tuple((r, width - 1 - c) for r, c in coords))


def _flip_vertical(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _ in coords) + 1
    return _normalize_template(tuple((height - 1 - r, c) for r, c in coords))


def _template_variants(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {
        _normalize_template(coords),
        _flip_horizontal(coords),
        _flip_vertical(coords),
        _flip_vertical(_flip_horizontal(coords)),
    }
    return sorted(variants)


def _template_anchor(template: tuple[tuple[int, int], ...], side: str) -> tuple[int, int]:
    if side == "left":
        edge = min(c for _, c in template)
        candidates = [(r, c) for r, c in template if c == edge]
        center = sum(r for r, _ in template) / len(template)
        candidates.sort(key=lambda cell: (abs(cell[0] - center), cell[0], cell[1]))
        return candidates[0]
    if side == "right":
        edge = max(c for _, c in template)
        candidates = [(r, c) for r, c in template if c == edge]
        center = sum(r for r, _ in template) / len(template)
        candidates.sort(key=lambda cell: (abs(cell[0] - center), cell[0], cell[1]))
        return candidates[0]
    if side == "top":
        edge = min(r for r, _ in template)
        candidates = [(r, c) for r, c in template if r == edge]
        center = sum(c for _, c in template) / len(template)
        candidates.sort(key=lambda cell: (abs(cell[1] - center), cell[1], cell[0]))
        return candidates[0]
    if side == "bottom":
        edge = max(r for r, _ in template)
        candidates = [(r, c) for r, c in template if r == edge]
        center = sum(c for _, c in template) / len(template)
        candidates.sort(key=lambda cell: (abs(cell[1] - center), cell[1], cell[0]))
        return candidates[0]
    raise ValueError(f"invalid side: {side}")


def _place_template(
    template: tuple[tuple[int, int], ...],
    side: str,
    target_anchor: tuple[int, int],
) -> list[tuple[int, int]]:
    ar, ac = _template_anchor(template, side)
    base_r = int(target_anchor[0] - ar)
    base_c = int(target_anchor[1] - ac)
    return [(base_r + r, base_c + c) for r, c in template]


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    r1 = int(arr[:, 0].max())
    c1 = int(arr[:, 1].max())
    return r0, c0, r1, c1


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _shape_patch(coords: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return patch


def _is_horizontally_symmetric(coords: list[tuple[int, int]] | np.ndarray) -> bool:
    patch = _shape_patch(coords)
    return bool(np.array_equal(patch, np.flipud(patch)))


def _expand8(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if in_bounds(shape, nr, nc):
                expanded.add((nr, nc))
    return expanded


def _find_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(end)
    pq: list[tuple[int, int, tuple[int, int]]] = [(0, 0, start)]
    best_cost = {start: 0}
    parent = {start: None}
    while pq:
        _score, cost, cell = heapq.heappop(pq)
        if cell == end:
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        if cost != best_cost.get(cell):
            continue
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not in_bounds(shape, nr, nc) or nxt in blocked:
                continue
            new_cost = cost + 1
            if new_cost >= best_cost.get(nxt, 10**9):
                continue
            best_cost[nxt] = new_cost
            parent[nxt] = cell
            heuristic = abs(nr - end[0]) + abs(nc - end[1])
            heapq.heappush(pq, (new_cost + heuristic, new_cost, nxt))
    return None


def _route_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    waypoints: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    stops = [start, *waypoints, end]
    built: list[tuple[int, int]] = []
    own_cells: set[tuple[int, int]] = set()
    for a, b in zip(stops, stops[1:]):
        seg_blocked = set(blocked)
        seg_blocked -= own_cells
        seg = _find_path(shape, a, b, seg_blocked)
        if seg is None:
            return None
        if built:
            seg = seg[1:]
        built.extend(seg)
        own_cells.update(seg)
    return built


def _analyze(grid: np.ndarray) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    objects = find_components(g, bg=BG, connectivity=4, monochromatic=True, colors=OBJECT_COLORS)
    paths = find_components(g, bg=BG, connectivity=8, monochromatic=True, colors=[PATH])
    if not objects:
        raise ValueError("no objects found")
    if not paths:
        raise ValueError("no paths found")

    objects = sorted(objects, key=lambda comp: (comp["bbox"], comp["color"]))
    obj_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(objects):
        coords = np.asarray(comp["coords"], dtype=int)
        comp["id"] = idx
        comp["coords"] = coords
        comp["bbox"] = tuple(map(int, comp["bbox"]))
        comp["center"] = tuple(coords.mean(axis=0))
        comp["symmetric"] = _is_horizontally_symmetric(coords)
        if not 3 <= int(comp["area"]) <= 6:
            raise ValueError("object area out of range")
        for r, c in map(tuple, coords):
            obj_lookup[(int(r), int(c))] = idx

    if len({int(comp["color"]) for comp in objects}) != len(objects):
        raise ValueError("object colors must be unique")

    pairs: list[tuple[int, int]] = []
    degree = [0] * len(objects)
    for path in paths:
        touched: set[int] = set()
        coords = np.asarray(path["coords"], dtype=int)
        path["coords"] = coords
        for r, c in map(tuple, coords):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS8:
                obj_id = obj_lookup.get((rr + dr, cc + dc))
                if obj_id is not None:
                    touched.add(obj_id)
        if len(touched) != 2:
            raise ValueError("every path must touch exactly two objects")
        a, b = sorted(touched)
        if objects[a]["symmetric"] == objects[b]["symmetric"]:
            raise ValueError("path does not connect one sender and one receiver")
        sender = a if objects[a]["symmetric"] else b
        receiver = b if sender == a else a
        degree[sender] += 1
        degree[receiver] += 1
        pairs.append((sender, receiver))

    if any(d != 1 for d in degree):
        raise ValueError("each object must participate in exactly one pair")
    if len(pairs) * 2 != len(objects):
        raise ValueError("pair count mismatch")
    return objects, paths, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each symmetric sender's color onto its path-linked receiver, then mark the receiver with its original outer ring.

    Input structure:
    - The background is black.
    - There are 4 to 6 small 4-connected monochrome objects in distinct non-gray colors.
    - Gray single-cell paths form disjoint 8-connected connectors between object pairs.
    - In each pair, exactly one object is horizontally symmetric inside its own bounding box.

    Rule:
    - Detect horizontal symmetry in each object's cropped bounding box; the symmetric object is the sender and the non-symmetric one is the receiver.
    - Use the gray path connectivity to determine which sender is paired with which receiver.
    - Recolor every receiver pixel to the paired sender's color.
    - Paint the one-cell outer border of the receiver's expanded bounding box using the receiver's original color.
    - Erase all gray path pixels and erase all sender objects.

    Key insight:
    - Symmetry decides direction, but the gray path decides pairing, so each surviving object keeps two identities at once: the sender's color inside and its own original color on the outer ring.
    """
    g = np.asarray(grid, dtype=int)
    objects, _paths, pairs = _analyze(g)
    out = g.copy()
    out[out == PATH] = BG

    for sender, _receiver in pairs:
        for r, c in map(tuple, objects[sender]["coords"]):
            out[int(r), int(c)] = BG

    for sender, receiver in pairs:
        sender_color = int(objects[sender]["color"])
        receiver_color = int(objects[receiver]["color"])
        for r, c in map(tuple, objects[receiver]["coords"]):
            out[int(r), int(c)] = sender_color
        r0, c0, r1, c1 = objects[receiver]["bbox"]
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if not in_bounds(out.shape, r, c):
                    continue
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                out[r, c] = receiver_color
    return out


def _sample_template(
    base_templates: list[tuple[tuple[int, int], ...]],
    used: set[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...] | None:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for template in base_templates:
        candidates.extend(_template_variants(template))
    random.shuffle(candidates)
    for candidate in candidates:
        if candidate in used:
            continue
        used.add(candidate)
        return candidate
    return None


def _slot_definitions(h: int, w: int, n_pairs: int) -> dict[str, dict]:
    left_col = random.choice([3, 4])
    right_col = w - random.choice([4, 5])
    center_col = w // 2 + random.choice([-1, 0, 1])
    top_row = random.randint(3, 4)
    upper_center = h // 2 - random.choice([3, 4])
    lower_center = h // 2 + random.choice([3, 4])
    slots = {
        "lt": {"anchor": (top_row, left_col), "side": "right"},
        "rt": {"anchor": (top_row, right_col), "side": "left"},
        "ct": {"anchor": (upper_center, center_col), "side": "bottom"},
        "cb": {"anchor": (lower_center, center_col), "side": "top"},
    }
    if n_pairs == 3:
        bottom_row = random.randint(h - 5, h - 4)
        slots["lb"] = {"anchor": (bottom_row, left_col), "side": "right"}
        slots["rb"] = {"anchor": (bottom_row, right_col), "side": "left"}
    return slots


def _build_objects(h: int, w: int, n_pairs: int) -> tuple[list[dict], dict[str, dict]] | None:
    slots = _slot_definitions(h, w, n_pairs)
    pair_specs = [
        ("lt", "rt", "top"),
        ("ct", "cb", "center"),
    ]
    if n_pairs == 3:
        pair_specs.append(("lb", "rb", "bottom"))

    colors = random.sample(OBJECT_COLORS, 2 * n_pairs)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    objects: list[dict] = []
    color_index = 0

    for pair_index, (slot_a, slot_b, route_key) in enumerate(pair_specs):
        sender_on_a = bool(random.getrandbits(1))
        sender_template = _sample_template(SYMMETRIC_TEMPLATES, used_shapes)
        receiver_template = _sample_template(ASYMMETRIC_TEMPLATES, used_shapes)
        if sender_template is None or receiver_template is None:
            return None

        for slot_name, template, is_sender in [
            (slot_a, sender_template if sender_on_a else receiver_template, sender_on_a),
            (slot_b, receiver_template if sender_on_a else sender_template, not sender_on_a),
        ]:
            slot = slots[slot_name]
            coords = _place_template(template, slot["side"], slot["anchor"])
            if not all(in_bounds((h, w), r, c) for r, c in coords):
                return None
            bbox = _bbox(coords)
            start_r = slot["anchor"][0] + OUTWARD[slot["side"]][0]
            start_c = slot["anchor"][1] + OUTWARD[slot["side"]][1]
            if not in_bounds((h, w), start_r, start_c):
                return None
            obj = {
                "slot": slot_name,
                "pair_index": pair_index,
                "route_key": route_key,
                "coords": coords,
                "bbox": bbox,
                "color": colors[color_index],
                "symmetric": is_sender,
                "anchor": slot["anchor"],
                "side": slot["side"],
                "path_cell": (start_r, start_c),
            }
            color_index += 1
            for prior in objects:
                if _bbox_overlap(obj["bbox"], prior["bbox"], margin=2):
                    return None
            objects.append(obj)

    return objects, slots


def _build_grid_and_paths(h: int, w: int, n_pairs: int) -> np.ndarray | None:
    built = _build_objects(h, w, n_pairs)
    if built is None:
        return None
    objects, _slots = built
    grid = np.zeros((h, w), dtype=int)
    object_cells = {cell for obj in objects for cell in obj["coords"]}
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r, c] = int(obj["color"])

    pair_to_objs: dict[int, list[dict]] = {}
    for obj in objects:
        pair_to_objs.setdefault(int(obj["pair_index"]), []).append(obj)

    routed_paths: list[list[tuple[int, int]]] = []
    route_order = [0, 1, 2] if n_pairs == 3 else [0, 1]
    for pair_index in route_order:
        pair_objects = pair_to_objs[pair_index]
        a_obj, b_obj = sorted(pair_objects, key=lambda obj: obj["slot"])
        start = a_obj["path_cell"]
        end = b_obj["path_cell"]
        if a_obj["route_key"] in {"top", "bottom"}:
            if start[0] != end[0]:
                return None
            step = 1 if end[1] >= start[1] else -1
            path = [(start[0], c) for c in range(start[1], end[1] + step, step)]
        else:
            if start[1] != end[1]:
                return None
            step = 1 if end[0] >= start[0] else -1
            path = [(r, start[1]) for r in range(start[0], end[0] + step, step)]
        if any(cell in object_cells for cell in path):
            return None
        new_path_moat = _expand8((h, w), set(path))
        for prev in routed_paths:
            if new_path_moat & set(prev):
                return None
        for obj in objects:
            if obj["pair_index"] == pair_index:
                continue
            if new_path_moat & set(obj["coords"]):
                return None
        routed_paths.append(path)

    for path in routed_paths:
        for r, c in path:
            grid[r, c] = PATH
    return grid


def _scene_is_valid(inp: np.ndarray, expected_pairs: int) -> bool:
    try:
        objects, paths, pairs = _analyze(inp)
    except Exception:
        return False
    if len(objects) != expected_pairs * 2 or len(paths) != expected_pairs or len(pairs) != expected_pairs:
        return False
    if sum(int(obj["symmetric"]) for obj in objects) != expected_pairs:
        return False
    if not any(abs(float(objects[sender]["center"][0]) - float(objects[receiver]["center"][0])) >= 3.0 for sender, receiver in pairs):
        return False
    out = main(inp)
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(out == PATH)) != 0:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([2, 2, 3])
    if n_pairs == 2:
        h = random.randint(22, 28)
        w = random.randint(20, 29)
    else:
        h = random.randint(24, 29)
        w = random.randint(20, 29)
    grid = _build_grid_and_paths(h, w, n_pairs)
    if grid is None:
        return None
    if not _scene_is_valid(grid, n_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle instance")
