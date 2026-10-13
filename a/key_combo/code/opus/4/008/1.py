from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_008_1",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
find_components = _ARC_UTILS.find_components


BG = 0
DIR_BY_COLOR = {
    1: (0, 1),   # blue -> right
    2: (-1, 0),  # red -> up
    3: (1, 0),   # green -> down
    4: (0, -1),  # yellow -> left
}
ALL_COLORS = tuple(sorted(DIR_BY_COLOR))
GRID_ROWS = (15, 19)
GRID_COLS = (15, 19)
MIN_CLUSTER_GAP = 2
GEN_ATTEMPTS = 500


def _normalize_shape(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in cells}
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape({(c, -r) for r, c in shape})


def _flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape({(r, -c) for r, c in shape})


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    cur = shape
    for _ in range(4):
        variants.add(_normalize_shape(cur))
        variants.add(_flip_shape(cur))
        cur = _rotate_shape(cur)
    return sorted(variants)


def _enumerate_shapes(max_size: int = 5) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    free_shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, max_size + 1):
        grown: set[tuple[tuple[int, int], ...]] = set()
        for shape in free_shapes[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                frontier.update(
                    {
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    }
                )
            frontier -= cells
            for nxt in frontier:
                variants = _shape_variants(_normalize_shape(cells | {nxt}))
                grown.add(min(variants))
        free_shapes[size] = grown

    fixed_shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {}
    for size, shapes in free_shapes.items():
        fixed_shapes[size] = set()
        for shape in shapes:
            fixed_shapes[size].update(_shape_variants(shape))
    return {size: sorted(fixed_shapes[size]) for size in fixed_shapes if size >= 2}


SHAPES_BY_SIZE = _enumerate_shapes()


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _shape_bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _emitters_for_shape(shape: tuple[tuple[int, int], ...], color: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _shape_bbox(shape)
    if color == 2:
        keep = {(r, c) for r, c in shape if r == r0}
    elif color == 1:
        keep = {(r, c) for r, c in shape if c == c1}
    elif color == 3:
        keep = {(r, c) for r, c in shape if r == r1}
    elif color == 4:
        keep = {(r, c) for r, c in shape if c == c0}
    else:
        raise ValueError(f"unsupported color: {color}")
    return sorted(keep)


def _axis_span_for_shape(shape: tuple[tuple[int, int], ...], color: int) -> int:
    r0, c0, r1, c1 = _shape_bbox(shape)
    if color in (2, 3):
        return r1 - r0 + 1
    return c1 - c0 + 1


def _shape_options(
    size: int,
    color: int,
    *,
    unique_emitter: bool | None = None,
    multi_emitter: bool | None = None,
    span_differs_from_area: bool = False,
) -> list[tuple[tuple[int, int], ...]]:
    out = []
    for shape in SHAPES_BY_SIZE[size]:
        emitters = _emitters_for_shape(shape, color)
        if unique_emitter is True and len(emitters) != 1:
            continue
        if multi_emitter is True and len(emitters) < 2:
            continue
        if span_differs_from_area and _axis_span_for_shape(shape, color) == size:
            continue
        out.append(shape)
    return out


def _absolute_coords(shape: tuple[tuple[int, int], ...], top: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, top)
    return sorted((r0 + r, c0 + c) for r, c in shape)


def _make_cluster(color: int, shape: tuple[tuple[int, int], ...], top: tuple[int, int]) -> dict:
    coords = _absolute_coords(shape, top)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    emitters = _absolute_coords(tuple(_emitters_for_shape(shape, color)), top)
    return {
        "color": int(color),
        "shape": shape,
        "top": (int(top[0]), int(top[1])),
        "coords": coords,
        "coord_set": set(coords),
        "bbox": bbox,
        "emitters": emitters,
        "area": len(shape),
    }


def _in_bounds_cluster(cluster: dict, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in cluster["coords"])


def _cluster_gap_ok(a: dict, b: dict, min_gap: int = MIN_CLUSTER_GAP) -> bool:
    for ra, ca in a["coords"]:
        for rb, cb in b["coords"]:
            if max(abs(ra - rb), abs(ca - cb)) <= int(min_gap):
                return False
    return True


def _can_place_cluster(
    cluster: dict,
    shape: tuple[int, int],
    placed: list[dict],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    if not _in_bounds_cluster(cluster, shape):
        return False
    if any(cell in reserved_empty for cell in cluster["coords"]):
        return False
    for other in placed:
        if not _cluster_gap_ok(cluster, other):
            return False
    return True


def _ray_cells(emitter: tuple[int, int], color: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIR_BY_COLOR[int(color)]
    return [(int(emitter[0] + dr * step), int(emitter[1] + dc * step)) for step in range(1, int(length) + 1)]


def _cluster_path_cells(cluster: dict) -> list[tuple[int, int]]:
    cells = []
    for emitter in cluster["emitters"]:
        cells.extend(_ray_cells(emitter, cluster["color"], cluster["area"]))
    return cells


def _full_clear_path_exists(
    cluster: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    h, w = map(int, shape)
    for emitter in cluster["emitters"]:
        ok = True
        for cell in _ray_cells(emitter, cluster["color"], cluster["area"]):
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if cell in occupied or cell in reserved_empty:
                ok = False
                break
        if ok:
            return True
    return False


def _any_visible_path_exists(
    cluster: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    h, w = map(int, shape)
    for emitter in cluster["emitters"]:
        for cell in _ray_cells(emitter, cluster["color"], cluster["area"]):
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                continue
            if cell in occupied or cell in reserved_empty:
                break
            return True
    return False


def _grid_from_clusters(shape: tuple[int, int], clusters: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    for cluster in clusters:
        for r, c in cluster["coords"]:
            grid[r, c] = int(cluster["color"])
    return grid


def _clusters_from_grid(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        colors=ALL_COLORS,
        connectivity=4,
        monochromatic=True,
    )
    records = []
    for comp in comps:
        color = int(comp["color"])
        if color not in DIR_BY_COLOR:
            continue
        coords = sorted((int(r), int(c)) for r, c in np.asarray(comp["coords"], dtype=int))
        bbox = tuple(int(v) for v in comp["bbox"])
        emitters = []
        if color == 2:
            emitters = [cell for cell in coords if cell[0] == bbox[0]]
        elif color == 1:
            emitters = [cell for cell in coords if cell[1] == bbox[3]]
        elif color == 3:
            emitters = [cell for cell in coords if cell[0] == bbox[2]]
        elif color == 4:
            emitters = [cell for cell in coords if cell[1] == bbox[1]]
        records.append(
            {
                "color": color,
                "coords": coords,
                "coord_set": set(coords),
                "bbox": bbox,
                "area": int(comp["area"]),
                "emitters": sorted(emitters),
                "dir": DIR_BY_COLOR[color],
            }
        )
    records.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1], rec["bbox"][2], rec["bbox"][3], rec["color"]))
    for idx, rec in enumerate(records):
        rec["id"] = int(idx)
    return records


def _ray_length(record: dict, mode: str) -> int:
    if mode == "area":
        return int(record["area"])
    if mode == "emitters":
        return int(len(record["emitters"]))
    if mode == "bbox":
        r0, c0, r1, c1 = record["bbox"]
        if record["color"] in (2, 3):
            return int(r1 - r0 + 1)
        return int(c1 - c0 + 1)
    if mode == "constant3":
        return 3
    raise ValueError(f"unknown length mode: {mode}")


def _simulate(grid: np.ndarray, *, order: str = "sorted", length_mode: str = "area") -> dict:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    records = _clusters_from_grid(g)
    if order == "sorted":
        proc = records
    elif order == "reverse":
        proc = list(reversed(records))
    else:
        raise ValueError(f"unknown order: {order}")

    owner: dict[tuple[int, int], tuple[str, int]] = {}
    stats: dict[int, dict] = {}
    for rec in records:
        stats[int(rec["id"])] = {
            "visible": set(),
            "collisions": [],
            "runs": [],
        }
        for cell in rec["coords"]:
            owner[cell] = ("cluster", int(rec["id"]))

    for rec in proc:
        rec_id = int(rec["id"])
        color = int(rec["color"])
        length = _ray_length(rec, length_mode)
        dr, dc = DIR_BY_COLOR[color]
        for emitter in rec["emitters"]:
            drawn = 0
            bounds_skips = 0
            collision = None
            for step in range(1, length + 1):
                rr = int(emitter[0] + dr * step)
                cc = int(emitter[1] + dc * step)
                if not (0 <= rr < h and 0 <= cc < w):
                    bounds_skips += 1
                    continue
                if int(out[rr, cc]) != BG:
                    prev_kind, prev_owner = owner.get((rr, cc), ("cluster", -1))
                    collision = {
                        "at": (rr, cc),
                        "step": int(step),
                        "kind": prev_kind,
                        "owner": int(prev_owner),
                    }
                    stats[rec_id]["collisions"].append(collision)
                    break
                out[rr, cc] = color
                owner[(rr, cc)] = ("ray", rec_id)
                stats[rec_id]["visible"].add((rr, cc))
                drawn += 1
            stats[rec_id]["runs"].append(
                {
                    "emitter": emitter,
                    "drawn": int(drawn),
                    "bounds_skips": int(bounds_skips),
                    "collision": collision,
                }
            )

    return {
        "out": out,
        "records": records,
        "stats": stats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each foreground cluster into a bounded color-directed ray whose length equals the cluster size.

    Input structure:
    - The background is black.
    - Foreground objects are 4-connected monochromatic clusters in blue, red, green, or yellow.
    - Each cluster may have one or several emitter pixels on the bbox edge pointed to by its color.

    Rule:
    - Find every 4-connected foreground cluster and count its pixels.
    - Map color to direction: red up, blue right, green down, yellow left.
    - For every emitter pixel on that cluster's emitting bbox edge, draw the cluster color for exactly N attempted steps, where N is the cluster area; the emitter pixel itself is unchanged.
    - Process clusters by bbox top-left row-major order, and stop an emitter ray before any in-bounds non-background cell; out-of-bounds attempted steps are skipped but still consume length.

    Key insight:
    - The hidden quantity is the cluster's pixel count: it sets the finite ray length, while color picks the direction and earlier rays can block later ones.
    """
    return _simulate(np.asarray(grid, dtype=int), order="sorted", length_mode="area")["out"]


def _place_collision_pair(shape: tuple[int, int]) -> tuple[list[dict], set[tuple[int, int]]] | None:
    h, w = map(int, shape)
    area = random.choice([4, 5])
    down_shapes = _shape_options(area, 3, unique_emitter=True)
    left_shapes = _shape_options(area, 4, unique_emitter=True)
    if not down_shapes or not left_shapes:
        return None

    for _ in range(200):
        a_shape = random.choice(down_shapes)
        b_shape = random.choice(left_shapes)
        a_emit = _emitters_for_shape(a_shape, 3)[0]
        b_emit = _emitters_for_shape(b_shape, 4)[0]
        d_row = random.randint(2, area)
        d_col = random.randint(2, area)

        a_h, a_w = _shape_dims(a_shape)
        b_h, b_w = _shape_dims(b_shape)
        a_r = random.randint(1, max(1, min(h - a_h - area - 2, h // 3)))
        a_c = random.randint(1, max(1, min(w - a_w - area - 2, w // 3)))
        b_r = a_r + a_emit[0] + d_row - b_emit[0]
        b_c = a_c + a_emit[1] + d_col - b_emit[1]

        a_cluster = _make_cluster(3, a_shape, (a_r, a_c))
        b_cluster = _make_cluster(4, b_shape, (b_r, b_c))
        if not _in_bounds_cluster(a_cluster, shape) or not _in_bounds_cluster(b_cluster, shape):
            continue
        if not _cluster_gap_ok(a_cluster, b_cluster):
            continue
        if (a_cluster["bbox"][0], a_cluster["bbox"][1]) >= (b_cluster["bbox"][0], b_cluster["bbox"][1]):
            continue
        if not _full_clear_path_exists(a_cluster, shape, b_cluster["coord_set"], set()):
            continue
        if not _full_clear_path_exists(b_cluster, shape, a_cluster["coord_set"], set()):
            continue

        a_path = _ray_cells(a_cluster["emitters"][0], 3, area)
        b_path = _ray_cells(b_cluster["emitters"][0], 4, area)
        intersection = set(a_path) & set(b_path)
        if len(intersection) != 1:
            continue
        inter = next(iter(intersection))
        if inter in a_cluster["coord_set"] or inter in b_cluster["coord_set"]:
            continue

        reserved = {cell for cell in a_path + b_path if 0 <= cell[0] < h and 0 <= cell[1] < w}
        return [a_cluster, b_cluster], reserved
    return None


def _candidate_tops(
    shape: tuple[int, int],
    grid_shape: tuple[int, int],
    zone: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    h, w = map(int, grid_shape)
    sh, sw = shape
    if zone is None:
        r0, r1, c0, c1 = 0, h - sh, 0, w - sw
    else:
        r0, r1, c0, c1 = zone
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(h - sh, r1)
        c1 = min(w - sw, c1)
    if r0 > r1 or c0 > c1:
        return []
    tops = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(tops)
    return tops


def _place_special_cluster(
    color: int,
    size: int,
    grid_shape: tuple[int, int],
    placed: list[dict],
    reserved_empty: set[tuple[int, int]],
    *,
    zone: tuple[int, int, int, int] | None = None,
    span_differs_from_area: bool = False,
) -> tuple[dict, set[tuple[int, int]]] | None:
    occupied = set().union(*(cluster["coord_set"] for cluster in placed)) if placed else set()
    options = _shape_options(size, color, unique_emitter=True, span_differs_from_area=span_differs_from_area)
    random.shuffle(options)
    for shape in options:
        for top in _candidate_tops(_shape_dims(shape), grid_shape, zone):
            cluster = _make_cluster(color, shape, top)
            if not _can_place_cluster(cluster, grid_shape, placed, reserved_empty):
                continue
            if not _full_clear_path_exists(cluster, grid_shape, occupied, reserved_empty):
                continue
            path_cells = {cell for cell in _cluster_path_cells(cluster) if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]}
            return cluster, path_cells
    return None


def _place_extra_cluster(
    grid_shape: tuple[int, int],
    placed: list[dict],
    reserved_empty: set[tuple[int, int]],
    *,
    multi_emitter: bool,
) -> tuple[dict, set[tuple[int, int]]] | None:
    occupied = set().union(*(cluster["coord_set"] for cluster in placed)) if placed else set()
    sizes = [2, 3, 4, 5]
    random.shuffle(sizes)
    for size in sizes:
        colors = list(ALL_COLORS)
        random.shuffle(colors)
        for color in colors:
            options = _shape_options(size, color, multi_emitter=multi_emitter if multi_emitter else None)
            if not options:
                continue
            random.shuffle(options)
            for shape in options:
                for top in _candidate_tops(_shape_dims(shape), grid_shape):
                    cluster = _make_cluster(color, shape, top)
                    if not _can_place_cluster(cluster, grid_shape, placed, reserved_empty):
                        continue
                    path_cells = {
                        cell
                        for cell in _cluster_path_cells(cluster)
                        if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]
                    }
                    if path_cells & reserved_empty:
                        continue
                    if not _any_visible_path_exists(cluster, grid_shape, occupied, reserved_empty):
                        continue
                    return cluster, path_cells
    return None


def _scene_semantics_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if set(map(int, np.unique(g))) - {BG} - set(ALL_COLORS):
        return False

    truth = _simulate(g, order="sorted", length_mode="area")
    records = truth["records"]
    stats = truth["stats"]
    if not (5 <= len(records) <= 6):
        return False
    if len({rec["color"] for rec in records}) < 4:
        return False
    if any(not (2 <= rec["area"] <= 5) for rec in records):
        return False
    if len({rec["area"] for rec in records}) < 2:
        return False
    if not any(len(rec["emitters"]) >= 2 for rec in records):
        return False

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if not _cluster_gap_ok(records[i], records[j]):
                return False

    pair_ok = False
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["area"] == records[j]["area"] and records[i]["color"] != records[j]["color"]:
                pair_ok = True
    if not pair_ok:
        return False

    if np.array_equal(truth["out"], g):
        return False

    added = int(np.count_nonzero(truth["out"] != g))
    if not (8 <= added <= (g.shape[0] * g.shape[1]) // 4):
        return False

    if any(len(stats[int(rec["id"])]["visible"]) == 0 for rec in records):
        return False
    if not any(
        run["collision"] is None and run["drawn"] > 0
        for rec in records
        for run in stats[int(rec["id"])]["runs"]
    ):
        return False
    if not any(
        collision["kind"] == "ray"
        for rec in records
        for collision in stats[int(rec["id"])]["collisions"]
    ):
        return False

    reverse = _simulate(g, order="reverse", length_mode="area")["out"]
    if np.array_equal(reverse, truth["out"]):
        return False
    emitters_alt = _simulate(g, order="sorted", length_mode="emitters")["out"]
    bbox_alt = _simulate(g, order="sorted", length_mode="bbox")["out"]
    const_alt = _simulate(g, order="sorted", length_mode="constant3")["out"]
    if np.array_equal(emitters_alt, truth["out"]):
        return False
    if np.array_equal(bbox_alt, truth["out"]):
        return False
    if np.array_equal(const_alt, truth["out"]):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    grid_shape = (h, w)

    pair = _place_collision_pair(grid_shape)
    if pair is None:
        return None
    clusters, reserved_empty = pair

    area_ab = clusters[0]["area"]
    area_c = 5 if area_ab == 4 else 4
    c_zone = (h // 2, h - 3, 1, max(2, w // 3))
    c_result = _place_special_cluster(
        1,
        area_c,
        grid_shape,
        clusters,
        reserved_empty,
        zone=c_zone,
        span_differs_from_area=True,
    )
    if c_result is None:
        return None
    c_cluster, c_reserved = c_result
    clusters.append(c_cluster)
    reserved_empty |= c_reserved

    d_size = random.choice([2, 3, 5])
    d_zone = (h // 2, h - 2, max(w // 2, 1), w - 2)
    d_result = _place_special_cluster(
        2,
        d_size,
        grid_shape,
        clusters,
        reserved_empty,
        zone=d_zone,
    )
    if d_result is None:
        return None
    d_cluster, d_reserved = d_result
    clusters.append(d_cluster)
    reserved_empty |= d_reserved

    extra1 = _place_extra_cluster(grid_shape, clusters, reserved_empty, multi_emitter=True)
    if extra1 is None:
        return None
    extra_cluster, extra_reserved = extra1
    clusters.append(extra_cluster)
    reserved_empty |= extra_reserved

    if random.random() < 0.55:
        extra2 = _place_extra_cluster(grid_shape, clusters, reserved_empty, multi_emitter=False)
        if extra2 is not None:
            extra_cluster2, extra_reserved2 = extra2
            clusters.append(extra_cluster2)
            reserved_empty |= extra_reserved2

    grid = _grid_from_clusters(grid_shape, clusters)
    if not _scene_semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cluster-ray input")
