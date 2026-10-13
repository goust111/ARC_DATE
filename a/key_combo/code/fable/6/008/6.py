from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

SIZE_CHOICES = (5, 7, 9)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    use8 = int(connectivity) == 8
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                if use8:
                    for dr in (-1, 1):
                        for dc in (-1, 1):
                            nr, nc = rr + dr, cc + dc
                            if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                                seen[nr, nc] = True
                                stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _object_interior(patch: np.ndarray, bg: int) -> np.ndarray:
    mask = (patch != int(bg)).astype(int)
    h, w = patch.shape
    exterior = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] == 0 and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] == 0 and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and mask[nr, nc] == 0 and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return ~exterior


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components_mask(grid == int(color), connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_ray_aligned(src: tuple[int, int], dst: tuple[int, int]) -> bool:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (abs(dr) == abs(dc) and dr != 0)


def _step_toward(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sdr, sdc


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for comp in _find_components(grid, bg):
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        local = coords - np.array([r0, c0], dtype=int)
        patch[local[:, 0], local[:, 1]] = color

        interior = _object_interior(patch, bg)
        holes_local = np.argwhere((patch == bg) & interior)
        assert 1 <= len(holes_local) <= 2, "each object must have one or two enclosed single-pixel holes"

        holes = [(int(r0 + rr), int(c0 + cc)) for rr, cc in holes_local]
        silhouette = np.vstack([coords, holes_local + np.array([r0, c0], dtype=int)])
        centroid_arr = np.rint(silhouette.mean(axis=0)).astype(int)
        centroid = (int(centroid_arr[0]), int(centroid_arr[1]))

        shots = []
        for hole in sorted(holes):
            assert _is_ray_aligned(centroid, hole), "hole must lie on an 8-direction ray from the centroid"
            direction = _step_toward(centroid, hole)
            shots.append({"hole": hole, "direction": direction})

        objects.append(
            {
                "color": color,
                "coords": {tuple(map(int, rc)) for rc in coords},
                "silhouette": {tuple(map(int, rc)) for rc in coords} | set(holes),
                "bbox": (r0, c0, r1, c1),
                "centroid": centroid,
                "shots": shots,
            }
        )
    return objects


def _ray_start(obj: dict, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    rr, cc = obj["centroid"]
    while (rr, cc) in obj["silhouette"]:
        rr += dr
        cc += dc
    return int(rr), int(cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each enclosed single-pixel hole as a cannon muzzle and project that direction onto another object.

    Input structure:
    - The grid has one dominant background color.
    - It contains several separated monochrome solid objects with different colors.
    - Each object contains one or two enclosed background single-pixel holes lying on an 8-direction ray from the object's centroid cell.

    Rule:
    - For each hole, take the 8-direction from the object's centroid cell toward that hole.
    - Starting from the first cell just outside that object's outer boundary in the same direction, paint a ray in the object's color through background cells.
    - Stop at the cell before a hit; if the ray hits another object, change the hit cell on that target object to the background color to carve a new hole there.
    - Original objects, original holes, and all other background cells remain unchanged; newly carved holes do not fire again.

    Key insight:
    - The hole does not mark where to start drawing; it encodes only the firing direction, so the ray must be launched from the object's outer boundary and collisions are resolved against the original objects only.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    out = g.copy()

    for obj in objects:
        color = int(obj["color"])
        for shot in obj["shots"]:
            dr, dc = shot["direction"]
            rr, cc = _ray_start(obj, (dr, dc))
            while _in_bounds(g.shape, rr, cc):
                if int(g[rr, cc]) != bg:
                    out[rr, cc] = bg
                    break
                out[rr, cc] = color
                rr += dr
                cc += dc
    return out


def _bbox_from_center(center: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    half = int(size) // 2
    r, c = center
    return int(r - half), int(c - half), int(r + half), int(c + half)


def _bbox_clear(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _cell_in_bbox(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _trace_from_spec(
    specs: list[dict],
    shape: tuple[int, int],
    src_idx: int,
    direction: tuple[int, int],
) -> dict:
    src = specs[src_idx]
    half = int(src["size"]) // 2
    dr, dc = direction
    rr = int(src["center"][0] + (half + 1) * dr)
    cc = int(src["center"][1] + (half + 1) * dc)
    path: list[tuple[int, int]] = []
    while _in_bounds(shape, rr, cc):
        for dst_idx, dst in enumerate(specs):
            if dst_idx == src_idx:
                continue
            if _cell_in_bbox((rr, cc), dst["bbox"]):
                return {
                    "src_idx": src_idx,
                    "direction": direction,
                    "path": path,
                    "hit_idx": dst_idx,
                    "impact": (rr, cc),
                }
        path.append((rr, cc))
        rr += dr
        cc += dc
    return {
        "src_idx": src_idx,
        "direction": direction,
        "path": path,
        "hit_idx": None,
        "impact": None,
    }


def _render_input(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
        for hole in spec["holes"]:
            grid[hole] = bg
    return grid


def _simulate_specs(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = _render_input(specs, shape, bg)
    out = grid.copy()
    for spec in specs:
        for shot in spec["shots"]:
            for cell in shot["trace"]["path"]:
                out[cell] = int(spec["color"])
            if shot["trace"]["impact"] is not None:
                out[shot["trace"]["impact"]] = bg
    return out


def _semantic_ok(specs: list[dict], shape: tuple[int, int], bg: int) -> bool:
    if len(specs) < 3:
        return False
    if len({int(spec["color"]) for spec in specs}) != len(specs):
        return False

    boxes = [spec["bbox"] for spec in specs]
    for i, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        if not (0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]):
            return False
        if not _bbox_clear(box, boxes[:i] + boxes[i + 1 :], pad=1):
            return False

    hit_count = 0
    miss_count = 0
    used_targets: set[int] = set()
    painted: set[tuple[int, int]] = set()
    impacts: set[tuple[int, int]] = set()
    multi_hole_seen = False
    for spec in specs:
        if not (1 <= len(spec["shots"]) <= 2):
            return False
        if len(spec["shots"]) == 2:
            multi_hole_seen = True
        seen_holes = set()
        seen_dirs = set()
        half = int(spec["size"]) // 2
        cr, cc = spec["center"]
        for shot in spec["shots"]:
            hole = shot["hole"]
            direction = shot["direction"]
            if hole in seen_holes or direction in seen_dirs:
                return False
            seen_holes.add(hole)
            seen_dirs.add(direction)
            if not _cell_in_bbox(hole, spec["bbox"]):
                return False
            if hole[0] in (spec["bbox"][0], spec["bbox"][2]) or hole[1] in (spec["bbox"][1], spec["bbox"][3]):
                return False
            if max(abs(hole[0] - cr), abs(hole[1] - cc)) >= half:
                return False
            tr = shot["trace"]
            if not tr["path"] and tr["impact"] is None:
                return False
            path_set = set(tr["path"])
            if path_set & painted:
                return False
            if path_set & impacts:
                return False
            for cell in tr["path"]:
                if any(_cell_in_bbox(cell, other["bbox"]) for other in specs):
                    return False
            painted |= path_set
            if tr["impact"] is None:
                miss_count += 1
            else:
                hit_count += 1
                if tr["hit_idx"] in used_targets or tr["impact"] in impacts or tr["impact"] in painted:
                    return False
                if not _cell_in_bbox(tr["impact"], specs[tr["hit_idx"]]["bbox"]):
                    return False
                used_targets.add(int(tr["hit_idx"]))
                impacts.add(tr["impact"])
    return hit_count >= 1 and miss_count >= 1 and multi_hole_seen


def _random_center(rng: np.random.Generator, shape: tuple[int, int], size: int) -> tuple[int, int]:
    half = int(size) // 2
    r = int(rng.integers(half + 1, shape[0] - half - 1))
    c = int(rng.integers(half + 1, shape[1] - half - 1))
    return r, c


def _place_specs(rng: np.random.Generator) -> tuple[list[dict], tuple[int, int], int] | None:
    h = int(rng.integers(20, 31))
    w = int(rng.integers(20, 31))
    shape = (h, w)
    bg = int(rng.integers(0, 10))
    palette = [c for c in range(10) if c != bg]
    n_objects = int(rng.integers(3, 6))
    colors = [int(c) for c in rng.choice(palette, size=n_objects, replace=False)]
    sizes = [int(rng.choice(SIZE_CHOICES)) for _ in range(n_objects)]

    specs: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    first_center = _random_center(rng, shape, sizes[0])
    first_box = _bbox_from_center(first_center, sizes[0])
    specs.append({"center": first_center, "size": sizes[0], "color": colors[0], "bbox": first_box})
    boxes.append(first_box)

    aligned_goal = min(n_objects - 1, int(rng.choice([1, 2, 2, 3])))
    placed = 1
    while placed < 1 + aligned_goal:
        size = sizes[placed]
        done = False
        for _ in range(80):
            anchor = specs[int(rng.integers(0, len(specs)))]
            direction = DIRS8[int(rng.integers(0, len(DIRS8)))]
            dist = int(anchor["size"] // 2 + size // 2 + 2 + rng.integers(1, 6))
            center = (
                int(anchor["center"][0] + direction[0] * dist),
                int(anchor["center"][1] + direction[1] * dist),
            )
            box = _bbox_from_center(center, size)
            if not _cell_in_bbox(center, (size // 2, size // 2, h - size // 2 - 1, w - size // 2 - 1)):
                continue
            if not _bbox_clear(box, boxes, pad=1):
                continue
            specs.append({"center": center, "size": size, "color": colors[placed], "bbox": box})
            boxes.append(box)
            placed += 1
            done = True
            break
        if not done:
            return None

    while placed < n_objects:
        size = sizes[placed]
        done = False
        for _ in range(150):
            center = _random_center(rng, shape, size)
            box = _bbox_from_center(center, size)
            if _bbox_clear(box, boxes, pad=1):
                specs.append({"center": center, "size": size, "color": colors[placed], "bbox": box})
                boxes.append(box)
                placed += 1
                done = True
                break
        if not done:
            return None

    return specs, shape, bg


def _choose_shots(specs: list[dict], shape: tuple[int, int], rng: np.random.Generator) -> bool:
    traces_by_obj: list[dict[tuple[int, int], dict]] = []
    both_candidates = []
    for idx in range(len(specs)):
        traces = {direction: _trace_from_spec(specs, shape, idx, direction) for direction in DIRS8}
        traces_by_obj.append(traces)
        hit_dirs = [d for d, tr in traces.items() if tr["hit_idx"] is not None]
        miss_dirs = [d for d, tr in traces.items() if tr["hit_idx"] is None and len(tr["path"]) >= 2]
        if hit_dirs and miss_dirs:
            both_candidates.append(idx)
    if not both_candidates:
        return False

    featured = int(rng.choice(both_candidates))
    hit_dirs = [d for d, tr in traces_by_obj[featured].items() if tr["hit_idx"] is not None]
    miss_dirs = [d for d, tr in traces_by_obj[featured].items() if tr["hit_idx"] is None and len(tr["path"]) >= 2]
    rng.shuffle(hit_dirs)
    rng.shuffle(miss_dirs)

    for spec in specs:
        spec["shots"] = []

    used_targets: set[int] = set()
    painted: set[tuple[int, int]] = set()
    impacts: set[tuple[int, int]] = set()

    def can_add(trace: dict) -> bool:
        path_set = set(trace["path"])
        if path_set & painted or path_set & impacts:
            return False
        if trace["impact"] is not None:
            if trace["hit_idx"] in used_targets or trace["impact"] in impacts or trace["impact"] in painted:
                return False
        return True

    def add_shot(obj_idx: int, direction: tuple[int, int]) -> bool:
        trace = traces_by_obj[obj_idx][direction]
        if not can_add(trace):
            return False
        specs[obj_idx]["shots"].append({"direction": direction, "trace": trace})
        painted.update(trace["path"])
        if trace["impact"] is not None:
            used_targets.add(int(trace["hit_idx"]))
            impacts.add(trace["impact"])
        return True

    featured_ok = False
    for hit_dir, miss_dir in itertools.product(hit_dirs, miss_dirs):
        if add_shot(featured, hit_dir) and add_shot(featured, miss_dir):
            featured_ok = True
            break
        specs[featured]["shots"] = []
        used_targets.clear()
        painted.clear()
        impacts.clear()
    if not featured_ok:
        return False

    order = list(range(len(specs)))
    rng.shuffle(order)
    for idx in order:
        if idx == featured:
            continue
        options = list(DIRS8)
        rng.shuffle(options)
        options.sort(key=lambda d: 0 if traces_by_obj[idx][d]["hit_idx"] is None else 1)
        if any(traces_by_obj[idx][d]["hit_idx"] is not None and can_add(traces_by_obj[idx][d]) for d in options):
            if rng.random() < 0.55:
                options.sort(key=lambda d: 0 if traces_by_obj[idx][d]["hit_idx"] is not None else 1)
        picked = False
        for direction in options:
            trace = traces_by_obj[idx][direction]
            if trace["hit_idx"] is None and len(trace["path"]) < 2:
                continue
            if add_shot(idx, direction):
                picked = True
                break
        if not picked:
            return False
    return True


def _carve_holes(specs: list[dict], rng: np.random.Generator) -> None:
    for spec in specs:
        half = int(spec["size"]) // 2
        used_holes = set()
        for shot in spec["shots"]:
            distance = int(rng.integers(1, half))
            hole = (
                int(spec["center"][0] + shot["direction"][0] * distance),
                int(spec["center"][1] + shot["direction"][1] * distance),
            )
            while hole in used_holes:
                distance = int(rng.integers(1, half))
                hole = (
                    int(spec["center"][0] + shot["direction"][0] * distance),
                    int(spec["center"][1] + shot["direction"][1] * distance),
                )
            used_holes.add(hole)
            shot["hole"] = hole
        spec["holes"] = [shot["hole"] for shot in spec["shots"]]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    placed = _place_specs(rng)
    if placed is None:
        return None
    specs, shape, bg = placed
    if not _choose_shots(specs, shape, rng):
        return None
    _carve_holes(specs, rng)
    if not _semantic_ok(specs, shape, bg):
        return None

    inp = _render_input(specs, shape, bg)
    expected = _simulate_specs(specs, shape, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
