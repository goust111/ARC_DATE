import numpy as np
from collections import deque


COLOR_ORDER = (2, 1, 3, 4)
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_FROM_VEC = {vec: name for name, vec in DIRS.items()}
SHAPE_COUNTS = (4, 5, 6, 7, 8)
GRID_MIN = 18
GRID_MAX = 26
MAX_GENERATION_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        queue = deque([(r, c)])
        visited[r, c] = True
        coords = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _tail_analysis(coords: np.ndarray) -> dict:
    cells = {tuple(map(int, p)) for p in coords}
    centroid = coords.mean(axis=0)
    candidates = []
    for cell in cells:
        nbrs = [nbr for nbr in _neighbors4(*cell) if nbr in cells]
        if len(nbrs) != 1:
            continue
        remain = cells - {cell}
        if remain and _connected(remain):
            dr = float(cell[0]) - float(centroid[0])
            dc = float(cell[1]) - float(centroid[1])
            dist2 = dr * dr + dc * dc
            candidates.append((dist2, cell, nbrs[0]))
    if not candidates:
        raise ValueError("shape has no removable endpoint tail")

    best_dist = max(item[0] for item in candidates)
    best = [item for item in candidates if abs(item[0] - best_dist) < 1e-9]
    best.sort(key=lambda item: item[1])
    _, tail, body_nbr = best[0]
    dr = int(tail[0] - body_nbr[0])
    dc = int(tail[1] - body_nbr[1])
    direction = DIR_FROM_VEC[(dr, dc)]
    return {
        "tail": tail,
        "direction": direction,
        "all_candidates": [item[1] for item in candidates],
        "best_candidates": [item[1] for item in best],
        "centroid": centroid,
    }


def _make_patch(body_cells, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(body_cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in body_cells:
        patch[r - r0, c - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _destination_top_left(tail, direction: str, distance: int, patch_shape, body_bbox):
    tr, tc = map(int, tail)
    r0, c0, _r1, _c1 = body_bbox
    h, w = map(int, patch_shape)
    if direction == "R":
        return int(r0), int(tc + distance)
    if direction == "L":
        return int(r0), int(tc - distance - w + 1)
    if direction == "D":
        return int(tr + distance), int(c0)
    return int(tr - distance - h + 1), int(c0)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        for coords in _components_of_color(g, color):
            analysis = _tail_analysis(coords)
            tail = analysis["tail"]
            direction = analysis["direction"]
            cells = {tuple(map(int, p)) for p in coords}
            body_cells = sorted(cells - {tail})
            patch, body_bbox = _make_patch(body_cells, color)
            dst_r0, dst_c0 = _destination_top_left(
                tail,
                direction,
                len(cells),
                patch.shape,
                body_bbox,
            )
            objects.append(
                {
                    "color": int(color),
                    "count": int(len(cells)),
                    "tail": tuple(map(int, tail)),
                    "direction": direction,
                    "coords": sorted(cells),
                    "body_cells": body_cells,
                    "patch": patch,
                    "body_bbox": body_bbox,
                    "dst": (int(dst_r0), int(dst_c0)),
                    "best_tail_count": len(analysis["best_candidates"]),
                }
            )
    return objects


def _priority_key(obj: dict):
    return (int(obj["count"]), -COLOR_RANK.get(int(obj["color"]), 999))


def _overlay_patches(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in sorted(objects, key=_priority_key):
        patch = obj["patch"]
        r0, c0 = obj["dst"]
        h, w = patch.shape
        for rr in range(h):
            for cc in range(w):
                if patch[rr, cc] != 0:
                    out[r0 + rr, c0 + cc] = patch[rr, cc]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each tailed shape's cropped body block in its tail direction, with larger shapes winning overlaps.

    Input structure:
    - Black (0) background contains 2 or 3 separated 4-connected single-color shapes.
    - Each shape has 4 to 8 pixels and exactly one removable endpoint pixel that acts as its tail.
    - Colors come from red, blue, green, and yellow, and different shapes use different colors.

    Rule:
    - For each shape, find the tail pixel: a degree-1 pixel whose removal leaves the rest 4-connected; if several exist, choose the one farthest from the shape centroid.
    - Remove that tail, crop the remaining same-color pixels to their tight bounding box, and keep black holes inside the crop as transparent cells.
    - Launch this cropped rectangle in the tail direction by a distance equal to the full shape's pixel count, placing the rectangle's emitting side that far away from the tail.
    - Erase all original shapes and draw only the launched crops; when launched colored pixels overlap, the shape with more total pixels wins, and equal counts break by the fixed color order red, blue, green, yellow.

    Key insight:
    - The same quantity, a shape's total pixel count, controls both how far its cropped body travels and which color survives when multiple launched crops collide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _extract_objects(g)
    return _overlay_patches(g.shape, objects)


def _random_body(rng: np.random.Generator, size: int) -> tuple[tuple[int, int], ...]:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            frontier = list(frontier)
            rng.shuffle(frontier)
            added = False
            for cand in frontier:
                trial = cells | {cand}
                nr0, nc0, nr1, nc1 = _bbox(trial)
                if nr1 - nr0 + 1 <= 4 and nc1 - nc0 + 1 <= 4:
                    cells = trial
                    added = True
                    break
            if not added:
                break
        if len(cells) != size:
            continue
        body = _normalize(cells)
        if len({r for r, _ in body}) >= 2 and len({c for _, c in body}) >= 2:
            return body
    raise RuntimeError("failed to generate compact body")


def _generate_shape_spec(rng: np.random.Generator, color: int, count: int) -> dict:
    direction = str(rng.choice(tuple(DIRS)))
    dr, dc = DIRS[direction]
    body_size = count - 1
    for _ in range(500):
        body = _random_body(rng, body_size)
        rows = [r for r, _ in body]
        cols = [c for _, c in body]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if direction == "R":
            attach = [cell for cell in body if cell[1] == max_c and (cell[0] + dr, cell[1] + dc) not in body]
        elif direction == "L":
            attach = [cell for cell in body if cell[1] == min_c and (cell[0] + dr, cell[1] + dc) not in body]
        elif direction == "D":
            attach = [cell for cell in body if cell[0] == max_r and (cell[0] + dr, cell[1] + dc) not in body]
        else:
            attach = [cell for cell in body if cell[0] == min_r and (cell[0] + dr, cell[1] + dc) not in body]
        rng.shuffle(attach)
        for ar, ac in attach:
            tail = (ar + dr, ac + dc)
            full = np.array(list(body) + [tail], dtype=int)
            analysis = _tail_analysis(full)
            if tuple(analysis["tail"]) != tail or analysis["direction"] != direction:
                continue
            if len(analysis["best_candidates"]) != 1:
                continue
            patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
            for r, c in body:
                patch[r - min_r, c - min_c] = int(color)
            return {
                "color": int(color),
                "count": int(count),
                "direction": direction,
                "body_local": tuple(sorted(body)),
                "tail_local": tail,
                "patch": patch,
            }
    raise RuntimeError("failed to generate shape spec")


def _rect_overlap(a_r0, a_c0, a_h, a_w, b_r0, b_c0, b_h, b_w):
    rr0 = max(a_r0, b_r0)
    cc0 = max(a_c0, b_c0)
    rr1 = min(a_r0 + a_h, b_r0 + b_h)
    cc1 = min(a_c0 + a_w, b_c0 + b_w)
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return rr0, cc0, rr1, cc1


def _has_colored_overlap(spec_a: dict, pos_a, spec_b: dict, pos_b) -> bool:
    a_patch = spec_a["patch"]
    b_patch = spec_b["patch"]
    overlap = _rect_overlap(pos_a[0], pos_a[1], a_patch.shape[0], a_patch.shape[1], pos_b[0], pos_b[1], b_patch.shape[0], b_patch.shape[1])
    if overlap is None:
        return False
    rr0, cc0, rr1, cc1 = overlap
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            if a_patch[r - pos_a[0], c - pos_a[1]] != 0 and b_patch[r - pos_b[0], c - pos_b[1]] != 0:
                return True
    return False


def _derive_source_bbox(pos, spec: dict) -> tuple[int, int]:
    r0, c0 = map(int, pos)
    h, w = spec["patch"].shape
    distance = int(spec["count"])
    direction = spec["direction"]
    if direction == "R":
        return int(r0), int(c0 - distance - w)
    if direction == "L":
        return int(r0), int(c0 + distance + w)
    if direction == "D":
        return int(r0 - distance - h), int(c0)
    return int(r0 + distance + h), int(c0)


def _instantiate_objects(shape: tuple[int, int], specs: list[dict], target_positions: list[tuple[int, int]]) -> list[dict] | None:
    h, w = shape
    objects = []
    occupied = set()
    for spec, pos in zip(specs, target_positions):
        src_r0, src_c0 = _derive_source_bbox(pos, spec)
        body_abs = {(src_r0 + r, src_c0 + c) for r, c in spec["body_local"]}
        tail_abs = (src_r0 + spec["tail_local"][0], src_c0 + spec["tail_local"][1])
        coords = body_abs | {tail_abs}
        if any(not _in_bounds(shape, r, c) for r, c in coords):
            return None
        for r, c in coords:
            if (r, c) in occupied:
                return None
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in occupied:
                    return None
        occupied |= coords

        patch = spec["patch"].copy()
        dst_r0, dst_c0 = pos
        if dst_r0 < 0 or dst_c0 < 0 or dst_r0 + patch.shape[0] > h or dst_c0 + patch.shape[1] > w:
            return None

        objects.append(
            {
                "color": int(spec["color"]),
                "count": int(spec["count"]),
                "direction": spec["direction"],
                "coords": sorted(coords),
                "tail": tail_abs,
                "body_cells": sorted(body_abs),
                "patch": patch,
                "dst": (int(dst_r0), int(dst_c0)),
            }
        )
    return objects


def _wrong_priority_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in sorted(objects, key=lambda item: (-int(item["count"]), COLOR_RANK.get(int(item["color"]), 999))):
        patch = obj["patch"]
        r0, c0 = obj["dst"]
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                if patch[rr, cc] != 0:
                    out[r0 + rr, c0 + cc] = patch[rr, cc]
    return out


def _wrong_distance_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    shifted = []
    for obj in objects:
        distance = int(obj["count"]) - 1
        r0, c0, r1, c1 = _bbox(obj["body_cells"])
        dst_r0, dst_c0 = _destination_top_left(obj["tail"], obj["direction"], distance, obj["patch"].shape, (r0, c0, r1, c1))
        shifted.append({**obj, "dst": (dst_r0, dst_c0)})
    return _overlay_patches(shape, shifted)


def _pixel_shift_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in sorted(objects, key=_priority_key):
        dr, dc = DIRS[obj["direction"]]
        distance = int(obj["count"])
        for r, c in obj["body_cells"]:
            rr = r + dr * distance
            cc = c + dc * distance
            if _in_bounds(shape, rr, cc):
                out[rr, cc] = int(obj["color"])
    return out


def _validate_semantics(inp: np.ndarray, objects: list[dict]) -> bool:
    try:
        parsed = _extract_objects(inp)
    except Exception:
        return False
    if len(parsed) != len(objects):
        return False
    if any(obj["count"] < 4 or obj["count"] > 8 for obj in parsed):
        return False
    counts = [obj["count"] for obj in parsed]
    if len(set(counts)) != len(counts):
        return False
    if any(obj["best_tail_count"] != 1 for obj in parsed):
        return False

    out = main(inp)
    overlap_pairs = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _has_colored_overlap(objects[i], objects[i]["dst"], objects[j], objects[j]["dst"]):
                overlap_pairs += 1
    if overlap_pairs == 0:
        return False

    if np.array_equal(out, _wrong_priority_output(inp.shape, objects)):
        return False
    if np.array_equal(out, _wrong_distance_output(inp.shape, objects)):
        return False
    if np.array_equal(out, _pixel_shift_output(inp.shape, objects)):
        return False
    return True


def _sample_positions(rng: np.random.Generator, shape: tuple[int, int], specs: list[dict]) -> list[tuple[int, int]] | None:
    h, w = shape
    first, second = specs[0], specs[1]
    h1, w1 = first["patch"].shape
    h2, w2 = second["patch"].shape
    for _ in range(300):
        pos1 = (int(rng.integers(0, h - h1 + 1)), int(rng.integers(0, w - w1 + 1)))
        drow = int(rng.integers(-(h2 - 1), h1))
        dcol = int(rng.integers(-(w2 - 1), w1))
        if drow == 0 and dcol == 0:
            continue
        pos2 = (pos1[0] + drow, pos1[1] + dcol)
        if pos2[0] < 0 or pos2[1] < 0 or pos2[0] + h2 > h or pos2[1] + w2 > w:
            continue
        overlap = _rect_overlap(pos1[0], pos1[1], h1, w1, pos2[0], pos2[1], h2, w2)
        if overlap is None:
            continue
        rr0, cc0, rr1, cc1 = overlap
        area = (rr1 - rr0) * (cc1 - cc0)
        if area > min(h1 * w1, h2 * w2) - 1:
            continue
        if not _has_colored_overlap(first, pos1, second, pos2):
            continue
        positions = [pos1, pos2]
        ok = True
        for spec in specs[2:]:
            ph, pw = spec["patch"].shape
            placed = False
            for _ in range(150):
                pos = (int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1)))
                positions.append(pos)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return positions
    return None


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
    )
    obj_count = 2 if rng.random() < 0.55 else 3
    counts = sorted(rng.choice(SHAPE_COUNTS, size=obj_count, replace=False).tolist())
    rng.shuffle(counts)
    colors = rng.choice(np.array(COLOR_ORDER, dtype=int), size=obj_count, replace=False).tolist()
    specs = [_generate_shape_spec(rng, int(color), int(count)) for color, count in zip(colors, counts)]
    rng.shuffle(specs)

    positions = _sample_positions(rng, shape, specs)
    if positions is None:
        return None
    objects = _instantiate_objects(shape, specs, positions)
    if objects is None:
        return None

    inp = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c in obj["coords"]:
            inp[r, c] = int(obj["color"])
    if not _validate_semantics(inp, objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
