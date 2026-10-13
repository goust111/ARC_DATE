import numpy as np
from collections import deque


BG = 0
RAY = 1
REPAINT = 2
OBJECT_COLORS = (3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SEED_LENGTHS = (2, 3)
GRID_RANGE = (18, 23)  # upper bound is exclusive
MAX_GENERATION_ATTEMPTS = 500
HIT_PATTERNS = (
    ("frame", "solid"),
    ("frame", "solid", "frame"),
    ("frame", "frame", "solid"),
    ("frame", "solid", "solid"),
    ("frame", "solid", "frame", "solid"),
    ("frame", "frame", "solid", "solid"),
    ("frame", "solid", "solid", "frame"),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the blue diagonal ray so hollow 4x4 boxes reflect it while solid 4x4 blocks let it pass through and later turn red.

    Input structure:
    - A black background contains exactly one short blue diagonal seed of length 2 or 3.
    - It also contains several separated monochrome 4x4 objects using non-blue foreground colors.
    - Each 4x4 object is either a hollow frame with a black 2x2 interior or a fully filled solid block.

    Rule:
    - Classify every 4x4 object by its 2x2 interior: all-black interior means frame, otherwise solid.
    - Start from the seed endpoint whose diagonal continuation stays inside the grid, and extend the ray in that same diagonal direction.
    - Paint every visited non-frame cell blue; when the next cell is a frame border, do not paint it and reflect by flipping the row direction on horizontal-edge hits, the column direction on vertical-edge hits, or both on corner hits; when the next cell is a solid block, pass through it and remember that block.
    - After the ray exits the grid, repaint every remembered solid block entirely red.

    Key insight:
    - Geometry, not color, chooses the ray physics: the same collision is reflective for hollow 4x4 frames but transparent for solid 4x4 blocks.
    """
    g = np.asarray(grid, dtype=int)
    seed, objects = _parse_grid(g)
    out, _details = _trace_ray(g, seed, objects)
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_seed(g: np.ndarray) -> dict:
    comps = _components(g == RAY, connectivity=8)
    assert len(comps) == 1, "expected exactly one blue seed component"
    coords = comps[0]
    assert len(coords) in SEED_LENGTHS, "seed must have length 2 or 3"
    seed_cells = {tuple(map(int, cell)) for cell in coords}

    endpoints: list[tuple[int, int]] = []
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in seed_cells:
        neigh = []
        for dr, dc in DIAGONALS:
            other = (cell[0] + dr, cell[1] + dc)
            if other in seed_cells:
                neigh.append(other)
        neighbor_map[cell] = neigh
        if len(neigh) == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "seed must be a single diagonal chain"

    starts = []
    for endpoint in endpoints:
        neighbor = neighbor_map[endpoint][0]
        dr = endpoint[0] - neighbor[0]
        dc = endpoint[1] - neighbor[1]
        assert abs(dr) == 1 and abs(dc) == 1
        nr, nc = endpoint[0] + dr, endpoint[1] + dc
        if _in_bounds(g.shape, nr, nc):
            starts.append((endpoint, dr, dc))
    assert len(starts) == 1, "seed direction must be uniquely determined"

    start, dr, dc = starts[0]
    return {
        "cells": seed_cells,
        "start": start,
        "direction": (dr, dc),
        "length": len(seed_cells),
    }


def _find_objects(g: np.ndarray) -> list[dict]:
    comps = _components((g != BG) & (g != RAY), connectivity=4)
    objects = []
    for idx, coords in enumerate(comps):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert r1 - r0 + 1 == 4 and c1 - c0 + 1 == 4, "objects must have 4x4 bbox"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.ones((4, 4), dtype=bool)
        border_mask[1:3, 1:3] = False
        if int(coords.shape[0]) == 12 and np.all(sub[border_mask] == color) and np.all(sub[1:3, 1:3] == BG):
            kind = "frame"
        elif int(coords.shape[0]) == 16 and np.all(sub == color):
            kind = "solid"
        else:
            raise AssertionError("objects must be hollow 4x4 frames or solid 4x4 blocks")
        objects.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "kind": kind,
                "color": color,
                "cells": {tuple(map(int, cell)) for cell in coords},
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    for idx, obj in enumerate(objects):
        obj["id"] = idx
    return objects


def _parse_grid(g: np.ndarray) -> tuple[dict, list[dict]]:
    assert g.ndim == 2, "grid must be 2D"
    seed = _find_seed(g)
    objects = _find_objects(g)
    return seed, objects


def _trace_ray(g: np.ndarray, seed: dict, objects: list[dict]) -> tuple[np.ndarray, dict]:
    object_by_cell = {}
    object_by_id = {}
    for obj in objects:
        object_by_id[obj["id"]] = obj
        for cell in obj["cells"]:
            object_by_cell[cell] = obj

    out = g.copy()
    r, c = seed["start"]
    dr, dc = seed["direction"]
    seen_states: set[tuple[int, int, int, int]] = set()
    hit_order: list[int] = []
    hit_once: set[int] = set()
    penetrated_solids: set[int] = set()
    reflection_styles: list[str] = []
    path_cells = set(seed["cells"])

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray path must not loop"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        value = int(g[nr, nc])
        if value in (BG, RAY):
            out[nr, nc] = RAY
            r, c = nr, nc
            path_cells.add((r, c))
            continue

        obj = object_by_cell.get((nr, nc))
        assert obj is not None, "foreground cell must belong to a classified object"

        if obj["id"] not in hit_once:
            hit_once.add(obj["id"])
            hit_order.append(obj["id"])

        if obj["kind"] == "solid":
            penetrated_solids.add(obj["id"])
            out[nr, nc] = RAY
            r, c = nr, nc
            path_cells.add((r, c))
            continue

        r0, c0, _r1, _c1 = obj["bbox"]
        lr, lc = nr - r0, nc - c0
        flip_r = lr in (0, 3)
        flip_c = lc in (0, 3)
        assert flip_r or flip_c, "ray can only hit a frame on its border"
        if flip_r and flip_c:
            reflection_styles.append("corner")
        elif flip_r:
            reflection_styles.append("horizontal")
        else:
            reflection_styles.append("vertical")
        if flip_r:
            dr = -dr
        if flip_c:
            dc = -dc

    for solid_id in penetrated_solids:
        r0, c0, r1, c1 = object_by_id[solid_id]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = REPAINT

    details = {
        "path_cells": path_cells,
        "hit_object_ids": hit_order,
        "hit_kinds_in_order": [object_by_id[obj_id]["kind"] for obj_id in hit_order],
        "penetrated_solids": penetrated_solids,
        "reflection_count": len(reflection_styles),
        "reflection_styles": reflection_styles,
        "objects_by_id": object_by_id,
    }
    return out, details


def _bbox_cells(r0: int, c0: int, margin: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r in range(max(0, r0 - margin), min(h, r0 + 4 + margin)):
        for c in range(max(0, c0 - margin), min(w, c0 + 4 + margin)):
            cells.add((r, c))
    return cells


def _render_grid(shape: tuple[int, int], seed_cells: list[tuple[int, int]], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in seed_cells:
        grid[r, c] = RAY
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        color = obj["color"]
        if obj["kind"] == "solid":
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
    return grid


def _distance_to_exit(shape: tuple[int, int], start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    steps = 0
    while _in_bounds(shape, r + dr, c + dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _sample_seed(rng: np.random.Generator, shape: tuple[int, int], length: int, dr: int, dc: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    candidates = []

    edge_r = 0 if dr == 1 else h - 1
    for c in range(w):
        end_r = edge_r + (length - 1) * dr
        end_c = c + (length - 1) * dc
        if _in_bounds(shape, end_r, end_c):
            candidates.append((edge_r, c))

    edge_c = 0 if dc == 1 else w - 1
    for r in range(h):
        end_r = r + (length - 1) * dr
        end_c = edge_c + (length - 1) * dc
        if _in_bounds(shape, end_r, end_c):
            candidates.append((r, edge_c))

    deduped = sorted(set(candidates))
    filtered = []
    for outer in deduped:
        start = (outer[0] + (length - 1) * dr, outer[1] + (length - 1) * dc)
        if _distance_to_exit(shape, start, dr, dc) >= 9:
            filtered.append((outer, start))
    choices = filtered if filtered else [(outer, (outer[0] + (length - 1) * dr, outer[1] + (length - 1) * dc)) for outer in deduped]
    assert choices
    outer, start = choices[int(rng.integers(len(choices)))]
    seed_cells = [(outer[0] + i * dr, outer[1] + i * dc) for i in range(length)]
    return seed_cells, start


def _solid_local_candidates(dr: int, dc: int) -> list[tuple[int, int]]:
    near_r = 0 if dr == 1 else 3
    near_c = 0 if dc == 1 else 3
    locals_set = {(near_r, c) for c in range(4)} | {(r, near_c) for r in range(4)}
    candidates = []
    for lr, lc in locals_set:
        rr, cc = lr, lc
        length = 0
        while 0 <= rr < 4 and 0 <= cc < 4:
            length += 1
            rr += dr
            cc += dc
        if length >= 2:
            candidates.append((lr, lc))
    return candidates


def _frame_local_candidates(dr: int, dc: int) -> list[tuple[tuple[int, int], str]]:
    near_r = 0 if dr == 1 else 3
    near_c = 0 if dc == 1 else 3
    return [
        ((near_r, 1), "horizontal"),
        ((near_r, 2), "horizontal"),
        ((1, near_c), "vertical"),
        ((2, near_c), "vertical"),
        ((near_r, near_c), "corner"),
    ]


def _place_hit_objects(
    rng: np.random.Generator,
    shape: tuple[int, int],
    seed_cells: list[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    hit_types: tuple[str, ...],
    colors: list[int],
) -> list[dict] | None:
    objects: list[dict] = []
    object_buffer = set()
    used_path = set(seed_cells)
    current = start
    dr, dc = direction

    for index, (kind, color) in enumerate(zip(hit_types, colors)):
        remaining = len(hit_types) - index - 1
        placed = False
        for _ in range(120):
            lead = int(rng.integers(1, 5))
            pre_cells = []
            ok = True
            rr, cc = current
            for _step in range(lead):
                rr += dr
                cc += dc
                if not _in_bounds(shape, rr, cc) or (rr, cc) in object_buffer:
                    ok = False
                    break
                pre_cells.append((rr, cc))
            if not ok:
                continue

            before_hit = pre_cells[-1]
            target = (before_hit[0] + dr, before_hit[1] + dc)
            if not _in_bounds(shape, target[0], target[1]):
                continue

            if kind == "frame":
                candidates = _frame_local_candidates(dr, dc)
                rng.shuffle(candidates)
                for (lr, lc), _style in candidates:
                    r0 = target[0] - lr
                    c0 = target[1] - lc
                    if not (0 <= r0 and 0 <= c0 and r0 + 3 < shape[0] and c0 + 3 < shape[1]):
                        continue
                    bbox_cells = {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}
                    if bbox_cells & object_buffer:
                        continue
                    if bbox_cells & used_path:
                        continue
                    new_dr, new_dc = dr, dc
                    if lr in (0, 3):
                        new_dr = -new_dr
                    if lc in (0, 3):
                        new_dc = -new_dc
                    if _distance_to_exit(shape, before_hit, new_dr, new_dc) < max(2, 3 * remaining):
                        continue
                    used_path.update(pre_cells)
                    objects.append(
                        {
                            "bbox": (r0, c0, r0 + 3, c0 + 3),
                            "kind": "frame",
                            "color": color,
                        }
                    )
                    object_buffer.update(_bbox_cells(r0, c0, margin=1, shape=shape))
                    current = before_hit
                    dr, dc = new_dr, new_dc
                    placed = True
                    break
            else:
                candidates = _solid_local_candidates(dr, dc)
                rng.shuffle(candidates)
                for lr, lc in candidates:
                    r0 = target[0] - lr
                    c0 = target[1] - lc
                    if not (0 <= r0 and 0 <= c0 and r0 + 3 < shape[0] and c0 + 3 < shape[1]):
                        continue
                    bbox_cells = {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}
                    if bbox_cells & object_buffer:
                        continue
                    if bbox_cells & used_path:
                        continue
                    traversed = []
                    rr, cc = target
                    while r0 <= rr <= r0 + 3 and c0 <= cc <= c0 + 3:
                        traversed.append((rr, cc))
                        rr += dr
                        cc += dc
                    if len(traversed) < 2:
                        continue
                    if _distance_to_exit(shape, traversed[-1], dr, dc) < max(2, 3 * remaining):
                        continue
                    used_path.update(pre_cells)
                    used_path.update(traversed)
                    objects.append(
                        {
                            "bbox": (r0, c0, r0 + 3, c0 + 3),
                            "kind": "solid",
                            "color": color,
                        }
                    )
                    object_buffer.update(_bbox_cells(r0, c0, margin=1, shape=shape))
                    current = traversed[-1]
                    placed = True
                    break

            if placed:
                break
        if not placed:
            return None

    return objects


def _place_decoys(
    rng: np.random.Generator,
    shape: tuple[int, int],
    seed_cells: list[tuple[int, int]],
    existing_objects: list[dict],
    path_cells: set[tuple[int, int]],
    colors: list[int],
) -> list[dict] | None:
    objects = list(existing_objects)
    object_buffer = set()
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        object_buffer.update(_bbox_cells(r0, c0, margin=1, shape=shape))

    decoy_kinds = ["frame", "solid"]
    extra = min(len(colors) - 2, int(rng.integers(0, 2)))
    for _ in range(extra):
        decoy_kinds.append("frame" if rng.random() < 0.5 else "solid")

    if len(colors) < len(decoy_kinds):
        return None

    for kind, color in zip(decoy_kinds, colors):
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(0, shape[0] - 3))
            c0 = int(rng.integers(0, shape[1] - 3))
            bbox_cells = {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}
            if bbox_cells & object_buffer:
                continue
            if bbox_cells & path_cells:
                continue
            if bbox_cells & set(seed_cells):
                continue
            objects.append(
                {
                    "bbox": (r0, c0, r0 + 3, c0 + 3),
                    "kind": kind,
                    "color": color,
                }
            )
            object_buffer.update(_bbox_cells(r0, c0, margin=1, shape=shape))
            placed = True
            break
        if not placed:
            return None
    return objects


def _analyze_grid(g: np.ndarray) -> dict:
    seed, objects = _parse_grid(g)
    out, trace = _trace_ray(g, seed, objects)
    total_frames = sum(obj["kind"] == "frame" for obj in objects)
    total_solids = sum(obj["kind"] == "solid" for obj in objects)
    hit_ids = trace["hit_object_ids"]
    hit_frame_ids = [obj_id for obj_id in hit_ids if trace["objects_by_id"][obj_id]["kind"] == "frame"]
    hit_solid_ids = [obj_id for obj_id in hit_ids if trace["objects_by_id"][obj_id]["kind"] == "solid"]
    seed_cells = seed["cells"]
    return {
        "seed": seed,
        "objects": objects,
        "out": out,
        "path_cells": trace["path_cells"],
        "hit_object_ids": hit_ids,
        "hit_kinds_in_order": trace["hit_kinds_in_order"],
        "reflection_count": trace["reflection_count"],
        "reflection_styles": trace["reflection_styles"],
        "hit_frame_ids": hit_frame_ids,
        "hit_solid_ids": hit_solid_ids,
        "unhit_frames": total_frames - len(hit_frame_ids),
        "unhit_solids": total_solids - len(hit_solid_ids),
        "new_blue_cells": len(trace["path_cells"] - set(seed_cells)),
        "red_cells": int(np.count_nonzero(out == REPAINT)),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_RANGE[0], GRID_RANGE[1])),
        int(rng.integers(GRID_RANGE[0], GRID_RANGE[1])),
    )
    dr, dc = DIAGONALS[int(rng.integers(len(DIAGONALS)))]
    seed_length = SEED_LENGTHS[int(rng.integers(len(SEED_LENGTHS)))]
    seed_cells, start = _sample_seed(rng, shape, seed_length, dr, dc)

    hit_types = HIT_PATTERNS[int(rng.integers(len(HIT_PATTERNS)))]
    palette = list(OBJECT_COLORS)
    rng.shuffle(palette)
    hit_colors = palette[: len(hit_types)]
    decoy_colors = palette[len(hit_types) :]

    hit_objects = _place_hit_objects(
        rng=rng,
        shape=shape,
        seed_cells=seed_cells,
        start=start,
        direction=(dr, dc),
        hit_types=hit_types,
        colors=hit_colors,
    )
    if hit_objects is None:
        return None

    hit_only_grid = _render_grid(shape, seed_cells, hit_objects)
    try:
        hit_analysis = _analyze_grid(hit_only_grid)
    except AssertionError:
        return None

    if hit_analysis["hit_kinds_in_order"] != list(hit_types):
        return None
    if hit_analysis["reflection_count"] < 1:
        return None
    if not hit_analysis["hit_frame_ids"] or not hit_analysis["hit_solid_ids"]:
        return None

    all_objects = _place_decoys(
        rng=rng,
        shape=shape,
        seed_cells=seed_cells,
        existing_objects=hit_objects,
        path_cells=hit_analysis["path_cells"],
        colors=decoy_colors,
    )
    if all_objects is None:
        return None

    grid = _render_grid(shape, seed_cells, all_objects)
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return None

    if len(analysis["objects"]) < 4 or len(analysis["objects"]) > 6:
        return None
    if analysis["hit_kinds_in_order"] != list(hit_types):
        return None
    if analysis["reflection_count"] < 1:
        return None
    if len(analysis["hit_object_ids"]) < 2:
        return None
    if analysis["unhit_frames"] < 1 or analysis["unhit_solids"] < 1:
        return None
    if analysis["new_blue_cells"] < 6:
        return None
    if analysis["red_cells"] < 16:
        return None
    if np.array_equal(grid, analysis["out"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
