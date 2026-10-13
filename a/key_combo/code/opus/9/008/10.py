from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_task10",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


BG = Color.BLACK
SOURCE = Color.GREY
TARGET = Color.ORANGE
PATH_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

TARGET_SIZE = 9
TARGET_SLOTS = ((1, 1), (1, 5), (5, 1), (5, 5))
HOLE_BANK = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
)


def _norm_coords(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> frozenset[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for dr, dc in shape:
        rr, cc = top + int(dr), left + int(dc)
        grid[rr, cc] = int(color)
        coords.add((rr, cc))
    return coords


def _component_coord_set(comp: dict) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in comp["coords"]}


def _adjacent_non_object_cells(grid: np.ndarray, coords: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in coords:
        for rr, cc in neighbors4(r, c):
            if not in_bounds(grid.shape, rr, cc) or (rr, cc) in coords or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            out.append((int(rr), int(cc), int(grid[rr, cc])))
    return out


def _right_anchor_candidates(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in sorted(coords):
        rr, cc = r, c + 1
        if (rr, cc) in coords:
            continue
        neighbor_count = sum((nr, nc) in coords for nr, nc in neighbors4(rr, cc))
        if neighbor_count == 1:
            candidates.append((rr, cc))
    return sorted(set(candidates))


def _extract_target_holes(grid: np.ndarray, target_comp: dict) -> list[dict]:
    r0, c0, r1, c1 = map(int, target_comp["bbox"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    black = sub == BG
    seen = np.zeros_like(black, dtype=bool)
    holes: list[dict] = []

    for sr, sc in np.argwhere(black):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        region: list[tuple[int, int]] = []
        touches_border = False
        while stack:
            r, c = stack.pop()
            region.append((r, c))
            if r in (0, black.shape[0] - 1) or c in (0, black.shape[1] - 1):
                touches_border = True
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < black.shape[0] and 0 <= cc < black.shape[1] and black[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if touches_border:
            continue
        global_coords = [(r0 + r, c0 + c) for r, c in region]
        holes.append(
            {
                "coords": set(global_coords),
                "shape": _norm_coords(region),
            }
        )
    return holes


def _trace_path(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    nodes = {(int(r), int(c)) for r, c in np.argwhere(grid == int(color))}
    assert start in nodes
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in nodes and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _target_adjacency_map(targets: list[dict], shape: tuple[int, int]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, target in enumerate(targets):
        for r, c in _component_coord_set(target):
            owner[r, c] = idx
    return owner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the matching hole in each reached orange object using the guide path color from its gray source shape.

    Input structure:
    - A black background contains 2-3 small gray source shapes, 1-2 larger orange target objects, and sparse colored guide pixels.
    - Each real source shape has exactly one adjacent guide pixel, and same-color guide pixels continue on a two-cell lattice to one target.
    - Each target object is a 4-connected orange region whose bounding box contains 2-3 black hole components of different shapes.

    Rule:
    - Find every gray component and keep only those with a unique adjacent non-black guide pixel.
    - Follow that guide color through cardinal steps of length two to identify the path component and the orange target object touched by the path endpoint.
    - Normalize the gray source shape by cropping it to its bounding box, normalize every hole inside the reached target's bounding box, and compare exact coordinate sets.
    - Fill the unique matching hole with the guide path color and leave all other cells unchanged.

    Key insight:
    - The path decides which target object is relevant, the source geometry decides which hole is relevant, and the guide color decides the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    targets = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[TARGET],
        return_format="dict",
    )
    assert 1 <= len(targets) <= 2
    target_owner = _target_adjacency_map(targets, g.shape)

    target_holes: list[list[dict]] = [_extract_target_holes(g, target) for target in targets]
    out = g.copy()

    sources = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SOURCE],
        return_format="dict",
    )

    for source in sources:
        src_coords = _component_coord_set(source)
        guide_neighbors = [
            (r, c, color)
            for r, c, color in _adjacent_non_object_cells(g, src_coords)
            if color not in {BG, SOURCE, TARGET}
        ]
        if len(guide_neighbors) != 1:
            continue

        start_r, start_c, path_color = guide_neighbors[0]
        path_nodes = _trace_path(g, (start_r, start_c), path_color)
        touched_targets: set[int] = set()
        for r, c in path_nodes:
            for rr, cc in neighbors4(r, c):
                if in_bounds(g.shape, rr, cc) and target_owner[rr, cc] >= 0:
                    touched_targets.add(int(target_owner[rr, cc]))
        assert len(touched_targets) == 1
        target_idx = next(iter(touched_targets))

        src_shape = _norm_coords(src_coords)
        matches = [hole for hole in target_holes[target_idx] if hole["shape"] == src_shape]
        assert len(matches) == 1
        for r, c in matches[0]["coords"]:
            out[r, c] = int(path_color)

    return out


def _segment_coords(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 2 if c1 > c0 else -2
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 2 if r1 > r0 else -2
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _polyline_path(start: tuple[int, int], end: tuple[int, int], pivot_c: int) -> list[tuple[int, int]]:
    if start[1] == pivot_c and end[1] == pivot_c:
        return _segment_coords(start, end)
    a = _segment_coords(start, (start[0], pivot_c))
    b = _segment_coords((start[0], pivot_c), (end[0], pivot_c))
    c = _segment_coords((end[0], pivot_c), end)
    return a + b[1:] + c[1:]


def _path_midpoints(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mids: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        mids.append(((r0 + r1) // 2, (c0 + c1) // 2))
    return mids


def _path_ok(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    allowed = {start, end}
    for node in path:
        if not in_bounds(grid.shape, node[0], node[1]):
            return False
        if grid[node] != BG and node not in allowed:
            return False
        if node in blocked and node not in allowed:
            return False
    for mid in _path_midpoints(path):
        if not in_bounds(grid.shape, mid[0], mid[1]) or grid[mid] != BG or mid in blocked:
            return False
    return True


def _path_blockers(path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set(path)
    blocked.update(_path_midpoints(path))
    for r, c in path:
        for rr, cc in neighbors4(r, c):
            if in_bounds((30, 30), rr, cc):
                blocked.add((rr, cc))
    return blocked


def _shape_slots(rng: np.random.Generator, hole_shapes: list[tuple[tuple[int, int], ...]]) -> tuple[np.ndarray, list[dict]]:
    patch = np.full((TARGET_SIZE, TARGET_SIZE), TARGET, dtype=int)
    chosen_slots = rng.choice(len(TARGET_SLOTS), size=len(hole_shapes), replace=False)
    holes: list[dict] = []
    for shape, slot_idx in zip(hole_shapes, chosen_slots):
        slot_r, slot_c = TARGET_SLOTS[int(slot_idx)]
        h, w = _shape_dims(shape)
        off_r = int(rng.integers(0, 3 - h + 1))
        off_c = int(rng.integers(0, 3 - w + 1))
        hole_cells: list[tuple[int, int]] = []
        for dr, dc in shape:
            rr = slot_r + off_r + int(dr)
            cc = slot_c + off_c + int(dc)
            patch[rr, cc] = BG
            hole_cells.append((rr, cc))
        holes.append({"coords_local": hole_cells, "shape": _norm_coords(shape)})
    return patch, holes


def _can_place_nonbg(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    *,
    padding: int,
) -> bool:
    own = set(coords)
    for r, c in own:
        if not in_bounds(grid.shape, r, c) or grid[r, c] != BG:
            return False
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if not in_bounds(grid.shape, rr, cc):
                    continue
                if (rr, cc) in own:
                    continue
                if grid[rr, cc] != BG:
                    return False
    return True


def _place_patch_in_region(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    *,
    padding: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    nonbg_local = {(int(r), int(c)) for r, c in np.argwhere(patch != BG)}
    for top in range(max(0, row_min), min(row_max, grid.shape[0] - h) + 1):
        for left in range(max(0, col_min), min(col_max, grid.shape[1] - w) + 1):
            coords = {(top + r, left + c) for r, c in nonbg_local}
            if _can_place_nonbg(grid, coords, padding=padding):
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _object_neighbor_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    out: set[tuple[int, int]] = set()
    for r, c in occupied:
        for rr, cc in neighbors4(r, c):
            if in_bounds(grid.shape, rr, cc) and (rr, cc) not in occupied:
                out.add((rr, cc))
    return out


def _expected_output(case: dict) -> np.ndarray:
    out = case["grid"].copy()
    for src in case["sources"]:
        hole = case["targets"][src["target_index"]]["holes"][src["hole_index"]]
        for r, c in hole["coords"]:
            out[r, c] = int(src["path_color"])
    return out


def _analyze_connected_sources(grid: np.ndarray) -> list[tuple[frozenset[tuple[int, int]], int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SOURCE],
        return_format="dict",
    )
    linked: list[tuple[frozenset[tuple[int, int]], int]] = []
    for comp in comps:
        coords = _component_coord_set(comp)
        neighbors = [
            color
            for _r, _c, color in _adjacent_non_object_cells(grid, coords)
            if color not in {BG, SOURCE, TARGET}
        ]
        if len(neighbors) == 1:
            linked.append((_norm_coords(coords), int(neighbors[0])))
        else:
            assert len(neighbors) == 0
    return linked


def _validate_case(case: dict) -> bool:
    grid = case["grid"]
    out = main(grid)
    expected = _expected_output(case)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    linked = _analyze_connected_sources(grid)
    if len(linked) != len(case["sources"]):
        return False

    real_pairs = {(src["shape"], src["path_color"]) for src in case["sources"]}
    if set(linked) != real_pairs:
        return False

    for target in case["targets"]:
        hole_shapes = [hole["shape"] for hole in target["holes"]]
        if len(hole_shapes) != len(set(hole_shapes)):
            return False
        assigned = [src for src in case["sources"] if src["target_index"] == target["index"]]
        assigned_ids = {src["hole_index"] for src in assigned}
        if len(assigned_ids) != len(assigned):
            return False
        if len(assigned_ids) >= len(target["holes"]):
            return False
        for src in assigned:
            matches = [i for i, hole in enumerate(target["holes"]) if hole["shape"] == src["shape"]]
            if matches != [src["hole_index"]]:
                return False

    for distractor in case["distractors"]:
        neighbors = [
            color
            for _r, _c, color in _adjacent_non_object_cells(grid, distractor["coords"])
            if color not in {BG, SOURCE, TARGET}
        ]
        if neighbors:
            return False

    return True


def _pattern_spec(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    pattern = random.choice(("single_target_shared", "unused_target", "mixed"))
    if pattern == "single_target_shared":
        return [3], [0, 0]
    if pattern == "unused_target":
        return [3, 2], [0, 0]
    if random.random() < 0.5:
        return [3, 2], [0, 0, 1]
    return [3, 2], [0, 1, 1]


def _build_case(rng: np.random.Generator) -> dict | None:
    h = int(rng.integers(22, 29))
    w = int(rng.integers(22, 29))
    grid = np.full((h, w), BG, dtype=int)

    target_hole_counts, assignments = _pattern_spec(rng)
    n_targets = len(target_hole_counts)
    n_sources = len(assignments)
    path_colors = list(PATH_COLORS[:n_sources])
    rng.shuffle(path_colors)

    hole_shapes_all = [HOLE_BANK[i] for i in rng.choice(len(HOLE_BANK), size=sum(target_hole_counts), replace=False)]
    per_target_shapes: list[list[tuple[tuple[int, int], ...]]] = []
    cursor = 0
    for count in target_hole_counts:
        per_target_shapes.append(hole_shapes_all[cursor : cursor + count])
        cursor += count

    targets: list[dict] = []
    if n_targets == 1:
        target_row_ranges = [(h // 2 - 4, h // 2 - 2)]
    else:
        target_row_ranges = [(1, max(1, h // 4)), (h // 2 + 1, h - TARGET_SIZE - 1)]

    target_col_min = max(2, w - TARGET_SIZE - 3)
    target_col_max = w - TARGET_SIZE - 1

    for idx, hole_shapes in enumerate(per_target_shapes):
        patch, holes = _shape_slots(rng, hole_shapes)
        row_min, row_max = target_row_ranges[idx]
        placed = _place_patch_in_region(
            rng,
            grid,
            patch,
            row_min,
            row_max,
            target_col_min,
            target_col_max,
            padding=2,
        )
        if placed is None:
            return None
        top, left = placed
        target_coords = {
            (top + int(r), left + int(c))
            for r, c in np.argwhere(patch == TARGET)
        }
        for r, c in target_coords:
            grid[r, c] = TARGET
        target_info = {
            "index": idx,
            "top_left": (top, left),
            "coords": target_coords,
            "holes": [],
            "left_endpoints": [],
        }
        for hole in holes:
            global_coords = {(top + r, left + c) for r, c in hole["coords_local"]}
            target_info["holes"].append(
                {
                    "coords": global_coords,
                    "shape": hole["shape"],
                }
            )
        for rr in range(top + 1, top + TARGET_SIZE - 1):
            target_info["left_endpoints"].append((rr, left - 1))
        targets.append(target_info)

    source_records: list[dict] = []
    source_bands = [(1, h // 3 - 2), (h // 3 + 1, 2 * h // 3 - 2), (2 * h // 3 + 1, h - 5)]
    rng.shuffle(source_bands)

    available_hole_refs: dict[int, list[int]] = {
        t_idx: list(range(len(targets[t_idx]["holes"])))
        for t_idx in range(n_targets)
    }
    for i, target_idx in enumerate(assignments):
        hole_index = random.choice(available_hole_refs[target_idx])
        available_hole_refs[target_idx].remove(hole_index)
        shape = per_target_shapes[target_idx][hole_index]
        patch = _shape_patch(shape, SOURCE)
        band = source_bands[i]
        placed = _place_patch_in_region(
            rng,
            grid,
            patch,
            band[0],
            band[1],
            1,
            max(1, w // 3),
            padding=2,
        )
        if placed is None:
            return None
        top, left = placed
        coords = _paint_shape(grid, shape, top, left, SOURCE)
        anchors = [anchor for anchor in _right_anchor_candidates(coords) if in_bounds(grid.shape, anchor[0], anchor[1]) and grid[anchor] == BG]
        if not anchors:
            return None
        source_records.append(
            {
                "shape": _norm_coords(shape),
                "coords": coords,
                "anchors": anchors,
                "target_index": target_idx,
                "hole_index": hole_index,
                "path_color": path_colors[i],
            }
        )

    distractors: list[dict] = []
    distractor_shape = random.choice(per_target_shapes[random.randrange(n_targets)])
    distractor_patch = _shape_patch(distractor_shape, SOURCE)
    distractor_row_min = max(1, h // 3 - 2)
    distractor_row_max = min(h - 4, 2 * h // 3 + 2)
    distractor_col_min = max(1, w // 3)
    distractor_col_max = max(distractor_col_min, w // 2)
    placed = _place_patch_in_region(
        rng,
        grid,
        distractor_patch,
        distractor_row_min,
        distractor_row_max,
        distractor_col_min,
        distractor_col_max,
        padding=2,
    )
    if placed is not None:
        top, left = placed
        coords = _paint_shape(grid, distractor_shape, top, left, SOURCE)
        distractors.append({"coords": coords, "shape": _norm_coords(distractor_shape)})

    blocked = _object_neighbor_cells(grid)
    pivot_pool = [c for c in range(max(6, w // 3), w - TARGET_SIZE - 2) if 1 <= c < w - 1]
    random.shuffle(pivot_pool)
    used_endpoints: set[tuple[int, int]] = set()

    for src in sorted(source_records, key=lambda item: min(r for r, _ in item["coords"])):
        target = targets[src["target_index"]]
        start_options = src["anchors"][:]
        random.shuffle(start_options)
        end_options = [p for p in target["left_endpoints"] if p not in used_endpoints]
        random.shuffle(end_options)
        found = None
        for start in start_options:
            for end in end_options:
                if (start[0] % 2, start[1] % 2) != (end[0] % 2, end[1] % 2):
                    continue
                pivot_choices = [
                    c for c in pivot_pool
                    if c % 2 == start[1] % 2 and start[1] + 2 <= c <= end[1] - 2
                ]
                random.shuffle(pivot_choices)
                for pivot_c in pivot_choices:
                    path = _polyline_path(start, end, pivot_c)
                    if _path_ok(grid, path, blocked, start, end):
                        found = (start, end, path)
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return None
        start, end, path = found
        for r, c in path:
            grid[r, c] = src["path_color"]
        src["path_start"] = start
        src["path_end"] = end
        src["path"] = path
        used_endpoints.add(end)
        blocked.update(_path_blockers(path))

    case = {
        "grid": grid,
        "targets": targets,
        "sources": source_records,
        "distractors": distractors,
    }
    if not _validate_case(case):
        return None
    return case


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        case = _build_case(rng)
        if case is not None:
            return case["grid"]
    raise RuntimeError("failed to generate a valid input for puzzle 10")


if __name__ == "__main__":
    grid = generate_input()
    print(grid)
    print(main(grid))
