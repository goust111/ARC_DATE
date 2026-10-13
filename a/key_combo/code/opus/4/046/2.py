from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components


BEND_TO_COLOR = {
    1: Color.BLUE,
    2: Color.PINK,
    3: Color.ORANGE,
}

INPUT_BG = Color.BLACK
INPUT_START = Color.RED
INPUT_SHAPE = Color.GREEN

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
PATH_LENGTH_RANGE = (4, 5)
GEN_RETRIES = 400
GAP_OPTIONS_NEAR = (2, 3, 4)
GAP_OPTIONS_FAR = (4, 5, 6)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _count_bends(mask: np.ndarray) -> int:
    h, w = mask.shape
    bends = 0
    for r in range(h - 1):
        for c in range(w - 1):
            bends += int(int(mask[r, c]) + int(mask[r + 1, c]) + int(mask[r, c + 1]) + int(mask[r + 1, c + 1]) == 3)
    return bends


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _all_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = list(cells)
    variants = []
    seen = set()
    for rot in range(4):
        cur = pts
        for _ in range(rot):
            cur = [(-c, r) for r, c in cur]
        for flip in (False, True):
            transformed = {(r, -c) if flip else (r, c) for r, c in cur}
            norm = _normalize_cells(transformed)
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)
    return variants


def _polyomino_library() -> dict[str, dict[int, list[np.ndarray]]]:
    polyominoes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(2, 9):
        next_set: set[tuple[tuple[int, int], ...]] = set()
        for poly in polyominoes[area - 1]:
            cells = set(poly)
            frontier = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            for extra in frontier:
                next_set.add(min(_all_variants(_normalize_cells(cells | {extra}))))
        polyominoes[area] = next_set

    library = {
        "all": {1: [], 2: [], 3: []},
        "h2": {1: [], 2: [], 3: []},
        "w2": {1: [], 2: [], 3: []},
    }
    seen_masks = set()
    for area in range(3, 9):
        for poly in polyominoes[area]:
            for variant in _all_variants(poly):
                max_r = max(r for r, _c in variant)
                max_c = max(c for _r, c in variant)
                mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
                for r, c in variant:
                    mask[r, c] = True
                bends = _count_bends(mask)
                if bends not in (1, 2, 3):
                    continue
                if mask.shape[0] < 2 or mask.shape[1] < 2 or max(mask.shape) > 6:
                    continue
                key = (bends, mask.shape, tuple(mask.astype(int).ravel()))
                if key in seen_masks:
                    continue
                seen_masks.add(key)
                library["all"][bends].append(mask.copy())
                if mask.shape[0] == 2:
                    library["h2"][bends].append(mask.copy())
                if mask.shape[1] == 2:
                    library["w2"][bends].append(mask.copy())

    for axis_key in library:
        for bends in library[axis_key]:
            library[axis_key][bends].sort(key=lambda m: (m.shape[0], m.shape[1], int(m.sum()), tuple(m.astype(int).ravel())))
    return library


_SHAPE_LIBRARY = _polyomino_library()


def _shape_record(mask: np.ndarray, top: int, left: int, bends: int, shape_id: int) -> dict:
    coords_local = np.argwhere(mask)
    coords = coords_local + np.array([top, left], dtype=int)
    h, w = mask.shape
    return {
        "id": shape_id,
        "kind": "shape",
        "mask": mask.copy(),
        "top": int(top),
        "left": int(left),
        "bbox": (int(top), int(left), int(top + h - 1), int(left + w - 1)),
        "coords": coords.astype(int),
        "bend": int(bends),
    }


def _point_record(r: int, c: int) -> dict:
    coords = np.array([[int(r), int(c)]], dtype=int)
    return {
        "id": -1,
        "kind": "point",
        "bbox": (int(r), int(c), int(r), int(c)),
        "coords": coords,
    }


def _bbox_separated(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    if a1 < b0:
        return int(a1 + 1), int(b0 - 1)
    if b1 < a0:
        return int(b1 + 1), int(a0 - 1)
    return None


def _coord_distance(a: dict, b: dict) -> int:
    da = a["coords"][:, None, :]
    db = b["coords"][None, :, :]
    return int(np.abs(da - db).sum(axis=2).min())


def _center_r(node: dict) -> float:
    r0, _c0, r1, _c1 = node["bbox"]
    return (r0 + r1) / 2.0


def _center_c(node: dict) -> float:
    _r0, c0, _r1, c1 = node["bbox"]
    return (c0 + c1) / 2.0


def _best_row_band(g: np.ndarray, bg: int, current: dict, target: dict) -> tuple[int, int] | None:
    cur_r0, cur_c0, cur_r1, cur_c1 = current["bbox"]
    tar_r0, tar_c0, tar_r1, tar_c1 = target["bbox"]
    gap = _bbox_separated(cur_c0, cur_c1, tar_c0, tar_c1)
    if gap is None:
        return None
    gap_c0, gap_c1 = gap
    if gap_c0 > gap_c1:
        return None

    row_tops: list[int] = []
    if current["kind"] == "point":
        point_r = cur_r0
        for top in (point_r - 1, point_r):
            if tar_r0 <= top and top + 1 <= tar_r1 and 0 <= top and top + 1 < g.shape[0]:
                row_tops.append(int(top))
    else:
        lo = max(cur_r0, tar_r0)
        hi = min(cur_r1, tar_r1)
        row_tops.extend(range(lo, hi))

    scored = []
    for top in sorted(set(row_tops)):
        if np.all(g[top : top + 2, gap_c0 : gap_c1 + 1] == bg):
            mid = top + 0.5
            score = abs(mid - _center_r(current)) + abs(mid - _center_r(target))
            scored.append((score, top))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return int(scored[0][1]), int(gap_c1 - gap_c0 + 1)


def _best_col_band(g: np.ndarray, bg: int, current: dict, target: dict) -> tuple[int, int] | None:
    cur_r0, cur_c0, cur_r1, cur_c1 = current["bbox"]
    tar_r0, tar_c0, tar_r1, tar_c1 = target["bbox"]
    gap = _bbox_separated(cur_r0, cur_r1, tar_r0, tar_r1)
    if gap is None:
        return None
    gap_r0, gap_r1 = gap
    if gap_r0 > gap_r1:
        return None

    col_lefts: list[int] = []
    if current["kind"] == "point":
        point_c = cur_c0
        for left in (point_c - 1, point_c):
            if tar_c0 <= left and left + 1 <= tar_c1 and 0 <= left and left + 1 < g.shape[1]:
                col_lefts.append(int(left))
    else:
        lo = max(cur_c0, tar_c0)
        hi = min(cur_c1, tar_c1)
        col_lefts.extend(range(lo, hi))

    scored = []
    for left in sorted(set(col_lefts)):
        if np.all(g[gap_r0 : gap_r1 + 1, left : left + 2] == bg):
            mid = left + 0.5
            score = abs(mid - _center_c(current)) + abs(mid - _center_c(target))
            scored.append((score, left))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return int(scored[0][1]), int(gap_r1 - gap_r0 + 1)


def _visible_step(g: np.ndarray, bg: int, current: dict, target: dict) -> dict | None:
    horizontal = _best_row_band(g, bg, current, target)
    vertical = _best_col_band(g, bg, current, target)
    if horizontal is None and vertical is None:
        return None

    distance = _coord_distance(current, target)
    candidates = []
    if horizontal is not None:
        band_top, gap_len = horizontal
        candidates.append(("h", gap_len, band_top))
    if vertical is not None:
        band_left, gap_len = vertical
        candidates.append(("v", gap_len, band_left))
    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    axis, gap_len, band = candidates[0]
    return {
        "axis": axis,
        "band": int(band),
        "gap_len": int(gap_len),
        "distance": int(distance),
    }


def _paint_corridor(out: np.ndarray, current: dict, target: dict, axis: str, band: int, color: int) -> None:
    cur_r0, cur_c0, cur_r1, cur_c1 = current["bbox"]
    tar_r0, tar_c0, tar_r1, tar_c1 = target["bbox"]
    if axis == "h":
        gap_c0, gap_c1 = _bbox_separated(cur_c0, cur_c1, tar_c0, tar_c1)
        out[band : band + 2, gap_c0 : gap_c1 + 1] = int(color)
        return
    gap_r0, gap_r1 = _bbox_separated(cur_r0, cur_r1, tar_r0, tar_r1)
    out[gap_r0 : gap_r1 + 1, band : band + 2] = int(color)


def _parse_input(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    start_components = []
    shape_components = []
    for comp in components:
        color = int(comp["color"])
        if color == INPUT_START:
            start_components.append(comp)
        elif color == INPUT_SHAPE:
            shape_components.append(comp)
        else:
            raise AssertionError("unexpected input color")

    assert len(start_components) == 1, "expected exactly one red start marker"
    assert int(start_components[0]["area"]) == 1, "start marker must be a single pixel"
    assert shape_components, "expected green shapes"

    sr, sc = map(int, start_components[0]["coords"][0])
    start = _point_record(sr, sc)

    shapes = []
    for shape_id, comp in enumerate(shape_components):
        bbox = tuple(map(int, comp["bbox"]))
        sprite = crop_bbox(g, bbox)
        mask = sprite != bg
        bends = _count_bends(mask)
        assert bends in (1, 2, 3), "each shape must have one to three bends"
        shapes.append({
            "id": int(shape_id),
            "kind": "shape",
            "bbox": bbox,
            "coords": comp["coords"].astype(int),
            "bend": int(bends),
        })
    return bg, start, shapes


def _trace_walk(
    grid: np.ndarray,
    *,
    choose_by_bend: bool = True,
    corridor_color_by_bend: bool = True,
    paint_corridors: bool = True,
    recolor_shapes: bool = True,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg, start, shapes = _parse_input(g)
    out = g.copy()
    current = start
    unvisited = [shape.copy() for shape in shapes]
    steps = []

    while True:
        candidates = []
        for shape in unvisited:
            step = _visible_step(g, bg, current, shape)
            if step is None:
                continue
            candidates.append({
                "shape": shape,
                "axis": step["axis"],
                "band": step["band"],
                "gap_len": step["gap_len"],
                "distance": step["distance"],
                "bend": int(shape["bend"]),
                "bbox": tuple(map(int, shape["bbox"])),
                "id": int(shape["id"]),
            })

        if not candidates:
            break

        if choose_by_bend:
            candidates.sort(key=lambda item: (item["bend"], item["distance"], item["bbox"], item["id"]))
        else:
            candidates.sort(key=lambda item: (item["distance"], item["bend"], item["bbox"], item["id"]))

        chosen = candidates[0]
        second_key = None
        first_key = None
        if len(candidates) >= 2:
            if choose_by_bend:
                first_key = (candidates[0]["bend"], candidates[0]["distance"])
                second_key = (candidates[1]["bend"], candidates[1]["distance"])
            else:
                first_key = (candidates[0]["distance"], candidates[0]["bend"])
                second_key = (candidates[1]["distance"], candidates[1]["bend"])

        color = BEND_TO_COLOR[chosen["bend"]] if corridor_color_by_bend else Color.ORANGE
        if paint_corridors:
            _paint_corridor(out, current, chosen["shape"], chosen["axis"], chosen["band"], color)
        if recolor_shapes:
            coords = chosen["shape"]["coords"]
            out[coords[:, 0], coords[:, 1]] = int(color)

        steps.append({
            "current_kind": current["kind"],
            "current_bbox": tuple(map(int, current["bbox"])),
            "visible": candidates,
            "chosen": chosen,
            "ambiguous": bool(first_key is not None and first_key == second_key),
        })
        current = chosen["shape"]
        unvisited = [shape for shape in unvisited if int(shape["id"]) != int(chosen["id"])]

    return out, steps, shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse visible green bend-shapes greedily by lowest bend count and encode each corridor and visited shape with that bend color.

    Input structure:
    - The grid has a black background, one red single-pixel start marker, and several 4-connected green shapes.
    - Each green shape has one, two, or three L-shaped 2x2 bend patterns inside its cropped bounding box.
    - Visibility is along a shared two-row or two-column band whose gap cells are all background.

    Rule:
    - Detect every green shape, crop its bounding box, and count the number of 2x2 windows containing exactly three shape cells; that bend count B is 1, 2, or 3.
    - Starting from the red marker, repeatedly look at all unvisited green shapes visible from the current position in a clear two-row or two-column band.
    - Choose the visible target with the smallest bend count B; if several visible targets share B, choose the one with the smallest Manhattan distance to the current object.
    - Paint the background gap between the current position and the chosen target with color B=1->blue, B=2->magenta, B=3->orange, then recolor the whole target shape to that same color and continue from it.
    - Stop when the current position has no visible unvisited green shape.

    Key insight:
    - The same topological property, bend count, controls both the walk order and the corridor color, so geometry changes the traversal itself instead of only a final recoloring.
    """
    out, _steps, _shapes = _trace_walk(np.asarray(grid, dtype=int))
    return out


def _bbox_of_shape(shape: dict) -> tuple[int, int, int, int]:
    return tuple(map(int, shape["bbox"]))


def _rect_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _choose_mask(rng: random.Random, bends: int, axis: str) -> np.ndarray:
    key = "h2" if axis == "h" else "w2"
    masks = _SHAPE_LIBRARY[key][bends]
    assert masks, f"missing masks for bend {bends} axis {axis}"
    return masks[rng.randrange(len(masks))].copy()


def _candidate_positions_for_relation(
    current: dict,
    mask: np.ndarray,
    axis: str,
    direction: int,
    gap: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    h, w = mask.shape
    candidates = []
    cur_r0, cur_c0, cur_r1, cur_c1 = current["bbox"]

    if axis == "h":
        if h != 2:
            return candidates
        if current["kind"] == "point":
            row_tops = [cur_r0 - 1, cur_r0]
        else:
            row_tops = list(range(cur_r0, cur_r1))
        for top in row_tops:
            if not (0 <= top and top + h <= H):
                continue
            if direction > 0:
                left = cur_c1 + gap + 1
            else:
                left = cur_c0 - gap - w
            if 0 <= left and left + w <= W:
                candidates.append((int(top), int(left)))
        return candidates

    if w != 2:
        return candidates
    if current["kind"] == "point":
        col_lefts = [cur_c0 - 1, cur_c0]
    else:
        col_lefts = list(range(cur_c0, cur_c1))
    for left in col_lefts:
        if not (0 <= left and left + w <= W):
            continue
        if direction > 0:
            top = cur_r1 + gap + 1
        else:
            top = cur_r0 - gap - h
        if 0 <= top and top + h <= H:
            candidates.append((int(top), int(left)))
    return candidates


def _place_shape(
    rng: random.Random,
    current: dict,
    bends: int,
    axis: str,
    preferred_gaps: tuple[int, ...],
    occupied: list[dict],
    grid_shape: tuple[int, int],
    shape_id: int,
) -> dict | None:
    mask = _choose_mask(rng, bends, axis)
    options = []
    directions = [1, -1]
    rng.shuffle(directions)
    gaps = list(preferred_gaps)
    rng.shuffle(gaps)
    for direction in directions:
        for gap in gaps:
            options.extend(_candidate_positions_for_relation(current, mask, axis, direction, gap, grid_shape))
    rng.shuffle(options)
    for top, left in options:
        candidate = _shape_record(mask, top, left, bends, shape_id)
        bbox = _bbox_of_shape(candidate)
        if not all(_rect_gap_ok(bbox, _bbox_of_shape(other), padding=1) for other in occupied):
            continue
        if current["kind"] == "point":
            pr, pc = map(int, current["coords"][0])
            if bbox[0] <= pr <= bbox[2] and bbox[1] <= pc <= bbox[3]:
                continue
        return candidate
    return None


def _render_input(shape: tuple[int, int], start: dict, shapes: list[dict]) -> np.ndarray:
    grid = np.full(shape, INPUT_BG, dtype=int)
    sr, sc = map(int, start["coords"][0])
    grid[sr, sc] = INPUT_START
    for shape_obj in shapes:
        coords = shape_obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = INPUT_SHAPE
    return grid


def _nearest_visible_output(grid: np.ndarray) -> np.ndarray:
    out, _steps, _shapes = _trace_walk(grid, choose_by_bend=False)
    return out


def _orange_only_output(grid: np.ndarray) -> np.ndarray:
    out, _steps, _shapes = _trace_walk(grid, corridor_color_by_bend=False)
    return out


def _recolor_only_output(grid: np.ndarray) -> np.ndarray:
    out, _steps, _shapes = _trace_walk(grid, paint_corridors=False)
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, steps, shapes = _trace_walk(grid)
    except AssertionError:
        return False
    if not steps or any(step["ambiguous"] for step in steps):
        return False

    bends_present = {int(shape["bend"]) for shape in shapes}
    if bends_present != {1, 2, 3}:
        return False

    if sum(len(step["visible"]) >= 2 for step in steps) < 2:
        return False

    bend_priority_needed = False
    distance_tie_needed = False
    multi_colors_visited = set()
    for step in steps:
        chosen = step["chosen"]
        visible = step["visible"]
        multi_colors_visited.add(int(chosen["bend"]))
        nearest_distance = min(item["distance"] for item in visible)
        if chosen["distance"] > nearest_distance:
            closer_items = [item for item in visible if item["distance"] == nearest_distance]
            if min(item["bend"] for item in closer_items) > chosen["bend"]:
                bend_priority_needed = True
        same_bend = [item for item in visible if item["bend"] == chosen["bend"]]
        if len(same_bend) >= 2:
            same_bend.sort(key=lambda item: item["distance"])
            if same_bend[0]["id"] == chosen["id"] and same_bend[0]["distance"] < same_bend[1]["distance"]:
                distance_tie_needed = True

    if not bend_priority_needed or not distance_tie_needed:
        return False

    if np.array_equal(_nearest_visible_output(grid), out):
        return False
    if np.array_equal(_orange_only_output(grid), out):
        return False
    if np.array_equal(_recolor_only_output(grid), out):
        return False
    if len(multi_colors_visited) < 2:
        return False
    return True


def _fill_missing_bends(rng: random.Random, start: dict, shapes: list[dict], grid_shape: tuple[int, int], next_id: int) -> list[dict] | None:
    present = {shape["bend"] for shape in shapes}
    need = [bend for bend in (1, 2, 3) if bend not in present]
    occupied = shapes[:]
    current = start if not shapes else shapes[-1]
    out = []
    shape_id = next_id
    for bend in need:
        for _ in range(8):
            axis = rng.choice(["h", "v"])
            extra = _place_shape(rng, current, bend, axis, GAP_OPTIONS_NEAR + GAP_OPTIONS_FAR, occupied, grid_shape, shape_id)
            if extra is None:
                continue
            occupied.append(extra)
            out.append(extra)
            current = extra
            shape_id += 1
            break
        else:
            return None
    return out


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    start = _point_record(rng.randint(4, H - 5), rng.randint(4, W - 5))

    path_len = rng.randint(*PATH_LENGTH_RANGE)
    path_bends = [rng.randint(1, 3) for _ in range(path_len)]
    competitor0_bend = rng.choice([bend for bend in (1, 2, 3) if bend > path_bends[0]]) if path_bends[0] < 3 else 2
    competitor1_bend = path_bends[1] if path_len >= 2 else path_bends[0]

    occupied: list[dict] = []
    shapes: list[dict] = []
    current: dict = start
    prev_axis: str | None = None
    next_id = 0

    for idx, bend in enumerate(path_bends):
        axis_choices = ["h", "v"] if prev_axis is None else ([prev_axis, "v" if prev_axis == "h" else "h"] if rng.random() < 0.35 else ["v" if prev_axis == "h" else "h", prev_axis])
        placed = None
        for axis in axis_choices:
            gaps = GAP_OPTIONS_FAR if idx == 0 else GAP_OPTIONS_NEAR + GAP_OPTIONS_FAR
            placed = _place_shape(rng, current, bend, axis, gaps, occupied, (H, W), next_id)
            if placed is not None:
                prev_axis = axis
                break
        if placed is None:
            return None
        occupied.append(placed)
        shapes.append(placed)
        next_id += 1

        if idx == 0:
            comp_axis = "v" if prev_axis == "h" else "h"
            competitor = _place_shape(rng, current, competitor0_bend, comp_axis, GAP_OPTIONS_NEAR, occupied, (H, W), next_id)
            if competitor is None:
                return None
            occupied.append(competitor)
            shapes.append(competitor)
            next_id += 1

        if idx == 1:
            comp_axis = "v" if prev_axis == "h" else "h"
            competitor = _place_shape(rng, current, competitor1_bend, comp_axis, GAP_OPTIONS_FAR, occupied, (H, W), next_id)
            if competitor is None:
                return None
            occupied.append(competitor)
            shapes.append(competitor)
            next_id += 1

        current = placed

    extras = _fill_missing_bends(rng, start, shapes, (H, W), next_id)
    if extras is None:
        return None
    shapes.extend(extras)

    grid = _render_input((H, W), start, shapes)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bend-priority corridor puzzle")
