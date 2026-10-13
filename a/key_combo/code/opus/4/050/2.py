from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4


def _frame_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr in range(3) for dc in range(3)]


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    frames = []
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            center = int(patch[1, 1])
            if center == bg:
                continue
            ring = [
                int(patch[0, 0]),
                int(patch[0, 1]),
                int(patch[0, 2]),
                int(patch[1, 0]),
                int(patch[1, 2]),
                int(patch[2, 0]),
                int(patch[2, 1]),
                int(patch[2, 2]),
            ]
            if len(set(ring)) != 1:
                continue
            border = ring[0]
            if border == bg or border == center:
                continue
            frames.append(
                {
                    "top": r,
                    "left": c,
                    "center": (r + 1, c + 1),
                    "border_color": border,
                    "center_color": center,
                    "cells": _frame_cells(r, c),
                }
            )
    return frames


def _holes_in_mask(mask: np.ndarray) -> list[np.ndarray]:
    bg_components = components_mask(~mask, connectivity=4)
    holes = []
    h, w = mask.shape
    for comp in bg_components:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _find_object_holes(coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    holes = _holes_in_mask(mask)
    return [hole + np.array([r0, c0], dtype=int) for hole in holes]


def _extract_objects(grid: np.ndarray, frames: list[dict], bg: int) -> list[dict]:
    frame_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        for r, c in frame["cells"]:
            frame_mask[r, c] = True

    mask = (grid != bg) & ~frame_mask
    objects = []
    for coords in components_mask(mask, connectivity=4):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "foreground objects must be monochromatic"
        holes = _find_object_holes(coords)
        objects.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "color": next(iter(colors)),
                "holes": holes,
            }
        )
    return objects


def _dominant_direction(frame_center: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    dr2 = 2 * int(frame_center[0]) - int(r0 + r1)
    dc2 = 2 * int(frame_center[1]) - int(c0 + c1)
    if abs(dc2) > abs(dr2):
        return "right" if dc2 > 0 else "left"
    return "down" if dr2 > 0 else "up"


def _collect_ray_cells(
    grid: np.ndarray,
    obj_coords: np.ndarray,
    holes: list[np.ndarray],
    direction: str,
    bg: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    row_bounds: dict[int, tuple[int, int]] = {}
    col_bounds: dict[int, tuple[int, int]] = {}
    for r, c in obj_coords:
        rr, cc = int(r), int(c)
        lo, hi = row_bounds.get(rr, (cc, cc))
        row_bounds[rr] = (min(lo, cc), max(hi, cc))
        lo, hi = col_bounds.get(cc, (rr, rr))
        col_bounds[cc] = (min(lo, rr), max(hi, rr))

    cells: list[tuple[int, int]] = []
    if direction in {"left", "right"}:
        rows = sorted({int(r) for hole in holes for r, _c in hole})
        step = -1 if direction == "left" else 1
        for r in rows:
            edge = row_bounds[r][0] if direction == "left" else row_bounds[r][1]
            c = edge + step
            while 0 <= c < w and int(grid[r, c]) == bg:
                cells.append((r, c))
                c += step
    else:
        cols = sorted({int(c) for hole in holes for _r, c in hole})
        step = -1 if direction == "up" else 1
        for c in cols:
            edge = col_bounds[c][0] if direction == "up" else col_bounds[c][1]
            r = edge + step
            while 0 <= r < h and int(grid[r, c]) == bg:
                cells.append((r, c))
                r += step
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect enclosed holes and shoot color rays whose hue comes from a nearby 3x3 frame center and whose direction comes from that frame's relative side.

    Input structure:
    - A black-background grid contains 2-3 monochromatic 4-connected foreground objects and 2-3 detached 3x3 monochrome frames.
    - Some foreground objects contain one or two enclosed background holes; others have none.
    - Each valid 3x3 frame has one border color on all eight ring cells and a different non-background center color.

    Rule:
    - Find every detached 3x3 frame, remove those cells from object detection, and identify the remaining monochromatic 4-connected objects.
    - For each object, detect enclosed background holes inside its own bounding box; objects without holes stay unchanged.
    - For a hole-containing object, choose the frame whose center has the unique smallest Manhattan distance to the object's bbox center.
    - The ray color is that frame's center color, and the ray direction is the frame's dominant relative side to the object center: left, right, up, or down.
    - For horizontal rays, use every row touched by a hole; for vertical rays, use every column touched by a hole. Start from the object's outer edge on that row or column and paint background cells until a non-background cell or the grid boundary is reached.

    Key insight:
    - The holes decide where rays are emitted, the frame center decides which color to paint, and the frame's relative position decides which way the rays travel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    frames = _detect_frames(g, bg)
    assert frames, "expected at least one 3x3 frame"
    objects = _extract_objects(g, frames, bg)
    assert objects, "expected at least one foreground object"

    paints: list[tuple[list[tuple[int, int]], int]] = []
    for obj in objects:
        if not obj["holes"]:
            continue
        r0, c0, r1, c1 = obj["bbox"]
        obj_center2 = (r0 + r1, c0 + c1)
        distances = [
            (
                abs(2 * frame["center"][0] - obj_center2[0]) + abs(2 * frame["center"][1] - obj_center2[1]),
                idx,
            )
            for idx, frame in enumerate(frames)
        ]
        distances.sort()
        assert len(distances) == 1 or distances[0][0] != distances[1][0], "nearest frame must be unique"
        frame = frames[distances[0][1]]
        direction = _dominant_direction(frame["center"], obj["bbox"])
        ray_cells = _collect_ray_cells(g, obj["coords"], obj["holes"], direction, bg)
        paints.append((ray_cells, int(frame["center_color"])))

    out = g.copy()
    for cells, color in paints:
        for r, c in cells:
            out[r, c] = color
    return out


_GEN_ATTEMPTS = 400
_BG = 0
_REGION_NAMES = ("TL", "TR", "BL", "BR")
_DIR_TO_REGIONS = {
    "left": ("TL", "BL"),
    "right": ("TR", "BR"),
    "up": ("TL", "TR"),
    "down": ("BL", "BR"),
}


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_hole_object(hole_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    base = np.ones((3, 3), dtype=bool) if hole_count == 1 else np.ones((3, 5), dtype=bool)
    if hole_count == 1:
        base[1, 1] = False
        area_target = int(np.random.randint(10, 16))
    else:
        base[1, 1] = False
        base[1, 3] = False
        area_target = int(np.random.randint(13, 19))

    rot = int(np.random.randint(0, 4))
    base = np.rot90(base, rot)
    canvas = np.zeros((9, 9), dtype=bool)
    br = (canvas.shape[0] - base.shape[0]) // 2
    bc = (canvas.shape[1] - base.shape[1]) // 2
    canvas[br : br + base.shape[0], bc : bc + base.shape[1]] = base

    attempts = 0
    while int(canvas.sum()) < area_target and attempts < 80:
        attempts += 1
        frontier = set()
        for r, c in np.argwhere(canvas):
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and not canvas[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        candidates = list(frontier)
        np.random.shuffle(candidates)
        placed = False
        for nr, nc in candidates:
            trial = canvas.copy()
            trial[nr, nc] = True
            holes = _holes_in_mask(trial)
            if len(holes) != hole_count:
                continue
            cropped = _crop_mask(trial)
            ch, cw = cropped.shape
            if ch > 6 or cw > 6:
                continue
            canvas = trial
            placed = True
            break
        if not placed:
            break

    cropped = _crop_mask(canvas)
    holes = _holes_in_mask(cropped)
    assert len(holes) == hole_count
    return cropped, holes


def _make_solid_object() -> np.ndarray:
    for _ in range(200):
        canvas = np.zeros((7, 7), dtype=bool)
        start = (3, 3)
        canvas[start] = True
        target = int(np.random.randint(10, 17))
        while int(canvas.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(canvas):
                for nr, nc in neighbors4(int(r), int(c)):
                    if 0 <= nr < 7 and 0 <= nc < 7 and not canvas[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cell = list(frontier)[int(np.random.randint(0, len(frontier)))]
            canvas[cell] = True
        cropped = _crop_mask(canvas)
        holes = _holes_in_mask(cropped)
        if holes:
            continue
        ch, cw = cropped.shape
        if ch > 6 or cw > 6:
            continue
        if int(cropped.sum()) == ch * cw:
            continue
        return cropped
    raise RuntimeError("failed to build a hole-free object")


def _regions(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return {
        "TL": (1, 1, mid_r - 2, mid_c - 2),
        "TR": (1, mid_c + 1, mid_r - 2, w - 2),
        "BL": (mid_r + 1, 1, h - 2, mid_c - 2),
        "BR": (mid_r + 1, mid_c + 1, h - 2, w - 2),
    }


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask).copy()
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _frame_array(border_color: int, center_color: int) -> np.ndarray:
    frame = np.full((3, 3), int(border_color), dtype=int)
    frame[1, 1] = int(center_color)
    return frame


def _can_place_cells(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = forbidden.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbidden[r, c]:
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, coords: np.ndarray, pad: int = 1) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        forbidden[r0:r1, c0:c1] = True


def _place_mask(
    grid: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> np.ndarray:
    coords = _coords_from_mask(mask, top, left)
    for r, c in coords:
        grid[r, c] = int(color)
    _mark_forbidden(forbidden, coords)
    return coords


def _place_frame(
    grid: np.ndarray,
    forbidden: np.ndarray,
    top: int,
    left: int,
    border_color: int,
    center_color: int,
) -> dict:
    cells = _frame_cells(top, left)
    arr = _frame_array(border_color, center_color)
    for dr in range(3):
        for dc in range(3):
            grid[top + dr, left + dc] = int(arr[dr, dc])
    coords = np.array(cells, dtype=int)
    _mark_forbidden(forbidden, coords)
    return {
        "top": top,
        "left": left,
        "center": (top + 1, left + 1),
        "border_color": int(border_color),
        "center_color": int(center_color),
        "cells": cells,
    }


def _region_positions(region: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    h, w = shape
    positions = [(r, c) for r in range(r0, r1 - h + 2) for c in range(c0, c1 - w + 2)]
    np.random.shuffle(positions)
    return positions


def _associated_frame_candidates(
    forbidden: np.ndarray,
    region: tuple[int, int, int, int],
    obj_bbox: tuple[int, int, int, int],
    hole_coords: list[np.ndarray],
    direction: str,
) -> list[tuple[int, int]]:
    rr0, cc0, rr1, cc1 = region
    r0, c0, r1, c1 = obj_bbox
    hole_rows = {int(r) for hole in hole_coords for r, _c in hole}
    hole_cols = {int(c) for hole in hole_coords for _r, c in hole}
    candidates = []
    for top in range(rr0, rr1 - 1):
        for left in range(cc0, cc1 - 1):
            cells = _frame_cells(top, left)
            if not _can_place_cells(forbidden, cells):
                continue
            if direction == "left" and top <= rr1 and left + 2 >= c0:
                pass
            if direction == "right" and left <= c1:
                continue
            if direction == "left" and left + 2 >= c0:
                continue
            if direction == "down" and top <= r1:
                continue
            if direction == "up" and top + 2 >= r0:
                continue

            center = (top + 1, left + 1)
            d = _dominant_direction(center, obj_bbox)
            if d != direction:
                continue
            if direction in {"left", "right"} and any(top <= r <= top + 2 for r in hole_rows):
                continue
            if direction in {"up", "down"} and any(left <= c <= left + 2 for c in hole_cols):
                continue
            candidates.append((top, left))
    np.random.shuffle(candidates)
    return candidates


def _assign_hole_regions(directions: list[str]) -> dict[int, str] | None:
    regions = list(_REGION_NAMES)
    order = list(range(len(directions)))
    np.random.shuffle(order)

    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        choices = list(_DIR_TO_REGIONS[directions[idx]])
        np.random.shuffle(choices)
        for region in choices:
            if region in used:
                continue
            used.add(region)
            assignment[idx] = region
            if backtrack(k + 1):
                return True
            used.remove(region)
            del assignment[idx]
        return False

    return assignment if backtrack(0) else None


def _choose_counts() -> tuple[int, int, int]:
    obj_count = int(np.random.choice([2, 3]))
    hole_obj_count = 1
    if obj_count == 2:
        hole_obj_count = 2 if np.random.rand() < 0.2 else 1
    else:
        hole_obj_count = 2 if np.random.rand() < 0.5 else 1
    frame_count = int(np.random.choice([max(2, hole_obj_count), 3]))
    return obj_count, hole_obj_count, frame_count


def _latent_output(grid: np.ndarray, objects: list[dict], bg: int) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    out = grid.copy()
    ray_sets: list[set[tuple[int, int]]] = []
    for obj in objects:
        if not obj["holes"]:
            continue
        cells = _collect_ray_cells(grid, obj["coords"], obj["holes"], obj["direction"], bg)
        cell_set = set(cells)
        ray_sets.append(cell_set)
        for r, c in cells:
            out[r, c] = int(obj["ray_color"])
    return out, ray_sets


def _semantic_check(grid: np.ndarray, objects: list[dict], frames: list[dict]) -> bool:
    bg = get_bg(grid)
    if bg != _BG:
        return False
    detected_frames = _detect_frames(grid, bg)
    if len(detected_frames) != len(frames):
        return False
    detected_objects = _extract_objects(grid, detected_frames, bg)
    if len(detected_objects) != len(objects):
        return False

    hole_counts = sorted(len(obj["holes"]) for obj in detected_objects if obj["holes"])
    expected_hole_counts = sorted(len(obj["holes"]) for obj in objects if obj["holes"])
    if hole_counts != expected_hole_counts:
        return False

    oracle, ray_sets = _latent_output(grid, objects, bg)
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(oracle, solved):
        return False
    if np.array_equal(grid, oracle):
        return False

    non_empty = [s for s in ray_sets if s]
    if len(non_empty) != len(ray_sets):
        return False
    union: set[tuple[int, int]] = set()
    for ray_set in ray_sets:
        if union & ray_set:
            return False
        union |= ray_set

    for obj in detected_objects:
        if not obj["holes"]:
            continue
        r0, c0, r1, c1 = obj["bbox"]
        obj_center2 = (r0 + r1, c0 + c1)
        distances = sorted(
            abs(2 * fr["center"][0] - obj_center2[0]) + abs(2 * fr["center"][1] - obj_center2[1])
            for fr in detected_frames
        )
        if len(distances) > 1 and distances[0] == distances[1]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(20, 25))
    w = int(np.random.randint(22, 27))
    grid = np.full((h, w), _BG, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)
    regions = _regions(h, w)

    obj_count, hole_obj_count, frame_count = _choose_counts()
    directions = list(np.random.choice(["left", "right", "up", "down"], size=hole_obj_count, replace=True))
    hole_region_map = _assign_hole_regions(directions)
    if hole_region_map is None:
        return None

    hole_regions = [hole_region_map[i] for i in range(hole_obj_count)]
    remaining_regions = [name for name in _REGION_NAMES if name not in hole_regions]
    np.random.shuffle(remaining_regions)
    non_hole_count = obj_count - hole_obj_count
    if len(remaining_regions) < non_hole_count:
        return None
    object_regions = hole_regions + remaining_regions[:non_hole_count]

    color_pool = np.random.permutation(np.arange(1, 10))
    object_colors = list(map(int, color_pool[:obj_count]))
    frame_border_colors = list(map(int, color_pool[obj_count : obj_count + frame_count]))
    frame_center_colors = list(map(int, color_pool[obj_count + frame_count : obj_count + 2 * frame_count]))

    objects: list[dict] = []
    frames: list[dict] = []
    frame_color_index = 0

    for idx in range(hole_obj_count):
        region_name = hole_regions[idx]
        region = regions[region_name]
        hole_count = 2 if np.random.rand() < 0.35 else 1
        mask, holes_local = _make_hole_object(hole_count)
        positions = _region_positions(region, mask.shape)
        placed = False
        for top, left in positions:
            coords = _coords_from_mask(mask, top, left)
            cells = [(int(r), int(c)) for r, c in coords]
            if not _can_place_cells(forbidden, cells):
                continue
            holes_global = [hole + np.array([top, left], dtype=int) for hole in holes_local]
            bbox = bbox_coords(coords)
            frame_candidates = _associated_frame_candidates(
                forbidden,
                region,
                bbox,
                holes_global,
                directions[idx],
            )
            if not frame_candidates:
                continue
            _place_mask(grid, forbidden, mask, top, left, object_colors[idx])
            frame_top, frame_left = frame_candidates[0]
            frame = _place_frame(
                grid,
                forbidden,
                frame_top,
                frame_left,
                frame_border_colors[frame_color_index],
                frame_center_colors[frame_color_index],
            )
            frame_color_index += 1
            frames.append(frame)
            objects.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "color": object_colors[idx],
                    "holes": holes_global,
                    "direction": directions[idx],
                    "ray_color": frame["center_color"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    for nh_idx in range(non_hole_count):
        region_name = remaining_regions[nh_idx]
        region = regions[region_name]
        mask = _make_solid_object()
        positions = _region_positions(region, mask.shape)
        placed = False
        obj_color = object_colors[hole_obj_count + nh_idx]
        for top, left in positions:
            coords = _coords_from_mask(mask, top, left)
            cells = [(int(r), int(c)) for r, c in coords]
            if not _can_place_cells(forbidden, cells):
                continue
            _place_mask(grid, forbidden, mask, top, left, obj_color)
            objects.append(
                {
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "color": obj_color,
                    "holes": [],
                    "direction": None,
                    "ray_color": None,
                }
            )
            placed = True
            break
        if not placed:
            return None

    extra_frames = frame_count - hole_obj_count
    free_regions = [name for name in _REGION_NAMES if name not in object_regions]
    np.random.shuffle(free_regions)
    if len(free_regions) < extra_frames:
        return None
    for ef_idx in range(extra_frames):
        region = regions[free_regions[ef_idx]]
        positions = _region_positions(region, (3, 3))
        placed = False
        for top, left in positions:
            cells = _frame_cells(top, left)
            if not _can_place_cells(forbidden, cells):
                continue
            frame = _place_frame(
                grid,
                forbidden,
                top,
                left,
                frame_border_colors[frame_color_index],
                frame_center_colors[frame_color_index],
            )
            frame_color_index += 1
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    if not _semantic_check(grid, objects, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
