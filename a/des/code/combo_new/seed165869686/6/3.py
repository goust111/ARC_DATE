from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return out


def _frame_mask_from_bbox(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    coords = comp["coords"]  # type: ignore[assignment]
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return bool(np.array_equal(mask, _frame_mask_from_bbox((r0, c0, r1, c1))))


def _bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _extract_scene(grid: np.ndarray) -> tuple[
    dict[int, dict[str, object]],
    list[tuple[int, int, int]],
    list[dict[str, object]],
    dict[tuple[int, int], int],
    set[tuple[int, int]],
    dict[tuple[int, int], int],
]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]

    frames: dict[int, dict[str, object]] = {}
    seeds: list[tuple[int, int, int]] = []
    objects: list[dict[str, object]] = []

    for color in colors:
        comps = _components(g == color)
        frame_comps = [comp for comp in comps if _is_rect_frame(comp)]
        if frame_comps:
            assert len(frame_comps) == 1, "each frame color must have one rectangular frame component"
            frame = frame_comps[0]
            frames[color] = frame
            for comp in comps:
                if comp is frame:
                    continue
                assert int(comp["area"]) == 1, "non-frame components of a frame color must be singleton seeds"
                sr, sc = map(int, comp["coords"][0])
                seeds.append((sr, sc, color))
        else:
            for comp in comps:
                area = int(comp["area"])
                assert 2 <= area <= 4, "filter objects must be small 2-4 pixel components"
                obj = dict(comp)
                obj["color"] = int(color)
                objects.append(obj)

    assert frames, "expected at least one frame"
    assert seeds, "expected at least one seed"

    frame_by_cell: dict[tuple[int, int], int] = {}
    for color, frame in frames.items():
        for cell in map(tuple, frame["coords"]):  # type: ignore[index]
            assert cell not in frame_by_cell, "frames must not overlap"
            frame_by_cell[cell] = int(color)

    seed_cells = {(r, c) for r, c, _color in seeds}

    object_by_cell: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        assert int(obj["color"]) not in frames, "filter colors must differ from frame colors"
        for cell in map(tuple, obj["coords"]):  # type: ignore[index]
            assert cell not in object_by_cell, "filter objects must not overlap"
            object_by_cell[cell] = int(idx)

    seeds.sort(key=lambda item: (item[2], item[0], item[1]))
    return frames, seeds, objects, frame_by_cell, seed_cells, object_by_cell


def _trace_direction(
    shape: tuple[int, int],
    frame_by_cell: dict[tuple[int, int], int],
    seed_cells: set[tuple[int, int]],
    seed: tuple[int, int],
    direction: tuple[int, int],
    target_color: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    sr, sc = seed
    dr, dc = direction
    r, c = sr + dr, sc + dc
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cell = (int(r), int(c))
        if cell in seed_cells:
            return None
        hit_color = frame_by_cell.get(cell)
        if hit_color is not None:
            if hit_color == target_color:
                path.append(cell)
                return path
            return None
        path.append(cell)
        r += dr
        c += dc
    return None


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, seeds, objects, frame_by_cell, seed_cells, object_by_cell = _extract_scene(g)

    crossed_objects: set[int] = set()
    ray_pixels: dict[tuple[int, int], int] = {}

    for sr, sc, color in seeds:
        assert color in frames, "every seed color must match a frame color"
        candidates = []
        for direction in DIAG_DIRS:
            path = _trace_direction(g.shape, frame_by_cell, seed_cells, (sr, sc), direction, color)
            if path is not None:
                candidates.append((direction, path))
        assert len(candidates) == 1, "each seed must have exactly one diagonal direction to its matching frame"

        _direction, path = candidates[0]
        active_color = int(color)
        for cell in path:
            obj_idx = object_by_cell.get(cell)
            if obj_idx is not None:
                active_color = int(objects[obj_idx]["color"])
                crossed_objects.add(int(obj_idx))
            assert cell not in ray_pixels, "rays must not overlap"
            ray_pixels[cell] = int(active_color)

    out = g.copy()
    for obj_idx in crossed_objects:
        for r, c in map(tuple, objects[obj_idx]["coords"]):  # type: ignore[index]
            out[r, c] = BG
    for (r, c), color in ray_pixels.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed diagonally into its matching frame and let crossed objects recolor the later ray segments.

    Input structure:
    - The grid background is black.
    - Several non-overlapping one-pixel-thick rectangular frames appear in distinct colors.
    - Outside the frames sit singleton seed pixels whose colors match frame colors, plus a few small monochrome 2-4 pixel filter objects in other colors.

    Rule:
    - Pair every seed with the unique rectangular frame of the same color.
    - From each seed, follow the only diagonal direction that first reaches that matching frame, and draw the ray through the hit frame pixel.
    - Start the ray in the seed color, but whenever the path crosses a filter object, recolor that crossed pixel and every later ray pixel with the object's color; later crossed objects override earlier ones.
    - Erase every filter object that was crossed by a ray, then draw the recolored ray segments on top.

    Key insight:
    - Color identifies which frame belongs to a seed, geometry identifies which diagonal works, and the crossed objects act as visible color filters that update the rest of the ray.
    """
    return _solve(np.asarray(grid, dtype=int))


GEN_HEIGHT_RANGE = (18, 28)
GEN_WIDTH_RANGE = (18, 28)
GEN_FRAME_COUNT = (2, 3)
GEN_FRAME_H_RANGE = (4, 8)
GEN_FRAME_W_RANGE = (4, 8)
GEN_FRAME_PAD = 2
GEN_FRAME_RETRIES = 120
GEN_ATTEMPTS = 400

GEN_EXTRA_OBJECTS = (0, 2)
GEN_EXTRA_RETRIES = 60

DIR_TO_SIDES = {
    (-1, -1): ("bottom", "right"),
    (-1, 1): ("bottom", "left"),
    (1, -1): ("top", "right"),
    (1, 1): ("top", "left"),
}

SIDE_HIT_DIRS = {
    "top": ((1, -1), (1, 1)),
    "bottom": ((-1, -1), (-1, 1)),
    "left": ((-1, 1), (1, 1)),
    "right": ((-1, -1), (1, -1)),
}

OBJECT_PATTERNS = (
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, -1)),
    ((0, 0), (1, 0), (1, 1), (0, 1)),
    ((0, 0), (1, 0), (1, -1), (0, -1)),
    ((0, 0), (-1, 0), (-1, 1), (0, 1)),
    ((0, 0), (-1, 0), (-1, -1), (0, -1)),
)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _frame_cells_from_bbox(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _place_frames(h: int, w: int, colors: list[int]) -> list[dict[str, object]] | None:
    frames: list[dict[str, object]] = []
    for color in colors:
        for _ in range(GEN_FRAME_RETRIES):
            fh = random.randint(*GEN_FRAME_H_RANGE)
            fw = random.randint(*GEN_FRAME_W_RANGE)
            if fh >= h - 4 or fw >= w - 4:
                continue
            r0 = random.randint(2, h - fh - 2)
            c0 = random.randint(2, w - fw - 2)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(bbox, frame["bbox"], pad=GEN_FRAME_PAD) for frame in frames):
                continue
            frames.append({"color": int(color), "bbox": bbox, "cells": _frame_cells_from_bbox(bbox)})
            break
        else:
            return None
    return frames


def _side_hits(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    assert side == "right"
    return [(r, c1) for r in range(r0 + 1, r1)]


def _frame_lookup(frames: list[dict[str, object]]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for frame in frames:
        color = int(frame["color"])
        for cell in frame["cells"]:  # type: ignore[index]
            lookup[cell] = color
    return lookup


def _path_for_seed(seed: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = seed
    dr, dc = direction
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _anchor_options(path_len: int, count: int) -> list[tuple[int, ...]]:
    if count == 0:
        return [()]
    eligible = list(range(1, path_len - 1))
    combos = [
        combo
        for combo in itertools.combinations(eligible, count)
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]
    return [tuple(map(int, combo)) for combo in combos]


def _choose_seed_and_path(
    shape: tuple[int, int],
    frames: list[dict[str, object]],
    frame: dict[str, object],
    reserved_cells: set[tuple[int, int]],
    direction: tuple[int, int],
    desired_objects: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    target_color = int(frame["color"])
    bbox = frame["bbox"]  # type: ignore[assignment]
    compatible_sides = list(DIR_TO_SIDES[direction])
    random.shuffle(compatible_sides)

    frame_lookup = _frame_lookup(frames)
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    for side in compatible_sides:
        assert direction in SIDE_HIT_DIRS[side]
        hits = _side_hits(bbox, side)
        random.shuffle(hits)
        for hit in hits:
            max_len_row = hit[0] if direction[0] == 1 else shape[0] - 1 - hit[0]
            max_len_col = hit[1] if direction[1] == 1 else shape[1] - 1 - hit[1]
            max_len = int(min(max_len_row, max_len_col))
            if max_len < 2:
                continue
            lengths = list(range(2, max_len + 1))
            random.shuffle(lengths)
            for length in lengths:
                seed = (hit[0] - direction[0] * length, hit[1] - direction[1] * length)
                if seed in reserved_cells:
                    continue
                if any(_bbox_contains(other["bbox"], seed) for other in frames):  # type: ignore[index]
                    continue
                path = _path_for_seed(seed, direction, length)
                if any(cell in reserved_cells for cell in path):
                    continue
                if any(any(_bbox_contains(other["bbox"], cell) for other in frames) for cell in path[:-1]):  # type: ignore[index]
                    continue
                if _anchor_options(len(path), desired_objects) == []:
                    continue
                valid_dirs = 0
                for test_dir in DIAG_DIRS:
                    traced = _trace_direction(shape, frame_lookup, {seed}, seed, test_dir, target_color)
                    if traced is not None:
                        valid_dirs += 1
                if valid_dirs != 1:
                    continue
                candidates.append((seed, path))
    return random.choice(candidates) if candidates else None


def _cells_touch_occupied(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied and (nr, nc) not in cell_set:
                    return True
    return False


def _try_place_object(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    anchor: tuple[int, int],
    color: int,
) -> list[tuple[int, int]] | None:
    patterns = list(OBJECT_PATTERNS)
    random.shuffle(patterns)
    for pattern in patterns:
        cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in pattern]
        if any(not _in_bounds(grid.shape, r, c) for r, c in cells):
            continue
        if cells[0] != anchor:
            continue
        if any(grid[r, c] != BG for r, c in cells):
            continue
        if any(cell != anchor and cell in blocked for cell in cells):
            continue
        if _cells_touch_occupied(cells, occupied):
            continue
        for r, c in cells:
            grid[r, c] = color
        occupied.update(cells)
        return cells
    return None


def _place_ray_objects(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    path_blocked: set[tuple[int, int]],
    ray: dict[str, object],
    filter_colors: list[int],
) -> bool:
    count = int(ray["object_count"])
    path = ray["path"]  # type: ignore[assignment]
    if count == 0:
        return True

    combos = _anchor_options(len(path), count)
    random.shuffle(combos)
    for combo in combos:
        local_grid = grid.copy()
        local_occupied = set(occupied)
        colors = random.sample(filter_colors, count) if count <= len(filter_colors) else [random.choice(filter_colors) for _ in range(count)]
        success = True
        placed: list[list[tuple[int, int]]] = []
        for anchor_index, color in zip(combo, colors):
            anchor = path[anchor_index]
            cells = _try_place_object(local_grid, local_occupied, path_blocked, anchor, int(color))
            if cells is None:
                success = False
                break
            placed.append(cells)
        if success:
            grid[:, :] = local_grid
            occupied.clear()
            occupied.update(local_occupied)
            ray["objects"] = placed
            return True
    return False


def _place_extra_objects(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    filter_colors: list[int],
) -> None:
    count = random.randint(*GEN_EXTRA_OBJECTS)
    for _ in range(count):
        for _ in range(GEN_EXTRA_RETRIES):
            anchor = (random.randrange(grid.shape[0]), random.randrange(grid.shape[1]))
            if anchor in blocked or anchor in occupied:
                continue
            color = random.choice(filter_colors)
            cells = _try_place_object(grid, occupied, blocked, anchor, int(color))
            if cells is not None:
                break


def _object_count_pattern(frame_count: int) -> list[int]:
    if frame_count == 2:
        pair = random.choice(((0, 1), (1, 2), (0, 2)))
        out = list(pair)
        random.shuffle(out)
        return out
    out = [0, 1, 2]
    random.shuffle(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    frame_count = random.randint(*GEN_FRAME_COUNT)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_colors = palette[:frame_count]
    filter_colors = palette[frame_count:]

    frames = _place_frames(h, w, frame_colors)
    if frames is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for frame in frames:
        _draw_frame(grid, frame["bbox"], int(frame["color"]))  # type: ignore[index]
        occupied.update(frame["cells"])  # type: ignore[arg-type]

    directions = random.sample(list(DIAG_DIRS), frame_count)
    counts = _object_count_pattern(frame_count)
    order = list(range(frame_count))
    order.sort(key=lambda idx: counts[idx], reverse=True)

    rays: list[dict[str, object]] = [None] * frame_count  # type: ignore[assignment]
    reserved: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()

    for idx in order:
        frame = frames[idx]
        seed_and_path = _choose_seed_and_path(grid.shape, frames, frame, reserved, directions[idx], counts[idx])
        if seed_and_path is None:
            return None
        seed, path = seed_and_path
        grid[seed] = int(frame["color"])
        occupied.add(seed)
        reserved.add(seed)
        reserved.update(path)
        path_cells.update(path)
        rays[idx] = {
            "seed": seed,
            "path": path,
            "color": int(frame["color"]),
            "object_count": int(counts[idx]),
            "direction": directions[idx],
        }

    blocked_for_extras = set(path_cells) | {ray["seed"] for ray in rays}
    for idx in sorted(range(frame_count), key=lambda item: int(rays[item]["object_count"]), reverse=True):
        if not _place_ray_objects(grid, occupied, blocked_for_extras, rays[idx], filter_colors):
            return None

    _place_extra_objects(grid, occupied, blocked_for_extras, filter_colors)

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid diagonal-filter ARC input")
