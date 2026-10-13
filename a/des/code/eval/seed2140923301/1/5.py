from __future__ import annotations

import math
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": tuple(coords), "area": len(coords)})
    return components


def _object_center(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return _round_half_up(sum(rows) / len(rows)), _round_half_up(sum(cols) / len(cols))


def _pick_anchor(center: tuple[int, int], anchors: list[dict[str, object]]) -> int:
    cr, cc = center
    candidates: list[tuple[int, int]] = []
    for idx, anchor in enumerate(anchors):
        ar, ac = anchor["coords"][0]
        if ar == cr and ac != cc:
            candidates.append((abs(ac - cc), idx))
        elif ac == cc and ar != cr:
            candidates.append((abs(ar - cr), idx))
    if not candidates:
        raise AssertionError("object center must align with an anchor")
    candidates.sort()
    if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
        raise AssertionError("nearest aligned anchor must be unique")
    return int(candidates[0][1])


def _translation(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], anchor: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    center = _object_center(coords)
    ar, ac = anchor
    cr, cc = center
    h, w = shape
    size = len(coords)

    if ar == cr:
        step = 1 if ac > cc else -1
        if step > 0:
            max_shift = w - 1 - max(c for _, c in coords)
        else:
            max_shift = min(c for _, c in coords)
        return 0, int(step * min(size, max_shift))

    assert ac == cc
    step = 1 if ar > cr else -1
    if step > 0:
        max_shift = h - 1 - max(r for r, _ in coords)
    else:
        max_shift = min(r for r, _ in coords)
    return int(step * min(size, max_shift)), 0


def _translate(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], delta: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    dr, dc = delta
    return tuple((int(r + dr), int(c + dc)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scatter objects drift toward their aligned anchor pixels by a number of steps equal to object size,
    then take on the anchor color.

    Input structure:
    - bg is the dominant color.
    - Non-bg singleton components are isolated anchor pixels with distinct colors.
    - Every other non-bg component is a 4-connected monochrome object of size 2..6.

    Rule:
    - For each object, round the mean row and mean column of its pixels to get its visual center.
    - Assign the object to the unique nearest anchor that shares that center's row or column.
    - Translate the whole object along that axis toward the anchor by exactly its pixel count, clamping at the grid edge.
    - Recolor the translated object to the anchor's color. Anchors themselves remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    components = _find_components(g, bg)

    anchors = [comp for comp in components if int(comp["area"]) == 1]
    objects = [comp for comp in components if int(comp["area"]) > 1]

    out = np.full_like(g, bg)
    for anchor in anchors:
        (ar, ac) = anchor["coords"][0]
        out[ar, ac] = int(anchor["color"])

    for obj in objects:
        center = _object_center(obj["coords"])
        anchor_idx = _pick_anchor(center, anchors)
        anchor = anchors[anchor_idx]
        anchor_pos = tuple(anchor["coords"][0])
        delta = _translation(obj["coords"], anchor_pos, (h, w))
        moved = _translate(obj["coords"], delta)
        for r, c in moved:
            out[r, c] = int(anchor["color"])

    for anchor in anchors:
        (ar, ac) = anchor["coords"][0]
        out[ar, ac] = int(anchor["color"])
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _enumerate_shapes() -> list[dict[str, object]]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        next_shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes_by_size[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    candidate = (r + dr, c + dc)
                    if candidate not in cells:
                        frontier.add(candidate)
            for candidate in frontier:
                next_shapes.add(_normalize_shape(cells | {candidate}))
        shapes_by_size[size] = next_shapes

    catalog: list[dict[str, object]] = []
    for size in range(2, 7):
        for shape in sorted(shapes_by_size[size]):
            rows = [r for r, _ in shape]
            cols = [c for _, c in shape]
            center = (_round_half_up(sum(rows) / size), _round_half_up(sum(cols) / size))
            rel = tuple(sorted((r - center[0], c - center[1]) for r, c in shape))
            catalog.append(
                {
                    "size": size,
                    "coords": shape,
                    "center": center,
                    "rel": rel,
                }
            )
    return catalog


SHAPES = _enumerate_shapes()


def _shape_cells(shape: dict[str, object], center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    cr, cc = center
    local_center_r, local_center_c = shape["center"]
    return tuple(
        (int(cr - local_center_r + r), int(cc - local_center_c + c))
        for r, c in shape["coords"]
    )


def _cells_in_bounds(cells: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_blocked(cells: tuple[tuple[int, int], ...], blocked: set[tuple[int, int]], include_diagonal: bool = False) -> bool:
    for r, c in cells:
        if (r, c) in blocked:
            return True
        for dr, dc in DIR4:
            if (r + dr, c + dc) in blocked:
                return True
        if include_diagonal:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    if (r + dr, c + dc) in blocked:
                        return True
    return False


def _assignment_idx(cells: tuple[tuple[int, int], ...], anchors: list[tuple[int, int]]) -> int | None:
    center = _object_center(cells)
    candidates: list[tuple[int, int]] = []
    for idx, (ar, ac) in enumerate(anchors):
        if ar == center[0] and ac != center[1]:
            candidates.append((abs(ac - center[1]), idx))
        elif ac == center[1] and ar != center[0]:
            candidates.append((abs(ar - center[0]), idx))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
        return None
    return int(candidates[0][1])


def _scene_valid(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    components = _find_components(grid, bg)
    anchors = [comp for comp in components if int(comp["area"]) == 1]
    objects = [comp for comp in components if int(comp["area"]) > 1]
    if len(anchors) < 2 or len(objects) < 2:
        return False
    anchor_colors = [int(comp["color"]) for comp in anchors]
    if len(set(anchor_colors)) != len(anchor_colors):
        return False
    anchor_positions = [tuple(comp["coords"][0]) for comp in anchors]
    for ar, ac in anchor_positions:
        for dr, dc in DIR4:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != bg:
                return False
    for comp in objects:
        if not (2 <= int(comp["area"]) <= 6):
            return False
        if int(comp["color"]) in anchor_colors:
            return False
        if _assignment_idx(tuple(comp["coords"]), anchor_positions) is None:
            return False
    return True


def _expected_output(
    h: int,
    w: int,
    bg: int,
    anchors: list[tuple[int, int, int]],
    placed_objects: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((h, w), int(bg), dtype=int)
    for ar, ac, color in anchors:
        out[ar, ac] = int(color)
    for obj in placed_objects:
        anchor_color = int(obj["anchor_color"])
        for r, c in obj["output_cells"]:
            out[r, c] = anchor_color
    for ar, ac, color in anchors:
        out[ar, ac] = int(color)
    return out


def _special_band(length: int) -> list[int]:
    return sorted({1, 2, length - 3, length - 2})


def _specs_for_scene(object_count: int) -> list[dict[str, object]] | None:
    for _ in range(100):
        orientations = [random.choice(["h", "v"]) for _ in range(object_count)]
        if object_count >= 2 and len(set(orientations)) == 1:
            orientations[0] = "h"
            orientations[1] = "v"
            random.shuffle(orientations)

        clips = [bool(random.getrandbits(1)) for _ in range(object_count)]
        if object_count >= 2 and len(set(clips)) == 1:
            clips[0] = False
            clips[1] = True
            random.shuffle(clips)

        if sum(ori == "h" and clip for ori, clip in zip(orientations, clips)) > 4:
            continue
        if sum(ori == "v" and clip for ori, clip in zip(orientations, clips)) > 4:
            continue
        return [{"orientation": ori, "clip": clip} for ori, clip in zip(orientations, clips)]
    return None


def _sample_anchors(h: int, w: int, specs: list[dict[str, object]], anchor_count: int) -> list[tuple[int, int]] | None:
    all_rows = list(range(1, h - 1))
    all_cols = list(range(1, w - 1))
    if anchor_count > len(all_rows) or anchor_count > len(all_cols):
        return None

    need_edge_rows = sum(1 for spec in specs if spec["orientation"] == "v" and spec["clip"])
    need_edge_cols = sum(1 for spec in specs if spec["orientation"] == "h" and spec["clip"])
    edge_rows = [v for v in _special_band(h) if 1 <= v < h - 1]
    edge_cols = [v for v in _special_band(w) if 1 <= v < w - 1]
    if need_edge_rows > len(edge_rows) or need_edge_cols > len(edge_cols):
        return None

    used_rows = set(random.sample(edge_rows, need_edge_rows))
    used_cols = set(random.sample(edge_cols, need_edge_cols))
    free_rows = [r for r in all_rows if r not in used_rows]
    free_cols = [c for c in all_cols if c not in used_cols]
    remaining = anchor_count - len(specs)

    extra_rows_needed = len(specs) - need_edge_rows + remaining
    extra_cols_needed = len(specs) - need_edge_cols + remaining
    if len(free_rows) < extra_rows_needed or len(free_cols) < extra_cols_needed:
        return None

    pool_rows = random.sample(free_rows, extra_rows_needed)
    pool_cols = random.sample(free_cols, extra_cols_needed)
    row_iter = iter(pool_rows)
    col_iter = iter(pool_cols)
    edge_row_iter = iter(random.sample(list(used_rows), len(used_rows)))
    edge_col_iter = iter(random.sample(list(used_cols), len(used_cols)))

    anchors: list[tuple[int, int]] = []
    for spec in specs:
        if spec["orientation"] == "v" and spec["clip"]:
            row = int(next(edge_row_iter))
        else:
            row = int(next(row_iter))
        if spec["orientation"] == "h" and spec["clip"]:
            col = int(next(edge_col_iter))
        else:
            col = int(next(col_iter))
        anchors.append((row, col))

    leftover_rows = list(row_iter)
    leftover_cols = list(col_iter)
    for row, col in zip(leftover_rows[:remaining], leftover_cols[:remaining]):
        anchors.append((int(row), int(col)))

    if len({r for r, _ in anchors}) != len(anchors) or len({c for _, c in anchors}) != len(anchors):
        return None
    return anchors


def _preferred_sides(spec: dict[str, object], anchor: tuple[int, int], h: int, w: int) -> list[str]:
    ar, ac = anchor
    if spec["orientation"] == "h":
        if spec["clip"]:
            return ["right"] if ac <= w // 2 else ["left"]
        return random.sample(["left", "right"], 2)
    if spec["clip"]:
        return ["down"] if ar <= h // 2 else ["up"]
    return random.sample(["up", "down"], 2)


def _candidate_centers(
    spec: dict[str, object],
    anchor: tuple[int, int],
    side: str,
    anchor_rows: set[int],
    anchor_cols: set[int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    ar, ac = anchor
    if spec["orientation"] == "h":
        centers = []
        if side == "left":
            cols = [c for c in range(ac) if c not in anchor_cols]
        else:
            cols = [c for c in range(ac + 1, w) if c not in anchor_cols]
        random.shuffle(cols)
        centers.extend((ar, c) for c in cols)
        return centers

    centers = []
    if side == "up":
        rows = [r for r in range(ar) if r not in anchor_rows]
    else:
        rows = [r for r in range(ar + 1, h) if r not in anchor_rows]
    random.shuffle(rows)
    centers.extend((r, ac) for r in rows)
    return centers


def _output_conflict(
    moved_cells: tuple[tuple[int, int], ...],
    anchor_idx: int,
    anchors: list[tuple[int, int]],
    output_cells: set[tuple[int, int]],
) -> bool:
    other_anchors = {pos for idx, pos in enumerate(anchors) if idx != anchor_idx}
    if _touches_blocked(moved_cells, output_cells):
        return True
    if any(cell in other_anchors for cell in moved_cells):
        return True
    for r, c in moved_cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in other_anchors:
                return True
    return False


def _place_object(
    spec: dict[str, object],
    anchor_idx: int,
    anchors: list[tuple[int, int]],
    input_cells: set[tuple[int, int]],
    output_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> dict[str, object] | None:
    anchor = anchors[anchor_idx]
    anchor_rows = {r for r, _ in anchors}
    anchor_cols = {c for _, c in anchors}
    shape_pool = SHAPES.copy()
    random.shuffle(shape_pool)
    if spec["clip"]:
        shape_pool.sort(key=lambda shape: -int(shape["size"]))

    for shape in shape_pool:
        size = int(shape["size"])
        for side in _preferred_sides(spec, anchor, h, w):
            centers = _candidate_centers(spec, anchor, side, anchor_rows, anchor_cols, h, w)
            for center in centers:
                cells = _shape_cells(shape, center)
                if not _cells_in_bounds(cells, h, w):
                    continue
                if _touches_blocked(cells, input_cells):
                    continue
                if _assignment_idx(cells, anchors) != anchor_idx:
                    continue
                delta = _translation(cells, anchor, (h, w))
                shift = abs(delta[0]) + abs(delta[1])
                if shift == 0:
                    continue
                clipped = shift < size
                if clipped != bool(spec["clip"]):
                    continue
                moved = _translate(cells, delta)
                if _output_conflict(moved, anchor_idx, anchors, output_cells):
                    continue
                return {
                    "input_cells": cells,
                    "output_cells": moved,
                    "anchor_idx": anchor_idx,
                    "size": size,
                }
    return None


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    object_count = random.randint(2, 4)
    anchor_count = object_count + random.randint(0, 1)
    if 1 + anchor_count + object_count > len(ALL_COLORS):
        return None

    specs = _specs_for_scene(object_count)
    if specs is None:
        return None

    anchors_pos = _sample_anchors(h, w, specs, anchor_count)
    if anchors_pos is None:
        return None

    colors = random.sample(list(ALL_COLORS), 1 + anchor_count + object_count)
    bg = int(colors[0])
    anchor_colors = [int(color) for color in colors[1 : 1 + anchor_count]]
    object_colors = [int(color) for color in colors[1 + anchor_count :]]
    anchors = [(r, c, anchor_colors[i]) for i, (r, c) in enumerate(anchors_pos)]

    object_order = list(range(object_count))
    object_order.sort(key=lambda idx: (not bool(specs[idx]["clip"]), random.random()))

    input_blocked = {(r, c) for r, c, _color in anchors}
    output_blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * object_count
    for idx in object_order:
        placed_obj = _place_object(specs[idx], idx, anchors_pos, input_blocked, output_blocked, h, w)
        if placed_obj is None:
            return None
        placed[idx] = placed_obj
        input_blocked.update(placed_obj["input_cells"])
        output_blocked.update(placed_obj["output_cells"])

    if any(item is None for item in placed):
        return None
    placed_objects = [item for item in placed if item is not None]

    inp = np.full((h, w), int(bg), dtype=int)
    for r, c, color in anchors:
        inp[r, c] = int(color)
    for obj_idx, obj in enumerate(placed_objects):
        color = object_colors[obj_idx]
        for r, c in obj["input_cells"]:
            inp[r, c] = int(color)
        obj["anchor_color"] = anchor_colors[obj_idx]

    if not _scene_valid(inp):
        return None

    expected = _expected_output(h, w, bg, anchors, placed_objects)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
