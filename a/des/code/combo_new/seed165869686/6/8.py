from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIR_ORDER = ("up", "down", "left", "right")
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400
GEN_OBJECT_COUNT = (2, 3)
GEN_SHAPE_SIDE = (4, 6)
GEN_REPEAT_COUNT = (1, 4)
GEN_TRAIL_LIMIT = 20
GEN_GAP = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_pixel_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not bool(empty[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            count = 0
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                count += 1
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                total += count
    return int(total)


def _pointer_direction(pointer_rel: tuple[int, int], shape: tuple[int, int]) -> str:
    pr, pc = pointer_rel
    h, w = shape
    sides = []
    if pr == 0:
        sides.append("up")
    if pr == h - 1:
        sides.append("down")
    if pc == 0:
        sides.append("left")
    if pc == w - 1:
        sides.append("right")
    assert sides, "pointer pixel must lie on the object's outer edge"
    for side in DIR_ORDER:
        if side in sides:
            return side
    raise AssertionError("unreachable")


def _parse_object(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    counts = Counter(int(grid[r, c]) for r, c in coords)
    body_color = int(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
    pointer_cells = sorted((r, c) for r, c in coords if int(grid[r, c]) != body_color)
    assert len(pointer_cells) == 1, "each object must contain exactly one pointer pixel"
    pointer = pointer_cells[0]

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    rel_cells: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        mask[rr, cc] = True
        rel_cells.append((rr, cc))
    rel_cells.sort()

    direction = _pointer_direction((pointer[0] - r0, pointer[1] - c0), mask.shape)
    repeats = _hole_pixel_count(mask)
    return {
        "body_color": body_color,
        "direction": direction,
        "repeats": int(repeats),
        "top": int(r0),
        "left": int(c0),
        "height": int(h),
        "width": int(w),
        "rel_cells": rel_cells,
    }


def _paint_stamp(out: np.ndarray, top: int, left: int, rel_cells: list[tuple[int, int]], color: int) -> None:
    h, w = out.shape
    for dr, dc in rel_cells:
        rr = int(top + dr)
        cc = int(left + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(color)


def _apply_object(out: np.ndarray, obj: dict[str, object]) -> None:
    direction = str(obj["direction"])
    repeats = int(obj["repeats"])
    step = int(obj["width"] if direction in ("left", "right") else obj["height"])
    vec_r, vec_c = DIR_TO_VEC[direction]
    for k in range(repeats + 1):
        top = int(obj["top"]) + k * step * vec_r
        left = int(obj["left"]) + k * step * vec_c
        _paint_stamp(out, top, left, obj["rel_cells"], int(obj["body_color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked object into a one-color stamp and repeat it along its edge-encoded direction.

    Input structure:
    - The background is black.
    - The grid contains several disjoint 4-connected foreground objects.
    - Each object has one dominant body color, exactly one different-colored pointer pixel on one outer side, and may contain black cells inside its bounding box.

    Rule:
    - For each foreground object, find the unique pointer pixel and read its side of the bounding box as the stamp direction.
    - Count the black pixels inside the object's bounding box that belong to enclosed holes rather than to boundary-connected exterior gaps.
    - Replace the pointer pixel with the body color, then stamp that one-color silhouette in the pointer direction once per enclosed-hole pixel, shifting by one full object width or height each time.
    - Keep the original object in place and leave the background unchanged elsewhere.

    Key insight:
    - The same object encodes two separate control signals at once: the pointer's side gives the translation direction, and the total enclosed-hole area gives how many repeated imprints to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    objects = [_parse_object(g, comp) for comp in _components(g != BG)]
    objects.sort(key=lambda obj: (int(obj["top"]), int(obj["left"])))
    for obj in objects:
        _apply_object(out, obj)
    return out


def _trail_geometry(height: int, width: int, direction: str, repeats: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction == "right":
        return (height, width * (repeats + 1)), (0, 0)
    if direction == "left":
        return (height, width * (repeats + 1)), (0, width * repeats)
    if direction == "down":
        return (height * (repeats + 1), width), (0, 0)
    return (height * (repeats + 1), width), (height * repeats, 0)


def _sample_holes(height: int, width: int, count: int) -> set[tuple[int, int]]:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    assert count <= len(interior)
    if count == 0:
        return set()

    if count == 1 or random.random() < 0.45:
        return set(random.sample(interior, count))

    holes = {random.choice(interior)}
    while len(holes) < count:
        frontier = []
        for r, c in holes:
            for nr, nc in _neighbors4(r, c, height, width):
                if 1 <= nr < height - 1 and 1 <= nc < width - 1 and (nr, nc) not in holes:
                    frontier.append((nr, nc))
        if frontier and random.random() < 0.8:
            holes.add(random.choice(frontier))
        else:
            remaining = [cell for cell in interior if cell not in holes]
            holes.add(random.choice(remaining))
    return holes


def _side_cells(height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "up":
        return [(0, c) for c in range(1, width - 1)]
    if side == "down":
        return [(height - 1, c) for c in range(1, width - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, height - 1)]
    return [(r, width - 1) for r in range(1, height - 1)]


def _inward_neighbor(cell: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = cell
    if side == "up":
        return r + 1, c
    if side == "down":
        return r - 1, c
    if side == "left":
        return r, c + 1
    return r, c - 1


def _sample_notches(height: int, width: int, direction: str, holes: set[tuple[int, int]]) -> set[tuple[int, int]]:
    candidates = []
    for side in DIR_ORDER:
        if side == direction:
            continue
        for cell in _side_cells(height, width, side):
            inward = _inward_neighbor(cell, side)
            if inward in holes:
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    count = random.randint(0, min(2, len(candidates)))
    return set(candidates[:count])


def _make_object(direction: str, body_color: int, pointer_color: int) -> dict[str, object] | None:
    for _ in range(80):
        repeats = random.randint(*GEN_REPEAT_COUNT)
        max_h = GEN_SHAPE_SIDE[1]
        max_w = GEN_SHAPE_SIDE[1]
        if direction in ("left", "right"):
            max_w = min(max_w, GEN_TRAIL_LIMIT // (repeats + 1))
        else:
            max_h = min(max_h, GEN_TRAIL_LIMIT // (repeats + 1))
        if max_h < GEN_SHAPE_SIDE[0] or max_w < GEN_SHAPE_SIDE[0]:
            continue

        height = random.randint(GEN_SHAPE_SIDE[0], max_h)
        width = random.randint(GEN_SHAPE_SIDE[0], max_w)
        if (height - 2) * (width - 2) < repeats:
            continue

        holes = _sample_holes(height, width, repeats)
        notches = _sample_notches(height, width, direction, holes)
        pointer = random.choice(_side_cells(height, width, direction))

        mask = np.ones((height, width), dtype=bool)
        for r, c in holes | notches:
            mask[r, c] = False
        if len(_components(mask)) != 1:
            continue
        if _hole_pixel_count(mask) != repeats:
            continue
        patch = np.zeros((height, width), dtype=int)
        patch[mask] = int(body_color)
        patch[pointer] = int(pointer_color)

        trail_shape, origin_offset = _trail_geometry(height, width, direction, repeats)
        return {
            "direction": direction,
            "repeats": int(repeats),
            "body_color": int(body_color),
            "pointer_color": int(pointer_color),
            "patch": patch,
            "mask": mask,
            "height": int(height),
            "width": int(width),
            "trail_shape": trail_shape,
            "origin_offset": origin_offset,
        }
    return None


def _grid_choices(boxes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_h = max(h for h, _w in boxes)
    min_w = max(w for _h, w in boxes)
    needed_area = sum((h + GEN_GAP) * (w + GEN_GAP) for h, w in boxes)
    choices = []
    for height in range(max(12, min_h), 31):
        for width in range(max(12, min_w), 31):
            if height * width >= needed_area:
                choices.append((height, width))
    return choices


def _place_boxes(grid_h: int, grid_w: int, boxes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = list(range(len(boxes)))
    random.shuffle(order)
    order.sort(key=lambda idx: boxes[idx][0] * boxes[idx][1], reverse=True)

    for idx in order:
        box_h, box_w = boxes[idx]
        positions = [(r0, c0) for r0 in range(grid_h - box_h + 1) for c0 in range(grid_w - box_w + 1)]
        random.shuffle(positions)
        found = None
        for r0, c0 in positions:
            rr0 = max(0, r0 - GEN_GAP)
            rr1 = min(grid_h, r0 + box_h + GEN_GAP)
            cc0 = max(0, c0 - GEN_GAP)
            cc1 = min(grid_w, c0 + box_w + GEN_GAP)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue
            reserved[r0 : r0 + box_h, c0 : c0 + box_w] = True
            found = (int(r0), int(c0))
            break
        if found is None:
            return None
        placements[idx] = found

    return [placement for placement in placements if placement is not None]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    window = grid[top : top + h, left : left + w]
    mask = patch != BG
    window[mask] = patch[mask]


def _expected_output(inp: np.ndarray, placed_objects: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for obj in placed_objects:
        _apply_object(out, obj)
    return out


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.randint(*GEN_OBJECT_COUNT)
    directions = random.sample(list(DIR_ORDER), obj_count)
    palette = random.sample(list(range(1, 10)), 2 * obj_count)

    objects: list[dict[str, object]] = []
    for idx in range(obj_count):
        spec = _make_object(directions[idx], palette[2 * idx], palette[2 * idx + 1])
        if spec is None:
            return None
        objects.append(spec)

    box_shapes = [tuple(map(int, obj["trail_shape"])) for obj in objects]
    choices = _grid_choices(box_shapes)
    if not choices:
        return None
    random.shuffle(choices)

    placements = None
    grid_shape = None
    for height, width in choices[:120]:
        placements = _place_boxes(height, width, box_shapes)
        if placements is not None:
            grid_shape = (height, width)
            break
    if placements is None or grid_shape is None:
        return None

    grid = np.zeros(grid_shape, dtype=int)
    placed_for_expected: list[dict[str, object]] = []
    for spec, (trail_top, trail_left) in zip(objects, placements):
        off_r, off_c = spec["origin_offset"]
        top = int(trail_top + off_r)
        left = int(trail_left + off_c)
        _paste_patch(grid, spec["patch"], top, left)
        placed_for_expected.append(
            {
                "direction": spec["direction"],
                "repeats": spec["repeats"],
                "body_color": spec["body_color"],
                "top": top,
                "left": left,
                "height": spec["height"],
                "width": spec["width"],
                "rel_cells": [tuple(map(int, cell)) for cell in np.argwhere(spec["mask"]).tolist()],
            }
        )

    expected = _expected_output(grid, placed_for_expected)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping puzzle input")
