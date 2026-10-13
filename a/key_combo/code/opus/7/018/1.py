import random
from itertools import combinations

import numpy as np


DIRS_BY_DEPTH = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

COLOR_POOL = [1, 3, 4, 5, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            anchor = min(coords)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "anchor": anchor,
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _annotate_depths(objects: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for obj in objects:
        depth = sum(
            1 for other in objects if other is not obj and _strict_contains(other["bbox"], obj["bbox"])
        )
        item = dict(obj)
        item["depth"] = depth
        annotated.append(item)
    return annotated


def _sorted_objects(objects: list[dict]) -> list[dict]:
    return sorted(objects, key=lambda o: (o["depth"], o["anchor"][0], o["anchor"][1]))


def _apply_rule(grid: np.ndarray, *, max_depth: int | None = None) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    objects = _sorted_objects(_annotate_depths(_extract_objects(out)))
    h, w = out.shape

    for obj in objects:
        if max_depth is not None and obj["depth"] > max_depth:
            continue
        dr, dc = DIRS_BY_DEPTH[obj["depth"] % 4]
        rr = obj["anchor"][0] + dr
        cc = obj["anchor"][1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            if int(out[rr, cc]) != 0:
                break
            out[rr, cc] = obj["color"]
            rr += dr
            cc += dc
    return out


def _paint_shape(grid: np.ndarray, top: int, left: int, coords: list[tuple[int, int]], color: int) -> None:
    for dr, dc in coords:
        grid[top + dr, left + dc] = color


def _shape_top_right(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords.extend((r, w - 1) for r in range(1, h))
    return coords


def _shape_top_left(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords.extend((r, 0) for r in range(1, h))
    return coords


def _shape_left_bottom(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(r, 0) for r in range(h)]
    coords.extend((h - 1, c) for c in range(1, w))
    return coords


def _shape_small_up_blocker() -> tuple[int, int, list[tuple[int, int]]]:
    h, w = random.choice([(2, 2), (2, 3), (3, 2)])
    return h, w, _shape_left_bottom(h, w)


def _shape_depth2_core() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (1, 1)]


def _has_cross_color_adjacency(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(g[nr, nc])
                    if other != 0 and other != color:
                        return True
    return False


def _anchor_has_two_bg_steps(grid: np.ndarray, obj: dict) -> bool:
    dr, dc = DIRS_BY_DEPTH[obj["depth"] % 4]
    r, c = obj["anchor"]
    for step in (1, 2):
        rr = r + step * dr
        cc = c + step * dc
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != 0:
            return False
    return True


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            return False
        if max(grid.shape) > 30:
            return False
        if _has_cross_color_adjacency(grid):
            return False

        objects = _sorted_objects(_annotate_depths(_extract_objects(grid)))
        if not (4 <= len(objects) <= 5):
            return False
        if len({obj["color"] for obj in objects}) != len(objects):
            return False

        depth_counts = {d: sum(obj["depth"] == d for obj in objects) for d in range(4)}
        if depth_counts[1] != 1 or depth_counts[2] != 1 or depth_counts[0] < 2:
            return False

        for a, b in combinations(objects, 2):
            a_contains_b = _strict_contains(a["bbox"], b["bbox"])
            b_contains_a = _strict_contains(b["bbox"], a["bbox"])
            if _bboxes_overlap(a["bbox"], b["bbox"]) and not (a_contains_b or b_contains_a):
                return False

        if not all(_anchor_has_two_bg_steps(grid, obj) for obj in objects):
            return False

        out = _apply_rule(grid)
        if np.array_equal(out, grid):
            return False

        depth1_obj = next(obj for obj in objects if obj["depth"] == 1)
        prefix = _apply_rule(grid, max_depth=0)
        dr, dc = DIRS_BY_DEPTH[1]
        rr = depth1_obj["anchor"][0] + dr
        cc = depth1_obj["anchor"][1] + dc
        bg_steps = 0
        blocked_by_prior_ray = False
        while _in_bounds(grid.shape, rr, cc):
            if int(prefix[rr, cc]) != 0:
                blocked_by_prior_ray = int(grid[rr, cc]) == 0
                break
            bg_steps += 1
            rr += dr
            cc += dc
        if bg_steps < 2 or not blocked_by_prior_ray:
            return False

        depth2_obj = next(obj for obj in objects if obj["depth"] == 2)
        dr2, dc2 = DIRS_BY_DEPTH[2]
        rr = depth2_obj["anchor"][0] + dr2
        cc = depth2_obj["anchor"][1] + dc2
        painted = 0
        while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == 0:
            painted += 1
            rr += dr2
            cc += dc2
        if painted < 2:
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(20, 26)
    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(COLOR_POOL, k=random.choice([4, 5]))

    a_h = random.choice([8, 9])
    a_w = random.choice([8, 9])
    b_h = b_w = 5
    c_h = c_w = 2

    max_a_top = height - a_h - 5
    max_a_left = width - a_w - 7
    if max_a_top < 3 or max_a_left < 1:
        return None

    a_top = random.randint(3, max_a_top)
    a_left = random.randint(1, min(3, max_a_left))
    b_top = a_top + random.choice([2, 3] if a_h == 9 else [2])
    b_left = a_left + random.choice([2, 3] if a_w == 9 else [2])
    c_top = b_top + 1
    c_left = b_left + 2

    a_shape = _shape_top_left(a_h, a_w)
    b_shape = _shape_left_bottom(b_h, b_w)
    c_shape = _shape_depth2_core()

    _paint_shape(grid, a_top, a_left, a_shape, colors[0])
    _paint_shape(grid, b_top, b_left, b_shape, colors[1])
    _paint_shape(grid, c_top, c_left, c_shape, colors[2])

    d_h, d_w, d_shape = _shape_small_up_blocker()
    d_top_min = b_top + 4
    d_top_max = height - d_h - 2
    d_left_min = a_left + a_w + 2
    d_left_max = width - d_w - 2
    if d_top_min > d_top_max or d_left_min > d_left_max:
        return None
    d_top = random.randint(d_top_min, d_top_max)
    d_left = random.randint(d_left_min, d_left_max)
    _paint_shape(grid, d_top, d_left, d_shape, colors[3])

    if len(colors) == 5:
        e_h, e_w, e_shape = _shape_small_up_blocker()
        candidate_slots: list[tuple[int, int]] = []
        for top in range(max(a_top + a_h + 1, 3), height - e_h - 1):
            for left in range(1, min(a_left - 3, width - e_w - 2) + 1):
                bbox = (top, left, top + e_h - 1, left + e_w - 1)
                occupied = False
                for existing in _extract_objects(grid):
                    eb = existing["bbox"]
                    if _bboxes_overlap(bbox, eb):
                        occupied = True
                        break
                    if abs(bbox[0] - eb[2]) <= 1 and not (bbox[3] < eb[1] or eb[3] < bbox[1]):
                        occupied = True
                        break
                    if abs(bbox[1] - eb[3]) <= 1 and not (bbox[2] < eb[0] or eb[2] < bbox[0]):
                        occupied = True
                        break
                if not occupied:
                    candidate_slots.append((top, left))
        if not candidate_slots:
            return None
        e_top, e_left = random.choice(candidate_slots)
        _paint_shape(grid, e_top, e_left, e_shape, colors[4])

    if not _validate_generated_grid(grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast one orthogonal ray from each object, with the ray direction encoded by bounding-box nesting depth.

    Input structure:
    - The grid is black-background and contains several 4-connected monochromatic objects.
    - Objects are separated from each other, and some smaller objects lie strictly inside the bounding boxes of larger ones.
    - Each object is represented only by its visible pixels; the nesting relation must be inferred from all object bounding boxes.

    Rule:
    - Find every non-black 4-connected monochromatic object and compute its bounding box.
    - The nesting depth of an object is the number of other object bounding boxes that strictly contain its box.
    - Map depth mod 4 to a ray direction: 0 -> up, 1 -> right, 2 -> down, 3 -> left.
    - Process objects from shallow depth to deep depth; ties use the topmost anchor pixel and then the leftmost anchor pixel.
    - For each object, take its topmost pixel, breaking ties by the leftmost column, and extend a ray from the next cell in the encoded direction through background only, stopping before the first non-background cell or the grid boundary.

    Key insight:
    - The emitted direction is not a local property of shape or color; it comes from the object's global bounding-box containment depth, and earlier rays become real obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _apply_rule(g)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth ray puzzle")
