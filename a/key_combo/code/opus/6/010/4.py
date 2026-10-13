from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, grid.flatten()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[Coord] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def find_components(grid: np.ndarray, *, bg: int, connectivity: int = 4) -> list[dict]:
    if connectivity != 4:
        raise ValueError("this solver only uses 4-connectivity")
    h, w = grid.shape
    fg = grid != int(bg)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _recolor_patch(patch: np.ndarray, src_base: int, src_pattern: int, dst_base: int, dst_pattern: int) -> np.ndarray:
    out = patch.copy()
    out[patch == int(src_base)] = int(dst_base)
    out[patch == int(src_pattern)] = int(dst_pattern)
    return out


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4)
    assert len(comps) == 4, "expected four foreground objects"

    h, w = g.shape
    grid_center_r = (h - 1) / 2.0
    grid_center_c = (w - 1) / 2.0
    objects: list[dict] = []
    quadrants: set[tuple[int, int]] = set()
    for comp in comps:
        box = comp["bbox"]
        r0, c0, r1, c1 = box
        patch = crop_bbox(g, box)
        assert np.all(patch != bg), "objects must be solid rectangles"
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        assert obj_h % 2 == 0, "object height must be even"
        cols = colors_present(patch, exclude={bg})
        assert len(cols) == 2, "each object must use exactly two colors"
        counts = sorted(((int((patch == col).sum()), int(col)) for col in cols), reverse=True)
        assert counts[0][0] > counts[1][0], "base color must be the strict majority"
        base_color = counts[0][1]
        pattern_color = counts[1][1]
        pattern_mask = patch == pattern_color
        assert 3 <= int(pattern_mask.sum()) <= 6, "pattern size must be 3..6"
        assert is_contiguous(pattern_mask), "pattern must be 4-connected"

        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        sign_r = -1 if center_r < grid_center_r else 1
        sign_c = -1 if center_c < grid_center_c else 1
        quadrants.add((sign_r, sign_c))

        half_h = obj_h // 2
        objects.append(
            {
                "bbox": box,
                "patch": patch,
                "upper": patch[:half_h].copy(),
                "lower": patch[half_h:].copy(),
                "half_h": half_h,
                "height": obj_h,
                "width": obj_w,
                "base_color": base_color,
                "pattern_color": pattern_color,
                "center_r": center_r,
                "center_c": center_c,
                "angle": math.atan2(center_r - grid_center_r, center_c - grid_center_c),
            }
        )
    assert len(quadrants) == 4, "objects must occupy four distinct quadrants"
    return bg, objects


def _render_from_order(shape: tuple[int, int], bg: int, objects: list[dict], order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    n = len(order)
    for i, idx in enumerate(order):
        obj = objects[idx]
        nxt = objects[order[(i + 1) % n]]
        r0, c0, _r1, _c1 = obj["bbox"]
        w = int(obj["width"])
        half_h = int(obj["half_h"])
        upper = _recolor_patch(
            obj["upper"],
            obj["base_color"],
            obj["pattern_color"],
            nxt["base_color"],
            nxt["pattern_color"],
        )
        paste(out, obj["lower"], r0 + half_h, c0)
        paste(out, upper, r0, c0 + w)
    return out


def _correct_order(objects: list[dict]) -> list[int]:
    return sorted(range(len(objects)), key=lambda i: objects[i]["angle"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each quadrant rectangle, recolor its upper half with the next rectangle's palette, then diagonally rejoin it.

    Input structure:
    - The grid has a dominant background and exactly four separated foreground objects.
    - Each object is a solid even-height rectangle occupying a distinct quadrant around the grid center.
    - Every rectangle uses exactly two unique non-background colors: a majority base color and a 4-connected minority pattern color.

    Rule:
    - Compute each rectangle's bbox center and its polar angle around the grid center using the grid's row/column coordinate frame.
    - Sort those angles from small to large to get the clockwise ring order.
    - Split every rectangle into equal upper and lower halves across its horizontal midpoint.
    - Recolor only the upper half: base color becomes the next rectangle's base color, and pattern color becomes the next rectangle's pattern color.
    - Keep the lower half fixed, and paste the recolored upper half shifted right by exactly one rectangle width so its lower-left corner is diagonally adjacent to the lower half's upper-right corner.

    Key insight:
    - The palette flow is not fixed by color ids or reading order; it is encoded by the rectangles' spatial clockwise order around the grid center.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _extract_objects(g)
    order = _correct_order(objects)
    return _render_from_order(g.shape, bg, objects, order)


def _rects_separated(a: BBox, b: BBox, pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_pattern_mask(h: int, w: int) -> np.ndarray | None:
    half_h = h // 2
    for _ in range(200):
        target = random.randint(3, 6)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < target:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if not any(r < half_h for r in rows) or not any(r >= half_h for r in rows):
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if is_contiguous(mask):
            return mask
    return None


def _make_object_patch(h: int, w: int, base_color: int, pattern_color: int) -> np.ndarray | None:
    pattern = _random_pattern_mask(h, w)
    if pattern is None:
        return None
    patch = np.full((h, w), int(base_color), dtype=int)
    patch[pattern] = int(pattern_color)
    return patch


def _quadrant_ok(r0: int, c0: int, h: int, w: int, grid_h: int, grid_w: int, quadrant: str) -> bool:
    center_r = r0 + (h - 1) / 2.0
    center_c = c0 + (w - 1) / 2.0
    grid_center_r = (grid_h - 1) / 2.0
    grid_center_c = (grid_w - 1) / 2.0
    if quadrant == "ul":
        return center_r < grid_center_r and center_c < grid_center_c
    if quadrant == "ur":
        return center_r < grid_center_r and center_c > grid_center_c
    if quadrant == "br":
        return center_r > grid_center_r and center_c > grid_center_c
    if quadrant == "bl":
        return center_r > grid_center_r and center_c < grid_center_c
    raise ValueError("bad quadrant")


def _place_objects(grid_h: int, grid_w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    input_boxes: list[BBox] = []
    output_boxes: list[BBox] = []
    for spec in sorted(specs, key=lambda s: (s["width"], s["height"]), reverse=True):
        h = int(spec["height"])
        w = int(spec["width"])
        if grid_w - 2 * w - 2 <= 0 or grid_h - h - 2 <= 0:
            return None
        found = False
        for _ in range(400):
            r0 = random.randint(1, grid_h - h - 1)
            c0 = random.randint(1, grid_w - 2 * w - 1)
            if not _quadrant_ok(r0, c0, h, w, grid_h, grid_w, spec["quadrant"]):
                continue
            in_box = (r0, c0, r0 + h - 1, c0 + w - 1)
            out_box = (r0, c0, r0 + h - 1, c0 + 2 * w - 1)
            if any(not _rects_separated(in_box, other, 2) for other in input_boxes):
                continue
            if any(not _rects_separated(out_box, other, 1) for other in output_boxes):
                continue
            placed.append({**spec, "r0": r0, "c0": c0})
            input_boxes.append(in_box)
            output_boxes.append(out_box)
            found = True
            break
        if not found:
            return None
    return placed


def _build_candidate() -> np.ndarray | None:
    grid_h = random.randint(20, 28)
    grid_w = random.randint(26, 30)
    bg = random.randrange(10)
    fg_pool = [c for c in range(10) if c != bg]
    if len(fg_pool) < 8:
        return None
    fg_colors = random.sample(fg_pool, 8)
    quadrants = ["ul", "ur", "br", "bl"]
    specs: list[dict] = []
    for i, quadrant in enumerate(quadrants):
        height = random.choice([4, 6])
        width = random.choice([5, 6, 7])
        base_color = fg_colors[2 * i]
        pattern_color = fg_colors[2 * i + 1]
        patch = _make_object_patch(height, width, base_color, pattern_color)
        if patch is None:
            return None
        specs.append(
            {
                "quadrant": quadrant,
                "height": height,
                "width": width,
                "base_color": base_color,
                "pattern_color": pattern_color,
                "patch": patch,
            }
        )
    placed = _place_objects(grid_h, grid_w, specs)
    if placed is None:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    for spec in placed:
        paste(grid, spec["patch"], int(spec["r0"]), int(spec["c0"]))
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    bg, objects = _extract_objects(grid)
    order = _correct_order(objects)
    out = _render_from_order(grid.shape, bg, objects, order)
    if not np.array_equal(main(grid), out):
        return False

    row_major = sorted(range(4), key=lambda i: (objects[i]["center_r"], objects[i]["center_c"]))
    col_major = sorted(range(4), key=lambda i: (objects[i]["center_c"], objects[i]["center_r"]))
    reverse_angle = sorted(range(4), key=lambda i: objects[i]["angle"], reverse=True)
    wrong_orders = [row_major, col_major, reverse_angle]
    if any(order == wrong for wrong in wrong_orders):
        return False
    if any(np.array_equal(out, _render_from_order(grid.shape, bg, objects, wrong)) for wrong in wrong_orders):
        return False

    seen_colors = set()
    for obj in objects:
        cols = {int(obj["base_color"]), int(obj["pattern_color"])}
        if seen_colors & cols:
            return False
        seen_colors |= cols
        upper = obj["upper"]
        lower = obj["lower"]
        if not np.any(upper == obj["pattern_color"]):
            return False
        if not np.any(lower == obj["pattern_color"]):
            return False
        r0, c0, _r1, _c1 = obj["bbox"]
        w = int(obj["width"])
        half_h = int(obj["half_h"])
        shifted = out[r0 : r0 + half_h, c0 + w : c0 + 2 * w]
        if set(colors_present(shifted, exclude={bg})) != {objects[order[(order.index(objects.index(obj)) + 1) % 4]]["base_color"], objects[order[(order.index(objects.index(obj)) + 1) % 4]]["pattern_color"]}:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _build_candidate()
        if grid is not None and _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
