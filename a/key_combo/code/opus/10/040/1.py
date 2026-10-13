from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
OBJECT_COLORS = (1, 3, 6)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_DIRS = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "ll": (1, -1),
    "lr": (1, 1),
}
TOP_PAIR = ("ul", "ur")
LEFT_PAIR = ("ul", "ll")
BOX_CHOICES = ((4, 4), (4, 5), (5, 4))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
PATCH_RETRIES = 200
SUBSET_RETRIES = 60
MIN_MARGIN = 2
MIN_BBOX_GAP = 2
GRID_RANGE = (17, 22)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _border_cells(h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    return cells


def _corner_regions_visible(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    regions = [
        _rect_cells(0, 0, min(2, h), min(2, w)),
        _rect_cells(0, max(0, w - 2), min(2, h), min(2, w)),
        _rect_cells(max(0, h - 2), 0, min(2, h), min(2, w)),
        _rect_cells(max(0, h - 2), max(0, w - 2), min(2, h), min(2, w)),
    ]
    return all(bool(region & cells) for region in regions)


def _host_options(h: int, w: int, orientation: str) -> list[tuple[int, int, int, int]]:
    options = []
    if orientation == "wide":
        for oh, ow in ((1, 2), (1, 3)):
            if h < oh + 2 or w < ow + 2:
                continue
            for r0 in range(1, h - oh):
                for c0 in range(1, w - ow):
                    options.append((r0, c0, oh, ow))
    else:
        for oh, ow in ((2, 1), (3, 1)):
            if h < oh + 2 or w < ow + 2:
                continue
            for r0 in range(1, h - oh):
                for c0 in range(1, w - ow):
                    options.append((r0, c0, oh, ow))
    return options


def _try_remove_border_cells(
    rng: random.Random,
    border: set[tuple[int, int]],
    host: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    removable = sorted(border - corners)
    base = border | host
    min_target = max(8, len(base) - 3)
    max_target = min(15, len(base))
    if min_target > max_target:
        return None

    targets = list(range(min_target, max_target + 1))
    rng.shuffle(targets)
    for target in targets:
        remove_count = len(base) - target
        if remove_count < 0 or remove_count > len(removable):
            continue
        if remove_count == 0:
            candidates = [()]
        else:
            candidates = []
            seen: set[tuple[tuple[int, int], ...]] = set()
            all_subsets = list(combinations(removable, remove_count))
            rng.shuffle(all_subsets)
            for subset in all_subsets[:SUBSET_RETRIES]:
                key = tuple(sorted(subset))
                if key not in seen:
                    seen.add(key)
                    candidates.append(subset)

        for subset in candidates:
            removed = set(subset)
            visible = border - removed
            full = base - removed
            if not _is_connected(visible):
                continue
            if not _is_connected(full):
                continue
            if not _corner_regions_visible(visible, h, w):
                continue
            return full
    return None


def _build_patch(rng: random.Random, orientation: str) -> dict | None:
    for _ in range(PATCH_RETRIES):
        h, w = rng.choice(BOX_CHOICES)
        options = _host_options(h, w, orientation)
        if not options:
            continue
        occ_r0, occ_c0, occ_h, occ_w = rng.choice(options)
        border = _border_cells(h, w)
        host = _rect_cells(occ_r0, occ_c0, occ_h, occ_w)
        cells = _try_remove_border_cells(rng, border, host, h, w)
        if cells is None or len(cells) > 15:
            continue
        visible = cells - host
        if not _is_connected(visible):
            continue

        patch = np.zeros((h, w), dtype=int)
        for r, c in visible:
            patch[r, c] = 1
        occ_mask = np.zeros((h, w), dtype=bool)
        for r, c in host:
            occ_mask[r, c] = True

        return {
            "h": h,
            "w": w,
            "visible_cells": visible,
            "full_cells": cells,
            "occ_bbox_rel": (occ_r0, occ_c0, occ_r0 + occ_h - 1, occ_c0 + occ_w - 1),
            "occ_shape": (occ_h, occ_w),
            "orientation": orientation,
            "patch": patch,
            "occ_mask": occ_mask,
        }
    return None


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + MIN_BBOX_GAP < br0
        or br1 + MIN_BBOX_GAP < ar0
        or ac1 + MIN_BBOX_GAP < bc0
        or bc1 + MIN_BBOX_GAP < ac0
    )


def _corner_map_from_bbox(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "ul": (r0, c0),
        "ur": (r0, c1),
        "ll": (r1, c0),
        "lr": (r1, c1),
    }


def _ray_span(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = start
    dr, dc = step
    span = 0
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        span += 1
        r += dr
        c += dc
    return span


def _render_input(scene: dict) -> np.ndarray:
    grid = np.zeros((scene["H"], scene["W"]), dtype=int)
    for obj in scene["objects"]:
        r0, c0 = obj["top_left"]
        h, w = obj["patch"].shape
        region = grid[r0 : r0 + h, c0 : c0 + w]
        region[obj["patch"] != 0] = obj["color"]
        or0, oc0, or1, oc1 = obj["occ_bbox_rel"]
        grid[r0 + or0 : r0 + or1 + 1, c0 + oc0 : c0 + oc1 + 1] = GREY
    return grid


def _erase_occluders(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BG
    return out


def _paint_from_specs(
    base: np.ndarray,
    objects: list[dict],
    *,
    pair_mode: str = "correct",
    reverse_order: bool = False,
) -> np.ndarray:
    out = base.copy()
    order = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    if reverse_order:
        order = list(reversed(order))
    for obj in order:
        corners = _corner_map_from_bbox(obj["bbox"])
        if pair_mode == "all":
            selected = ("ul", "ur", "ll", "lr")
        else:
            if pair_mode == "swapped":
                selected = LEFT_PAIR if obj["occ_shape"][1] > obj["occ_shape"][0] else TOP_PAIR
            else:
                selected = TOP_PAIR if obj["occ_shape"][1] > obj["occ_shape"][0] else LEFT_PAIR
        for name in selected:
            r, c = corners[name]
            dr, dc = CORNER_DIRS[name]
            while _in_bounds(base.shape, r, c):
                base_val = int(base[r, c])
                if base_val in (BG, obj["color"]):
                    out[r, c] = obj["color"]
                r += dr
                c += dc
    return out


def _trace_paintable_rays(base: np.ndarray, obj: dict) -> list[list[tuple[int, int]]]:
    corners = _corner_map_from_bbox(obj["bbox"])
    selected = TOP_PAIR if obj["occ_shape"][1] > obj["occ_shape"][0] else LEFT_PAIR
    paths = []
    for name in selected:
        r, c = corners[name]
        dr, dc = CORNER_DIRS[name]
        path = []
        while _in_bounds(base.shape, r, c):
            if int(base[r, c]) in (BG, obj["color"]):
                path.append((r, c))
            else:
                path.append((r, c))
            r += dr
            c += dc
        paths.append(path)
    return paths


def _wrong_order_changes(base: np.ndarray, objects: list[dict]) -> bool:
    a = _paint_from_specs(base, objects, pair_mode="correct", reverse_order=False)
    b = _paint_from_specs(base, objects, pair_mode="correct", reverse_order=True)
    return not np.array_equal(a, b)


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for coords in _components(grid == GREY):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(coords) != h * w:
            raise AssertionError("grey component must be a solid rectangle")
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            raise AssertionError("grey rectangle bbox must be filled")
        rects.append({"coords": coords, "bbox": bbox, "shape": (h, w)})
    return rects


def _detect_objects_from_grid(grid: np.ndarray, rects: list[dict]) -> list[dict]:
    objects = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)):
        coords = np.argwhere(grid == color)
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        matches = []
        for rect in rects:
            rr0, rc0, rr1, rc1 = rect["bbox"]
            if r0 < rr0 and c0 < rc0 and rr1 < r1 and rc1 < c1:
                matches.append(rect)
        if len(matches) != 1:
            raise AssertionError("each color object must contain exactly one internal grey rectangle")
        objects.append(
            {
                "color": color,
                "bbox": bbox,
                "occ_shape": matches[0]["shape"],
                "rect_bbox": matches[0]["bbox"],
            }
        )
    if len(objects) != len(rects):
        raise AssertionError("expected one rectangle per foreground object")
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each internal grey rectangle, then extend diagonal rays from bbox corners selected by the rectangle's aspect ratio.

    Input structure:
    - A black background contains 2-3 differently colored foreground objects.
    - Each object has exactly one solid grey rectangle fully inside its bounding box, occluding part of the object.
    - The visible colored pixels still determine each object's full bounding box and its four corner coordinates.

    Rule:
    - Find every grey connected component, verify it is a filled rectangle, and erase it to black.
    - For each foreground color, take the bounding box of all visible pixels of that color and associate the unique grey rectangle strictly inside it.
    - If that rectangle is wider than tall, launch rays from the object's upper-left and upper-right bbox corners; if taller than wide, launch rays from the upper-left and lower-left corners.
    - Each selected corner emits a diagonal ray away from the bbox center (UL->up-left, UR->up-right, LL->down-left, LR->down-right) until the grid boundary. Rays repaint background and previous rays, but never overwrite original pixels of other foreground objects. Objects are processed in bbox top-left row-major order, so later rays win at crossings.

    Key insight:
    - The occluder is not just noise to erase: its width-vs-height encodes which pair of bbox corners should emit the diagonal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _detect_rectangles(g)
    assert 2 <= len(rects) <= 3
    base = _erase_occluders(g)
    objects = _detect_objects_from_grid(base, rects)

    out = base.copy()
    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    for obj in ordered:
        corners = _corner_map_from_bbox(obj["bbox"])
        if obj["occ_shape"][1] > obj["occ_shape"][0]:
            selected = TOP_PAIR
        else:
            selected = LEFT_PAIR
        for name in selected:
            r, c = corners[name]
            dr, dc = CORNER_DIRS[name]
            while _in_bounds(g.shape, r, c):
                cell = int(base[r, c])
                if cell in (BG, obj["color"]):
                    out[r, c] = obj["color"]
                r += dr
                c += dc

    return out


def _make_scene(rng: random.Random) -> dict | None:
    H = rng.randint(*GRID_RANGE)
    W = rng.randint(*GRID_RANGE)
    count = rng.choice((2, 3))
    colors = rng.sample(list(OBJECT_COLORS), count)
    orientations = ["wide", "tall"]
    if count == 3:
        orientations.append(rng.choice(("wide", "tall")))
    rng.shuffle(orientations)

    objects = []
    for color, orientation in zip(colors, orientations):
        spec = _build_patch(rng, orientation)
        if spec is None:
            return None
        spec = spec.copy()
        spec["color"] = color
        objects.append(spec)

    need_cross = count >= 2 and rng.random() < 0.45
    for _ in range(PLACEMENT_RETRIES):
        placed = []
        boxes = []
        for obj in objects:
            h, w = obj["patch"].shape
            candidates = [
                (r0, c0)
                for r0 in range(MIN_MARGIN, H - h - MIN_MARGIN + 1)
                for c0 in range(MIN_MARGIN, W - w - MIN_MARGIN + 1)
            ]
            rng.shuffle(candidates)
            choices = []
            for r0, c0 in candidates:
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_bbox_gap_ok(bbox, other) for other in boxes):
                    all_corners = _corner_map_from_bbox(bbox)
                    if min(
                        _ray_span((H, W), all_corners[name], CORNER_DIRS[name])
                        for name in ("ul", "ur", "ll", "lr")
                    ) >= 2:
                        choices.append((r0, c0, bbox))
            if not choices:
                break
            r0, c0, bbox = rng.choice(choices)
            placed_obj = obj.copy()
            placed_obj["top_left"] = (r0, c0)
            placed_obj["bbox"] = bbox
            placed.append(placed_obj)
            boxes.append(bbox)
        if len(placed) != count:
            continue

        scene = {"H": H, "W": W, "objects": placed, "need_cross": need_cross}
        inp = _render_input(scene)
        base = _erase_occluders(inp)

        blocked = False
        for obj in placed:
            paths = _trace_paintable_rays(base, obj)
            for path in paths:
                for r, c in path[1:]:
                    cell = int(base[r, c])
                    if cell not in (BG, obj["color"]):
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue

        ray_sets = []
        for obj in placed:
            cells = set()
            for path in _trace_paintable_rays(base, obj):
                cells.update((r, c) for r, c in path if int(base[r, c]) == BG)
            ray_sets.append(cells)
        cross_count = sum(bool(ray_sets[i] & ray_sets[j]) for i in range(len(ray_sets)) for j in range(i + 1, len(ray_sets)))
        if need_cross and cross_count == 0:
            continue

        scene["cross_count"] = cross_count
        return scene
    return None


def _semantic_checks(scene: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    objects = scene["objects"]
    rects = _detect_rectangles(inp)
    if len(rects) != len(objects):
        return False

    base = _erase_occluders(inp)
    if np.any(base == GREY):
        return False

    detected = _detect_objects_from_grid(base, rects)
    detected_orients = {"wide" if obj["occ_shape"][1] > obj["occ_shape"][0] else "tall" for obj in detected}
    if detected_orients != {"wide", "tall"}:
        return False

    if np.array_equal(expected, base):
        return False
    if np.array_equal(expected, _paint_from_specs(base, detected, pair_mode="all")):
        return False
    if np.array_equal(expected, _paint_from_specs(base, detected, pair_mode="swapped")):
        return False

    if scene["need_cross"]:
        if scene.get("cross_count", 0) == 0:
            return False
        if not _wrong_order_changes(base, detected):
            return False

    changed = int(np.count_nonzero(expected != inp))
    if changed < len(objects) * 4:
        return False
    if changed > inp.size // 2:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        scene = _make_scene(rng)
        if scene is None:
            continue
        inp = _render_input(scene)
        expected = _paint_from_specs(_erase_occluders(inp), scene["objects"])
        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        if not _semantic_checks(scene, inp, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
