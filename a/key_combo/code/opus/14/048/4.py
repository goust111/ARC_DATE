from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7)
SIDES = ("top", "bottom", "left", "right")
GRID_BOUNDS = (18, 22)
OBJECT_COUNT_RANGE = (2, 3)
MIN_OBJECT_GAP = 5
GENERATE_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _count_layers(rect: np.ndarray) -> tuple[int, list[int]]:
    sub = np.asarray(rect, dtype=int)
    colors: list[int] = []
    while True:
        assert sub.size > 0, "empty rectangle"
        border = [sub[0, :], sub[-1, :]]
        if sub.shape[0] > 2:
            border.append(sub[1:-1, 0])
            border.append(sub[1:-1, -1])
        border_vals = np.concatenate(border)
        color = int(border_vals[0])
        assert np.all(border_vals == color), "each peel boundary must be uniform"
        assert color not in colors, "nested layer colors must be distinct"
        colors.append(color)
        if np.all(sub == color):
            return len(colors), colors
        assert sub.shape[0] >= 3 and sub.shape[1] >= 3, "non-core layer must leave an interior"
        sub = sub[1:-1, 1:-1]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objs: list[dict] = []
    for coords in _components(g != bg):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        rect = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(rect != bg), "foreground object must fill its bbox"
        depth, layer_colors = _count_layers(rect)
        objs.append(
            {
                "id": bbox,
                "bbox": bbox,
                "rect": rect,
                "depth": depth,
                "layer_colors": layer_colors,
                "outer": int(layer_colors[0]),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    assert 2 <= len(objs) <= 4, "expected 2-4 nested rectangles"
    return objs


def _ray_cells(bbox: tuple[int, int, int, int], side: str, step: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 - step, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1 + step, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0 - step) for r in range(r0 + 1, r1)]
    return [(r, c1 + step) for r in range(r0 + 1, r1)]


def _simulate(
    grid: np.ndarray,
    *,
    sort_mode: str = "col_row",
    return_lengths: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[tuple[int, int, int, int], dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objs = _extract_objects(g)
    if sort_mode == "col_row":
        objs = sorted(objs, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    elif sort_mode == "row_col":
        objs = sorted(objs, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    else:
        raise ValueError(sort_mode)

    out = g.copy()
    lengths: dict[tuple[int, int, int, int], dict[str, int]] = {}
    for obj in objs:
        bbox = obj["bbox"]
        depth = int(obj["depth"])
        color = int(obj["outer"])
        lengths[obj["id"]] = {}
        for side in SIDES:
            actual = 0
            for step in range(1, depth + 1):
                cells = _ray_cells(bbox, side, step)
                if not cells:
                    break
                if any((not _in_bounds(out.shape, r, c)) or out[r, c] != bg for r, c in cells):
                    break
                for r, c in cells:
                    out[r, c] = color
                actual = step
            lengths[obj["id"]][side] = actual
    if return_lengths:
        return out, lengths
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend four outer-color ray stripes from each nested rectangle by a length equal to its nesting depth.

    Input structure:
    - The grid has a black background and 2-4 separated solid rectangular foreground objects.
    - Each foreground object is a fully filled rectangle made from 2-4 inward nested 1-pixel color layers with a single-color core.
    - Different objects may have different depths, sizes, and outer colors; the depth is implicit in the visible layer nesting.

    Rule:
    - For each object, count its nesting depth by repeatedly peeling a uniform outer border until only the single-color core remains.
    - Process objects by the upper-left bbox corner from left to right, then top to bottom, using that depth D as the ray length.
    - From each side of the bbox, extend an outer-color stripe outward for up to D cells; each stripe omits the side's two corner-aligned cells, and stops early if the next stripe slice would hit any non-background cell.

    Key insight:
    - The hidden numeric quantity is the number of nested layers, and that count directly controls a spatial extension length whose exact result can depend on inter-object ray blocking order.
    """
    return _simulate(np.asarray(grid, dtype=int), sort_mode="col_row")


def _build_nested_patch(depth: int, h: int, w: int, colors: list[int]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    top = left = 0
    bottom, right = h, w
    for i, color in enumerate(colors):
        patch[top:bottom, left:right] = int(color)
        if i < depth - 1:
            top += 1
            left += 1
            bottom -= 1
            right -= 1
    return patch


def _random_colors(depth: int, forbidden_outer: set[int] | None = None) -> list[int]:
    forbidden_outer = set() if forbidden_outer is None else set(forbidden_outer)
    outer_choices = [c for c in PALETTE if c not in forbidden_outer]
    outer = random.choice(outer_choices)
    inner_choices = [c for c in PALETTE if c != outer]
    return [outer] + random.sample(inner_choices, depth - 1)


def _make_object(depth: int, h: int, w: int, *, forbidden_outer: set[int] | None = None) -> dict:
    colors = _random_colors(depth, forbidden_outer)
    patch = _build_nested_patch(depth, h, w, colors)
    return {
        "depth": int(depth),
        "h": int(h),
        "w": int(w),
        "outer": int(colors[0]),
        "patch": patch,
    }


def _make_anchor_pair() -> tuple[dict, dict]:
    if random.random() < 0.6:
        h = random.randint(5, 6)
        w = random.randint(5, 6)
        first = _make_object(3, h, w)
        second = _make_object(2, h, w, forbidden_outer={first["outer"]})
        return first, second

    shared_h = random.randint(5, 7)
    first = _make_object(3, shared_h, 5)
    second_w = random.randint(4, 6)
    second = _make_object(2, shared_h, second_w, forbidden_outer={first["outer"]})
    return first, second


def _make_extra_object() -> dict:
    depth = random.choices((2, 3, 4), weights=(4, 4, 1))[0]
    min_h = 2 * depth - 1
    min_w = 2 * depth - 1
    h = random.randint(min_h, min(8, min_h + 2))
    w = random.randint(min_w, min(8, min_w + 2))
    return _make_object(depth, h, w)


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[int, int]:
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    row_gap = max(0, r0b - r1a - 1, r0a - r1b - 1)
    col_gap = max(0, c0b - c1a - 1, c0a - c1b - 1)
    return int(row_gap), int(col_gap)


def _expanded_bbox(bbox: tuple[int, int, int, int], depth: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - depth, c0 - depth, r1 + depth, c1 + depth


def _boxes_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _extra_position_valid(candidate: tuple[int, int, int, int], depth: int, placed: list[dict]) -> bool:
    r0, c0, r1, c1 = candidate
    if r0 < depth or c0 < depth:
        return False
    for obj in placed:
        row_gap, col_gap = _bbox_gap(candidate, obj["bbox"])
        if max(row_gap, col_gap) < MIN_OBJECT_GAP:
            return False
        if _boxes_overlap(_expanded_bbox(candidate, depth), _expanded_bbox(obj["bbox"], obj["depth"])):
            return False
    return True


def _place_objects(h: int, w: int, objects: list[dict]) -> bool:
    placed: list[dict] = []
    for obj in sorted(objects, key=lambda item: (item["h"] * item["w"], item["depth"]), reverse=True):
        if h - obj["h"] - obj["depth"] < obj["depth"] or w - obj["w"] - obj["depth"] < obj["depth"]:
            return False
        for _ in range(200):
            r0 = random.randint(obj["depth"], h - obj["depth"] - obj["h"])
            c0 = random.randint(obj["depth"], w - obj["depth"] - obj["w"])
            bbox = (r0, c0, r0 + obj["h"] - 1, c0 + obj["w"] - 1)
            if _extra_position_valid(bbox, obj["depth"], placed):
                obj["bbox"] = bbox
                placed.append(obj)
                break
        else:
            return False
    return True


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        objects = _extract_objects(inp)
        out, lengths = _simulate(inp, sort_mode="col_row", return_lengths=True)
    except AssertionError:
        return False

    if np.array_equal(inp, out):
        return False
    depths = [obj["depth"] for obj in objects]
    if len(set(depths)) < 2:
        return False
    same_size_diff_depth = any(
        a["h"] == b["h"] and a["w"] == b["w"] and a["depth"] != b["depth"]
        for i, a in enumerate(objects)
        for b in objects[i + 1 :]
    )
    if not same_size_diff_depth:
        return False

    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            row_gap, col_gap = _bbox_gap(a["bbox"], b["bbox"])
            if max(row_gap, col_gap) < MIN_OBJECT_GAP:
                return False

    for obj in objects:
        if not any(lengths[obj["id"]][side] > 0 for side in SIDES):
            return False
        if any(ray_len < obj["depth"] for ray_len in lengths[obj["id"]].values()):
            return False

    anti_shortcut = any(
        a["depth"] > b["depth"] and not (a["h"] > b["h"] and a["w"] > b["w"])
        for a in objects
        for b in objects
        if a is not b
    )
    if not anti_shortcut:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_BOUNDS[0], GRID_BOUNDS[1])
    w = random.randint(GRID_BOUNDS[0], GRID_BOUNDS[1])
    count = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])

    first, second = _make_anchor_pair()
    objects = [first, second]
    while len(objects) < count:
        objects.append(_make_extra_object())
    if not _place_objects(h, w, objects):
        return None

    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        _paint_patch(grid, obj["patch"], r0, c0)

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATE_ATTEMPTS} attempts")
