from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


try:
    from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "new" / "solvers"
        if (candidate / "arc_utils.py").exists():
            sys.path.append(str(candidate))
            break
    from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite


BACKGROUND = 0
RECTANGLE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SPRITE_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]
MAX_GENERATION_ATTEMPTS = 400

_K_BAG: list[int] = []
_SPRITE_SHAPE_BAG: list[tuple[int, int]] = []


def _bagged_k() -> int:
    global _K_BAG
    if not _K_BAG:
        _K_BAG = random.sample([2, 3, 4], 3)
    return int(_K_BAG.pop())


def _bagged_sprite_shape() -> tuple[int, int]:
    global _SPRITE_SHAPE_BAG
    if not _SPRITE_SHAPE_BAG:
        _SPRITE_SHAPE_BAG = random.sample(SPRITE_SHAPES, len(SPRITE_SHAPES))
    return tuple(_SPRITE_SHAPE_BAG.pop())


def _is_hollow_rectangle(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    h, w = map(int, patch.shape)
    if h < 4 or w < 4:
        return None
    expected = np.full((h, w), color, dtype=int)
    expected[1:-1, 1:-1] = int(bg)
    if not np.array_equal(patch, expected):
        return None
    inner_h = h - 2
    inner_w = w - 2
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "inner_shape": (inner_h, inner_w),
        "inner_area": inner_h * inner_w,
    }


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    hollows = [info for comp in mono if (info := _is_hollow_rectangle(g, comp, bg)) is not None]
    assert len(hollows) >= 2, "expected at least two hollow rectangles"

    max_area = max(int(item["inner_area"]) for item in hollows)
    largest = [item for item in hollows if int(item["inner_area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest hollow interior"
    canvas = largest[0]

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    multicolor = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) > 1:
            multicolor.append(comp)
    assert len(multicolor) == 1, "expected one multicolor sprite component"
    sprite_bbox = tuple(map(int, multicolor[0]["bbox"]))
    sprite = crop_bbox(g, sprite_bbox)

    return {
        "bg": bg,
        "K": len(hollows),
        "hollows": hollows,
        "canvas": canvas,
        "sprite_bbox": sprite_bbox,
        "sprite": sprite,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Count hollow rectangles to get the upscale factor, then fill the largest hollow frame with the enlarged sprite.

    Input structure:
    - A black background contains several separated monochrome rectangles, each in its own color.
    - Some rectangles are exact one-pixel-thick hollow frames with enclosed black interiors; others are solid distractor rectangles.
    - One separate small multicolor sprite component appears away from the rectangles.

    Rule:
    - Detect every hollow rectangle by checking that its bbox is a complete monochrome border ring around a black interior.
    - Let K be the number of such hollow rectangles, and select the one whose enclosed interior area is uniquely largest.
    - Crop the multicolor sprite, enlarge it by K with pixel replication, and place it into the selected frame's interior aligned to the interior's top-left corner.
    - Return only that selected frame, cropped to its bbox, with the enlarged sprite inside.

    Key insight:
    - The hollow rectangles serve two roles at once: their count encodes the scale factor, and the largest enclosed hole provides the output canvas.
    """
    info = _scene_info(np.asarray(grid, dtype=int))
    canvas = info["canvas"]
    sprite = info["sprite"]
    scale = int(info["K"])

    enlarged = scale_sprite(sprite, scale)
    inner_h, inner_w = map(int, canvas["inner_shape"])
    assert enlarged.shape == (inner_h, inner_w), "largest hollow interior must match the enlarged sprite"

    out = crop_bbox(np.asarray(grid, dtype=int), canvas["bbox"])
    out[1:-1, 1:-1] = enlarged
    return out


def _hollow_patch(inner_h: int, inner_w: int, color: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(color), dtype=int)
    patch[1:-1, 1:-1] = BACKGROUND
    return patch


def _solid_patch(height: int, width: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=int)


def _sprite_patch(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    assert h * w >= len(colors) >= 2
    while True:
        cells = list(colors)
        while len(cells) < h * w:
            cells.append(random.choice(colors))
        random.shuffle(cells)
        patch = np.array(cells, dtype=int).reshape(h, w)
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        if all(len(set(map(int, row))) == 1 for row in patch):
            continue
        if all(len(set(map(int, col))) == 1 for col in patch.T):
            continue
        return patch


def _smaller_hollow_dims(target_area: int, count: int) -> list[tuple[int, int]]:
    candidates = [
        (h, w)
        for h in range(2, 9)
        for w in range(2, 9)
        if h * w < target_area
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for inner_h, inner_w in candidates:
        area = inner_h * inner_w
        if area in used_areas:
            continue
        chosen.append((inner_h, inner_w))
        used_areas.add(area)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("not enough smaller hollow rectangle sizes")


def _solid_dims(target_outer_h: int, target_outer_w: int, count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    large_h_hi = min(10, max(3, target_outer_h + 1))
    large_w_hi = min(10, max(3, target_outer_w + 1))
    large_h_lo = min(large_h_hi, max(3, min(10, target_outer_h - 2)))
    large_w_lo = min(large_w_hi, max(3, min(10, target_outer_w - 2)))
    dims.append((random.randint(large_h_lo, large_h_hi), random.randint(large_w_lo, large_w_hi)))
    while len(dims) < count:
        dims.append((random.randint(3, 9), random.randint(3, 9)))
    random.shuffle(dims)
    return dims


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _pair_pad(kind_a: str, kind_b: str) -> int:
    return 2 if "sprite" in {kind_a, kind_b} else 1


def _place_items(items: list[dict]) -> tuple[int, int, list[dict]] | None:
    max_h = max(int(item["patch"].shape[0]) for item in items)
    max_w = max(int(item["patch"].shape[1]) for item in items)
    footprint = 0
    for item in items:
        pad = 4 if item["kind"] == "sprite" else 2
        h, w = map(int, item["patch"].shape)
        footprint += (h + pad) * (w + pad)
    base_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(footprint * 1.2))))
    lo = max(18, base_side - 1)
    hi = min(30, base_side + 4)
    if lo > hi:
        return None
    sizes = []
    for _ in range(8):
        h = random.randint(lo, hi)
        w = random.randint(lo, hi)
        sizes.append((h, w))
    random.shuffle(sizes)

    ordered = sorted(items, key=lambda item: int(item["patch"].shape[0]) * int(item["patch"].shape[1]), reverse=True)
    for height, width in sizes:
        placed: list[dict] = []
        ok = True
        for item in ordered:
            patch = item["patch"]
            ph, pw = map(int, patch.shape)
            candidates = [
                (r, c)
                for r in range(height - ph + 1)
                for c in range(width - pw + 1)
            ]
            random.shuffle(candidates)

            chosen = None
            for r0, c0 in candidates:
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(
                    _boxes_separated(box, prev["bbox"], _pair_pad(item["kind"], prev["kind"]))
                    for prev in placed
                ):
                    chosen = box
                    break
            if chosen is None:
                ok = False
                break
            placed.append({**item, "bbox": chosen})
        if ok:
            return height, width, placed
    return None


def _make_expected_output(target_patch: np.ndarray, sprite_patch: np.ndarray, k: int) -> np.ndarray:
    out = target_patch.copy()
    out[1:-1, 1:-1] = scale_sprite(sprite_patch, int(k))
    return out


def _attempt_scene() -> tuple[np.ndarray, dict] | None:
    k = _bagged_k()
    sprite_shape = _bagged_sprite_shape()
    sprite_h, sprite_w = map(int, sprite_shape)

    target_inner_h = sprite_h * k
    target_inner_w = sprite_w * k
    total_rectangles_hi = 5 if target_inner_h * target_inner_w >= 72 else 6
    total_rectangles = random.randint(max(4, k + 1), total_rectangles_hi)
    solid_count = total_rectangles - k

    sprite_color_count = random.randint(2, min(3, len(RECTANGLE_COLORS) - total_rectangles))
    colors = random.sample(RECTANGLE_COLORS, total_rectangles + sprite_color_count)
    rect_colors = colors[:total_rectangles]
    sprite_colors = colors[total_rectangles:]

    target_patch = _hollow_patch(target_inner_h, target_inner_w, rect_colors[0])
    smaller_hollows = _smaller_hollow_dims(target_inner_h * target_inner_w, k - 1)
    solid_sizes = _solid_dims(target_patch.shape[0], target_patch.shape[1], solid_count)
    sprite_patch = _sprite_patch(sprite_shape, sprite_colors)

    items = [{"kind": "target_hollow", "patch": target_patch, "name": "target"}]
    for idx, (inner_h, inner_w) in enumerate(smaller_hollows, start=1):
        items.append(
            {
                "kind": "hollow",
                "patch": _hollow_patch(inner_h, inner_w, rect_colors[idx]),
                "name": f"hollow_{idx}",
            }
        )
    for idx, (h, w) in enumerate(solid_sizes, start=1):
        items.append(
            {
                "kind": "solid",
                "patch": _solid_patch(h, w, rect_colors[k - 1 + idx]),
                "name": f"solid_{idx}",
            }
        )
    items.append({"kind": "sprite", "patch": sprite_patch, "name": "sprite"})

    placed = _place_items(items)
    if placed is None:
        return None
    height, width, placements = placed

    grid = np.full((height, width), BACKGROUND, dtype=int)
    meta: dict[str, object] = {
        "K": k,
        "sprite_patch": sprite_patch,
        "rectangles": [],
        "sprite_bbox": None,
        "target_bbox": None,
        "target_patch": target_patch,
    }

    for item in placements:
        r0, c0, r1, c1 = map(int, item["bbox"])
        paste(grid, item["patch"], r0, c0)
        if item["kind"] == "sprite":
            meta["sprite_bbox"] = (r0, c0, r1, c1)
        else:
            rect_info = {
                "kind": item["kind"],
                "bbox": (r0, c0, r1, c1),
                "shape": tuple(map(int, item["patch"].shape)),
                "color": int(np.max(item["patch"])),
            }
            cast_rectangles = list(meta["rectangles"])
            cast_rectangles.append(rect_info)
            meta["rectangles"] = cast_rectangles
            if item["kind"] == "target_hollow":
                meta["target_bbox"] = (r0, c0, r1, c1)

    meta["expected"] = _make_expected_output(target_patch, sprite_patch, k)
    return grid, meta


def _validate_scene(grid: np.ndarray, meta: dict) -> bool:
    try:
        info = _scene_info(grid)
        out = main(grid)
    except Exception:
        return False

    k = int(meta["K"])
    if int(info["K"]) != k:
        return False

    target_bbox = tuple(meta["target_bbox"])
    if tuple(map(int, info["canvas"]["bbox"])) != target_bbox:
        return False

    sprite_bbox = tuple(meta["sprite_bbox"])
    if tuple(map(int, info["sprite_bbox"])) != sprite_bbox:
        return False

    sprite_patch = np.asarray(meta["sprite_patch"], dtype=int)
    if not np.array_equal(info["sprite"], sprite_patch):
        return False

    expected = np.asarray(meta["expected"], dtype=int)
    if not np.array_equal(out, expected):
        return False

    rectangles = list(meta["rectangles"])
    if len(rectangles) < 4 or len(rectangles) > 6:
        return False
    if len([item for item in rectangles if item["kind"] in {"target_hollow", "hollow"}]) != k:
        return False
    if not any(item["kind"] == "solid" for item in rectangles):
        return False

    target_patch = np.asarray(meta["target_patch"], dtype=int)
    if target_patch.shape != expected.shape:
        return False

    scaled = scale_sprite(sprite_patch, k)
    if scaled.shape != tuple(map(int, info["canvas"]["inner_shape"])):
        return False
    if not np.array_equal(expected[1:-1, 1:-1], scaled):
        return False

    largest_hole = max(int(item["inner_area"]) for item in info["hollows"])
    if sum(int(item["inner_area"]) == largest_hole for item in info["hollows"]) != 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _attempt_scene()
        if built is None:
            continue
        grid, meta = built
        if _validate_scene(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid hole-count upscale scene")
