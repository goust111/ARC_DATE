from __future__ import annotations

from importlib.machinery import SourceFileLoader
import math
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
crop_bbox = _ARC_UTILS.crop_bbox
bbox_coords = _ARC_UTILS.bbox_coords
in_bounds = _ARC_UTILS.in_bounds


BG = 0
PALETTE = tuple(range(1, 10))
GRID_MIN = 22
GRID_MAX = 28
PAIR_COUNT_CHOICES = (2, 2, 3, 3, 4)
LAYER_COUNT_CHOICES = (2, 3, 4)
PLACEMENT_MARGIN = 1
PAIR_GAP_MIN = 2
PAIR_GAP_MAX = 4
MAX_GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 250

_RNG = np.random.default_rng()

_BASE_SHAPES = [
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], dtype=int),
]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _direction_from_layer_count(layer_count: int) -> tuple[str, tuple[int, int]]:
    mod = int(layer_count) % 4
    if mod == 1:
        return "right", (0, 1)
    if mod == 2:
        return "down", (1, 0)
    if mod == 3:
        return "left", (0, -1)
    return "up", (-1, 0)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    norm = np.array(coords, dtype=int)
    norm[:, 0] -= norm[:, 0].min()
    norm[:, 1] -= norm[:, 1].min()
    order = np.lexsort((norm[:, 1], norm[:, 0]))
    return norm[order]


def _random_shape() -> np.ndarray:
    coords = _BASE_SHAPES[int(_RNG.integers(len(_BASE_SHAPES)))]
    mask = np.zeros((coords[:, 0].max() + 1, coords[:, 1].max() + 1), dtype=int)
    mask[coords[:, 0], coords[:, 1]] = 1
    if bool(_RNG.integers(2)):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=int(_RNG.integers(4)))
    return _normalize_coords(np.argwhere(mask == 1))


def _shape_bbox(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _shell_coords(r0: int, c0: int, h: int, w: int, depth: int) -> list[tuple[int, int]]:
    rr0 = int(r0 + depth)
    cc0 = int(c0 + depth)
    rr1 = int(r0 + h - 1 - depth)
    cc1 = int(c0 + w - 1 - depth)
    coords: set[tuple[int, int]] = set()
    for cc in range(cc0, cc1 + 1):
        coords.add((rr0, cc))
        coords.add((rr1, cc))
    for rr in range(rr0, rr1 + 1):
        coords.add((rr, cc0))
        coords.add((rr, cc1))
    return sorted(coords)


def _render_nested_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    colors: list[int],
) -> list[list[tuple[int, int]]]:
    layers: list[list[tuple[int, int]]] = []
    for depth, color in enumerate(colors):
        coords = _shell_coords(r0, c0, h, w, depth)
        for rr, cc in coords:
            grid[rr, cc] = int(color)
        layers.append(coords)
    return layers


def _translate_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _has_4_contact(coords: list[tuple[int, int]], target: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r - 1, c) in target or (r + 1, c) in target or (r, c - 1) in target or (r, c + 1) in target:
            return True
    return False


def _slide_distance(
    coords: list[tuple[int, int]],
    occupied_grid: np.ndarray,
    dr: int,
    dc: int,
    bg: int,
) -> int:
    shift = 0
    while True:
        next_coords = _translate_coords(coords, (shift + 1) * dr, (shift + 1) * dc)
        if not all(in_bounds(occupied_grid.shape, r, c) for r, c in next_coords):
            break
        if any(int(occupied_grid[r, c]) != int(bg) for r, c in next_coords):
            break
        shift += 1
    return int(shift)


def _analyze_nested_component(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    bbox = tuple(map(int, component["bbox"]))
    sub = crop_bbox(grid, bbox)
    layers: list[dict] = []
    base_r, base_c = bbox[0], bbox[1]
    current = sub.copy()
    offset_r = 0
    offset_c = 0

    while True:
        fg = np.argwhere(current != int(bg))
        if fg.size == 0:
            break
        inner_bbox = bbox_coords(fg)
        r0, c0, r1, c1 = map(int, inner_bbox)
        if (r0, c0, r1, c1) != (0, 0, current.shape[0] - 1, current.shape[1] - 1):
            return None
        border = _border_mask(current.shape)
        border_vals = current[border]
        if np.any(border_vals == int(bg)):
            return None
        colors = sorted(set(int(v) for v in border_vals.tolist()))
        if len(colors) != 1:
            return None
        layer_color = int(colors[0])
        layer_local = np.argwhere(border)
        layer_coords = [(base_r + offset_r + int(rr), base_c + offset_c + int(cc)) for rr, cc in layer_local]
        layers.append({"color": layer_color, "coords": layer_coords})
        current = current.copy()
        current[border] = int(bg)
        rem = np.argwhere(current != int(bg))
        if rem.size == 0:
            break
        shrink_bbox = bbox_coords(rem)
        offset_r += int(shrink_bbox[0])
        offset_c += int(shrink_bbox[1])
        current = crop_bbox(current, shrink_bbox)

    if len(layers) < 2:
        return None
    if len(set(layer["color"] for layer in layers)) != len(layers):
        return None
    if bbox[2] - bbox[0] + 1 < 4 or bbox[3] - bbox[1] + 1 < 4:
        return None

    coord_set = {tuple(map(int, rc)) for rc in component["coords"]}
    reconstructed = {coord for layer in layers for coord in layer["coords"]}
    if reconstructed != coord_set:
        return None

    return {
        "bbox": bbox,
        "bbox_area": _bbox_area(bbox),
        "layer_count": len(layers),
        "layers": layers,
        "coords": sorted(coord_set),
        "coord_set": coord_set,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    rectangles: list[dict] = []
    objects: list[dict] = []
    for comp in components:
        rect = _analyze_nested_component(g, comp, bg)
        if rect is not None:
            rectangles.append(rect)
            continue

        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        values = sorted(set(int(g[r, c]) for r, c in coords))
        if len(values) != 1:
            raise AssertionError("non-rectangle components must be monochrome small objects")
        objects.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": sorted(coords),
                "coord_set": set(coords),
                "color": int(values[0]),
                "area": len(coords),
            }
        )

    rectangles.sort(key=lambda rect: (rect["bbox_area"], rect["bbox"][0], rect["bbox"][1]))
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))

    if not rectangles or len(rectangles) != len(objects):
        raise AssertionError("expected a bijection between nested rectangles and small objects")
    return rectangles, objects, bg


def _pair_objects(rectangles: list[dict], objects: list[dict]) -> list[dict]:
    pairings: list[dict] = []
    used_rects: set[int] = set()
    for obj_index, obj in enumerate(objects):
        orow, ocol = _bbox_center(obj["bbox"])
        distances: list[tuple[float, int]] = []
        for rect_index, rect in enumerate(rectangles):
            rrow, rcol = _bbox_center(rect["bbox"])
            dist = math.hypot(orow - rrow, ocol - rcol)
            distances.append((dist, rect_index))
        distances.sort()
        if len(distances) > 1 and math.isclose(distances[0][0], distances[1][0], rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError("pairing by nearest bbox center must be unique")
        rect_index = int(distances[0][1])
        if rect_index in used_rects:
            raise AssertionError("pairing must be bijective")
        used_rects.add(rect_index)
        pairings.append({"rect_index": rect_index, "object_index": obj_index})
    if len(used_rects) != len(rectangles):
        raise AssertionError("every rectangle must receive exactly one object")
    pairings.sort(key=lambda pair: pair["rect_index"])
    return pairings


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each small object to its paired nested rectangle using the rectangle's layer count, then reverse the rectangle's layer colors.

    Input structure:
    - A black background contains 2-4 multicolor nested rectangular ring objects and the same number of small monochrome 4-connected objects.
    - Each nested rectangle is made of 2 or more concentric one-cell-thick rectangular layers with different colors.
    - Each small object is a separate nearby component, and it pairs with the nested rectangle whose bbox center is strictly closest.

    Rule:
    - For each nested rectangle, count its layers by repeatedly peeling off the outer uniform-color border.
    - Map the layer count L to a slide direction using L mod 4: 1 -> right, 2 -> down, 3 -> left, 0 -> up.
    - Process paired rectangles in ascending bbox-area order: erase the paired object's original pixels, slide it step by step in the mapped direction until the next step would collide, and keep the stop position where it first 4-neighbor touches the rectangle.
    - Reverse every rectangle's layer-color order on the same occupied shells while keeping the small objects' colors and shapes unchanged.

    Key insight:
    - The rectangle's structural depth determines another object's motion; solving requires reading the nesting depth from the input component structure before repainting the layers in reversed order.
    """
    g = np.asarray(grid, dtype=int)
    rectangles, objects, bg = _parse_scene(g)
    pairings = _pair_objects(rectangles, objects)
    object_for_rect = {pair["rect_index"]: pair["object_index"] for pair in pairings}

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = int(bg)

    for rect in rectangles:
        reversed_colors = [layer["color"] for layer in rect["layers"]][::-1]
        for new_color, layer in zip(reversed_colors, rect["layers"]):
            for r, c in layer["coords"]:
                out[r, c] = int(new_color)

    for rect_index, rect in enumerate(rectangles):
        obj = objects[object_for_rect[rect_index]]
        _direction_name, (dr, dc) = _direction_from_layer_count(rect["layer_count"])
        shift = _slide_distance(obj["coords"], out, dr, dc, bg)
        moved = _translate_coords(obj["coords"], shift * dr, shift * dc)
        if shift <= 0:
            raise AssertionError("small object must move before touching its rectangle")
        if not _has_4_contact(moved, rect["coord_set"]):
            raise AssertionError("slid object must stop in 4-neighbor contact with its paired rectangle")
        for r, c in moved:
            out[r, c] = int(obj["color"])

    return out


def _make_pair_blueprint(layer_count: int) -> dict:
    direction, _vec = _direction_from_layer_count(layer_count)
    object_coords = _random_shape()
    obj_h, obj_w = _shape_bbox(object_coords)
    hole_h = int(_RNG.integers(0, 3))
    hole_w = int(_RNG.integers(0, 3))
    rect_h = int(2 * layer_count + hole_h)
    rect_w = int(2 * layer_count + hole_w)
    colors = [int(v) for v in _RNG.choice(PALETTE, size=layer_count, replace=False)]
    remaining = [color for color in PALETTE if color not in colors]
    object_color = int(_RNG.choice(remaining))
    return {
        "layer_count": int(layer_count),
        "direction": direction,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "rect_colors": colors,
        "object_coords_local": object_coords,
        "obj_h": obj_h,
        "obj_w": obj_w,
        "object_color": object_color,
    }


def _try_place_blueprints(grid_h: int, grid_w: int, blueprints: list[dict]) -> list[dict] | None:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []

    order = sorted(range(len(blueprints)), key=lambda idx: blueprints[idx]["rect_h"] * blueprints[idx]["rect_w"], reverse=True)
    for idx in order:
        bp = blueprints[idx]
        success = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            gap = int(_RNG.integers(PAIR_GAP_MIN, PAIR_GAP_MAX + 1))
            rect_h = int(bp["rect_h"])
            rect_w = int(bp["rect_w"])
            obj_h = int(bp["obj_h"])
            obj_w = int(bp["obj_w"])
            direction = bp["direction"]

            if direction == "right":
                r_lo = PLACEMENT_MARGIN
                r_hi = grid_h - rect_h - PLACEMENT_MARGIN
                c_lo = PLACEMENT_MARGIN + obj_w + gap
                c_hi = grid_w - rect_w - PLACEMENT_MARGIN
            elif direction == "left":
                r_lo = PLACEMENT_MARGIN
                r_hi = grid_h - rect_h - PLACEMENT_MARGIN
                c_lo = PLACEMENT_MARGIN
                c_hi = grid_w - rect_w - gap - obj_w - PLACEMENT_MARGIN
            elif direction == "down":
                r_lo = PLACEMENT_MARGIN + obj_h + gap
                r_hi = grid_h - rect_h - PLACEMENT_MARGIN
                c_lo = PLACEMENT_MARGIN
                c_hi = grid_w - rect_w - PLACEMENT_MARGIN
            else:
                r_lo = PLACEMENT_MARGIN
                r_hi = grid_h - rect_h - gap - obj_h - PLACEMENT_MARGIN
                c_lo = PLACEMENT_MARGIN
                c_hi = grid_w - rect_w - PLACEMENT_MARGIN

            if r_lo > r_hi or c_lo > c_hi:
                continue

            rect_r0 = int(_RNG.integers(r_lo, r_hi + 1))
            rect_c0 = int(_RNG.integers(c_lo, c_hi + 1))
            if direction in ("right", "left"):
                obj_r0 = int(_RNG.integers(rect_r0, rect_r0 + rect_h - obj_h + 1))
                obj_c0 = rect_c0 - gap - obj_w if direction == "right" else rect_c0 + rect_w + gap
            else:
                obj_c0 = int(_RNG.integers(rect_c0, rect_c0 + rect_w - obj_w + 1))
                obj_r0 = rect_r0 - gap - obj_h if direction == "down" else rect_r0 + rect_h + gap

            rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)
            object_bbox = (obj_r0, obj_c0, obj_r0 + obj_h - 1, obj_c0 + obj_w - 1)
            rect_area = _bbox_area(rect_bbox)
            if any(_bbox_area(other["rect_bbox"]) == rect_area for other in placed):
                continue
            union_bbox = (
                min(rect_bbox[0], object_bbox[0]),
                min(rect_bbox[1], object_bbox[1]),
                max(rect_bbox[2], object_bbox[2]),
                max(rect_bbox[3], object_bbox[3]),
            )
            pad_bbox = (
                max(0, union_bbox[0] - 1),
                max(0, union_bbox[1] - 1),
                min(grid_h - 1, union_bbox[2] + 1),
                min(grid_w - 1, union_bbox[3] + 1),
            )
            if reserved[pad_bbox[0] : pad_bbox[2] + 1, pad_bbox[1] : pad_bbox[3] + 1].any():
                continue

            object_coords = [(obj_r0 + int(r), obj_c0 + int(c)) for r, c in bp["object_coords_local"]]
            if not all(in_bounds((grid_h, grid_w), r, c) for r, c in object_coords):
                continue
            obj_center = _bbox_center(object_bbox)
            rect_center = _bbox_center(rect_bbox)
            own_dist = math.hypot(obj_center[0] - rect_center[0], obj_center[1] - rect_center[1])
            if any(
                own_dist >= math.hypot(obj_center[0] - _bbox_center(other["rect_bbox"])[0], obj_center[1] - _bbox_center(other["rect_bbox"])[1]) - 1e-9
                for other in placed
            ):
                continue
            if any(
                math.hypot(_bbox_center(other["object_bbox"])[0] - _bbox_center(other["rect_bbox"])[0], _bbox_center(other["object_bbox"])[1] - _bbox_center(other["rect_bbox"])[1])
                >= math.hypot(_bbox_center(other["object_bbox"])[0] - rect_center[0], _bbox_center(other["object_bbox"])[1] - rect_center[1]) - 1e-9
                for other in placed
            ):
                continue

            reserved[pad_bbox[0] : pad_bbox[2] + 1, pad_bbox[1] : pad_bbox[3] + 1] = True
            placed.append(
                {
                    **bp,
                    "pair_index": idx,
                    "gap": gap,
                    "rect_bbox": rect_bbox,
                    "object_bbox": object_bbox,
                    "object_coords": sorted(object_coords),
                    "reserved_bbox": pad_bbox,
                }
            )
            success = True
            break
        if not success:
            return None

    placed.sort(key=lambda spec: spec["pair_index"])
    return placed


def _render_specs(grid_h: int, grid_w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        layers = _render_nested_rectangle(
            grid,
            spec["rect_bbox"][0],
            spec["rect_bbox"][1],
            spec["rect_h"],
            spec["rect_w"],
            spec["rect_colors"],
        )
        spec["layer_coords"] = layers
        spec["rect_coords"] = sorted({coord for layer in layers for coord in layer})
        for r, c in spec["object_coords"]:
            grid[r, c] = int(spec["object_color"])
    return grid


def _side_matches_direction(spec: dict) -> bool:
    rr0, cc0, rr1, cc1 = spec["rect_bbox"]
    or0, oc0, or1, oc1 = spec["object_bbox"]
    direction = spec["direction"]
    if direction == "right":
        return oc1 < cc0 and rr0 <= or0 <= rr1 and rr0 <= or1 <= rr1
    if direction == "left":
        return oc0 > cc1 and rr0 <= or0 <= rr1 and rr0 <= or1 <= rr1
    if direction == "down":
        return or1 < rr0 and cc0 <= oc0 <= cc1 and cc0 <= oc1 <= cc1
    return or0 > rr1 and cc0 <= oc0 <= cc1 and cc0 <= oc1 <= cc1


def _specs_expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["object_coords"]:
            out[r, c] = BG
    for spec in specs:
        reversed_colors = list(spec["rect_colors"])[::-1]
        for color, coords in zip(reversed_colors, spec["layer_coords"]):
            for r, c in coords:
                out[r, c] = int(color)

    order = sorted(specs, key=lambda spec: (_bbox_area(spec["rect_bbox"]), spec["rect_bbox"][0], spec["rect_bbox"][1]))
    for spec in order:
        _direction_name, (dr, dc) = _direction_from_layer_count(spec["layer_count"])
        shift = _slide_distance(spec["object_coords"], out, dr, dc, BG)
        moved = _translate_coords(spec["object_coords"], shift * dr, shift * dc)
        rect_set = set(spec["rect_coords"])
        if shift <= 0 or not _has_4_contact(moved, rect_set):
            raise AssertionError("spec-driven slide must end in first contact")
        for r, c in moved:
            out[r, c] = int(spec["object_color"])
        spec["moved_object_coords"] = moved
    return out


def _validate_generated_scene(inp: np.ndarray, specs: list[dict]) -> None:
    parsed_rects, parsed_objs, bg = _parse_scene(inp)
    assert int(bg) == BG
    assert len(parsed_rects) == len(specs) == len(parsed_objs)
    assert len({tuple(spec["rect_bbox"]) for spec in specs}) == len(specs)
    assert len({_bbox_area(spec["rect_bbox"]) for spec in specs}) == len(specs)
    assert len({spec["layer_count"] for spec in specs}) > 1

    for spec in specs:
        assert 2 <= int(spec["layer_count"]) <= 4
        assert 3 <= len(spec["object_coords"]) <= 6
        assert _side_matches_direction(spec)

    for spec in specs:
        orow, ocol = _bbox_center(spec["object_bbox"])
        dists = []
        for other in specs:
            rrow, rcol = _bbox_center(other["rect_bbox"])
            dists.append((math.hypot(orow - rrow, ocol - rcol), other["pair_index"]))
        dists.sort()
        assert dists[0][1] == spec["pair_index"]
        assert not math.isclose(dists[0][0], dists[1][0], rel_tol=0.0, abs_tol=1e-9)

    expected = _specs_expected_output(inp, specs)
    actual = main(inp)
    assert np.array_equal(expected, actual)
    assert not np.array_equal(inp, actual)

    for spec in specs:
        moved_set = set(spec["moved_object_coords"])
        for r, c in spec["object_coords"]:
            if (r, c) not in moved_set:
                assert int(actual[r, c]) == BG
        for color, coords in zip(spec["rect_colors"][::-1], spec["layer_coords"]):
            for r, c in coords:
                assert int(actual[r, c]) == int(color)
        for r, c in spec["moved_object_coords"]:
            assert int(actual[r, c]) == int(spec["object_color"])


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    grid_w = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    pair_count = int(_RNG.choice(PAIR_COUNT_CHOICES))
    layer_counts = [int(_RNG.choice(LAYER_COUNT_CHOICES)) for _ in range(pair_count)]
    if len(set(layer_counts)) < 2:
        return None

    blueprints = [_make_pair_blueprint(layer_count) for layer_count in layer_counts]
    specs = _try_place_blueprints(grid_h, grid_w, blueprints)
    if specs is None:
        return None

    grid = _render_specs(grid_h, grid_w, specs)
    try:
        _validate_generated_scene(grid, specs)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-layer sliding ARC input")
