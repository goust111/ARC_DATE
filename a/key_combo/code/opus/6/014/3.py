from __future__ import annotations

from collections import Counter, defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS = SourceFileLoader("_arc_utils_shared_for_arc_014_3", str(_ARC_UTILS_PATH)).load_module()

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = 0
COLORS = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_OBJECT_HEIGHT = 4
MAX_OBJECT_WIDTH = 5
AREA_RANGE = range(3, 8)
GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 30)
GLOBAL_GAP_CHOICES = (1, 2, 3)
LAYOUT_PATTERNS = (
    ((2,), 3),
    ((3,), 2),
    ((2,), 4),
    ((3,), 3),
    ((2, 2), 2),
    ((2, 2), 3),
    ((3, 2), 2),
    ((2, 2), 4),
    ((3, 2), 3),
    ((2, 2, 2), 2),
)
MAX_GENERATION_ATTEMPTS = 600

_RNG = random.Random()


def _tight_mask_from_cells(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            v = np.rot90(base, k)
            key = (tuple(map(int, v.shape)), v.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(v.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for v in _mask_variants(mask):
        key = (tuple(map(int, v.shape)), v.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _canonical_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    best_coords = None
    for v in _mask_variants(_tight_mask_from_cells(cells)):
        coords = tuple(sorted(map(tuple, np.argwhere(v).tolist())))
        if best_coords is None or coords < best_coords:
            best_coords = coords
    assert best_coords is not None
    return frozenset((int(r), int(c)) for r, c in best_coords)


def _shape_library() -> list[dict]:
    by_size: dict[int, set[frozenset[tuple[int, int]]]] = {1: {frozenset({(0, 0)})}}
    for size in range(2, max(AREA_RANGE) + 1):
        next_shapes: set[frozenset[tuple[int, int]]] = set()
        for shape in by_size[size - 1]:
            shape_set = set(shape)
            frontier = set()
            for r, c in shape_set:
                for dr, dc in DIR4:
                    cell = (r + dr, c + dc)
                    if cell not in shape_set:
                        frontier.add(cell)
            for cell in frontier:
                next_shapes.add(_canonical_cells(shape_set | {cell}))
        by_size[size] = next_shapes

    library: list[dict] = []
    for area in AREA_RANGE:
        for cells in by_size[area]:
            mask = _tight_mask_from_cells(cells)
            variants = [
                v
                for v in _mask_variants(mask)
                if int(v.shape[0]) <= MAX_OBJECT_HEIGHT and int(v.shape[1]) <= MAX_OBJECT_WIDTH
            ]
            if not variants:
                continue
            library.append(
                {
                    "area": int(area),
                    "shape_key": _canonical_shape_key(mask),
                    "variants": variants,
                }
            )
    return library


_SHAPE_LIBRARY = _shape_library()
_SHAPES_BY_AREA: dict[int, list[dict]] = defaultdict(list)
for _shape in _SHAPE_LIBRARY:
    _SHAPES_BY_AREA[int(_shape["area"])].append(_shape)


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    objects = []
    for comp in comps:
        mask = _mask_from_component(comp)
        color = int(comp["color"])
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = color
        r0, c0, r1, c1 = map(int, comp["bbox"])
        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "patch": patch,
                "shape_key": _canonical_shape_key(mask),
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": int(mask.sum()),
            }
        )
    return objects


def _translated_object(obj: dict, height: int, to_top: bool) -> dict:
    dest_r0 = 0 if to_top else int(height - obj["h"])
    r0, c0, _r1, _c1 = obj["bbox"]
    c0 = int(c0)
    return {
        **obj,
        "dest_r0": dest_r0,
        "dest_bbox": (dest_r0, c0, dest_r0 + obj["h"] - 1, c0 + obj["w"] - 1),
        "c0": c0,
    }


def _fill_group_gaps(out: np.ndarray, group: list[dict]) -> list[dict]:
    features = []
    ordered = sorted(group, key=lambda item: int(item["c0"]))
    for left, right in zip(ordered, ordered[1:]):
        left_r0, left_c0, left_r1, left_c1 = map(int, left["dest_bbox"])
        right_r0, right_c0, right_r1, right_c1 = map(int, right["dest_bbox"])
        gap_w = int(right_c0 - left_c1 - 1)
        if gap_w <= 0:
            continue
        span_r0 = min(left_r0, right_r0)
        span_r1 = max(left_r1, right_r1)
        span = gap_w // 2
        if span > 0:
            out[span_r0 : span_r1 + 1, left_c1 + 1 : left_c1 + 1 + span] = int(left["color"])
            out[span_r0 : span_r1 + 1, right_c0 - span : right_c0] = int(right["color"])
        features.append(
            {
                "gap_w": gap_w,
                "left_h": int(left["h"]),
                "right_h": int(right["h"]),
            }
        )
    return features


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort same-shape-frequency objects to the top or bottom, then half-fill the horizontal gaps between new neighbors.

    Input structure:
    - A black background contains 5-8 small monochromatic 4-connected objects, each in a distinct non-black color.
    - Objects are spatially separated, each has area 3-7, and each bbox height is at most 4.
    - Some objects share the same binary shape up to rotation/reflection, while at least one other shape is unique.

    Rule:
    - Canonicalize every object by its binary mask up to all rotations and reflections, then count how often each canonical shape appears.
    - Move every object whose shape frequency is at least 2 to the top edge and every frequency-1 object to the bottom edge, preserving each object's left column and exact pixels.
    - Within the top row group and within the bottom row group separately, sort objects by left column; for every adjacent pair, inspect the horizontal bbox gap between them after the move.
    - Fill the left half of that gap with the left object's color and the right half with the right object's color across the union of the two moved bbox row spans; if the gap width is odd, leave the middle column black.

    Key insight:
    - The gap-filling neighbors do not come from the input layout directly: shape-frequency grouping decides which objects share a target row, and that new row membership creates the adjacency that determines each filled corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    objects = _extract_objects(g)
    assert 5 <= len(objects) <= 8

    freq = Counter(obj["shape_key"] for obj in objects)
    translated = [_translated_object(obj, g.shape[0], int(freq[obj["shape_key"]]) >= 2) for obj in objects]

    out = np.full_like(g, BG)
    for obj in translated:
        paste(out, obj["patch"], obj["dest_r0"], obj["c0"], transparent=BG)

    top_group = [obj for obj in translated if int(freq[obj["shape_key"]]) >= 2]
    bottom_group = [obj for obj in translated if int(freq[obj["shape_key"]]) == 1]
    _fill_group_gaps(out, top_group)
    _fill_group_gaps(out, bottom_group)
    return out


def _choose_templates() -> tuple[tuple[int, ...], list[dict], list[dict]]:
    shared_sizes, unique_count = _RNG.choice(LAYOUT_PATTERNS)
    candidate_areas = [area for area, items in _SHAPES_BY_AREA.items() if len(items) >= 2]
    shared_area = _RNG.choice(candidate_areas)
    shared0, unique_same_area = _RNG.sample(_SHAPES_BY_AREA[shared_area], 2)

    used = {shared0["shape_key"], unique_same_area["shape_key"]}
    shared_templates = [shared0]
    for _ in range(len(shared_sizes) - 1):
        candidates = [shape for shape in _SHAPE_LIBRARY if shape["shape_key"] not in used]
        template = _RNG.choice(candidates)
        used.add(template["shape_key"])
        shared_templates.append(template)

    unique_templates = [unique_same_area]
    for _ in range(unique_count - 1):
        candidates = [shape for shape in _SHAPE_LIBRARY if shape["shape_key"] not in used]
        template = _RNG.choice(candidates)
        used.add(template["shape_key"])
        unique_templates.append(template)

    return shared_sizes, shared_templates, unique_templates


def _oriented_patch(shape: dict, color: int) -> dict:
    mask = _RNG.choice(shape["variants"]).copy()
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return {
        "shape_key": shape["shape_key"],
        "area": int(shape["area"]),
        "mask": mask,
        "patch": patch,
        "color": int(color),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
    }


def _make_objects() -> list[dict]:
    shared_sizes, shared_templates, unique_templates = _choose_templates()
    colors = _RNG.sample(COLORS, sum(shared_sizes) + len(unique_templates))
    objects = []
    color_idx = 0

    for group_id, (size, template) in enumerate(zip(shared_sizes, shared_templates)):
        for _ in range(size):
            obj = _oriented_patch(template, colors[color_idx])
            obj["group"] = "shared"
            obj["template_id"] = f"s{group_id}"
            objects.append(obj)
            color_idx += 1

    for unique_id, template in enumerate(unique_templates):
        obj = _oriented_patch(template, colors[color_idx])
        obj["group"] = "unique"
        obj["template_id"] = f"u{unique_id}"
        objects.append(obj)
        color_idx += 1

    _RNG.shuffle(objects)
    return objects


def _assign_columns(objects: list[dict]) -> tuple[int, list[int]] | None:
    widths = [int(obj["w"]) for obj in objects]
    gaps = [_RNG.choice(GLOBAL_GAP_CHOICES) for _ in range(len(objects) - 1)]
    base_width = sum(widths) + sum(gaps)
    if base_width > GRID_WIDTH_RANGE[1]:
        return None

    min_width = max(GRID_WIDTH_RANGE[0], base_width)
    width = _RNG.randint(min_width, GRID_WIDTH_RANGE[1])
    slack = width - base_width
    extra = [0] * (len(objects) + 1)
    for _ in range(slack):
        extra[_RNG.randrange(len(extra))] += 1

    cols = []
    c = extra[0]
    for idx, obj in enumerate(objects):
        cols.append(int(c))
        c += int(obj["w"])
        if idx < len(gaps):
            c += int(gaps[idx]) + int(extra[idx + 1])
    return width, cols


def _place_output_objects(objects: list[dict], height: int) -> tuple[np.ndarray, list[dict], dict[str, list[dict]], list[dict]]:
    out = np.full((height, max(int(obj["c0"] + obj["w"]) for obj in objects)), BG, dtype=int)
    groups: dict[str, list[dict]] = {"shared": [], "unique": []}
    placed = []

    for obj in objects:
        dest_r0 = 0 if obj["group"] == "shared" else int(height - obj["h"])
        placed_obj = {
            **obj,
            "dest_r0": dest_r0,
            "dest_bbox": (dest_r0, obj["c0"], dest_r0 + obj["h"] - 1, obj["c0"] + obj["w"] - 1),
        }
        placed.append(placed_obj)
        groups[obj["group"]].append(placed_obj)
        paste(out, placed_obj["patch"], dest_r0, int(obj["c0"]), transparent=BG)

    features = []
    features.extend(_fill_group_gaps(out, groups["shared"]))
    features.extend(_fill_group_gaps(out, groups["unique"]))
    return out, placed, groups, features


def _assign_input_rows(objects: list[dict], height: int) -> bool:
    for obj in objects:
        choices = [r for r in range(1, height - obj["h"] - 1) if r != int(obj["dest_r0"])]
        if not choices:
            return False
        obj["input_r0"] = int(_RNG.choice(choices))
        obj["input_bbox"] = (
            int(obj["input_r0"]),
            int(obj["c0"]),
            int(obj["input_r0"] + obj["h"] - 1),
            int(obj["c0"] + obj["w"] - 1),
        )
    return True


def _render_input(objects: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in objects:
        paste(grid, obj["patch"], int(obj["input_r0"]), int(obj["c0"]), transparent=BG)
    return grid


def _global_label_interleaving(objects: list[dict]) -> bool:
    labels = [obj["group"] for obj in sorted(objects, key=lambda item: int(item["c0"]))]
    return any(labels[i] == labels[i + 2] != labels[i + 1] for i in range(len(labels) - 2))


def _validate_layout(inp: np.ndarray, out: np.ndarray, objects: list[dict], groups: dict[str, list[dict]], features: list[dict]) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    extracted = _extract_objects(inp)
    if len(extracted) != len(objects):
        return False

    freq = Counter(obj["shape_key"] for obj in extracted)
    if not any(count >= 2 for count in freq.values()):
        return False
    if not any(count == 1 for count in freq.values()):
        return False

    if len(groups["shared"]) < 2 or len(groups["unique"]) < 2:
        return False
    if not features:
        return False
    if not any(int(feature["gap_w"]) >= 2 for feature in features):
        return False
    parities = {int(feature["gap_w"]) % 2 for feature in features if int(feature["gap_w"]) >= 2}
    if parities != {0, 1}:
        return False
    if not any(int(feature["left_h"]) != int(feature["right_h"]) for feature in features):
        return False

    class_by_area: dict[int, set[str]] = defaultdict(set)
    for obj in objects:
        class_by_area[int(obj["area"])].add(str(obj["group"]))
    if not any(classes == {"shared", "unique"} for classes in class_by_area.values()):
        return False

    if not _global_label_interleaving(objects):
        return False

    if np.array_equal(inp, out):
        return False
    return True


def _single_attempt() -> np.ndarray | None:
    objects = _make_objects()
    placement = _assign_columns(objects)
    if placement is None:
        return None
    width, cols = placement
    for obj, c0 in zip(objects, cols):
        obj["c0"] = int(c0)
        obj["bbox"] = (0, int(c0), obj["h"] - 1, int(c0) + obj["w"] - 1)

    max_top_h = max(int(obj["h"]) for obj in objects if obj["group"] == "shared")
    max_bottom_h = max(int(obj["h"]) for obj in objects if obj["group"] == "unique")
    min_height = max(GRID_HEIGHT_RANGE[0], max_top_h + max_bottom_h + 5)
    if min_height > GRID_HEIGHT_RANGE[1]:
        return None
    height = _RNG.randint(min_height, GRID_HEIGHT_RANGE[1])

    out, placed, groups, features = _place_output_objects(objects, height)
    if out.shape[1] != width:
        padded = np.full((height, width), BG, dtype=int)
        padded[:, : out.shape[1]] = out
        out = padded

    if not _assign_input_rows(placed, height):
        return None
    inp = _render_input(placed, height, width)

    if _validate_layout(inp, out, placed, groups, features):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _single_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
