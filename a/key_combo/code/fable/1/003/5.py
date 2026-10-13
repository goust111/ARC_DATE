from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG_COLOR = Color.BLACK
ADD_COLOR = Color.TEAL
SUB_COLOR = Color.RED
FILL_COLOR = Color.GREY
FRAME_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (20, 28)
BAR_MAX_HEIGHT = 6
BAR_BAND_HEIGHT = BAR_MAX_HEIGHT + 2
MAX_INNER_HEIGHT = 6
MAX_INNER_WIDTH = 10
MIN_KEY = 3
MAX_KEY = 8
MAX_ADD_SUM = 12
MAX_KEY_GAP = 4
FRAME_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 500


def _checkerboard_capacity_from_bbox(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    return (inner_h * inner_w + 1) // 2


def _paint_checkerboard_interior(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r - r0 - 1 + c - c0 - 1) % 2 == 0:
                grid[r, c] = int(color)


def _is_bottom_bar(comp: dict, shape: tuple[int, int]) -> bool:
    color = int(comp["color"])
    if color not in {ADD_COLOR, SUB_COLOR}:
        return False
    h, _w = map(int, shape)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return c0 == c1 and r1 == h - 1 and int(comp["area"]) == r1 - r0 + 1


def _frame_bbox_if_valid(grid: np.ndarray, bg: int, comp: dict) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 3 or outer_w < 3:
        return None
    expected_area = 2 * outer_h + 2 * outer_w - 4
    if int(comp["area"]) != expected_area:
        return None

    mask = np.zeros((outer_h, outer_w), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True

    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask, expected):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.all(interior == int(bg)):
        return None
    return r0, c0, r1, c1


def _shape_catalog(max_inner_h: int, max_inner_w: int) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    cap_to_shapes: dict[int, list[dict]] = defaultdict(list)
    area_to_shapes: dict[int, list[dict]] = defaultdict(list)
    for inner_h in range(1, max_inner_h + 1):
        for inner_w in range(1, max_inner_w + 1):
            area = inner_h * inner_w
            cap = (area + 1) // 2
            shape = {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "area": area,
                "cap": cap,
            }
            cap_to_shapes[cap].append(shape)
            area_to_shapes[area].append(shape)
    return cap_to_shapes, area_to_shapes


def _choose_shape(shapes: list[dict], predicate) -> dict | None:
    candidates = [shape for shape in shapes if predicate(shape)]
    if not candidates:
        return None
    return dict(random.choice(candidates))


def _random_partition(total: int, count: int, max_part: int) -> list[int] | None:
    if total < count or total > count * max_part:
        return None
    parts = [1] * count
    remaining = total - count
    while remaining > 0:
        choices = [idx for idx, value in enumerate(parts) if value < max_part]
        if not choices:
            return None
        idx = random.choice(choices)
        parts[idx] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _sample_bar_heights(add_sum: int, sub_sum: int) -> tuple[list[int], list[int]] | None:
    add_count_choices = [n for n in range(2, 5) if add_sum <= n * BAR_MAX_HEIGHT]
    sub_count_choices = [n for n in range(1, 4) if sub_sum <= n * BAR_MAX_HEIGHT]
    random.shuffle(add_count_choices)
    random.shuffle(sub_count_choices)
    for add_count in add_count_choices:
        add_heights = _random_partition(add_sum, add_count, BAR_MAX_HEIGHT)
        if add_heights is None:
            continue
        for sub_count in sub_count_choices:
            sub_heights = _random_partition(sub_sum, sub_count, BAR_MAX_HEIGHT)
            if sub_heights is not None:
                return add_heights, sub_heights
    return None


def _sample_spaced_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(120):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    parity_slots = list(range(random.choice((1, 2)), width - 1, 2))
    if len(parity_slots) < count:
        return None
    return sorted(random.sample(parity_slots, count))


def _place_frames(height: int, width: int, frame_region_h: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)
    for idx in order:
        outer_h = int(specs[idx]["outer_h"])
        outer_w = int(specs[idx]["outer_w"])
        locations: list[tuple[int, int]] = []
        for r0 in range(1, frame_region_h - outer_h):
            for c0 in range(1, width - outer_w):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + outer_h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + outer_w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                locations.append((r0, c0))
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        specs[idx]["bbox"] = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        occupied[max(0, r0 - 1) : min(height, r0 + outer_h + 1), max(0, c0 - 1) : min(width, c0 + outer_w + 1)] = True
    return specs


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _expected_from_metadata(grid: np.ndarray, metadata: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    _paint_checkerboard_interior(out, metadata["target_bbox"], FILL_COLOR)
    return out


def _validate_metadata(grid: np.ndarray, metadata: dict) -> bool:
    key = int(metadata["key"])
    add_sum = int(metadata["add_sum"])
    sub_sum = int(metadata["sub_sum"])
    frames = metadata["frames"]
    target_bbox = metadata["target_bbox"]

    if key <= 0 or add_sum <= sub_sum:
        return False
    if key != add_sum - sub_sum:
        return False

    capacities = [_checkerboard_capacity_from_bbox(frame["bbox"]) for frame in frames]
    if capacities.count(key) != 1:
        return False

    area_matches = [frame for frame in frames if int(frame["area"]) == key and frame["bbox"] != target_bbox]
    if len(area_matches) != 1:
        return False

    addsum_matches = [frame for frame in frames if int(frame["cap"]) == add_sum and frame["bbox"] != target_bbox]
    if len(addsum_matches) != 1:
        return False

    target = next(frame for frame in frames if frame["bbox"] == target_bbox)
    if int(target["area"]) == key:
        return False

    expected = _expected_from_metadata(grid, metadata)
    if np.array_equal(expected, grid):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the net height of bottom teal-minus-red bars as a key to select and checker-fill one hollow rectangle.

    Input structure:
    - The background is uniform.
    - Several one-cell-wide vertical bars are bottom-aligned: teal bars are addends and red bars are subtrahends.
    - Elsewhere there are several one-cell-thick hollow rectangular frames with empty interiors.

    Rule:
    - Compute key = sum(teal bar heights) - sum(red bar heights).
    - For each rectangular frame, count the checkerboard cells inside its border starting from the interior upper-left cell.
    - Exactly one frame has checkerboard capacity equal to the key; fill only those checkerboard cells with grey and leave every other pixel unchanged.

    Key insight:
    - The bars are a height-difference calculator, and the chosen frame is identified only by a quantity match between that arithmetic result and the frame's upper-left-starting checkerboard capacity.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    add_sum = 0
    sub_sum = 0
    frames: list[tuple[int, int, int, int]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if _is_bottom_bar(comp, g.shape):
            height = int(comp["bbox"][2] - comp["bbox"][0] + 1)
            if int(comp["color"]) == ADD_COLOR:
                add_sum += height
            else:
                sub_sum += height
            continue

        bbox = _frame_bbox_if_valid(g, bg, comp)
        assert bbox is not None, "unexpected foreground object"
        frames.append(bbox)

    assert add_sum > 0 and sub_sum > 0, "expected both teal and red bars"
    key = add_sum - sub_sum
    assert key > 0, "height difference must be positive"

    matches = [bbox for bbox in frames if _checkerboard_capacity_from_bbox(bbox) == key]
    assert len(matches) == 1, "checkerboard capacity match must be unique"

    out = g.copy()
    _paint_checkerboard_interior(out, matches[0], FILL_COLOR)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    frame_region_h = height - BAR_BAND_HEIGHT
    max_inner_h = min(MAX_INNER_HEIGHT, frame_region_h - 3)
    max_inner_w = min(MAX_INNER_WIDTH, width - 4)
    if max_inner_h < 1 or max_inner_w < 3:
        return None

    cap_to_shapes, area_to_shapes = _shape_catalog(max_inner_h, max_inner_w)
    viable_keys = [
        key
        for key in range(MIN_KEY, MAX_KEY + 1)
        if key in area_to_shapes
        and key in cap_to_shapes
        and any(shape["area"] != key for shape in cap_to_shapes[key])
    ]
    if not viable_keys:
        return None

    candidate_pairs = [
        (key, add_sum)
        for key in viable_keys
        for add_sum in sorted(cap_to_shapes)
        if key < add_sum <= MAX_ADD_SUM and add_sum - key <= MAX_KEY_GAP
    ]
    random.shuffle(candidate_pairs)

    for key, add_sum in candidate_pairs:
        sub_sum = add_sum - key
        bar_heights = _sample_bar_heights(add_sum, sub_sum)
        if bar_heights is None:
            continue
        add_heights, sub_heights = bar_heights

        target_shape = _choose_shape(cap_to_shapes[key], lambda shape: shape["area"] != key)
        area_distractor = _choose_shape(area_to_shapes[key], lambda _shape: True)
        sum_distractor = _choose_shape(cap_to_shapes[add_sum], lambda shape: shape["cap"] != key)
        if target_shape is None or area_distractor is None or sum_distractor is None:
            continue

        total_frames = random.randint(*FRAME_COUNT_RANGE)
        specs = [
            {**target_shape, "role": "target"},
            {**area_distractor, "role": "area_distractor"},
            {**sum_distractor, "role": "sum_distractor"},
        ]

        used_caps = {int(spec["cap"]) for spec in specs}
        while len(specs) < total_frames:
            extra_caps = [cap for cap in cap_to_shapes if cap not in used_caps]
            if not extra_caps:
                break
            cap = random.choice(extra_caps)
            shape = dict(random.choice(cap_to_shapes[cap]))
            if int(shape["area"]) == key:
                used_caps.add(cap)
                continue
            specs.append({**shape, "role": "extra"})
            used_caps.add(cap)

        if len(specs) < 3:
            continue

        random.shuffle(specs)
        colors = random.sample(FRAME_COLORS, len(specs))
        for spec, color in zip(specs, colors):
            spec["color"] = int(color)

        placed = _place_frames(height, width, frame_region_h, specs)
        if placed is None:
            continue

        bar_specs = [{"height": h, "color": ADD_COLOR} for h in add_heights]
        bar_specs.extend({"height": h, "color": SUB_COLOR} for h in sub_heights)
        random.shuffle(bar_specs)
        columns = _sample_spaced_columns(width, len(bar_specs))
        if columns is None:
            continue

        grid = np.full((height, width), BG_COLOR, dtype=int)
        metadata_frames: list[dict] = []
        target_bbox: tuple[int, int, int, int] | None = None

        for spec in placed:
            bbox = tuple(map(int, spec["bbox"]))
            _draw_frame(grid, bbox, int(spec["color"]))
            metadata_frames.append(
                {
                    "bbox": bbox,
                    "cap": int(spec["cap"]),
                    "area": int(spec["area"]),
                    "role": spec["role"],
                }
            )
            if spec["role"] == "target":
                target_bbox = bbox

        if target_bbox is None:
            continue

        for col, bar in zip(columns, bar_specs):
            top = height - int(bar["height"])
            grid[top:height, int(col)] = int(bar["color"])

        metadata = {
            "key": key,
            "add_sum": add_sum,
            "sub_sum": sub_sum,
            "target_bbox": target_bbox,
            "frames": metadata_frames,
        }
        if _validate_metadata(grid, metadata):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arithmetic-key frame input")
