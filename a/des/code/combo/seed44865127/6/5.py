from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import gcd
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            assert len(colors) == 1, "each foreground object must be monochromatic"
            color = next(iter(colors))
            coords_arr = np.asarray(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            obj_h = int(r1 - r0 + 1)
            obj_w = int(c1 - c0 + 1)
            bbox = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert coords_arr.shape[0] == obj_h * obj_w, "each object must be a solid rectangle"
            assert bool(np.all(bbox == color)), "rectangle bbox must be fully filled"

            objects.append(
                {
                    "color": int(color),
                    "h": obj_h,
                    "w": obj_w,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    objects.sort(key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))
    return objects


def _ratio_key(width: int, height: int) -> tuple[int, int]:
    d = gcd(width, height)
    return (width // d, height // d)


def _group_patch(group: list[dict[str, int | tuple[int, int, int, int]]], bg: int) -> np.ndarray:
    max_obj = max(group, key=lambda obj: int(obj["h"]) * int(obj["w"]))
    out = np.full((int(max_obj["h"]), int(max_obj["w"])), bg, dtype=int)
    for obj in group:
        out[: int(obj["h"]), : int(obj["w"])] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group solid rectangles by reduced width:height ratio, merge each group by
    left-aligned overlay, then order the merged rectangles by increasing ratio.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    objects = _find_objects(g, bg)
    assert objects, "expected at least one foreground rectangle"

    grouped: dict[tuple[int, int], list[dict[str, int | tuple[int, int, int, int]]]] = defaultdict(list)
    for obj in objects:
        ratio = _ratio_key(int(obj["w"]), int(obj["h"]))
        grouped[ratio].append(obj)

    ordered_ratios = sorted(grouped, key=lambda ratio: Fraction(ratio[0], ratio[1]))
    merged = [_group_patch(grouped[ratio], bg) for ratio in ordered_ratios]

    out_h = max(patch.shape[0] for patch in merged)
    out_w = sum(patch.shape[1] for patch in merged)
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for patch in merged:
        ph, pw = patch.shape
        out[:ph, col : col + pw] = patch
        col += pw

    return out


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))

TALL_RATIOS = ((1, 2), (2, 3))
SQUARE_RATIOS = ((1, 1),)
WIDE_RATIOS = ((3, 2), (2, 1))
ALL_RATIOS = TALL_RATIOS + SQUARE_RATIOS + WIDE_RATIOS

GROUP_SIZE_PROFILES = (
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (2, 1, 1),
    (2, 2, 1),
    (3, 1, 1),
    (2, 1, 1, 1),
)

MAX_GRID_SIDE = 30
MAX_OBJECT_SIDE = 9
MIN_GAP = 1
MAX_GAP = 3
MIN_MARGIN = 1
MAX_MARGIN = 4
EXTRA_SIDE_SLACK = 6
MAX_GENERATION_ATTEMPTS = 400


def _sample_group_sizes() -> list[int]:
    return list(random.choice(GROUP_SIZE_PROFILES))


def _sample_ratio_set(n_groups: int) -> list[tuple[int, int]]:
    ratios = [random.choice(TALL_RATIOS), random.choice(WIDE_RATIOS)]
    pool = [ratio for ratio in ALL_RATIOS if ratio not in ratios]
    if n_groups >= 3 and (1, 1) in pool and random.random() < 0.65:
        ratios.append((1, 1))
        pool.remove((1, 1))
    while len(ratios) < n_groups:
        pick = random.choice(pool)
        ratios.append(pick)
        pool.remove(pick)
    random.shuffle(ratios)
    return ratios


def _ordered_scales(count: int, ratio: tuple[int, int], featured: bool) -> list[int] | None:
    max_scale = min(3, MAX_OBJECT_SIDE // max(ratio))
    if count == 1:
        if max_scale < 1:
            return None
        return [random.randint(1, max_scale)]
    if max_scale < count:
        return None

    scales = sorted(random.sample(range(1, max_scale + 1), count))
    if featured:
        return list(reversed(scales))
    random.shuffle(scales)
    return scales


def _build_group_objects() -> list[list[dict[str, int | tuple[int, int]]]] | None:
    group_sizes = _sample_group_sizes()
    ratios = _sample_ratio_set(len(group_sizes))
    multi_groups = [idx for idx, size in enumerate(group_sizes) if size > 1]
    assert multi_groups
    featured_group = random.choice(multi_groups)

    group_objects: list[list[dict[str, int | tuple[int, int]]]] = []
    for idx, (ratio, count) in enumerate(zip(ratios, group_sizes)):
        scales = _ordered_scales(count, ratio, featured=idx == featured_group)
        if scales is None:
            return None
        objects = []
        for scale in scales:
            obj_w = ratio[0] * scale
            obj_h = ratio[1] * scale
            objects.append({"ratio": ratio, "w": obj_w, "h": obj_h})
        group_objects.append(objects)

    return group_objects


def _interleave_groups(group_objects: list[list[dict[str, int | tuple[int, int]]]]) -> list[dict[str, int | tuple[int, int]]]:
    queues = [group[:] for group in group_objects]
    sequence: list[dict[str, int | tuple[int, int]]] = []
    active = [idx for idx, group in enumerate(queues) if group]
    while active:
        idx = random.choice(active)
        sequence.append(queues[idx].pop(0))
        if not queues[idx]:
            active.remove(idx)
    return sequence


def _needs_more_structure(sequence: list[dict[str, int | tuple[int, int]]]) -> bool:
    ratio_order = [_ratio_key(int(obj["w"]), int(obj["h"])) for obj in sequence]
    seen: set[tuple[int, int]] = set()
    first_appearance: list[tuple[int, int]] = []
    for ratio in ratio_order:
        if ratio not in seen:
            seen.add(ratio)
            first_appearance.append(ratio)

    sorted_ratios = sorted(seen, key=lambda ratio: Fraction(ratio[0], ratio[1]))
    if first_appearance != sorted_ratios:
        return False

    for ratio in seen:
        positions = [idx for idx, cur in enumerate(ratio_order) if cur == ratio]
        if positions[-1] - positions[0] + 1 != len(positions):
            return False

    return True


def _assign_colors(sequence: list[dict[str, int | tuple[int, int]]], bg: int) -> None:
    palette = random.sample([color for color in COLOR_VALUES if color != bg], len(sequence))
    for obj, color in zip(sequence, palette):
        obj["color"] = int(color)


def _render_input(sequence: list[dict[str, int | tuple[int, int]]], bg: int) -> np.ndarray | None:
    widths = [int(obj["w"]) for obj in sequence]
    heights = [int(obj["h"]) for obj in sequence]
    fg_area = sum(int(obj["w"]) * int(obj["h"]) for obj in sequence)

    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(max(0, len(sequence) - 1))]
    left_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    right_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    base_width = left_margin + sum(widths) + sum(gaps) + right_margin
    if base_width > MAX_GRID_SIDE:
        return None

    max_height = max(heights)
    top_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    bottom_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    base_height = top_margin + max_height + bottom_margin
    if base_height > MAX_GRID_SIDE:
        return None

    extra_w = random.randint(0, min(EXTRA_SIDE_SLACK, MAX_GRID_SIDE - base_width))
    extra_h = random.randint(0, min(EXTRA_SIDE_SLACK, MAX_GRID_SIDE - base_height))
    width = base_width + extra_w
    height = base_height + extra_h
    if width * height <= 2 * fg_area:
        return None

    grid = np.full((height, width), bg, dtype=int)
    col = left_margin
    for idx, obj in enumerate(sequence):
        obj_h = int(obj["h"])
        obj_w = int(obj["w"])
        row = random.randint(top_margin, height - bottom_margin - obj_h)
        grid[row : row + obj_h, col : col + obj_w] = int(obj["color"])
        col += obj_w
        if idx < len(gaps):
            col += gaps[idx]

    return grid


def _visible_multicolor_group(group_objects: list[list[dict[str, int | tuple[int, int]]]], bg: int) -> bool:
    for group in group_objects:
        patch = _group_patch(group, bg)
        colors = set(int(color) for color in np.unique(patch) if int(color) != bg)
        if len(colors) >= 2:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_VALUES)
    group_objects = _build_group_objects()
    if group_objects is None:
        return None

    sequence = _interleave_groups(group_objects)
    if _needs_more_structure(sequence):
        return None

    _assign_colors(sequence, bg)
    ratio_to_objects: dict[tuple[int, int], list[dict[str, int | tuple[int, int]]]] = defaultdict(list)
    for obj in sequence:
        ratio_to_objects[_ratio_key(int(obj["w"]), int(obj["h"]))].append(obj)
    grouped = [ratio_to_objects[ratio] for ratio in ratio_to_objects]
    if not _visible_multicolor_group(grouped, bg):
        return None

    grid = _render_input(sequence, bg)
    if grid is None:
        return None

    counts = np.bincount(grid.ravel(), minlength=len(COLOR_VALUES))
    if int(np.argmax(counts)) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    out_bg = _get_bg(out)
    merged_color_counts = []
    ordered_ratios = sorted(ratio_to_objects, key=lambda ratio: Fraction(ratio[0], ratio[1]))
    offset = 0
    for ratio in ordered_ratios:
        patch = _group_patch(ratio_to_objects[ratio], bg)
        ph, pw = patch.shape
        window = out[:ph, offset : offset + pw]
        if not np.array_equal(window, patch):
            return None
        merged_color_counts.append(len(set(int(color) for color in np.unique(patch) if int(color) != bg)))
        offset += pw

    if max(merged_color_counts) < 2:
        return None
    if out_bg != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
