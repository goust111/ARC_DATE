from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.append(arc_utils_path)

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BBox = tuple[int, int, int, int]

BG = 0
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _bbox_inside(inner: BBox, outer: BBox) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _bbox_overlap(a: BBox, b: BBox, gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _largest_pattern_bbox(grid: np.ndarray, bg: int) -> BBox:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected a non-background pattern"
    areas = [int(comp["area"]) for comp in comps]
    largest = max(areas)
    assert areas.count(largest) == 1, "expected one unique largest foreground component"
    comp = comps[areas.index(largest)]
    return tuple(map(int, comp["bbox"]))


def _is_solid_square_component(grid: np.ndarray, comp: dict) -> bool:
    box = tuple(map(int, comp["bbox"]))
    h = bbox_h(box)
    w = bbox_w(box)
    color = int(comp["color"])
    if h != w or h < 2 or int(comp["area"]) != h * w:
        return False
    return bool(np.all(crop_bbox(grid, box) == color))


def _color_role_records(grid: np.ndarray, pattern_bbox: BBox, bg: int) -> list[tuple[int, tuple[int, int], BBox]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    records: list[tuple[int, tuple[int, int], BBox]] = []
    for color, color_comps in by_color.items():
        singles = [comp for comp in color_comps if int(comp["area"]) == 1]
        squares = [comp for comp in color_comps if _is_solid_square_component(grid, comp)]
        if len(singles) != 1 or len(squares) != 1:
            continue

        single = singles[0]
        square = squares[0]
        sr, sc = map(int, single["coords"][0])
        square_bbox = tuple(map(int, square["bbox"]))
        if _bbox_inside((sr, sc, sr, sc), pattern_bbox):
            continue
        if not _bbox_inside(square_bbox, pattern_bbox):
            continue
        records.append((color, (sr, sc), square_bbox))

    assert 2 <= len(records) <= 4, "expected two to four marker/occluder color pairs"
    return records


def _mirror_bbox_across_vertical(box: BBox, frame: BBox) -> BBox:
    r0, c0, r1, c1 = map(int, box)
    _fr0, fc0, _fr1, fc1 = map(int, frame)
    mirror_sum = fc0 + fc1
    return r0, mirror_sum - c1, r1, mirror_sum - c0


def _recover_patch(grid: np.ndarray, occluder_bbox: BBox, pattern_bbox: BBox) -> np.ndarray:
    mirrored_bbox = _mirror_bbox_across_vertical(occluder_bbox, pattern_bbox)
    patch = crop_bbox(grid, mirrored_bbox)
    return np.fliplr(patch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each mirrored occluded square patch and concatenate the recovered patches in marker order.

    Input structure:
    - Black background surrounds one large multicolor pattern that is mirror-symmetric about its vertical axis.
    - Two to four solid square occluders lie on the same side of that pattern; each occluder color is unique to that square and its matching single-pixel marker.
    - The matching marker pixels sit together on one bottom row outside the pattern.

    Rule:
    - Find the largest foreground component and use its bbox as the symmetric pattern frame.
    - For each marker color, locate the same-color solid square occluder inside the frame, reflect that bbox across the frame's vertical axis, and read the visible mirrored region.
    - Flip that mirrored region left-to-right to restore the hidden patch, then concatenate the restored patches from left to right in marker order.

    Key insight:
    - The occluder colors identify which patch to recover, but only the external marker row determines the output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    pattern_bbox = _largest_pattern_bbox(g, bg)
    records = _color_role_records(g, pattern_bbox, bg)

    marker_rows = {r for _color, (r, _c), _bbox in records}
    assert len(marker_rows) == 1, "markers must share one row"
    sorted_records = sorted(records, key=lambda item: item[1][1])

    sizes = {bbox_h(box) for _color, _marker, box in sorted_records}
    assert len(sizes) == 1, "all occluders must share one side length"

    patches = [_recover_patch(g, box, pattern_bbox) for _color, _marker, box in sorted_records]
    return np.concatenate(patches, axis=1)


def _band_index(idx: int, splits: list[int]) -> int:
    return sum(idx >= split for split in splits)


def _partition_splits(length: int, max_splits: int) -> list[int]:
    if length <= 2:
        return []
    upper = min(max_splits, length - 1)
    count = random.randint(1, upper) if upper > 0 else 0
    if count == 0:
        return []
    return sorted(random.sample(range(1, length), count))


def _make_half_pattern(height: int, half_width: int, colors: list[int]) -> np.ndarray:
    row_splits = _partition_splits(height, 3)
    col_splits = _partition_splits(half_width, 2)
    half = np.zeros((height, half_width), dtype=int)

    for r in range(height):
        for c in range(half_width):
            rid = _band_index(r, row_splits)
            cid = _band_index(c, col_splits)
            color = colors[(rid + 2 * cid + (r + c) % 2) % len(colors)]
            half[r, c] = color

    for _ in range(random.randint(3, 6)):
        r0 = random.randint(0, height - 1)
        c0 = random.randint(0, half_width - 1)
        r1 = random.randint(r0, min(height - 1, r0 + random.randint(1, 3)))
        c1 = random.randint(c0, min(half_width - 1, c0 + random.randint(1, 2)))
        half[r0 : r1 + 1, c0 : c1 + 1] = random.choice(colors)

    used = set(map(int, half.flat))
    for color in colors:
        if color not in used:
            r = random.randrange(height)
            c = random.randrange(half_width)
            half[r, c] = color

    return half


def _make_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    half = _make_half_pattern(height, width // 2, colors)
    return np.concatenate([half, np.fliplr(half)], axis=1)


def _sample_occluders(height: int, width: int, side: str, count: int, size: int) -> list[BBox] | None:
    half_width = width // 2
    if side == "left":
        cols = range(0, half_width - size + 1)
    else:
        cols = range(half_width, width - size + 1)
    candidates = [(r, c) for r in range(height - size + 1) for c in cols]
    random.shuffle(candidates)

    boxes: list[BBox] = []
    for r0, c0 in candidates:
        box = (r0, c0, r0 + size - 1, c0 + size - 1)
        if any(_bbox_overlap(box, other, gap=0) for other in boxes):
            continue
        boxes.append(box)
        if len(boxes) == count:
            return boxes
    return None


def _sample_marker_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(80):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _expected_output(pattern: np.ndarray, occluders: dict[int, BBox], marker_order: list[int]) -> np.ndarray:
    pattern_bbox = (0, 0, pattern.shape[0] - 1, pattern.shape[1] - 1)
    patches = [_recover_patch(pattern, occluders[color], pattern_bbox) for color in marker_order]
    return np.concatenate(patches, axis=1)


def _output_unique_under_permutations(patches: list[np.ndarray], expected: np.ndarray) -> bool:
    for perm in itertools.permutations(range(len(patches))):
        if perm == tuple(range(len(patches))):
            continue
        candidate = np.concatenate([patches[i] for i in perm], axis=1)
        if np.array_equal(candidate, expected):
            return False
    return True


def _semantic_checks(
    grid: np.ndarray,
    pattern_bbox: BBox,
    occluders: dict[int, BBox],
    marker_order: list[int],
    expected: np.ndarray,
) -> bool:
    size_set = {bbox_h(box) for box in occluders.values()}
    if len(size_set) != 1:
        return False

    for color, box in occluders.items():
        if int(np.count_nonzero(grid == color)) != bbox_h(box) * bbox_w(box) + 1:
            return False
        mirrored_bbox = _mirror_bbox_across_vertical(box, pattern_bbox)
        mirrored_patch = crop_bbox(grid, mirrored_bbox)
        if np.any(np.isin(mirrored_patch, list(occluders))):
            return False
        restored = np.fliplr(mirrored_patch)
        if len(set(map(int, restored.flat))) < 2:
            return False

    patches = [_recover_patch(grid, occluders[color], pattern_bbox) for color in marker_order]
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            if np.array_equal(patches[i], patches[j]):
                return False

    if not _output_unique_under_permutations(patches, expected):
        return False

    row_major = [color for color, _box in sorted(occluders.items(), key=lambda item: (item[1][0], item[1][1]))]
    if marker_order == row_major:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    size = random.choice([3, 4])
    side = random.choice(["left", "right"])

    pattern_height = random.randint(max(10, size + count + 3), 15)
    half_width = random.randint(max(size + 2, 5), 8)
    pattern_width = half_width * 2

    palette = list(ALL_COLORS)
    random.shuffle(palette)
    occluder_colors = palette[:count]
    pattern_color_count = random.randint(3, min(5, len(palette) - count))
    pattern_colors = palette[count : count + pattern_color_count]

    pattern = _make_pattern(pattern_height, pattern_width, pattern_colors)
    occluder_boxes = _sample_occluders(pattern_height, pattern_width, side, count, size)
    if occluder_boxes is None:
        return None

    color_to_box = dict(zip(occluder_colors, occluder_boxes))
    row_major = [color for color, _box in sorted(color_to_box.items(), key=lambda item: (item[1][0], item[1][1]))]
    perms = [list(p) for p in itertools.permutations(occluder_colors) if list(p) != row_major]
    if not perms:
        return None
    marker_order = random.choice(perms)

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    right = random.randint(1, 4)
    gap_to_markers = random.randint(2, 4)
    grid_height = top + pattern_height + gap_to_markers + 1
    grid_width = left + pattern_width + right
    if max(grid_height, grid_width) > 30:
        return None

    marker_cols = _sample_marker_columns(grid_width, count)
    if marker_cols is None:
        return None

    grid = np.zeros((grid_height, grid_width), dtype=int)
    paste(grid, pattern, top, left)

    for color, box in color_to_box.items():
        r0, c0, r1, c1 = box
        grid[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1] = color

    marker_row = grid_height - 1
    for col, color in zip(marker_cols, marker_order):
        grid[marker_row, col] = color

    pattern_bbox = (top, left, top + pattern_height - 1, left + pattern_width - 1)
    global_color_to_box = {
        color: (top + box[0], left + box[1], top + box[2], left + box[3])
        for color, box in color_to_box.items()
    }
    expected = _expected_output(pattern, color_to_box, marker_order)
    if max(expected.shape) > 30:
        return None
    if not _semantic_checks(grid, pattern_bbox, global_color_to_box, marker_order, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-patch sorting input")
