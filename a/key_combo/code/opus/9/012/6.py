from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, crop_bbox, find_components, get_bg, paste


BBox = tuple[int, int, int, int]

BG = Color.BLACK
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
TARGET_H_CHOICES = (4, 6, 8)
TARGET_W_RANGE = (4, 7)
LEGEND_COUNT_CHOICES = (1, 2)
SOURCE_MIN_PER_HALF = 2
GENERATE_ATTEMPTS = 400
NON_BG_COLORS = tuple(range(1, 10))


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: BBox, pad: int) -> BBox:
    r0, c0, r1, c1 = box
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _far_enough(box: BBox, others: list[BBox], pad: int = 2) -> bool:
    expanded = _expand_bbox(box, pad)
    return all(not _boxes_overlap(expanded, other) for other in others)


def _target_and_mapping(g: np.ndarray, bg: int) -> tuple[dict, dict[int, int]]:
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    target_candidates = []
    legends = []

    for comp in components:
        box = comp["bbox"]
        if comp["area"] == 2 and box[0] == box[2] and box[3] == box[1] + 1:
            legends.append(comp)
            continue

        crop = crop_bbox(g, box)
        if comp["area"] == _bbox_area(box) and len(colors_present(crop, exclude={bg})) >= 2:
            target_candidates.append(comp)

    assert len(target_candidates) == 1, "expected one filled multicolor rectangular target"
    assert 1 <= len(legends) <= 2, "expected one or two legend dominos"
    assert len(target_candidates) + len(legends) == len(components), "unexpected extra components"

    target = target_candidates[0]
    target_crop = crop_bbox(g, target["bbox"])
    target_colors = set(map(int, target_crop.ravel())) - {bg}
    height = target_crop.shape[0]
    assert height % 2 == 0 and height >= 2, "target height must be even"

    mapping: dict[int, int] = {}
    for comp in legends:
        r0, c0, r1, c1 = comp["bbox"]
        replacement = int(g[r0, c0])
        source = int(g[r0, c1])
        assert source != replacement, "legend colors must differ"
        assert source in target_colors, "legend source must appear in the target"
        assert replacement not in target_colors, "legend replacement must be absent from the unrecolored target"
        assert source not in mapping, "legend sources must be distinct"
        mapping[source] = replacement

    assert mapping, "expected at least one mapping"
    return target, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the multicolor target rectangle and recolor only its lower half using the isolated legend dominos.

    Input structure:
    - A dominant black background contains one filled multicolor rectangular target object.
    - One or two disconnected horizontal 1x2 legend dominos sit to the right of or below the target.
    - In each legend domino, the left color does not appear in the unrecolored target and the right color does.

    Rule:
    - Find the unique filled multicolor rectangle and crop the output to its bbox.
    - Read each horizontal legend left-to-right as replacement color R and source color S, defining S->R.
    - Split the cropped target at row H/2, where H is the crop height.
    - Replace source colors by their mapped replacements only in rows H/2 through H-1; keep the upper rows unchanged.

    Key insight:
    - The dominos define valid color substitutions globally, but the spatial rule limits those substitutions to the target's lower half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    target, mapping = _target_and_mapping(g, bg)
    crop = crop_bbox(g, target["bbox"])
    height = crop.shape[0]
    split = height // 2

    out = crop.copy()
    lower = np.arange(height)[:, None] >= split
    for source, replacement in mapping.items():
        out[(crop == source) & lower] = replacement
    return out


def _sample_target_dims() -> tuple[int, int]:
    return random.choice(TARGET_H_CHOICES), random.randint(*TARGET_W_RANGE)


def _choose_palette(n_legends: int) -> tuple[int, list[int], list[int]]:
    colors = random.sample(list(NON_BG_COLORS), 1 + 2 * n_legends)
    base = colors[0]
    sources = colors[1 : 1 + n_legends]
    replacements = colors[1 + n_legends :]
    return base, sources, replacements


def _paint_half(rows: int, width: int, base: int, sources: list[int]) -> np.ndarray:
    flat = np.full(rows * width, base, dtype=int)
    coords = list(range(flat.size))
    random.shuffle(coords)

    idx = 0
    for source in sources:
        for _ in range(SOURCE_MIN_PER_HALF):
            flat[coords[idx]] = source
            idx += 1

    extra_budget = flat.size - idx
    if extra_budget > 0:
        extra_count = random.randint(0, extra_budget)
        for pos in coords[idx : idx + extra_count]:
            flat[pos] = random.choice([base] + sources)

    arr = flat.reshape(rows, width)
    assert any(np.any(arr == source) for source in sources)
    return arr


def _build_target(height: int, width: int, base: int, sources: list[int]) -> np.ndarray:
    half = height // 2
    top = _paint_half(half, width, base, sources)
    bottom = _paint_half(half, width, base, sources)
    target = np.vstack([top, bottom])

    assert np.any(top == base) and np.any(bottom == base), "base color should remain visible in both halves"
    for source in sources:
        assert np.any(top == source), "each source must appear in the upper half"
        assert np.any(bottom == source), "each source must appear in the lower half"
    return target


def _apply_mapping_bottom_only(target: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = target.copy()
    split = target.shape[0] // 2
    for source, replacement in mapping.items():
        patch = out[split:, :]
        patch[patch == source] = replacement
        out[split:, :] = patch
    return out


def _sample_canvas_dims(height: int, width: int, need_right: bool, need_below: bool) -> tuple[int, int]:
    min_h = max(GRID_H_RANGE[0], height + (5 if need_below else 3))
    min_w = max(GRID_W_RANGE[0], width + (5 if need_right else 3))
    max_h = max(min_h, GRID_H_RANGE[1])
    max_w = max(min_w, GRID_W_RANGE[1])
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _side_plan(n_legends: int) -> list[str]:
    if n_legends == 1:
        return [random.choice(["right", "below"])]
    if random.random() < 0.5:
        return ["right", "below"]
    return [random.choice(["right", "below"]) for _ in range(2)]


def _candidate_positions(
    grid_shape: tuple[int, int],
    side: str,
    occupied_boxes: list[BBox],
    target_box: BBox,
) -> list[tuple[int, int]]:
    height, width = grid_shape
    target_r0, target_c0, target_r1, target_c1 = target_box
    candidates = []
    for r in range(height):
        for c in range(width - 1):
            box = (r, c, r, c + 1)
            if side == "right" and c < target_c1 + 3:
                continue
            if side == "below" and r < target_r1 + 3:
                continue
            if side not in {"right", "below"}:
                continue
            if c < 0 or c + 1 >= width:
                continue
            if not _far_enough(box, occupied_boxes, pad=2):
                continue
            candidates.append((r, c))
    return candidates


def _place_legends(
    canvas: np.ndarray,
    target_box: BBox,
    replacements: list[int],
    sources: list[int],
    sides: list[str],
) -> bool:
    occupied = [target_box]
    legend_specs = list(zip(replacements, sources, sides))
    random.shuffle(legend_specs)

    for replacement, source, side in legend_specs:
        candidates = _candidate_positions(canvas.shape, side, occupied, target_box)
        if not candidates:
            return False
        r, c = random.choice(candidates)
        paste(canvas, np.array([[replacement, source]], dtype=int), r, c)
        occupied.append((r, c, r, c + 1))
    return True


def _build_canvas(target: np.ndarray, replacements: list[int], sources: list[int], sides: list[str]) -> np.ndarray | None:
    target_h, target_w = target.shape
    need_right = "right" in sides
    need_below = "below" in sides
    canvas_h, canvas_w = _sample_canvas_dims(target_h, target_w, need_right, need_below)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    max_r = canvas_h - target_h - (3 if need_below else 1)
    max_c = canvas_w - target_w - (3 if need_right else 1)
    if max_r < 1 or max_c < 1:
        return None

    target_r = random.randint(1, max_r)
    left_cap = max(1, min(max_c, canvas_w // 2))
    target_c = random.randint(1, left_cap)
    paste(canvas, target, target_r, target_c)
    target_box = (target_r, target_c, target_r + target_h - 1, target_c + target_w - 1)

    if not _place_legends(canvas, target_box, replacements, sources, sides):
        return None
    return canvas


def _validate_case(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    bg = get_bg(inp)
    target, mapping = _target_and_mapping(inp, bg)
    crop = crop_bbox(inp, target["bbox"])
    height = crop.shape[0]
    split = height // 2
    top = crop[:split, :]
    bottom = crop[split:, :]

    if not (4 <= crop.shape[0] <= 8 and 4 <= crop.shape[1] <= 7 and crop.shape[0] % 2 == 0):
        return False
    if not (1 <= len(mapping) <= 2):
        return False

    for source, replacement in mapping.items():
        if replacement in crop:
            return False
        if not np.any(top == source):
            return False
        if not np.any(bottom == source):
            return False

    expected = _apply_mapping_bottom_only(crop, mapping)
    if not np.array_equal(out, expected):
        return False

    global_replace = crop.copy()
    for source, replacement in mapping.items():
        global_replace[global_replace == source] = replacement
    if np.array_equal(global_replace, out):
        return False
    if np.array_equal(crop, out):
        return False

    top_out = out[:split, :]
    bottom_out = out[split:, :]
    for source, replacement in mapping.items():
        if not np.any(top_out == source):
            return False
        if not np.any(bottom_out == replacement):
            return False

    components = find_components(inp, bg=bg, connectivity=4, monochromatic=False)
    target_box = target["bbox"]
    for comp in components:
        if comp["bbox"] == target_box:
            continue
        if not _far_enough(comp["bbox"], [target_box], pad=2):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_legends = random.choice(LEGEND_COUNT_CHOICES)
    sides = _side_plan(n_legends)
    target_h, target_w = _sample_target_dims()
    base, sources, replacements = _choose_palette(n_legends)
    target = _build_target(target_h, target_w, base, sources)
    canvas = _build_canvas(target, replacements, sources, sides)
    if canvas is None:
        return None
    if not _validate_case(canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
