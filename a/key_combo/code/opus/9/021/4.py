from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


def _row_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _count_solid_2x2(grid: np.ndarray, bg: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color == int(bg):
                continue
            if (
                int(grid[r + 1, c]) == color
                and int(grid[r, c + 1]) == color
                and int(grid[r + 1, c + 1]) == color
            ):
                counts[color] = counts.get(color, 0) + 1
    return counts


def _extract_templates(bottom: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    components = find_components(bottom, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected bottom templates"
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "template bbox must be 3x3"
        assert 3 <= int(comp["area"]) <= 7, "template area must be 3..7"
        assert color not in templates, "each color must have exactly one template"
        patch = np.full((3, 3), int(bg), dtype=int)
        coords = comp["coords"]
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
        templates[color] = patch

    mask_keys = {tuple((patch != bg).ravel().tolist()) for patch in templates.values()}
    assert len(mask_keys) == len(templates), "template masks must be distinct"
    return templates


def _render_histogram(
    count_by_color: dict[int, int],
    template_by_color: dict[int, np.ndarray],
    bg: int,
) -> np.ndarray:
    ranked = sorted(count_by_color.items(), key=lambda item: item[1], reverse=True)
    width = 3 * ranked[0][1]
    out = np.full((3 * len(ranked), width), int(bg), dtype=int)
    for row, (color, count) in enumerate(ranked):
        stamp = template_by_color[color]
        for rep in range(count):
            paste(out, stamp, row * 3, rep * 3, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count solid 2x2 blocks by color, match each color to its lower 3x3 stencil, and stamp a sorted histogram.

    Input structure:
    - The grid has a black background with an upper counting region and a lower template region.
    - In the upper region, each relevant color appears in several solid 2x2 blocks plus isolated same-color single-pixel distractors.
    - In the lower region, each relevant color has exactly one 3x3 connected stencil mask of that same color; the lower region is the bottom-most foreground row band.

    Rule:
    - Ignore the single-pixel distractors and scan only the upper region for 2x2 windows filled entirely by one non-background color.
    - Count those solid 2x2 windows per color.
    - Read the lower 3x3 stencil for each color and sort the colors by decreasing 2x2-block count.
    - Build the output with one 3-row band per color, stamping that color's 3x3 stencil left-to-right exactly as many times as its count.

    Key insight:
    - Three coupled cues are required: only full 2x2 windows count, color chooses the stencil shape, and the count order chooses the output row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    row_runs = _row_runs(np.any(g != bg, axis=1))
    assert row_runs, "expected foreground"
    bottom_r0, bottom_r1 = row_runs[-1]
    assert bottom_r1 - bottom_r0 + 1 == 3, "bottom template band must be 3 rows tall"

    upper = g[:bottom_r0, :]
    assert np.any(upper != bg), "expected upper counting region"
    upper_rows = np.flatnonzero(np.any(upper != bg, axis=1))
    assert upper_rows.size > 0
    assert bottom_r0 - int(upper_rows[-1]) >= 3, "upper and lower regions must be separated by at least 2 blank rows"

    count_by_color = _count_solid_2x2(upper, bg)
    assert count_by_color, "expected at least one counted color"
    counts = list(count_by_color.values())
    assert len(set(counts)) == len(counts), "2x2 counts must be unique"

    bottom = g[bottom_r0 : bottom_r1 + 1, :]
    template_by_color = _extract_templates(bottom, bg)
    assert set(template_by_color) == set(count_by_color), "counted colors and template colors must match"

    return _render_histogram(count_by_color, template_by_color, bg)


# --------------- generator ---------------

GEN_RETRIES = 400
OUTER_PAD_RANGE = (1, 3)
REGION_GAP_RANGE = (2, 4)
TOP_HEIGHT_RANGE = (8, 11)
TOP_WIDTH_RANGE = (18, 24)
COLOR_OPTIONS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
NOISE_RANGE = range(1, 7)
COUNT_START_OPTIONS = {2: [1, 2, 3], 3: [1, 2]}
COLOR_COUNT_OPTIONS = [2, 3]
PLACE_KWARGS = dict(transparent=Color.BLACK, padding=1, padding_connectivity=4)

_TEMPLATE_POOL: list[np.ndarray] | None = None


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(coords[0])}
    stack = [tuple(coords[0])]
    wanted = {tuple(coord) for coord in coords.tolist()}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in wanted and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(wanted)


def _template_pool() -> list[np.ndarray]:
    global _TEMPLATE_POOL
    if _TEMPLATE_POOL is not None:
        return _TEMPLATE_POOL

    pool: list[np.ndarray] = []
    for bits in range(1 << 9):
        count = bits.bit_count()
        if count < 3 or count > 7:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if _connected(mask):
            pool.append(mask.astype(int))
    _TEMPLATE_POOL = pool
    return pool


def _choose_counts(colors: list[int]) -> dict[int, int]:
    k = len(colors)
    start = random.choice(COUNT_START_OPTIONS[k])
    values = list(range(start, start + k))
    random.shuffle(values)
    return dict(zip(colors, values))


def _unique_ranking(value_by_color: dict[int, int]) -> list[int] | None:
    values = list(value_by_color.values())
    if len(set(values)) != len(values):
        return None
    return [color for color, _value in sorted(value_by_color.items(), key=lambda item: item[1], reverse=True)]


def _choose_noise_counts(count_by_color: dict[int, int]) -> dict[int, int] | None:
    correct_order = _unique_ranking(count_by_color)
    assert correct_order is not None
    colors = list(count_by_color)
    candidates: list[dict[int, int]] = []
    for noise_values in itertools.product(NOISE_RANGE, repeat=len(colors)):
        noise_by_color = dict(zip(colors, noise_values))
        object_rank = _unique_ranking({color: count_by_color[color] + noise_by_color[color] for color in colors})
        pixel_rank = _unique_ranking({color: 4 * count_by_color[color] + noise_by_color[color] for color in colors})
        if object_rank is None or pixel_rank is None:
            continue
        if object_rank != correct_order and pixel_rank != correct_order:
            candidates.append(noise_by_color)
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_templates(colors: list[int]) -> dict[int, np.ndarray]:
    masks = random.sample(_template_pool(), len(colors))
    template_by_color: dict[int, np.ndarray] = {}
    for color, mask in zip(colors, masks):
        template_by_color[color] = np.where(mask, color, Color.BLACK).astype(int)
    return template_by_color


def _bottom_order(colors: list[int], count_by_color: dict[int, int]) -> list[int]:
    target_order = [color for color, _count in sorted(count_by_color.items(), key=lambda item: item[1], reverse=True)]
    for _ in range(50):
        order = random.sample(colors, len(colors))
        if order != target_order:
            return order
    return list(reversed(target_order))


def _place_top_objects(count_by_color: dict[int, int], noise_by_color: dict[int, int]) -> np.ndarray | None:
    top_h = random.randint(*TOP_HEIGHT_RANGE)
    top_w = random.randint(*TOP_WIDTH_RANGE)
    canvas = np.zeros((top_h, top_w), dtype=int)

    objects: list[tuple[int, np.ndarray]] = []
    for color, count in count_by_color.items():
        block = np.full((2, 2), color, dtype=int)
        noise = np.array([[color]], dtype=int)
        for _ in range(count):
            objects.append((4, block))
        for _ in range(noise_by_color[color]):
            objects.append((1, noise))
    random.shuffle(objects)
    objects.sort(key=lambda item: item[0], reverse=True)

    for _size, patch in objects:
        try:
            r0, c0 = random_free_location(canvas, patch, **PLACE_KWARGS)
        except ValueError:
            return None
        paste(canvas, patch, r0, c0)
    return canvas


def _build_bottom_patch(order: list[int], template_by_color: dict[int, np.ndarray]) -> np.ndarray:
    width = 4 * len(order) - 1
    bottom = np.zeros((3, width), dtype=int)
    for idx, color in enumerate(order):
        paste(bottom, template_by_color[color], 0, idx * 4, transparent=Color.BLACK)
    return bottom


def _assemble_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top_pad = random.randint(*OUTER_PAD_RANGE)
    bottom_pad = random.randint(*OUTER_PAD_RANGE)
    left_pad = random.randint(*OUTER_PAD_RANGE)
    right_pad = random.randint(*OUTER_PAD_RANGE)
    gap = random.randint(*REGION_GAP_RANGE)

    min_width = max(top.shape[1], bottom.shape[1]) + left_pad + right_pad
    width = random.randint(min_width, min(30, min_width + 4))
    height = top_pad + top.shape[0] + gap + bottom.shape[0] + bottom_pad
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    top_c0 = random.randint(left_pad, width - right_pad - top.shape[1])
    bottom_c0 = random.randint(left_pad, width - right_pad - bottom.shape[1])
    top_r0 = top_pad
    bottom_r0 = top_r0 + top.shape[0] + gap

    paste(grid, top, top_r0, top_c0, transparent=Color.BLACK)
    paste(grid, bottom, bottom_r0, bottom_c0, transparent=Color.BLACK)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    count_by_color: dict[int, int],
    noise_by_color: dict[int, int],
    template_by_color: dict[int, np.ndarray],
    bottom_order: list[int],
) -> bool:
    bg = int(get_bg(grid))
    row_runs = _row_runs(np.any(grid != bg, axis=1))
    if not row_runs or row_runs[-1][1] - row_runs[-1][0] + 1 != 3:
        return False

    bottom_r0, bottom_r1 = row_runs[-1]
    upper = grid[:bottom_r0, :]
    actual_counts = _count_solid_2x2(upper, bg)
    if actual_counts != count_by_color:
        return False
    if len(set(actual_counts.values())) != len(actual_counts):
        return False

    bottom = grid[bottom_r0 : bottom_r1 + 1, :]
    actual_templates = _extract_templates(bottom, bg)
    if set(actual_templates) != set(template_by_color):
        return False
    for color in template_by_color:
        if not np.array_equal(actual_templates[color], template_by_color[color]):
            return False

    comps = find_components(upper, bg=bg, connectivity=4, monochromatic=True)
    object_totals = {color: 0 for color in count_by_color}
    pixel_totals = {color: int((upper == color).sum()) for color in count_by_color}
    for comp in comps:
        color = int(comp["color"])
        if color in object_totals:
            object_totals[color] += 1

    correct_rank = _unique_ranking(count_by_color)
    if correct_rank is None:
        return False
    if _unique_ranking(object_totals) == correct_rank:
        return False
    if _unique_ranking(pixel_totals) == correct_rank:
        return False
    if any(object_totals[color] != count_by_color[color] + noise_by_color[color] for color in count_by_color):
        return False

    left_to_right = []
    for comp in sorted(find_components(bottom, bg=bg, connectivity=4, monochromatic=True), key=lambda comp: comp["bbox"][1]):
        left_to_right.append(int(comp["color"]))
    if left_to_right != bottom_order:
        return False
    if left_to_right == correct_rank:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    color_count = random.choice(COLOR_COUNT_OPTIONS)
    colors = random.sample(COLOR_OPTIONS, color_count)
    count_by_color = _choose_counts(colors)
    noise_by_color = _choose_noise_counts(count_by_color)
    if noise_by_color is None:
        return None

    template_by_color = _choose_templates(colors)
    order = _bottom_order(colors, count_by_color)
    top = _place_top_objects(count_by_color, noise_by_color)
    if top is None:
        return None
    bottom = _build_bottom_patch(order, template_by_color)
    grid = _assemble_input(top, bottom)
    if grid is None:
        return None

    expected = _render_histogram(count_by_color, template_by_color, Color.BLACK)
    if not _semantic_checks(grid, count_by_color, noise_by_color, template_by_color, order):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
