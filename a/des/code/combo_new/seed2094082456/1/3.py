from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
CANVAS_ROW_RANGE = (8, 20)
CANVAS_COL_RANGE = (8, 20)
COLOR_COUNT_RANGE = (2, 5)
SEED_COUNT_RANGE = (2, 6)
MAX_TEMPLATE_SPAN = 6
CASE_OPTIONS = ("plain", "clip", "overlap", "both")
MAX_GENERATION_ATTEMPTS = 500
MAX_OUTPUT_FILL_RATIO = 0.8


def _decode_runs(line: np.ndarray) -> dict[int, int]:
    runs: dict[int, int] = {}
    i = 0
    n = int(line.shape[0])
    while i < n:
        color = int(line[i])
        if color == BLACK:
            i += 1
            continue
        j = i
        while j < n and int(line[j]) == color:
            j += 1
        assert color not in runs, "each border color must appear in exactly one run"
        runs[color] = j - i
        i = j
    return runs


def _paint_clipped(canvas: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    r1 = min(canvas.shape[0], r + h)
    c1 = min(canvas.shape[1], c + w)
    canvas[r:r1, c:c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each interior seed into the rectangle encoded for its color by the top row and left column.

    Input structure:
    - The background is black.
    - The top row and left column form a template-encoding border; the remaining cells are the canvas.
    - Each non-black color appears once as a horizontal run in the top row and once as a vertical run in the left column.
    - The canvas contains scattered single-pixel seeds whose colors match those border runs.

    Rule:
    - Read each top-row run length as that color's rectangle width and each left-column run length as that color's rectangle height.
    - Scan the canvas from top to bottom, then left to right.
    - For every non-black seed, draw a solid rectangle of the decoded height and width with the seed as its top-left corner.
    - Leave the border unchanged, clip rectangles to the canvas bounds, and let later seeds overwrite earlier ones.

    Key insight:
    - The row border and column border encode complementary template dimensions, so matching equal colors across the two borders reveals each growth rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "expected a bordered 2D grid"

    widths = _decode_runs(g[0, 1:])
    heights = _decode_runs(g[1:, 0])
    assert widths and set(widths) == set(heights), "top row and left column must encode the same colors"

    out = g.copy()
    src = g[1:, 1:]
    canvas = out[1:, 1:]
    canvas_h, canvas_w = map(int, src.shape)

    for r in range(canvas_h):
        for c in range(canvas_w):
            color = int(src[r, c])
            if color == BLACK:
                continue
            assert color in widths and color in heights, "seed color must have both width and height encodings"
            _paint_clipped(canvas, r, c, heights[color], widths[color], color)

    return out


def _sample_lengths(span: int, count: int) -> list[int] | None:
    max_run_sum = span - (count - 1)
    if max_run_sum < count:
        return None

    run_cap = min(MAX_TEMPLATE_SPAN, max_run_sum - count + 1)
    if run_cap < 1:
        return None

    lengths = [1] * count
    capacities = [run_cap - 1] * count
    total_extra = random.randint(0, min(sum(capacities), max_run_sum - count))
    while total_extra > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        idx = random.choice(choices)
        capacities[idx] -= 1
        lengths[idx] += 1
        total_extra -= 1
    return lengths


def _render_border_line(span: int, lengths_by_color: dict[int, int]) -> np.ndarray:
    colors = list(lengths_by_color)
    random.shuffle(colors)

    line = np.zeros(span, dtype=int)
    slack = span - sum(lengths_by_color[color] for color in colors) - (len(colors) - 1)
    assert slack >= 0
    extras = [0] * (len(colors) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    cursor = extras[0]
    for i, color in enumerate(colors):
        length = lengths_by_color[color]
        line[cursor : cursor + length] = int(color)
        cursor += length
        if i < len(colors) - 1:
            cursor += 1 + extras[i + 1]
    return line


def _footprint(r: int, c: int, h: int, w: int, canvas_h: int, canvas_w: int) -> tuple[int, int, int, int]:
    return r, c, min(canvas_h, r + h), min(canvas_w, c + w)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _seed_clips(seed: tuple[int, int, int], heights: dict[int, int], widths: dict[int, int], canvas_h: int, canvas_w: int) -> bool:
    r, c, color = seed
    return r + heights[color] > canvas_h or c + widths[color] > canvas_w


def _has_overlap(seeds: list[tuple[int, int, int]], heights: dict[int, int], widths: dict[int, int], canvas_h: int, canvas_w: int) -> bool:
    boxes = [
        _footprint(r, c, heights[color], widths[color], canvas_h, canvas_w)
        for r, c, color in seeds
    ]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                return True
    return False


def _safe_positions(color: int, heights: dict[int, int], widths: dict[int, int], canvas_h: int, canvas_w: int) -> list[tuple[int, int]]:
    h = heights[color]
    w = widths[color]
    return [(r, c) for r in range(canvas_h - h + 1) for c in range(canvas_w - w + 1)]


def _clip_positions(
    color: int,
    mode: str,
    heights: dict[int, int],
    widths: dict[int, int],
    canvas_h: int,
    canvas_w: int,
) -> list[tuple[int, int]]:
    h = heights[color]
    w = widths[color]
    positions: list[tuple[int, int]] = []
    for r in range(canvas_h):
        for c in range(canvas_w):
            clips_bottom = r + h > canvas_h
            clips_right = c + w > canvas_w
            if mode == "bottom" and clips_bottom and not clips_right:
                positions.append((r, c))
            elif mode == "right" and clips_right and not clips_bottom:
                positions.append((r, c))
            elif mode == "both" and clips_bottom and clips_right:
                positions.append((r, c))
    return positions


def _choose_overlap_pair(
    colors: list[int],
    heights: dict[int, int],
    widths: dict[int, int],
    canvas_h: int,
    canvas_w: int,
    occupied_positions: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]] | None:
    starters = [color for color in colors if heights[color] * widths[color] > 1]
    random.shuffle(starters)

    for color_a in starters:
        other_colors = [color for color in colors if color != color_a]
        random.shuffle(other_colors)
        h_a = heights[color_a]
        w_a = widths[color_a]
        a_positions = _safe_positions(color_a, heights, widths, canvas_h, canvas_w)
        random.shuffle(a_positions)

        for ra, ca in a_positions:
            if (ra, ca) in occupied_positions:
                continue
            box_a = _footprint(ra, ca, h_a, w_a, canvas_h, canvas_w)

            for color_b in other_colors:
                h_b = heights[color_b]
                w_b = widths[color_b]
                candidates: list[tuple[int, int, int, int]] = []
                for rb in range(box_a[0], box_a[2]):
                    for cb in range(box_a[1], box_a[3]):
                        if (rb, cb) == (ra, ca):
                            continue
                        if rb < ra or (rb == ra and cb <= ca):
                            continue
                        if (rb, cb) in occupied_positions:
                            continue
                        if rb + h_b > canvas_h or cb + w_b > canvas_w:
                            continue
                        box_b = _footprint(rb, cb, h_b, w_b, canvas_h, canvas_w)
                        if _boxes_overlap(box_a, box_b):
                            candidates.append((rb, cb, box_b[2], box_b[3]))
                if candidates:
                    rb, cb, r2, c2 = random.choice(candidates)
                    box_b = (rb, cb, r2, c2)
                    seeds = [(ra, ca, color_a), (rb, cb, color_b)]
                    return seeds, [box_a, box_b]
    return None


def _choose_clip_seed(
    colors: list[int],
    heights: dict[int, int],
    widths: dict[int, int],
    canvas_h: int,
    canvas_w: int,
    occupied_positions: set[tuple[int, int]],
    existing_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]] | None:
    feasible_modes = []
    for mode in ("bottom", "right", "both"):
        if any(_clip_positions(color, mode, heights, widths, canvas_h, canvas_w) for color in colors):
            feasible_modes.append(mode)
    if not feasible_modes:
        return None

    mode = random.choice(feasible_modes)
    color_order = colors[:]
    random.shuffle(color_order)
    for color in color_order:
        candidates = _clip_positions(color, mode, heights, widths, canvas_h, canvas_w)
        random.shuffle(candidates)
        for r, c in candidates:
            if (r, c) in occupied_positions:
                continue
            box = _footprint(r, c, heights[color], widths[color], canvas_h, canvas_w)
            if any(_boxes_overlap(box, other) for other in existing_boxes):
                continue
            return (r, c, color), box
    return None


def _add_safe_seed(
    seeds: list[tuple[int, int, int]],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    heights: dict[int, int],
    widths: dict[int, int],
    canvas_h: int,
    canvas_w: int,
) -> bool:
    occupied_positions = {(r, c) for r, c, _color in seeds}
    color_order = colors[:]
    random.shuffle(color_order)

    for color in color_order:
        candidates = _safe_positions(color, heights, widths, canvas_h, canvas_w)
        random.shuffle(candidates)
        for r, c in candidates:
            if (r, c) in occupied_positions:
                continue
            box = _footprint(r, c, heights[color], widths[color], canvas_h, canvas_w)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            seeds.append((r, c, color))
            boxes.append(box)
            return True
    return False


def _build_grid(
    canvas_h: int,
    canvas_w: int,
    heights: dict[int, int],
    widths: dict[int, int],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros((canvas_h + 1, canvas_w + 1), dtype=int)
    grid[0, 1:] = _render_border_line(canvas_w, widths)
    grid[1:, 0] = _render_border_line(canvas_h, heights)
    for r, c, color in seeds:
        grid[r + 1, c + 1] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_ROW_RANGE)
    canvas_w = random.randint(*CANVAS_COL_RANGE)
    max_colors = min(COLOR_COUNT_RANGE[1], (canvas_h + 1) // 2, (canvas_w + 1) // 2)
    if max_colors < COLOR_COUNT_RANGE[0]:
        return None

    color_count = random.randint(COLOR_COUNT_RANGE[0], max_colors)
    colors = random.sample(list(COLOR_POOL), color_count)
    height_values = _sample_lengths(canvas_h, color_count)
    width_values = _sample_lengths(canvas_w, color_count)
    if height_values is None or width_values is None:
        return None

    heights = {color: length for color, length in zip(colors, height_values)}
    widths = {color: length for color, length in zip(colors, width_values)}
    if max(heights[color] * widths[color] for color in colors) == 1:
        return None

    case = random.choice(CASE_OPTIONS)
    want_clip = case in {"clip", "both"}
    want_overlap = case in {"overlap", "both"}

    seeds: list[tuple[int, int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    occupied_positions: set[tuple[int, int]] = set()

    if want_overlap:
        pair = _choose_overlap_pair(colors, heights, widths, canvas_h, canvas_w, occupied_positions)
        if pair is None:
            return None
        pair_seeds, pair_boxes = pair
        seeds.extend(pair_seeds)
        boxes.extend(pair_boxes)
        occupied_positions |= {(r, c) for r, c, _color in pair_seeds}

    if want_clip:
        clip_seed = _choose_clip_seed(
            colors,
            heights,
            widths,
            canvas_h,
            canvas_w,
            occupied_positions,
            boxes,
        )
        if clip_seed is None:
            return None
        seed, box = clip_seed
        seeds.append(seed)
        boxes.append(box)
        occupied_positions.add((seed[0], seed[1]))

    min_seed_count = max(SEED_COUNT_RANGE[0], len(seeds))
    max_seed_count = min(SEED_COUNT_RANGE[1], len(seeds) + 2)
    if min_seed_count > max_seed_count:
        return None
    target_seed_count = random.randint(min_seed_count, max_seed_count)
    while len(seeds) < target_seed_count:
        if not _add_safe_seed(seeds, boxes, colors, heights, widths, canvas_h, canvas_w):
            return None

    grid = _build_grid(canvas_h, canvas_w, heights, widths, seeds)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if want_clip != any(_seed_clips(seed, heights, widths, canvas_h, canvas_w) for seed in seeds):
        return None
    if want_overlap != _has_overlap(seeds, heights, widths, canvas_h, canvas_w):
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out[1:, 1:]) > int(MAX_OUTPUT_FILL_RATIO * canvas_h * canvas_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bordered seed-growth puzzle")
