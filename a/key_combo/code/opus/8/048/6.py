from __future__ import annotations

import random

import numpy as np


MIN_SIDE = 4
NOISE_COLORS = (1, 2, 3, 7, 9)
PALETTE_COLORS = (4, 5, 6, 8)

GRID_HEIGHT_RANGE = (20, 27)
GRID_WIDTH_RANGE = (20, 28)
TARGET_COUNT_CHOICES = (3, 4)
TARGET_HEIGHT_RANGE = (4, 8)
TARGET_WIDTH_RANGE = (4, 9)
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_LONG_RANGE = (4, 9)
DISTRACTOR_THIN_RANGE = (2, 4)
TOP_MARGIN = 1
BOTTOM_MARGIN = 1
LEFT_OBJECT_MARGIN = 3
RIGHT_MARGIN = 1
CLEARANCE = 1
UNIFORM_WINDOW = 4
NOISE_REPAIR_PASSES = 80
MAX_GENERATION_ATTEMPTS = 500

Rect = tuple[int, int, int, int, int]


def _right_runs(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    runs = np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if int(grid[r, c]) == int(grid[r, c + 1]):
                runs[r, c] = runs[r, c + 1] + 1
    return runs


def _maximal_uniform_rectangles(grid: np.ndarray) -> list[Rect]:
    h, w = grid.shape
    runs = _right_runs(grid)
    rects: set[Rect] = set()

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            max_w = w - c0
            for r1 in range(r0, h):
                if int(grid[r1, c0]) != color:
                    break
                max_w = min(max_w, int(runs[r1, c0]))
                rect_h = r1 - r0 + 1
                if rect_h >= 1 and max_w >= 1:
                    rects.add((r0, c0, r1, c0 + max_w - 1, color))

    maximal: list[Rect] = []
    for rect in rects:
        r0, c0, r1, c1, color = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        contained = False
        for other in rects:
            if other == rect:
                continue
            or0, oc0, or1, oc1, ocolor = other
            if ocolor != color:
                continue
            other_area = (or1 - or0 + 1) * (oc1 - oc0 + 1)
            if other_area > area and or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1:
                contained = True
                break
        if not contained:
            maximal.append(rect)

    return sorted(maximal)


def _rankable_rectangles(grid: np.ndarray) -> list[Rect]:
    rects = []
    for rect in _maximal_uniform_rectangles(grid):
        r0, c0, r1, c1, _color = rect
        if r1 - r0 + 1 >= MIN_SIDE and c1 - c0 + 1 >= MIN_SIDE:
            rects.append(rect)
    return sorted(
        rects,
        key=lambda rect: (
            -((rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)),
            rect[0],
            rect[1],
        ),
    )


def _palette_from_column(grid: np.ndarray) -> list[int]:
    palette = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != 0:
            palette.append(color)
    return palette


def _paint_rectangles(grid: np.ndarray, rects: list[Rect], palette: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for (r0, c0, r1, c1, _color), new_color in zip(rects, palette):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(new_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor hidden large solid rectangles with palette colors according to descending area rank.

    Input structure:
    - A noisy multicolor grid contains 3-4 large solid monochrome axis-aligned rectangles.
    - Column 0 is a palette column: non-black cells in that column, read top to bottom,
      provide the ordered replacement colors, with black separator cells between them.
    - Smaller same-color rectangles and ordinary noise may also appear, but only the large
      maximal rectangles are targets.

    Rule:
    - Detect every maximal same-color solid rectangle in the grid.
    - Keep only those whose height and width are both at least 4.
    - Sort the kept rectangles by area from largest to smallest, breaking ties by top row
      then left column.
    - Read the palette colors from column 0 top to bottom, skipping black cells, and recolor
      the ranked rectangles in that same order.
    - Leave all non-target cells unchanged.

    Key insight:
    - The palette does not match rectangles by position or original color; it indexes them by
      the global ranking of detected rectangle areas, so detection, filtering, and ranking are
      all necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _rankable_rectangles(g)
    palette = _palette_from_column(g)
    assert rects, "expected at least one large target rectangle"
    assert len(rects) == len(palette), "palette length must match the number of target rectangles"
    return _paint_rectangles(g, rects, palette)


def _area(rect: Rect) -> int:
    return (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)


def _padded_slices(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    height: int,
    width: int,
    padding: int = CLEARANCE,
) -> tuple[slice, slice]:
    return (
        slice(max(0, r0 - padding), min(height, r1 + padding + 1)),
        slice(max(0, c0 - padding), min(width, c1 + padding + 1)),
    )


def _sample_target_specs(count: int) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    areas: set[int] = set()
    for _ in range(200):
        specs.clear()
        areas.clear()
        colors = random.sample(NOISE_COLORS, count)
        for color in colors:
            h = random.randint(*TARGET_HEIGHT_RANGE)
            w = random.randint(*TARGET_WIDTH_RANGE)
            specs.append((h, w, color))
            areas.add(h * w)
        if len(areas) != count:
            continue
        return specs[:]
    raise RuntimeError("failed to sample distinct target areas")


def _sample_distractor_specs(count: int) -> list[tuple[int, int, int]]:
    specs = []
    for _ in range(count):
        if random.choice((False, True)):
            h = random.randint(*DISTRACTOR_LONG_RANGE)
            w = random.randint(*DISTRACTOR_THIN_RANGE)
        else:
            h = random.randint(*DISTRACTOR_THIN_RANGE)
            w = random.randint(*DISTRACTOR_LONG_RANGE)
        specs.append((h, w, random.choice(NOISE_COLORS)))
    return specs


def _place_specs(
    height: int,
    width: int,
    target_specs: list[tuple[int, int, int]],
    distractor_specs: list[tuple[int, int, int]],
) -> tuple[list[Rect], list[Rect]] | None:
    specs = [("target", *spec) for spec in target_specs] + [("distractor", *spec) for spec in distractor_specs]
    random.shuffle(specs)
    specs.sort(key=lambda item: item[1] * item[2], reverse=True)

    padded_occupied = np.zeros((height, width), dtype=bool)
    targets: list[Rect] = []
    distractors: list[Rect] = []

    for kind, h, w, color in specs:
        positions = []
        max_r0 = height - BOTTOM_MARGIN - h
        max_c0 = width - RIGHT_MARGIN - w
        for r0 in range(TOP_MARGIN, max_r0 + 1):
            for c0 in range(LEFT_OBJECT_MARGIN, max_c0 + 1):
                r1 = r0 + h - 1
                c1 = c0 + w - 1
                rows, cols = _padded_slices(r0, c0, r1, c1, height, width)
                if not np.any(padded_occupied[rows, cols]):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        rows, cols = _padded_slices(r0, c0, r1, c1, height, width)
        padded_occupied[rows, cols] = True
        rect = (r0, c0, r1, c1, color)
        if kind == "target":
            targets.append(rect)
        else:
            distractors.append(rect)

    return targets, distractors


def _build_forbidden_colors(height: int, width: int, rects: list[Rect]) -> list[list[set[int]]]:
    forbidden = [[set() for _ in range(width)] for _ in range(height)]
    object_mask = np.zeros((height, width), dtype=bool)
    for r0, c0, r1, c1, _color in rects:
        object_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for r0, c0, r1, c1, color in rects:
        rows, cols = _padded_slices(r0, c0, r1, c1, height, width)
        for r in range(rows.start, rows.stop):
            for c in range(cols.start, cols.stop):
                if not object_mask[r, c]:
                    forbidden[r][c].add(color)
    return forbidden


def _noise_choices(forbidden: list[list[set[int]]], r: int, c: int) -> list[int]:
    return [color for color in NOISE_COLORS if color not in forbidden[r][c]]


def _uniform_noise_blocks(grid: np.ndarray, large_target_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    blocks = []
    for r0 in range(h - UNIFORM_WINDOW + 1):
        for c0 in range(w - UNIFORM_WINDOW + 1):
            if np.any(large_target_mask[r0 : r0 + UNIFORM_WINDOW, c0 : c0 + UNIFORM_WINDOW]):
                continue
            patch = grid[r0 : r0 + UNIFORM_WINDOW, c0 : c0 + UNIFORM_WINDOW]
            if np.all(patch == int(patch[0, 0])):
                blocks.append((r0, c0))
    return blocks


def _fill_noise(
    grid: np.ndarray,
    object_mask: np.ndarray,
    large_target_mask: np.ndarray,
    forbidden: list[list[set[int]]],
) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if object_mask[r, c] or c == 0:
                continue
            choices = _noise_choices(forbidden, r, c)
            if not choices:
                return False
            grid[r, c] = random.choice(choices)

    for _ in range(NOISE_REPAIR_PASSES):
        blocks = _uniform_noise_blocks(grid, large_target_mask)
        if not blocks:
            return True
        for r0, c0 in blocks:
            candidates = []
            uniform_color = int(grid[r0, c0])
            for r in range(r0, r0 + UNIFORM_WINDOW):
                for c in range(c0, c0 + UNIFORM_WINDOW):
                    if object_mask[r, c] or c == 0:
                        continue
                    choices = [color for color in _noise_choices(forbidden, r, c) if color != uniform_color]
                    if choices:
                        candidates.append((r, c, choices))
            if not candidates:
                return False
            r, c, choices = random.choice(candidates)
            grid[r, c] = random.choice(choices)

    return not _uniform_noise_blocks(grid, large_target_mask)


def _place_palette(grid: np.ndarray, palette: list[int]) -> bool:
    height = grid.shape[0]
    required = 2 * len(palette) - 1
    if required > height:
        return False
    start = random.randint(0, height - required)
    grid[:, 0] = 0
    for idx, color in enumerate(palette):
        grid[start + 2 * idx, 0] = int(color)
    return True


def _render_input(
    height: int,
    width: int,
    targets: list[Rect],
    distractors: list[Rect],
    palette: list[int],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    all_rects = targets + distractors
    object_mask = np.zeros((height, width), dtype=bool)
    large_target_mask = np.zeros((height, width), dtype=bool)

    for r0, c0, r1, c1, color in all_rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        object_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for r0, c0, r1, c1, _color in targets:
        large_target_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if not _place_palette(grid, palette):
        return None

    forbidden = _build_forbidden_colors(height, width, all_rects)
    if not _fill_noise(grid, object_mask, large_target_mask, forbidden):
        return None
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    palette = _palette_from_column(grid)
    rects = _rankable_rectangles(grid)
    return _paint_rectangles(grid, rects, palette)


def _sorted_by_row(rects: list[Rect]) -> list[Rect]:
    return sorted(rects, key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))


def _sorted_by_col(rects: list[Rect]) -> list[Rect]:
    return sorted(rects, key=lambda rect: (rect[1], rect[0], rect[2], rect[3]))


def _sorted_by_area_ascending(rects: list[Rect]) -> list[Rect]:
    return sorted(rects, key=lambda rect: (_area(rect), rect[0], rect[1]))


def _rect_id_set(rects: list[Rect]) -> set[tuple[int, int, int, int, int]]:
    return {tuple(map(int, rect)) for rect in rects}


def _validate_candidate(grid: np.ndarray, targets: list[Rect], distractors: list[Rect]) -> bool:
    detected_all = _maximal_uniform_rectangles(grid)
    detected_rankable = _rankable_rectangles(grid)
    palette = _palette_from_column(grid)
    expected = _expected_output(grid)

    target_set = _rect_id_set(targets)
    distractor_set = _rect_id_set(distractors)
    detected_all_set = _rect_id_set(detected_all)
    detected_rankable_set = _rect_id_set(detected_rankable)

    if len(palette) != len(targets):
        return False
    if detected_rankable_set != target_set:
        return False
    if not distractor_set:
        return False
    if not (distractor_set & detected_all_set):
        return False
    if distractor_set & detected_rankable_set:
        return False
    if len({_area(rect) for rect in detected_rankable}) != len(detected_rankable):
        return False

    true_order = detected_rankable
    row_order = _sorted_by_row(detected_rankable)
    col_order = _sorted_by_col(detected_rankable)
    asc_order = _sorted_by_area_ascending(detected_rankable)
    if true_order == row_order:
        return False
    if true_order == col_order:
        return False
    if true_order == asc_order:
        return False

    if np.array_equal(_paint_rectangles(grid, row_order, palette), expected):
        return False
    if np.array_equal(_paint_rectangles(grid, col_order, palette), expected):
        return False
    if np.array_equal(_paint_rectangles(grid, asc_order, palette), expected):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    target_count = random.choice(TARGET_COUNT_CHOICES)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)

    target_specs = _sample_target_specs(target_count)
    distractor_specs = _sample_distractor_specs(distractor_count)
    placed = _place_specs(height, width, target_specs, distractor_specs)
    if placed is None:
        return None
    targets, distractors = placed

    palette = random.sample(PALETTE_COLORS, target_count)
    grid = _render_input(height, width, targets, distractors, palette)
    if grid is None:
        return None

    if not _validate_candidate(grid, targets, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 048/6")
