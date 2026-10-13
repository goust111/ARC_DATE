import random
from itertools import combinations

import numpy as np


BG_COLOR = 8
FENCE_COLOR = 5
COLOR_POOL = tuple(color for color in range(10) if color not in {BG_COLOR, FENCE_COLOR})

MIN_REGIONS = 2
MAX_REGIONS = 4
MIN_REGION_SIZE = 3
MAX_REGION_SIZE = 7
GENERATION_RETRIES = 500
PAIR_COUNT_RANGE = (2, 4)
EXTRA_PAIR_MAX = 2


def _most_common_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


def _most_common_non_bg_color(grid: np.ndarray, bg: int) -> int:
    other = np.asarray(grid, dtype=int)[np.asarray(grid, dtype=int) != bg]
    assert other.size > 0, "expected at least one non-background color"
    return int(np.bincount(other).argmax())


def _color_points(grid: np.ndarray, bg: int, fence: int) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int, int]]]:
    colors = sorted(set(map(int, np.asarray(grid).ravel())) - {bg, fence})
    pairs: dict[int, list[tuple[int, int]]] = {}
    points: list[tuple[int, int, int]] = []
    for color in colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == color)]
        assert len(coords) == 2, "each non-structural color must appear exactly twice"
        pairs[int(color)] = coords
        for r, c in coords:
            points.append((r, c, int(color)))
    return pairs, points


def _crossed_fence_pixels(
    grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], fence: int
) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    pixels: list[tuple[int, int]] = []
    if r1 == r2:
        row = int(r1)
        for col in range(min(c1, c2) + 1, max(c1, c2)):
            if int(grid[row, col]) == fence:
                pixels.append((row, col))
    elif c1 == c2:
        col = int(c1)
        for row in range(min(r1, r2) + 1, max(r1, r2)):
            if int(grid[row, col]) == fence:
                pixels.append((row, col))
    else:
        raise AssertionError("paired points must share a row or a column")
    return pixels


def _analyze_operations(grid: np.ndarray, bg: int | None = None, fence: int | None = None) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g) if bg is None else int(bg)
    fence = _most_common_non_bg_color(g, bg) if fence is None else int(fence)
    pairs, points = _color_points(g, bg, fence)

    open_pixels: set[tuple[int, int]] = set()
    reinforce_row: set[tuple[int, int]] = set()
    reinforce_col: set[tuple[int, int]] = set()

    for coords in pairs.values():
        a, b = coords
        for pixel in _crossed_fence_pixels(g, a, b, fence):
            open_pixels.add(pixel)

    for (r1, c1, color1), (r2, c2, color2) in combinations(points, 2):
        if color1 == color2:
            continue
        if r1 == r2:
            for pixel in _crossed_fence_pixels(g, (r1, c1), (r2, c2), fence):
                reinforce_row.add(pixel)
        elif c1 == c2:
            for pixel in _crossed_fence_pixels(g, (r1, c1), (r2, c2), fence):
                reinforce_col.add(pixel)

    reinforce_base = reinforce_row | reinforce_col
    return {
        "bg": bg,
        "fence": fence,
        "open_pixels": open_pixels,
        "reinforce_row": reinforce_row,
        "reinforce_col": reinforce_col,
        "reinforce_base": reinforce_base,
        "color_positions": {(r, c) for r, c, _color in points},
        "open_count": len(open_pixels),
        "reinforce_count": len(reinforce_base),
        "conflict_count": len(open_pixels & reinforce_base),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same-color aligned point pairs open crossed grey fences; different-color aligned
    point pairs reinforce crossed grey fences; reinforcement overrides opening.
    All colored points are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_operations(g)
    bg = int(info["bg"])
    fence = int(info["fence"])

    out = g.copy()

    for r, c in info["open_pixels"] - info["reinforce_base"]:
        out[r, c] = bg

    for r, c in info["reinforce_row"]:
        for cc in range(c - 1, c + 2):
            if 0 <= cc < out.shape[1]:
                out[r, cc] = fence

    for r, c in info["reinforce_col"]:
        for rr in range(r - 1, r + 2):
            if 0 <= rr < out.shape[0]:
                out[rr, c] = fence

    for r, c in info["color_positions"]:
        out[r, c] = bg

    return out


def _build_spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    fences: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        start = cursor
        end = start + size - 1
        spans.append((start, end))
        cursor = end + 1
        if i < len(sizes) - 1:
            fences.append(cursor)
            cursor += 1
    return spans, fences, cursor


def _sample_layout() -> dict[str, object]:
    row_regions = random.randint(MIN_REGIONS, MAX_REGIONS)
    col_regions = random.randint(MIN_REGIONS, MAX_REGIONS)
    row_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(row_regions)]
    col_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(col_regions)]
    row_spans, fence_rows, height = _build_spans(row_sizes)
    col_spans, fence_cols, width = _build_spans(col_sizes)
    safe_rows = [list(range(r0 + 1, r1)) for r0, r1 in row_spans]
    safe_cols = [list(range(c0 + 1, c1)) for c0, c1 in col_spans]
    return {
        "height": height,
        "width": width,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "fence_rows": fence_rows,
        "fence_cols": fence_cols,
        "safe_rows": safe_rows,
        "safe_cols": safe_cols,
    }


def _blank_partitioned_grid(layout: dict[str, object]) -> np.ndarray:
    grid = np.full((layout["height"], layout["width"]), BG_COLOR, dtype=int)
    for row in layout["fence_rows"]:
        grid[row, :] = FENCE_COLOR
    for col in layout["fence_cols"]:
        grid[:, col] = FENCE_COLOR
    return grid


def _collect_used(placements: list[tuple[tuple[int, int], tuple[int, int]]]) -> tuple[set[tuple[int, int]], set[int], set[int]]:
    occupied: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for a, b in placements:
        for r, c in (a, b):
            occupied.add((r, c))
            used_rows.add(int(r))
            used_cols.add(int(c))
    return occupied, used_rows, used_cols


def _open_only_candidates(
    layout: dict[str, object],
    occupied: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    safe_rows: list[list[int]] = layout["safe_rows"]
    safe_cols: list[list[int]] = layout["safe_cols"]

    for row_region, rows in enumerate(safe_rows):
        for r in rows:
            if r in used_rows:
                continue
            for c1_region in range(len(safe_cols)):
                for c2_region in range(c1_region + 1, len(safe_cols)):
                    for c1 in safe_cols[c1_region]:
                        for c2 in safe_cols[c2_region]:
                            if c1 in used_cols or c2 in used_cols:
                                continue
                            pair = ((int(r), int(c1)), (int(r), int(c2)))
                            if pair[0] in occupied or pair[1] in occupied:
                                continue
                            candidates.append(pair)

    for col_region, cols in enumerate(safe_cols):
        for c in cols:
            if c in used_cols:
                continue
            for r1_region in range(len(safe_rows)):
                for r2_region in range(r1_region + 1, len(safe_rows)):
                    for r1 in safe_rows[r1_region]:
                        for r2 in safe_rows[r2_region]:
                            if r1 in used_rows or r2 in used_rows:
                                continue
                            pair = ((int(r1), int(c)), (int(r2), int(c)))
                            if pair[0] in occupied or pair[1] in occupied:
                                continue
                            candidates.append(pair)

    random.shuffle(candidates)
    return candidates


def _place_open_only_pairs(
    layout: dict[str, object],
    placements: list[tuple[tuple[int, int], tuple[int, int]]],
    target_count: int,
) -> bool:
    while len(placements) < target_count:
        occupied, used_rows, used_cols = _collect_used(placements)
        candidates = _open_only_candidates(layout, occupied, used_rows, used_cols)
        if not candidates:
            return False
        placements.append(candidates[0])
    return True


def _place_reinforce_no_conflict_core(
    layout: dict[str, object], orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    safe_rows: list[list[int]] = layout["safe_rows"]
    safe_cols: list[list[int]] = layout["safe_cols"]

    if orientation == "V":
        if len(safe_rows) < 2 or len(safe_cols) < 2:
            return None
        row_options = list(combinations(range(len(safe_rows)), 2))
        col_options = list(combinations(range(len(safe_cols)), 2))
        random.shuffle(row_options)
        random.shuffle(col_options)
        for r_top, r_bottom in row_options:
            for c_left, c_right in col_options:
                shared_on_top = random.choice([True, False])
                shared_rows = safe_rows[r_top] if shared_on_top else safe_rows[r_bottom]
                other_rows = safe_rows[r_bottom] if shared_on_top else safe_rows[r_top]
                if not shared_rows or not other_rows or not safe_cols[c_left] or not safe_cols[c_right]:
                    continue
                shared_r = random.choice(shared_rows)
                other_r1 = random.choice(other_rows)
                other_r2 = random.choice(other_rows)
                c1 = random.choice(safe_cols[c_left])
                c2 = random.choice(safe_cols[c_right])
                if c1 == c2:
                    continue
                if shared_on_top:
                    return [((shared_r, c1), (other_r1, c1)), ((shared_r, c2), (other_r2, c2))]
                return [((other_r1, c1), (shared_r, c1)), ((other_r2, c2), (shared_r, c2))]
        return None

    if len(safe_rows) < 2 or len(safe_cols) < 2:
        return None
    row_options = list(combinations(range(len(safe_rows)), 2))
    col_options = list(combinations(range(len(safe_cols)), 2))
    random.shuffle(row_options)
    random.shuffle(col_options)
    for r_top, r_bottom in row_options:
        for c_left, c_right in col_options:
            shared_on_left = random.choice([True, False])
            shared_cols = safe_cols[c_left] if shared_on_left else safe_cols[c_right]
            other_cols = safe_cols[c_right] if shared_on_left else safe_cols[c_left]
            if not shared_cols or not other_cols or not safe_rows[r_top] or not safe_rows[r_bottom]:
                continue
            shared_c = random.choice(shared_cols)
            other_c1 = random.choice(other_cols)
            other_c2 = random.choice(other_cols)
            r1 = random.choice(safe_rows[r_top])
            r2 = random.choice(safe_rows[r_bottom])
            if r1 == r2:
                continue
            if shared_on_left:
                return [((r1, shared_c), (r1, other_c1)), ((r2, shared_c), (r2, other_c2))]
            return [((r1, other_c1), (r1, shared_c)), ((r2, other_c2), (r2, shared_c))]
    return None


def _place_conflict_core(
    layout: dict[str, object], orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    safe_rows: list[list[int]] = layout["safe_rows"]
    safe_cols: list[list[int]] = layout["safe_cols"]

    if orientation == "H":
        if len(safe_cols) < 3:
            return None
        middle_cols = [idx for idx, cols in enumerate(safe_cols) if len(cols) >= 2 and 0 < idx < len(safe_cols) - 1]
        random.shuffle(middle_cols)
        row_regions = [idx for idx, rows in enumerate(safe_rows) if rows]
        random.shuffle(row_regions)
        for middle in middle_cols:
            left_choices = [idx for idx in range(middle) if safe_cols[idx]]
            right_choices = [idx for idx in range(middle + 1, len(safe_cols)) if safe_cols[idx]]
            if not left_choices or not right_choices:
                continue
            for row_region in row_regions:
                row = random.choice(safe_rows[row_region])
                left = random.choice(left_choices)
                right = random.choice(right_choices)
                mid_a, mid_b = random.sample(safe_cols[middle], 2)
                left_c = random.choice(safe_cols[left])
                right_c = random.choice(safe_cols[right])
                return [((row, left_c), (row, mid_a)), ((row, mid_b), (row, right_c))]
        return None

    if len(safe_rows) < 3:
        return None
    middle_rows = [idx for idx, rows in enumerate(safe_rows) if len(rows) >= 2 and 0 < idx < len(safe_rows) - 1]
    random.shuffle(middle_rows)
    col_regions = [idx for idx, cols in enumerate(safe_cols) if cols]
    random.shuffle(col_regions)
    for middle in middle_rows:
        above_choices = [idx for idx in range(middle) if safe_rows[idx]]
        below_choices = [idx for idx in range(middle + 1, len(safe_rows)) if safe_rows[idx]]
        if not above_choices or not below_choices:
            continue
        for col_region in col_regions:
            col = random.choice(safe_cols[col_region])
            above = random.choice(above_choices)
            below = random.choice(below_choices)
            mid_a, mid_b = random.sample(safe_rows[middle], 2)
            top_r = random.choice(safe_rows[above])
            bottom_r = random.choice(safe_rows[below])
            return [((top_r, col), (mid_a, col)), ((mid_b, col), (bottom_r, col))]
    return None


def _build_grid_from_pairs(
    layout: dict[str, object], placements: list[tuple[tuple[int, int], tuple[int, int]]]
) -> np.ndarray:
    grid = _blank_partitioned_grid(layout)
    colors = random.sample(COLOR_POOL, len(placements))
    for color, (a, b) in zip(colors, placements):
        grid[a] = color
        grid[b] = color
    return grid


def _valid_mode(grid: np.ndarray, mode: str) -> bool:
    info = _analyze_operations(grid, BG_COLOR, FENCE_COLOR)
    if info["open_count"] == 0:
        return False
    if mode == "open_only":
        return info["reinforce_count"] == 0
    if mode == "reinforce_no_conflict":
        return info["reinforce_count"] > 0 and info["conflict_count"] == 0
    if mode == "conflict":
        return info["conflict_count"] > 0
    raise AssertionError(f"unknown mode: {mode}")


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["open_only", "reinforce_no_conflict", "conflict"])
    orientation = random.choice(["H", "V"])
    layout = _sample_layout()
    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if mode == "open_only":
        target_pairs = random.randint(*PAIR_COUNT_RANGE)
        if not _place_open_only_pairs(layout, placements, target_pairs):
            return None
    elif mode == "reinforce_no_conflict":
        core = _place_reinforce_no_conflict_core(layout, orientation)
        if core is None:
            return None
        placements.extend(core)
        extra_pairs = random.randint(0, EXTRA_PAIR_MAX)
        _place_open_only_pairs(layout, placements, len(placements) + extra_pairs)
    else:
        core = _place_conflict_core(layout, orientation)
        if core is None:
            return None
        placements.extend(core)
        extra_pairs = random.randint(0, EXTRA_PAIR_MAX)
        _place_open_only_pairs(layout, placements, len(placements) + extra_pairs)

    if len(placements) > len(COLOR_POOL):
        return None

    grid = _build_grid_from_pairs(layout, placements)
    if not _valid_mode(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnet-fence puzzle input")
