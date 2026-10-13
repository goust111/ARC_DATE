from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TEAL = 8

SOURCE_DIMS = ((2, 3), (3, 2), (2, 2))
SOURCE_WEIGHTS = (4, 4, 1)
GRID_SIZE_RANGE = (14, 18)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _segments(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(int(i) for i in indices)
    spans: list[tuple[int, int]] = []
    start = end = ordered[0]
    for idx in ordered[1:]:
        if idx == end + 1:
            end = idx
        else:
            spans.append((start, end))
            start = end = idx
    spans.append((start, end))
    return spans


def _solid_source_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == TEAL
    box = _bbox(mask)
    assert box is not None, "expected one teal source rectangle"
    r0, c0, r1, c1 = box
    assert bool(mask[r0 : r1 + 1, c0 : c1 + 1].all()), "teal source must be solid"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "teal source must be rectangular"
    return box


def _paint_ray(base: np.ndarray, out: np.ndarray, r: int, c: int, dr: int, dc: int) -> None:
    height, width = map(int, base.shape)
    while 0 <= r < height and 0 <= c < width:
        value = int(base[r, c])
        if value == BLUE:
            break
        if value == BG:
            out[r, c] = YELLOW
        r += dr
        c += dc


def _cast_rays(grid: np.ndarray) -> np.ndarray:
    base = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _solid_source_bbox(base)
    out = base.copy()
    for r in range(r0, r1 + 1):
        _paint_ray(base, out, r, c0 - 1, 0, -1)
        _paint_ray(base, out, r, c1 + 1, 0, 1)
    for c in range(c0, c1 + 1):
        _paint_ray(base, out, r0 - 1, c, -1, 0)
        _paint_ray(base, out, r1 + 1, c, 1, 0)
    return out


def _marker_positions(values: np.ndarray) -> list[int]:
    return [
        i
        for i, value in enumerate(map(int, values))
        if value not in (BG, BLUE, YELLOW, TEAL)
    ]


def _line_moves(values: np.ndarray) -> list[tuple[int, int, int]]:
    vals = [int(v) for v in values]
    yellow_spans = _segments([i for i, v in enumerate(vals) if v == YELLOW])
    if len(yellow_spans) != 1:
        return []

    start, end = yellow_spans[0]
    if start == 0 or end == len(vals) - 1:
        return []

    left = [i for i in range(start) if vals[i] not in (BG, BLUE, YELLOW, TEAL)]
    right = [i for i in range(end + 1, len(vals)) if vals[i] not in (BG, BLUE, YELLOW, TEAL)]
    if len(left) != 1 or len(right) != 1:
        return []

    left_src, right_src = left[0], right[0]
    left_dst, right_dst = start - 1, end + 1
    if vals[left_dst] != BG and left_dst != left_src:
        return []
    if vals[right_dst] != BG and right_dst != right_src:
        return []
    return [
        (left_src, right_dst, vals[left_src]),
        (right_src, left_dst, vals[right_src]),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from the teal rectangle to build yellow barriers, then swap marker pixels across those computed barriers.

    Input structure:
    - A black-background grid contains one solid teal rectangle near the center and several blue single-pixel blockers.
    - Red and green single-pixel markers lie on rows or columns that will be separated by the yellow ray barriers.
    - Yellow is absent from the input; it is created only by ray casting.

    Rule:
    - From every row of the teal rectangle cast left/right rays, and from every column cast up/down rays.
    - A ray paints only black cells yellow and stops before a blue blocker; non-black cells are not recolored.
    - After the yellow barriers are formed, scan every row and column with exactly one contiguous yellow segment.
    - If such a line has exactly one marker on each side of that segment, erase both sources and place each marker
    - in the cell immediately beyond the opposite end of the yellow segment; lines with markers on only one side do nothing.

    Key insight:
    - The swap barrier is not present in the input: it must first be computed by ray casting from the teal rectangle, and
    - only then can the paired markers be identified and moved across the resulting yellow wall.
    """
    rayed = _cast_rays(np.asarray(grid, dtype=int))
    out = rayed.copy()
    clears: list[tuple[int, int]] = []
    paints: list[tuple[int, int, int]] = []
    height, width = map(int, rayed.shape)

    for r in range(height):
        for src, dst, color in _line_moves(rayed[r, :]):
            clears.append((int(r), int(src)))
            paints.append((int(r), int(dst), int(color)))

    for c in range(width):
        for src, dst, color in _line_moves(rayed[:, c]):
            clears.append((int(src), int(c)))
            paints.append((int(dst), int(c), int(color)))

    assert len(clears) == len(set(clears)), "a marker cannot participate in two swaps"
    destinations = [(r, c) for r, c, _ in paints]
    assert len(destinations) == len(set(destinations)), "swap destinations must be unique"

    for r, c in clears:
        out[r, c] = BG
    for r, c, color in paints:
        out[r, c] = int(color)
    return out


def _single_yellow_span(values: np.ndarray) -> tuple[int, int] | None:
    spans = _segments([i for i, v in enumerate(map(int, values)) if v == YELLOW])
    if len(spans) != 1:
        return None
    return spans[0]


def _choose_source_dims() -> tuple[int, int]:
    return random.choices(SOURCE_DIMS, weights=SOURCE_WEIGHTS, k=1)[0]


def _centered_start(length: int, size: int) -> int:
    lo = 3
    hi = length - size - 4
    if hi < lo:
        raise ValueError("grid too small for source margins")
    center = (length - size) // 2
    near_lo = max(lo, center - 1)
    near_hi = min(hi, center + 1)
    return random.randint(near_lo, near_hi)


def _place_source(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = _choose_source_dims()
    r0 = _centered_start(grid.shape[0], h)
    c0 = _centered_start(grid.shape[1], w)
    grid[r0 : r0 + h, c0 : c0 + w] = TEAL
    return r0, c0, r0 + h - 1, c0 + w - 1


def _sample_distances(maxima: list[int]) -> list[int]:
    distances = [random.randint(1, int(m)) for m in maxima]
    if all(dist == 1 for dist in distances):
        choices = [i for i, m in enumerate(maxima) if m >= 2]
        if not choices:
            raise ValueError("need room for a nonzero ray")
        idx = random.choice(choices)
        distances[idx] = random.randint(2, int(maxima[idx]))
    return distances


def _place_blockers(grid: np.ndarray, source: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = source
    row_sides = [random.choice((-1, 1)) for _ in range(r0, r1 + 1)]
    row_max = [c0 if side == -1 else grid.shape[1] - 1 - c1 for side in row_sides]
    row_dist = _sample_distances(row_max)

    col_sides = [random.choice((-1, 1)) for _ in range(c0, c1 + 1)]
    col_max = [r0 if side == -1 else grid.shape[0] - 1 - r1 for side in col_sides]
    col_dist = _sample_distances(col_max)

    blockers: list[tuple[int, int]] = []
    for row, side, dist in zip(range(r0, r1 + 1), row_sides, row_dist):
        col = c0 - dist if side == -1 else c1 + dist
        blockers.append((int(row), int(col)))
    for col, side, dist in zip(range(c0, c1 + 1), col_sides, col_dist):
        row = r0 - dist if side == -1 else r1 + dist
        blockers.append((int(row), int(col)))

    assert len(blockers) == len(set(blockers)), "blockers must be distinct"
    for r, c in blockers:
        grid[r, c] = BLUE
    return blockers


def _candidate_lines(base: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    cols: list[dict] = []
    height, width = map(int, base.shape)

    for r in range(height):
        span = _single_yellow_span(base[r, :])
        if span is None:
            continue
        start, end = span
        rows.append(
            {
                "line": int(r),
                "start": int(start),
                "end": int(end),
                "pair_ok": start >= 2 and end <= width - 3 and int(base[r, start - 1]) == BG and int(base[r, end + 1]) == BG,
            }
        )

    for c in range(width):
        span = _single_yellow_span(base[:, c])
        if span is None:
            continue
        start, end = span
        cols.append(
            {
                "line": int(c),
                "start": int(start),
                "end": int(end),
                "pair_ok": start >= 2 and end <= height - 3 and int(base[start - 1, c]) == BG and int(base[end + 1, c]) == BG,
            }
        )

    return rows, cols


def _choose_pair_mode(has_rows: bool, has_cols: bool) -> tuple[int, int]:
    modes: list[str] = []
    if has_rows:
        modes.append("row_only")
    if has_cols:
        modes.append("col_only")
    if has_rows and has_cols:
        modes.extend(["both", "both"])

    mode = random.choice(modes)
    if mode == "row_only":
        return 1, 0
    if mode == "col_only":
        return 0, 1
    return 1, 1


def _pick_pair_counts(row_pairable: list[dict], col_pairable: list[dict]) -> tuple[list[dict], list[dict]]:
    min_rows, min_cols = _choose_pair_mode(bool(row_pairable), bool(col_pairable))
    row_count = min(len(row_pairable), min_rows + random.randint(0, 1))
    col_count = min(len(col_pairable), min_cols + random.randint(0, 1))
    if row_count == 0 and col_count == 0:
        raise ValueError("need at least one pair")
    chosen_rows = random.sample(row_pairable, row_count) if row_count else []
    chosen_cols = random.sample(col_pairable, col_count) if col_count else []
    return chosen_rows, chosen_cols


def _pick_row_source(
    base: np.ndarray,
    line: dict,
    blocked_cols: set[int],
    used_cols: set[int],
    used_cells: set[tuple[int, int]],
    banned_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    r = int(line["line"])
    start = int(line["start"])
    end = int(line["end"])
    left_pool = [
        c
        for c in range(0, start - 1)
        if int(base[r, c]) == BG
        and c not in blocked_cols
        and c not in used_cols
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    right_pool = [
        c
        for c in range(end + 2, base.shape[1])
        if int(base[r, c]) == BG
        and c not in blocked_cols
        and c not in used_cols
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    if not left_pool or not right_pool:
        return None
    return random.choice(left_pool), random.choice(right_pool)


def _pick_col_source(
    base: np.ndarray,
    line: dict,
    blocked_rows: set[int],
    used_rows: set[int],
    used_cells: set[tuple[int, int]],
    banned_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    c = int(line["line"])
    start = int(line["start"])
    end = int(line["end"])
    top_pool = [
        r
        for r in range(0, start - 1)
        if int(base[r, c]) == BG
        and r not in blocked_rows
        and r not in used_rows
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    bottom_pool = [
        r
        for r in range(end + 2, base.shape[0])
        if int(base[r, c]) == BG
        and r not in blocked_rows
        and r not in used_rows
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    if not top_pool or not bottom_pool:
        return None
    return random.choice(top_pool), random.choice(bottom_pool)


def _place_row_pairs(
    grid: np.ndarray,
    base: np.ndarray,
    lines: list[dict],
    blocked_cols: set[int],
    banned_cells: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[int]] | None:
    destinations: list[tuple[int, int]] = []
    used_cols: set[int] = set()
    for line in lines:
        picked = _pick_row_source(base, line, blocked_cols, used_cols, used_cells, banned_cells)
        if picked is None:
            return None
        left_c, right_c = picked
        r = int(line["line"])
        left_color, right_color = (RED, GREEN) if random.random() < 0.5 else (GREEN, RED)
        grid[r, left_c] = left_color
        grid[r, right_c] = right_color
        used_cells.update({(r, left_c), (r, right_c)})
        used_cols.update({left_c, right_c})
        destinations.extend([(r, int(line["start"]) - 1), (r, int(line["end"]) + 1)])
    return destinations, used_cols


def _place_col_pairs(
    grid: np.ndarray,
    base: np.ndarray,
    lines: list[dict],
    blocked_rows: set[int],
    banned_cells: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[int]] | None:
    destinations: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    for line in lines:
        picked = _pick_col_source(base, line, blocked_rows, used_rows, used_cells, banned_cells)
        if picked is None:
            return None
        top_r, bottom_r = picked
        c = int(line["line"])
        top_color, bottom_color = (RED, GREEN) if random.random() < 0.5 else (GREEN, RED)
        grid[top_r, c] = top_color
        grid[bottom_r, c] = bottom_color
        used_cells.update({(top_r, c), (bottom_r, c)})
        used_rows.update({top_r, bottom_r})
        destinations.extend([(int(line["start"]) - 1, c), (int(line["end"]) + 1, c)])
    return destinations, used_rows


def _place_row_unpaired(
    grid: np.ndarray,
    base: np.ndarray,
    lines: list[dict],
    blocked_cols: set[int],
    used_cols: set[int],
    blocked_rows: set[int],
    used_cells: set[tuple[int, int]],
    banned_cells: set[tuple[int, int]],
) -> None:
    available = [line for line in lines if int(line["line"]) not in blocked_rows]
    if not available or random.random() < 0.5:
        return
    line = random.choice(available)
    r = int(line["line"])
    start = int(line["start"])
    end = int(line["end"])
    pools = []
    left_pool = [
        c
        for c in range(0, start - 1)
        if int(base[r, c]) == BG
        and c not in blocked_cols
        and c not in used_cols
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    right_pool = [
        c
        for c in range(end + 2, base.shape[1])
        if int(base[r, c]) == BG
        and c not in blocked_cols
        and c not in used_cols
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    if left_pool:
        pools.append(left_pool)
    if right_pool:
        pools.append(right_pool)
    if not pools:
        return
    c = random.choice(random.choice(pools))
    grid[r, c] = random.choice((RED, GREEN))
    used_cells.add((r, c))
    used_cols.add(c)


def _place_col_unpaired(
    grid: np.ndarray,
    base: np.ndarray,
    lines: list[dict],
    blocked_rows: set[int],
    used_rows: set[int],
    blocked_cols: set[int],
    used_cells: set[tuple[int, int]],
    banned_cells: set[tuple[int, int]],
) -> None:
    available = [line for line in lines if int(line["line"]) not in blocked_cols]
    if not available or random.random() < 0.5:
        return
    line = random.choice(available)
    c = int(line["line"])
    start = int(line["start"])
    end = int(line["end"])
    pools = []
    top_pool = [
        r
        for r in range(0, start - 1)
        if int(base[r, c]) == BG
        and r not in blocked_rows
        and r not in used_rows
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    bottom_pool = [
        r
        for r in range(end + 2, base.shape[0])
        if int(base[r, c]) == BG
        and r not in blocked_rows
        and r not in used_rows
        and (r, c) not in used_cells
        and (r, c) not in banned_cells
    ]
    if top_pool:
        pools.append(top_pool)
    if bottom_pool:
        pools.append(bottom_pool)
    if not pools:
        return
    r = random.choice(random.choice(pools))
    grid[r, c] = random.choice((RED, GREEN))
    used_cells.add((r, c))
    used_rows.add(r)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    source = _place_source(grid)
    blockers = _place_blockers(grid, source)
    assert 3 <= len(blockers) <= 5

    base = _cast_rays(grid)
    row_candidates, col_candidates = _candidate_lines(base)
    row_pairable = [line for line in row_candidates if bool(line["pair_ok"])]
    col_pairable = [line for line in col_candidates if bool(line["pair_ok"])]
    if not row_pairable and not col_pairable:
        return None

    try:
        chosen_rows, chosen_cols = _pick_pair_counts(row_pairable, col_pairable)
    except ValueError:
        return None

    banned_cells: set[tuple[int, int]] = set()
    for line in chosen_rows:
        banned_cells.add((int(line["line"]), int(line["start"]) - 1))
        banned_cells.add((int(line["line"]), int(line["end"]) + 1))
    for line in chosen_cols:
        banned_cells.add((int(line["start"]) - 1, int(line["line"])))
        banned_cells.add((int(line["end"]) + 1, int(line["line"])))
    if len(banned_cells) != 2 * (len(chosen_rows) + len(chosen_cols)):
        return None
    if any(int(grid[r, c]) != BG for r, c in banned_cells):
        return None

    blocked_cols = {int(line["line"]) for line in chosen_cols}
    blocked_rows = {int(line["line"]) for line in chosen_rows}
    used_cells: set[tuple[int, int]] = set()

    row_result = _place_row_pairs(grid, base, chosen_rows, blocked_cols, banned_cells, used_cells)
    if row_result is None:
        return None
    _row_destinations, row_used_cols = row_result

    col_result = _place_col_pairs(grid, base, chosen_cols, blocked_rows, banned_cells, used_cells)
    if col_result is None:
        return None
    _col_destinations, col_used_rows = col_result

    extra_row_lines = [line for line in row_candidates if line not in chosen_rows]
    extra_col_lines = [line for line in col_candidates if line not in chosen_cols]
    _place_row_unpaired(
        grid,
        base,
        extra_row_lines,
        blocked_cols,
        row_used_cols,
        blocked_rows | col_used_rows,
        used_cells,
        banned_cells,
    )
    _place_col_unpaired(
        grid,
        base,
        extra_col_lines,
        blocked_rows,
        col_used_rows,
        blocked_cols | row_used_cols,
        used_cells,
        banned_cells,
    )

    markers = np.argwhere(np.isin(grid, [RED, GREEN]))
    if markers.shape[0] < 2 * (len(chosen_rows) + len(chosen_cols)):
        return None

    out = main(grid)
    if YELLOW not in set(map(int, np.unique(out))):
        return None

    base_out = _cast_rays(grid)
    if not np.array_equal(out == YELLOW, base_out == YELLOW):
        return None

    r0, c0, r1, c1 = _solid_source_bbox(grid)
    for row in range(r0, r1 + 1):
        left_has = bool(np.any(grid[row, :c0] == BLUE))
        right_has = bool(np.any(grid[row, c1 + 1 :] == BLUE))
        if not (left_has or right_has):
            return None
    for col in range(c0, c1 + 1):
        up_has = bool(np.any(grid[:r0, col] == BLUE))
        down_has = bool(np.any(grid[r1 + 1 :, col] == BLUE))
        if not (up_has or down_has):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dynamic-barrier swap puzzle")
