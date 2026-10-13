import random
from collections import Counter

import numpy as np


BG = 0
ROW_PERIOD = 4
COL_PERIOD = 4
GEN_HEIGHT = 6
GEN_BAR_ROW = 3
GEN_STEM_ROW_OPTIONS = ((1, 2, 3), (0, 2, 3))
GEN_STEM_COL_OPTIONS = (1, 2, 3)
GEN_GROUP_COUNT_OPTIONS = (3, 4)
GEN_GROUP_WIDTH_MIN = 2
GEN_GROUP_WIDTH_MAX = 5
GEN_WIDTH_MIN = 9
GEN_WIDTH_MAX = 16
GEN_RETRY_LIMIT = 800


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    return sum(
        0 <= nr < h and 0 <= nc < w and mask[nr, nc]
        for nr, nc in _neighbors4(r, c)
    )


def _split_column_groups(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    nonempty = np.any(grid != bg, axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < grid.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c < grid.shape[1] and nonempty[c]:
            c += 1
        groups.append(grid[:, c0:c].copy())
    return groups


def _endpoint_row(group: np.ndarray, bg: int, side: str) -> int:
    mask = group != bg
    c = 0 if side == "left" else group.shape[1] - 1
    rows = [
        r
        for r in range(group.shape[0])
        if mask[r, c] and _degree(mask, r, c) == 1
    ]
    assert len(rows) == 1
    return int(rows[0])


def _top_row(group: np.ndarray, bg: int) -> int:
    rows = np.nonzero(group != bg)[0]
    assert len(rows) > 0
    return int(rows.min())


def _shift_group(group: np.ndarray, bg: int, dr: int) -> np.ndarray:
    out = np.full_like(group, bg)
    rows, cols = np.nonzero(group != bg)
    new_rows = rows + dr
    assert new_rows.min() >= 0 and new_rows.max() < group.shape[0]
    out[new_rows, cols] = group[rows, cols]
    return out


def _assemble_groups(
    groups: list[np.ndarray], bg: int, mode: str = "endpoint"
) -> np.ndarray:
    h = groups[0].shape[0]
    w = sum(group.shape[1] for group in groups)
    out = np.full((h, w), bg, dtype=int)

    first = groups[0]
    out[:, : first.shape[1]] = first
    col = first.shape[1]
    prev_endpoint = _endpoint_row(first, bg, "right")
    prev_top = _top_row(first, bg)

    for group in groups[1:]:
        if mode == "endpoint":
            dr = prev_endpoint - _endpoint_row(group, bg, "left")
        elif mode == "top":
            dr = prev_top - _top_row(group, bg)
        elif mode == "none":
            dr = 0
        else:
            raise ValueError(f"unknown assemble mode: {mode}")

        shifted = _shift_group(group, bg, dr) if dr != 0 else group
        out[:, col : col + group.shape[1]] = shifted
        prev_endpoint = _endpoint_row(shifted, bg, "right")
        prev_top = _top_row(shifted, bg)
        col += group.shape[1]
    return out


def _period_candidates(
    grid: np.ndarray, bg: int
) -> list[tuple[int, int, dict[tuple[int, int], int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            residue_colors: dict[tuple[int, int], int] = {}
            ok = True
            for r in range(h):
                for c in range(w):
                    color = int(grid[r, c])
                    if color == bg:
                        continue
                    key = (r % row_period, c % col_period)
                    if key in residue_colors and residue_colors[key] != color:
                        ok = False
                        break
                    residue_colors[key] = color
                if not ok:
                    break
            if ok:
                candidates.append((row_period, col_period, residue_colors))
    return candidates


def _periodic_completion(
    grid: np.ndarray, bg: int
) -> tuple[np.ndarray, int, int, dict[tuple[int, int], int]]:
    candidates = _period_candidates(grid, bg)
    assert candidates
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1

    row_period, col_period, residue_colors = best[0]
    out = np.full_like(grid, bg)
    for (rr, cc), color in residue_colors.items():
        out[rr::row_period, cc::col_period] = color
    return out, row_period, col_period, residue_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble separated path strips by endpoint alignment, then recover erased periodic cells.

    Input structure:
    - The grid is a short, wide black-background strip.
    - Foreground columns appear in 3-4 contiguous groups separated by all-black columns.
    - Each group is a multicolor path-like fragment whose left edge and right edge each contain exactly one degree-1 endpoint pixel.

    Rule:
    - Remove the all-black separator columns.
    - Keep the first group fixed, then slide each later group vertically so its left endpoint row matches the previous placed group's right endpoint row, and place it immediately to the right.
    - On the assembled strip, ignore black cells and find the unique smallest row/column period whose residue classes assign every visible foreground cell one consistent color.
    - Fill every black cell belonging to an observed foreground residue class with that class color; leave residue classes with no foreground evidence black.

    Key insight:
    - Endpoint alignment changes the assembled row coordinates, so the correct periodic residue classes are only visible after the groups are reconnected in the intended chain order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    groups = _split_column_groups(g, bg)
    assert 3 <= len(groups) <= 4

    assembled = _assemble_groups(groups, bg, mode="endpoint")
    completed, _row_period, _col_period, _residue_colors = _periodic_completion(assembled, bg)
    return completed


def _has_subperiod(values: list[int]) -> bool:
    n = len(values)
    for p in range(1, n):
        if n % p == 0 and all(values[i] == values[i % p] for i in range(n)):
            return True
    return False


def _valid_group(group: np.ndarray, bg: int) -> bool:
    mask = group != bg
    if not np.any(mask):
        return False
    if np.any(~np.any(mask, axis=0)):
        return False
    try:
        _endpoint_row(group, bg, "left")
        _endpoint_row(group, bg, "right")
    except AssertionError:
        return False
    return True


def _partitions(total: int, count: int) -> list[list[int]]:
    results: list[list[int]] = []

    def rec(remaining: int, groups_left: int, prefix: list[int]) -> None:
        if groups_left == 1:
            if GEN_GROUP_WIDTH_MIN <= remaining <= GEN_GROUP_WIDTH_MAX:
                results.append(prefix + [remaining])
            return
        for width in range(GEN_GROUP_WIDTH_MIN, GEN_GROUP_WIDTH_MAX + 1):
            min_tail = GEN_GROUP_WIDTH_MIN * (groups_left - 1)
            max_tail = GEN_GROUP_WIDTH_MAX * (groups_left - 1)
            if not (min_tail <= remaining - width <= max_tail):
                continue
            rec(remaining - width, groups_left - 1, prefix + [width])

    rec(total, count, [])
    return results


def _valid_partitions(width: int, group_count: int, stem_col: int) -> list[list[int]]:
    valid: list[list[int]] = []
    for parts in _partitions(width, group_count):
        col = 0
        ok = True
        for group_width in parts:
            left = col
            right = col + group_width - 1
            if left % COL_PERIOD == stem_col or right % COL_PERIOD == stem_col:
                ok = False
                break
            col += group_width
        if ok:
            valid.append(parts)
    return valid


def _sample_target(width: int, stem_col: int, stem_rows: tuple[int, ...]) -> np.ndarray | None:
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    a, b, c = palette
    if set(stem_rows) == {0, 2, 3}:
        stem_colors = {0: a, 2: b, 3: c}
    else:
        stem_colors = {1: a, 2: b, 3: c}

    bar_colors = [0] * COL_PERIOD
    center = stem_colors[GEN_BAR_ROW]
    others = [color for color in palette if color != center]
    bar_colors[stem_col] = center
    bar_colors[(stem_col + 1) % COL_PERIOD] = others[0]
    bar_colors[(stem_col + 2) % COL_PERIOD] = others[1]
    bar_colors[(stem_col + 3) % COL_PERIOD] = others[0]
    if _has_subperiod(bar_colors):
        return None

    target = np.zeros((GEN_HEIGHT, width), dtype=int)
    for c in range(width):
        target[GEN_BAR_ROW, c] = bar_colors[c % COL_PERIOD]
    for c in range(stem_col, width, COL_PERIOD):
        for r in range(GEN_HEIGHT):
            rr = r % ROW_PERIOD
            if rr in stem_rows:
                target[r, c] = stem_colors[rr]

    completed, row_period, col_period, _residue_colors = _periodic_completion(target, BG)
    if row_period != ROW_PERIOD or col_period != COL_PERIOD:
        return None
    if not np.array_equal(completed, target):
        return None
    return target


def _slice_target(target: np.ndarray, parts: list[int]) -> list[np.ndarray] | None:
    groups: list[np.ndarray] = []
    col = 0
    for width in parts:
        group = target[:, col : col + width].copy()
        if not _valid_group(group, BG):
            return None
        groups.append(group)
        col += width
    return groups


def _shifted_input_groups(groups: list[np.ndarray]) -> list[np.ndarray] | None:
    shifted = [groups[0]]
    moved = False
    for group in groups[1:]:
        rows = np.nonzero(group != BG)[0]
        min_dr = -int(rows.min())
        max_dr = int(group.shape[0] - 1 - rows.max())
        nonzero = [dr for dr in range(min_dr, max_dr + 1) if dr != 0]
        if not nonzero:
            return None
        options = [0] + nonzero
        dr = random.choice(options)
        if dr != 0:
            moved = True
        shifted.append(_shift_group(group, BG, dr))
    if not moved:
        idx = random.randrange(1, len(groups))
        group = groups[idx]
        rows = np.nonzero(group != BG)[0]
        min_dr = -int(rows.min())
        max_dr = int(group.shape[0] - 1 - rows.max())
        nonzero = [dr for dr in range(min_dr, max_dr + 1) if dr != 0]
        shifted[idx] = _shift_group(group, BG, random.choice(nonzero))
    return shifted


def _render_input(groups: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    width = sum(group.shape[1] for group in groups) + len(groups) - 1
    grid = np.zeros((groups[0].shape[0], width), dtype=int)
    spans: list[tuple[int, int]] = []
    col = 0
    for group in groups:
        grid[:, col : col + group.shape[1]] = group
        spans.append((col, col + group.shape[1]))
        col += group.shape[1] + 1
    return grid, spans


def _erase_cells(
    input_grid: np.ndarray, groups: list[np.ndarray], spans: list[tuple[int, int]], target: np.ndarray
) -> bool:
    assembled_col = 0
    input_to_assembled: dict[int, int] = {}
    input_col = 0
    for group in groups:
        for k in range(group.shape[1]):
            input_to_assembled[input_col + k] = assembled_col + k
        input_col += group.shape[1] + 1
        assembled_col += group.shape[1]

    class_counts: dict[tuple[int, int], int] = {}
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            if target[r, c] != BG:
                key = (r % ROW_PERIOD, c % COL_PERIOD)
                class_counts[key] = class_counts.get(key, 0) + 1

    visible_per_input_col = {
        c: int(np.count_nonzero(input_grid[:, c] != BG))
        for c in range(input_grid.shape[1])
        if np.any(input_grid[:, c] != BG)
    }

    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for c0, c1 in spans:
        for input_c in range(c0 + 2, c1 - 2):
            assembled_c = input_to_assembled[input_c]
            for r in range(input_grid.shape[0]):
                if input_grid[r, input_c] == BG:
                    continue
                key = (r % ROW_PERIOD, assembled_c % COL_PERIOD)
                if key in class_counts:
                    candidates.append((r, input_c, key))

    random.shuffle(candidates)
    if not candidates:
        return False

    max_erased = max(1, min(len(candidates), np.count_nonzero(target) // 5))
    erased_goal = random.randint(1, max_erased)
    erased = 0
    for r, c, key in candidates:
        if visible_per_input_col[c] <= 1:
            continue
        if class_counts[key] <= 1:
            continue
        input_grid[r, c] = BG
        class_counts[key] -= 1
        visible_per_input_col[c] -= 1
        erased += 1
        if erased >= erased_goal:
            break
    return erased > 0


def _alt_completion_from_mode(input_grid: np.ndarray, mode: str) -> np.ndarray | None:
    groups = _split_column_groups(input_grid, BG)
    try:
        assembled = _assemble_groups(groups, BG, mode=mode)
        completed, _row_period, _col_period, _residue_colors = _periodic_completion(assembled, BG)
    except AssertionError:
        return None
    return completed


def _validate_instance(input_grid: np.ndarray, target: np.ndarray) -> bool:
    groups = _split_column_groups(input_grid, BG)
    if len(groups) not in GEN_GROUP_COUNT_OPTIONS:
        return False
    if any(not _valid_group(group, BG) for group in groups):
        return False

    try:
        assembled = _assemble_groups(groups, BG, mode="endpoint")
    except AssertionError:
        return False
    if assembled.shape != target.shape:
        return False
    erased = int(np.count_nonzero((target != BG) & (assembled == BG)))
    if erased <= 0:
        return False
    if erased * 10 > 3 * int(np.count_nonzero(target)):
        return False

    try:
        solved = main(input_grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, target):
        return False

    completed, row_period, col_period, _residue_colors = _periodic_completion(target, BG)
    if row_period != ROW_PERIOD or col_period != COL_PERIOD:
        return False
    if not np.array_equal(completed, target):
        return False

    no_align = _alt_completion_from_mode(input_grid, mode="none")
    if no_align is not None and np.array_equal(no_align, target):
        return False

    top_align = _alt_completion_from_mode(input_grid, mode="top")
    if top_align is not None and np.array_equal(top_align, target):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    group_count = random.choice(GEN_GROUP_COUNT_OPTIONS)
    stem_col = random.choice(GEN_STEM_COL_OPTIONS)
    width_options = []
    for width in range(
        max(GEN_WIDTH_MIN, group_count * GEN_GROUP_WIDTH_MIN),
        min(GEN_WIDTH_MAX, group_count * GEN_GROUP_WIDTH_MAX) + 1,
    ):
        parts = _valid_partitions(width, group_count, stem_col)
        if parts:
            width_options.append((width, parts))
    if not width_options:
        return None
    width, parts = random.choice(width_options)
    stem_rows = random.choice(GEN_STEM_ROW_OPTIONS)
    target = _sample_target(width, stem_col, stem_rows)
    if target is None:
        return None
    groups = _slice_target(target, random.choice(parts))
    if groups is None:
        return None
    shifted_groups = _shifted_input_groups(groups)
    if shifted_groups is None:
        return None
    input_grid, spans = _render_input(shifted_groups)
    if not _erase_cells(input_grid, shifted_groups, spans, target):
        return None
    if not _validate_instance(input_grid, target):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 057-5")
