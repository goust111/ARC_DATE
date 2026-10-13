from __future__ import annotations

import random
import itertools
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
SEED_COLORS = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _has_adjacent_pair(values: np.ndarray, color: int = BLUE) -> bool:
    return bool(np.any((values[:-1] == color) & (values[1:] == color)))


def _detect_effective_lines(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    box = _bbox(grid == BLUE)
    assert box is not None, "expected a blue grid"
    r0, c0, r1, c1 = box
    rows = [r for r in range(r0, r1 + 1) if _has_adjacent_pair(grid[r, c0 : c1 + 1])]
    cols = [c for c in range(c0, c1 + 1) if _has_adjacent_pair(grid[r0 : r1 + 1, c])]
    assert 3 <= len(rows) <= 5
    assert 3 <= len(cols) <= 5
    assert rows == sorted(rows) and cols == sorted(cols)
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:]))
    return box, rows, cols


def _complete_grid(grid: np.ndarray, box: tuple[int, int, int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = box
    for r in rows:
        span = out[r, c0 : c1 + 1]
        span[span == BLACK] = BLUE
    for c in cols:
        span = out[r0 : r1 + 1, c]
        span[span == BLACK] = BLUE
    return out


def _fill_regions(grid: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for top, bottom in zip(rows, rows[1:]):
        for left, right in zip(cols, cols[1:]):
            assert bottom - top >= 2 and right - left >= 2
            region = out[top + 1 : bottom, left + 1 : right]
            seeds = np.argwhere((region != BLACK) & (region != BLUE))
            assert seeds.shape[0] in (0, 1), "each region must have at most one seed"
            if seeds.shape[0] == 1:
                seed_r, seed_c = map(int, seeds[0])
                color = int(region[seed_r, seed_c])
                region[region == BLACK] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the broken blue grid, then flood-fill each closed cell from its single seed pixel.

    Input structure:
    - A black background contains one blue orthogonal grid with 1-3 cell gaps on some line segments.
    - True horizontal grid lines are exactly the blue rows inside the blue bbox that contain a horizontal adjacent blue pair.
    - True vertical grid lines are exactly the blue columns inside the blue bbox that contain a vertical adjacent blue pair.
    - Some rectangular cells between the grid lines contain exactly one non-black, non-blue seed pixel.

    Rule:
    - Find the blue bbox, then detect the effective horizontal and vertical line coordinates using adjacent blue pairs.
    - Inside that bbox, recolor black cells on every effective row and every effective column to blue, completing the grid.
    - The completed grid partitions the bbox interior into rectangles between consecutive line rows and columns.
    - In each rectangle, if there is exactly one seed pixel, fill all black cells of that rectangle with the seed color; keep blue grid lines and the seed itself unchanged.

    Key insight:
    - The seed colors are not expanded in the broken input directly: the missing blue segments must be restored first, because those completed line segments create the closed rectangular boundaries that limit each flood fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    box, rows, cols = _detect_effective_lines(g)
    completed = _complete_grid(g, box, rows, cols)
    return _fill_regions(completed, rows, cols)


def _choose_line_positions(span: int, count: int) -> list[int]:
    candidates = []
    inner = range(1, span - 1)
    for mids in itertools.combinations(inner, count - 2):
        pos = (0, *mids, span - 1)
        if all(b - a >= 2 for a, b in zip(pos, pos[1:])):
            candidates.append(list(pos))
    assert candidates
    return random.choice(candidates)


def _region_records(rows: list[int], cols: list[int]) -> list[dict]:
    regions = []
    for ri, (top, bottom) in enumerate(zip(rows, rows[1:])):
        for ci, (left, right) in enumerate(zip(cols, cols[1:])):
            h = bottom - top - 1
            w = right - left - 1
            regions.append(
                {
                    "ri": ri,
                    "ci": ci,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "area": h * w,
                    "internal_sides": int(ri > 0) + int(ri < len(rows) - 2) + int(ci > 0) + int(ci < len(cols) - 2),
                }
            )
    return regions


def _paint_complete_grid(grid: np.ndarray, rows: list[int], cols: list[int]) -> None:
    for r in rows:
        grid[r, cols[0] : cols[-1] + 1] = BLUE
    for c in cols:
        grid[rows[0] : rows[-1] + 1, c] = BLUE


def _expected_output(base: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    return _fill_regions(base, rows, cols)


def _component_non_blue(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    q = deque([start])
    seen = {start}
    h, w = grid.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) != BLUE:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _interval_gap(slot: tuple[str, int, int, int]) -> tuple[int, int]:
    _, _, start, stop = slot
    length = random.randint(1, min(3, stop - start + 1))
    gap_start = random.randint(start, stop - length + 1)
    return gap_start, length


def _apply_gap(grid: np.ndarray, slot: tuple[str, int, int, int]) -> None:
    orient, fixed, start, stop = slot
    gap_start, length = _interval_gap(slot)
    if orient == "h":
        grid[fixed, gap_start : gap_start + length] = BLACK
    else:
        grid[gap_start : gap_start + length, fixed] = BLACK


def _candidate_slots(rows: list[int], cols: list[int]) -> tuple[list[tuple[str, int, int, int]], list[tuple[str, int, int, int]]]:
    h_slots = []
    for r in rows:
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                h_slots.append(("h", r, left + 1, right - 1))
    v_slots = []
    for c in cols:
        for top, bottom in zip(rows, rows[1:]):
            if bottom - top > 1:
                v_slots.append(("v", c, top + 1, bottom - 1))
    return h_slots, v_slots


def _leakage_slot(region: dict, rows: list[int], cols: list[int]) -> tuple[str, int, int, int]:
    sides = []
    ci = region["ci"]
    ri = region["ri"]
    sides.append(("h", rows[ri], cols[ci] + 1, cols[ci + 1] - 1, ri > 0))
    sides.append(("h", rows[ri + 1], cols[ci] + 1, cols[ci + 1] - 1, ri < len(rows) - 2))
    sides.append(("v", cols[ci], rows[ri] + 1, rows[ri + 1] - 1, ci > 0))
    sides.append(("v", cols[ci + 1], rows[ri] + 1, rows[ri + 1] - 1, ci < len(cols) - 2))
    internal = [side[:4] for side in sides if side[4]]
    if internal:
        return random.choice(internal)
    return random.choice([side[:4] for side in sides])


def _validate_gap_lengths(inp: np.ndarray, rows: list[int], cols: list[int]) -> bool:
    for r in rows:
        values = inp[r, cols[0] : cols[-1] + 1]
        run = 0
        for v in values:
            if int(v) == BLACK:
                run += 1
                if run > 3:
                    return False
            else:
                run = 0
    for c in cols:
        values = inp[rows[0] : rows[-1] + 1, c]
        run = 0
        for v in values:
            if int(v) == BLACK:
                run += 1
                if run > 3:
                    return False
            else:
                run = 0
    return True


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(3, 5)
    col_count = random.randint(3, 5)
    box_h = random.randint(max(10, 2 * (row_count - 1) + 1), 13)
    box_w = random.randint(max(10, 2 * (col_count - 1) + 1), 13)
    rows_rel = _choose_line_positions(box_h, row_count)
    cols_rel = _choose_line_positions(box_w, col_count)

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    h = box_h + top_margin + bottom_margin
    w = box_w + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    rows = [top_margin + r for r in rows_rel]
    cols = [left_margin + c for c in cols_rel]

    base = np.zeros((h, w), dtype=int)
    _paint_complete_grid(base, rows, cols)

    regions = _region_records(rows, cols)
    fillable = [region for region in regions if region["area"] >= 2]
    if len(fillable) < 4:
        return None

    max_seed_count = min(5, len(fillable), len(regions) - 1)
    if max_seed_count < 3:
        return None
    seed_count = random.randint(3, max_seed_count)
    seeded_regions = random.sample(fillable, seed_count)

    colors = [random.choice(SEED_COLORS) for _ in range(seed_count)]
    if len(set(colors)) < 2:
        colors[random.randrange(seed_count)] = random.choice([c for c in SEED_COLORS if c != colors[0]])

    seed_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for region, color in zip(seeded_regions, colors):
        r = random.randint(region["top"] + 1, region["bottom"] - 1)
        c = random.randint(region["left"] + 1, region["right"] - 1)
        base[r, c] = color
        seed_positions[(region["ri"], region["ci"])] = (r, c)

    expected = _expected_output(base, rows, cols)
    inp = base.copy()

    h_slots, v_slots = _candidate_slots(rows, cols)
    leak_region = max(seeded_regions, key=lambda region: (region["internal_sides"], region["area"]))
    used_slots = set()

    forced = _leakage_slot(leak_region, rows, cols)
    _apply_gap(inp, forced)
    used_slots.add(forced)

    if forced[0] == "h" and v_slots:
        extra = random.choice(v_slots)
        _apply_gap(inp, extra)
        used_slots.add(extra)
    elif forced[0] == "v" and h_slots:
        extra = random.choice(h_slots)
        _apply_gap(inp, extra)
        used_slots.add(extra)

    all_slots = h_slots + v_slots
    random.shuffle(all_slots)
    extra_count = random.randint(1, min(6, len(all_slots)))
    for slot in all_slots:
        if len(used_slots) >= extra_count + 1:
            break
        if slot in used_slots:
            continue
        if random.random() < 0.55:
            _apply_gap(inp, slot)
            used_slots.add(slot)

    box, det_rows, det_cols = _detect_effective_lines(inp)
    if det_rows != rows or det_cols != cols:
        return None
    if box != (rows[0], cols[0], rows[-1], cols[-1]):
        return None
    if not _validate_gap_lengths(inp, rows, cols):
        return None

    completed = _complete_grid(inp, box, det_rows, det_cols)
    if not np.array_equal(completed, base):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    if len(seed_positions) != seed_count or len(regions) == len(seeded_regions):
        return None

    leak_seed = seed_positions[(leak_region["ri"], leak_region["ci"])]
    comp = _component_non_blue(inp, leak_seed)
    target = {
        (r, c)
        for r in range(leak_region["top"] + 1, leak_region["bottom"])
        for c in range(leak_region["left"] + 1, leak_region["right"])
    }
    if not (comp - target):
        return None

    if np.count_nonzero((expected != base) & (base == BLACK)) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
