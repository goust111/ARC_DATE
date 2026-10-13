from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 5
SEPARATOR = 0
SEED_COLORS = tuple(c for c in range(10) if c not in (BACKGROUND, SEPARATOR))
QUADRANT_COUNT = 4
HALF_SIZE_RANGE = (5, 14)
COLOR_COUNT_RANGE = (2, 4)
COUNT_PER_COLOR_RANGE = (1, 4)
MAX_SEEDS = 10
GENERATION_ATTEMPTS = 400


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected at least one value"
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _find_separator_cross(g: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, g.shape)
    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(height)
        if bool(np.all(g[r, :] == int(g[r, 0])))
    ]
    solid_cols = [
        (c, int(g[0, c]))
        for c in range(width)
        if bool(np.all(g[:, c] == int(g[0, c])))
    ]

    black_rows = [r for r, color in solid_rows if color == SEPARATOR]
    black_cols = [c for c, color in solid_cols if color == SEPARATOR]
    if black_rows or black_cols:
        assert len(black_rows) == 1 and len(black_cols) == 1, (
            "expected one complete black separator row and column"
        )
        return int(black_rows[0]), int(black_cols[0]), SEPARATOR

    candidates = []
    for row, row_color in solid_rows:
        for col, col_color in solid_cols:
            if row_color == col_color:
                candidates.append((int(row), int(col), int(row_color)))
    assert len(candidates) == 1, "expected one solid separator cross"
    return candidates[0]


def _background_color(g: np.ndarray, sep: int) -> int:
    non_sep = g[g != int(sep)]
    assert non_sep.size > 0, "separator cannot fill the whole grid"
    return _unique_mode(non_sep)


def _quadrant(row: int, col: int, cross_row: int, cross_col: int) -> int:
    assert row != cross_row and col != cross_col, "seeds cannot lie on the separator cross"
    if row < cross_row and col < cross_col:
        return 0
    if row < cross_row and col > cross_col:
        return 1
    if row > cross_row and col < cross_col:
        return 2
    if row > cross_row and col > cross_col:
        return 3
    raise AssertionError("seed must be inside one quadrant")


def _raw_block_bounds(row: int, col: int, side: int, quadrant: int) -> tuple[int, int, int, int]:
    if quadrant == 0:
        return row - side + 1, row, col - side + 1, col
    if quadrant == 1:
        return row - side + 1, row, col, col + side - 1
    if quadrant == 2:
        return row, row + side - 1, col - side + 1, col
    if quadrant == 3:
        return row, row + side - 1, col, col + side - 1
    raise AssertionError("unknown quadrant")


def _clipped_block_bounds(
    row: int, col: int, side: int, quadrant: int, height: int, width: int
) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = _raw_block_bounds(row, col, side, quadrant)
    return max(0, r0), min(height - 1, r1), max(0, c0), min(width - 1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a same-color square away from the central cross.

    Input structure:
    - The background is gray and a complete black row plus a complete black column
      meet at the center, splitting the grid into four quadrants.
    - Every non-background, non-cross pixel in a quadrant is a one-cell colored seed.
    - The same seed color may appear multiple times anywhere in the grid.

    Rule:
    - Count the total number N of seeds of each color over the whole grid.
    - In row-major seed order, draw an N by N solid block from each seed in the
      diagonal direction away from the cross: upper-left, upper-right, lower-left,
      or lower-right according to the seed's quadrant.
    - Clip blocks at the grid boundary; later blocks overwrite earlier blocks.
    - Restore the black cross and all original seed pixels after drawing.

    Key insight:
    - A seed's color gives the square size through its global frequency, while
      the seed's quadrant gives the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    cross_row, cross_col, sep = _find_separator_cross(g)
    bg = _background_color(g, sep)

    seed_mask = (g != int(bg)) & (g != int(sep))
    seed_coords = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    assert seed_coords, "expected at least one colored seed"

    seeds = []
    for row, col in sorted(seed_coords):
        color = int(g[row, col])
        quad = _quadrant(row, col, cross_row, cross_col)
        seeds.append((int(row), int(col), color, quad))

    counts = Counter(color for _row, _col, color, _quad in seeds)
    out = g.copy()

    for row, col, color, quad in seeds:
        side = int(counts[color])
        r0, r1, c0, c1 = _clipped_block_bounds(row, col, side, quad, height, width)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    out[cross_row, :] = int(sep)
    out[:, cross_col] = int(sep)
    for row, col, color, _quad in seeds:
        out[row, col] = int(color)
    return out


# --------------- generator ---------------


def _quadrant_cells(quadrant: int, height: int, width: int, cross_row: int, cross_col: int):
    if quadrant == 0:
        rows = range(0, cross_row)
        cols = range(0, cross_col)
    elif quadrant == 1:
        rows = range(0, cross_row)
        cols = range(cross_col + 1, width)
    elif quadrant == 2:
        rows = range(cross_row + 1, height)
        cols = range(0, cross_col)
    else:
        rows = range(cross_row + 1, height)
        cols = range(cross_col + 1, width)
    return [(int(r), int(c)) for r in rows for c in cols]


def _away_step(quadrant: int) -> tuple[int, int]:
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)][int(quadrant)]


def _is_clipped(
    row: int, col: int, side: int, quadrant: int, height: int, width: int
) -> bool:
    r0, r1, c0, c1 = _raw_block_bounds(row, col, side, quadrant)
    return r0 < 0 or c0 < 0 or r1 >= height or c1 >= width


def _intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1]) and max(a[2], b[2]) <= min(a[3], b[3])


def _has_clipping(records: list[dict], counts: dict[int, int], height: int, width: int) -> bool:
    for rec in records:
        row, col = rec["pos"]
        if _is_clipped(row, col, counts[int(rec["color"])], int(rec["quadrant"]), height, width):
            return True
    return False


def _has_overlap(records: list[dict], counts: dict[int, int], height: int, width: int) -> bool:
    bounds = []
    for rec in records:
        row, col = rec["pos"]
        side = counts[int(rec["color"])]
        bounds.append(_clipped_block_bounds(row, col, side, int(rec["quadrant"]), height, width))
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            if _intersect(bounds[i], bounds[j]):
                return True
    return False


def _clear_position(pos: tuple[int, int], occupied: set[tuple[int, int]], clearance: int) -> bool:
    row, col = pos
    for r, c in occupied:
        if max(abs(row - r), abs(col - c)) <= int(clearance):
            return False
    return True


def _choose_position(
    candidates: list[tuple[int, int]], occupied: set[tuple[int, int]]
) -> tuple[int, int] | None:
    random.shuffle(candidates)
    for clearance in (1, 0):
        for pos in candidates:
            if pos not in occupied and _clear_position(pos, occupied, clearance):
                return pos
    return None


def _sample_counts(want_overlap: bool, want_clip: bool) -> dict[int, int] | None:
    for _ in range(100):
        n_colors = random.randint(*COLOR_COUNT_RANGE)
        colors = random.sample(SEED_COLORS, n_colors)
        counts = {
            int(color): random.randint(*COUNT_PER_COLOR_RANGE)
            for color in colors
        }
        while sum(counts.values()) < QUADRANT_COUNT:
            color = random.choice(colors)
            if counts[int(color)] < COUNT_PER_COLOR_RANGE[1]:
                counts[int(color)] += 1

        if want_overlap and not any(count >= 3 for count in counts.values()):
            counts[int(random.choice(colors))] = 3
        if want_clip and not any(count >= 2 for count in counts.values()):
            counts[int(random.choice(colors))] = 2

        total = sum(counts.values())
        overlap_eligible = sum(count for count in counts.values() if count >= 3)
        if QUADRANT_COUNT <= total <= MAX_SEEDS and (not want_overlap or overlap_eligible >= 2):
            return counts
    return None


def _make_records(counts: dict[int, int]) -> list[dict]:
    records = []
    for color, count in counts.items():
        for _ in range(int(count)):
            records.append({"color": int(color), "quadrant": None, "pos": None})
    random.shuffle(records)
    return records


def _assign_quadrants(
    records: list[dict], counts: dict[int, int], want_overlap: bool, want_clip: bool
) -> tuple[tuple[int, int] | None, int | None] | None:
    ids = list(range(len(records)))
    overlap_pair = None
    if want_overlap:
        eligible = [i for i in ids if counts[int(records[i]["color"])] >= 3]
        if len(eligible) < 2:
            return None
        a, b = random.sample(eligible, 2)
        overlap_pair = (int(a), int(b))
        q = random.randrange(QUADRANT_COUNT)
        records[a]["quadrant"] = q
        records[b]["quadrant"] = q

    clip_id = None
    if want_clip:
        excluded = set(overlap_pair or ())
        candidates = [
            i for i in ids
            if i not in excluded and counts[int(records[i]["color"])] >= 2
        ]
        if not candidates:
            candidates = [i for i in ids if counts[int(records[i]["color"])] >= 2]
        if not candidates:
            return None
        clip_id = int(random.choice(candidates))
        if records[clip_id]["quadrant"] is None:
            records[clip_id]["quadrant"] = random.randrange(QUADRANT_COUNT)

    present = {int(rec["quadrant"]) for rec in records if rec["quadrant"] is not None}
    missing = [q for q in range(QUADRANT_COUNT) if q not in present]
    available = [i for i in ids if records[i]["quadrant"] is None]
    if len(available) < len(missing):
        return None

    random.shuffle(available)
    for q, i in zip(missing, available):
        records[i]["quadrant"] = int(q)

    for i in ids:
        if records[i]["quadrant"] is None:
            records[i]["quadrant"] = random.randrange(QUADRANT_COUNT)
    return overlap_pair, clip_id


def _place_overlap_pair(
    records: list[dict],
    pair: tuple[int, int],
    counts: dict[int, int],
    height: int,
    width: int,
    cross_row: int,
    cross_col: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b = pair
    quadrant = int(records[a]["quadrant"])
    n_a = counts[int(records[a]["color"])]
    n_b = counts[int(records[b]["color"])]
    dr, dc = _away_step(quadrant)
    cells = set(_quadrant_cells(quadrant, height, width, cross_row, cross_col))

    preferred = []
    fallback = []
    for row, col in cells:
        pos_a = (int(row), int(col))
        pos_b = (int(row + 2 * dr), int(col + 2 * dc))
        if pos_b not in cells:
            continue
        b_a = _clipped_block_bounds(pos_a[0], pos_a[1], n_a, quadrant, height, width)
        b_b = _clipped_block_bounds(pos_b[0], pos_b[1], n_b, quadrant, height, width)
        if not _intersect(b_a, b_b):
            continue
        clipped = (
            _is_clipped(pos_a[0], pos_a[1], n_a, quadrant, height, width)
            or _is_clipped(pos_b[0], pos_b[1], n_b, quadrant, height, width)
        )
        (fallback if clipped else preferred).append((pos_a, pos_b))

    choices = preferred or fallback
    if not choices:
        return None
    return random.choice(choices)


def _place_records(
    records: list[dict],
    counts: dict[int, int],
    overlap_pair: tuple[int, int] | None,
    clip_id: int | None,
    height: int,
    width: int,
    cross_row: int,
    cross_col: int,
) -> bool:
    occupied: set[tuple[int, int]] = set()

    if overlap_pair is not None:
        pair_positions = _place_overlap_pair(
            records, overlap_pair, counts, height, width, cross_row, cross_col
        )
        if pair_positions is None:
            return False
        for idx, pos in zip(overlap_pair, pair_positions):
            records[idx]["pos"] = pos
            occupied.add(pos)

    if clip_id is not None and records[clip_id]["pos"] is None:
        rec = records[clip_id]
        q = int(rec["quadrant"])
        side = counts[int(rec["color"])]
        candidates = [
            pos for pos in _quadrant_cells(q, height, width, cross_row, cross_col)
            if _is_clipped(pos[0], pos[1], side, q, height, width)
        ]
        pos = _choose_position(candidates, occupied)
        if pos is None:
            return False
        rec["pos"] = pos
        occupied.add(pos)

    order = list(range(len(records)))
    random.shuffle(order)
    for idx in order:
        rec = records[idx]
        if rec["pos"] is not None:
            continue
        q = int(rec["quadrant"])
        candidates = _quadrant_cells(q, height, width, cross_row, cross_col)
        pos = _choose_position(candidates, occupied)
        if pos is None:
            return False
        rec["pos"] = pos
        occupied.add(pos)
    return True


def _render_input(records: list[dict], height: int, width: int, cross_row: int, cross_col: int) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[cross_row, :] = SEPARATOR
    grid[:, cross_col] = SEPARATOR
    for rec in records:
        row, col = rec["pos"]
        grid[row, col] = int(rec["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(*HALF_SIZE_RANGE)
    half_w = random.randint(*HALF_SIZE_RANGE)
    height = 2 * half_h + 1
    width = 2 * half_w + 1
    cross_row, cross_col = half_h, half_w

    want_overlap = random.random() < 0.5
    want_clip = random.random() < 0.5
    counts = _sample_counts(want_overlap, want_clip)
    if counts is None:
        return None

    records = _make_records(counts)
    assigned = _assign_quadrants(records, counts, want_overlap, want_clip)
    if assigned is None:
        return None
    overlap_pair, clip_id = assigned

    if not _place_records(records, counts, overlap_pair, clip_id, height, width, cross_row, cross_col):
        return None

    if want_overlap and not _has_overlap(records, counts, height, width):
        return None
    if want_clip and not _has_clipping(records, counts, height, width):
        return None

    used_quadrants = {int(rec["quadrant"]) for rec in records}
    if used_quadrants != set(range(QUADRANT_COUNT)):
        return None

    grid = _render_input(records, height, width, cross_row, cross_col)
    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/15/2 input")
