from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
ORANGE = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _guide_info(grid: np.ndarray) -> tuple[int, set[int]]:
    coords = np.argwhere(grid == ORANGE)
    assert coords.size > 0, "expected an orange guide bar"
    rows = {int(r) for r, _c in coords}
    assert len(rows) == 1, "guide bar must be horizontal"
    guide_row = next(iter(rows))
    guide_cols = {int(c) for _r, c in coords}
    assert 0 < guide_row < grid.shape[0] - 1, "guide row must leave vertical room"
    assert len(guide_cols) >= 4, "guide must span multiple columns"
    guide_band = {int(v) for v in np.unique(grid[guide_row, :])}
    assert guide_band <= {BG, ORANGE}, "guide row may only contain guide pixels and gaps"
    return guide_row, guide_cols


def _wall_segments(grid: np.ndarray, guide_row: int) -> list[tuple[int, int, int]]:
    grey_mask = grid == GREY
    assert not np.any(grey_mask[:-1, :] & grey_mask[1:, :]), "walls must be one cell high"

    segments: list[tuple[int, int, int]] = []
    for row in np.where(np.any(grey_mask, axis=1))[0]:
        r = int(row)
        assert r > guide_row, "walls must be below the guide"
        cols = [int(c) for c in np.where(grey_mask[r])[0]]
        start = prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
                continue
            segments.append((r, start, prev))
            start = prev = c
        segments.append((r, start, prev))

    assert segments, "expected wall segments"
    return segments


def _source_coords(grid: np.ndarray, guide_row: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert coords, "expected source pixels"
    cols = [c for _r, c in coords]
    assert len(set(cols)) == len(cols), "sources must occupy distinct columns"
    assert all(r > guide_row for r, _c in coords), "sources must lie below the guide"
    return coords


def _segment_bounds_on_row(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    left = int(col)
    right = int(col)
    while left - 1 >= 0 and int(grid[row, left - 1]) == GREY:
        left -= 1
    while right + 1 < grid.shape[1] and int(grid[row, right + 1]) == GREY:
        right += 1
    return left, right


def _classify_sources(
    grid: np.ndarray, guide_row: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    successful: list[tuple[int, int]] = []
    failed: list[tuple[int, int]] = []
    for r, c in _source_coords(grid, guide_row):
        if int(grid[guide_row, c]) == ORANGE:
            successful.append((r, c))
        else:
            failed.append((r, c))
    return successful, failed


def _trace_rays(
    grid: np.ndarray,
    base: np.ndarray,
    starts: list[tuple[int, int]],
    *,
    enable_branching: bool,
) -> tuple[np.ndarray, dict]:
    out = base.copy()
    h, w = grid.shape
    stack: list[tuple[int, int, int]] = [(int(r), int(c), 0) for r, c in starts]
    seen_starts: set[tuple[int, int]] = set()
    branch_starts: set[tuple[int, int]] = set()
    painted: set[tuple[int, int]] = set()
    wall_hits = 0
    max_depth = 0

    while stack:
        sr, sc, depth = stack.pop()
        if not _in_bounds(grid.shape, sr, sc):
            continue
        if (sr, sc) in seen_starts:
            continue
        seen_starts.add((sr, sc))
        max_depth = max(max_depth, depth)

        r = sr
        while r < h:
            cell = int(grid[r, sc])
            if cell == GREY:
                wall_hits += 1
                left, right = _segment_bounds_on_row(grid, r, sc)
                if enable_branching:
                    for open_c in (left - 1, right + 1):
                        if 0 <= open_c < w and int(grid[r, open_c]) == BG and r + 1 < h:
                            branch_starts.add((r + 1, open_c))
                            stack.append((r + 1, open_c, depth + 1))
                            max_depth = max(max_depth, depth + 1)
                break

            if cell == BG and int(out[r, sc]) == BG:
                out[r, sc] = RED
                painted.add((r, sc))
            r += 1

    return out, {
        "starts": {(int(r), int(c)) for r, c in starts if _in_bounds(grid.shape, r, c)},
        "seen_starts": seen_starts,
        "branch_starts": branch_starts,
        "branch_count": len(branch_starts),
        "wall_hits": wall_hits,
        "painted": painted,
        "max_depth": max_depth,
    }


def _solve_core(
    grid: np.ndarray,
    *,
    filter_projection: bool,
    start_mode: str,
    enable_branching: bool,
    emit_rays: bool,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, RED, GREY, ORANGE}, "unexpected colors"

    guide_row, _guide_cols = _guide_info(g)
    _wall_segments(g, guide_row)
    successful, failed = _classify_sources(g, guide_row)
    assert len(successful) >= 1, "expected at least one successful projection"
    assert len(failed) >= 1, "expected at least one failed projection"

    out = g.copy()
    active_sources = successful
    if filter_projection:
        for r, c in failed:
            out[r, c] = BG
    else:
        active_sources = successful + failed

    if not emit_rays:
        return out, {
            "guide_row": guide_row,
            "successful": successful,
            "failed": failed,
            "starts": set(),
            "seen_starts": set(),
            "branch_starts": set(),
            "branch_count": 0,
            "wall_hits": 0,
            "painted": set(),
            "max_depth": 0,
        }

    if start_mode == "guide":
        starts = [(guide_row + 1, c) for _r, c in active_sources if guide_row + 1 < g.shape[0]]
    elif start_mode == "source":
        starts = [(r + 1, c) for r, c in active_sources if r + 1 < g.shape[0]]
    else:
        raise ValueError(f"unknown start_mode: {start_mode}")

    traced, stats = _trace_rays(g, out, starts, enable_branching=enable_branching)
    stats["guide_row"] = guide_row
    stats["successful"] = successful
    stats["failed"] = failed
    return traced, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only successfully projected sources, then trace downward red rays that branch through wall openings.

    Input structure
    - The background is black.
    - One orange horizontal guide bar sits near the upper middle row and may contain black single-cell gaps.
    - Red source pixels lie below the guide in distinct columns, and grey horizontal wall segments lie farther below.

    Rule
    - Preserve the guide bar and all grey wall pixels.
    - A red source survives only if its column contains an orange guide pixel; sources aligned to guide gaps are erased.
    - Each surviving source launches a downward red ray from the cell immediately below the guide in that column.
    - A ray paints only black cells, stops just above the first grey wall segment it meets, and otherwise runs to the bottom edge.
    - When a ray stops on a wall, inspect the contiguous grey segment hit by that column; each black opening immediately outside the segment spawns a new downward ray from the cell below that opening.

    Key insight
    - The guide bar is both the projection filter and the shared launch surface, while the wall-segment endpoints convert horizontal openings into new vertical branches.
    """
    return _solve_core(
        grid,
        filter_projection=True,
        start_mode="guide",
        enable_branching=True,
        emit_rays=True,
    )[0]


# --------------- generator ---------------

HEIGHT_RANGE = (17, 24)
WIDTH_RANGE = (13, 18)
GUIDE_ROW_RANGE = (3, 5)
GUIDE_MARGIN = 1

WALL_COUNT_CHOICES = (2, 3)
WALL_LENGTH_RANGE = (3, 6)
WALL_ROW_GAP_RANGE = (2, 4)

SOURCE_COUNT_RANGE = (3, 6)
SUCCESS_COUNT_RANGE = (2, 4)
GAP_SOURCE_COUNT_RANGE = (1, 2)
GUIDE_GAP_COUNT_RANGE = (1, 3)

GENERATION_ATTEMPTS = 400


def _span_set(c0: int, c1: int) -> set[int]:
    return set(range(int(c0), int(c1) + 1))


def _choose_segment_containing(target: int, width: int) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []
    max_len = min(WALL_LENGTH_RANGE[1], width - 2)
    for length in range(WALL_LENGTH_RANGE[0], max_len + 1):
        for c0 in range(1, width - length):
            c1 = c0 + length - 1
            if c0 <= target <= c1:
                options.append((c0, c1))
    if not options:
        return None
    return random.choice(options)


def _spaced_candidates(candidates: list[int], chosen: list[int], min_gap: int = 2) -> list[int]:
    return [
        c
        for c in candidates
        if all(abs(int(c) - int(other)) >= min_gap for other in chosen)
    ]


def _pick_chain_layout(
    height: int, width: int, guide_row: int
) -> tuple[list[tuple[int, int, int]], list[int]] | None:
    wall_count = random.choice(WALL_COUNT_CHOICES)
    latest_first = height - (5 if wall_count == 3 else 3)
    first_row_min = guide_row + 4
    if first_row_min > latest_first:
        return None

    wall_rows = [random.randint(first_row_min, latest_first)]
    for idx in range(1, wall_count):
        remaining = wall_count - idx
        row_min = wall_rows[-1] + WALL_ROW_GAP_RANGE[0]
        row_max = height - (2 + 2 * remaining)
        if row_min > row_max:
            return None
        wall_rows.append(random.randint(row_min, row_max))

    max_len = min(WALL_LENGTH_RANGE[1], width - 2)
    if WALL_LENGTH_RANGE[0] > max_len:
        return None

    first_len = random.randint(WALL_LENGTH_RANGE[0], max_len)
    first_c0 = random.randint(1, width - first_len - 1)
    first_c1 = first_c0 + first_len - 1
    root_col = random.randint(first_c0, first_c1)
    segments = [(wall_rows[0], first_c0, first_c1)]

    direct_source_cols = [root_col]
    target_cols = [first_c0 - 1, first_c1 + 1]

    for row in wall_rows[1:]:
        random.shuffle(target_cols)
        placed = False
        for target in target_cols:
            segment = _choose_segment_containing(target, width)
            if segment is None:
                continue
            c0, c1 = segment
            existing_spans = [_span_set(sc0, sc1) for _r, sc0, sc1 in segments]
            direct_candidates = sorted(
                c
                for c in range(c0, c1 + 1)
                if all(c not in span for span in existing_spans)
            )
            direct_candidates = _spaced_candidates(direct_candidates, direct_source_cols)
            if not direct_candidates:
                continue
            segments.append((row, c0, c1))
            direct_source_cols.append(random.choice(direct_candidates))
            target_cols = [c0 - 1, c1 + 1]
            placed = True
            break
        if not placed:
            return None

    return segments, direct_source_cols


def _pick_success_columns(
    width: int, segments: list[tuple[int, int, int]], chain_cols: list[int]
) -> list[int] | None:
    target_count = random.randint(SUCCESS_COUNT_RANGE[0], SUCCESS_COUNT_RANGE[1])
    success_cols = list(chain_cols)
    wall_spans = [_span_set(c0, c1) for _row, c0, c1 in segments]
    columns = list(range(GUIDE_MARGIN, width - GUIDE_MARGIN))

    bottom_candidates = [
        c for c in columns if all(c not in span for span in wall_spans)
    ]
    bottom_candidates = _spaced_candidates(bottom_candidates, success_cols)
    if len(success_cols) < target_count and bottom_candidates:
        success_cols.append(random.choice(bottom_candidates))

    extra_hit_candidates = [
        c
        for _row, c0, c1 in segments
        for c in range(c0, c1 + 1)
        if c not in success_cols
    ]
    extra_hit_candidates = sorted(set(_spaced_candidates(extra_hit_candidates, success_cols)))
    while len(success_cols) < target_count and extra_hit_candidates:
        choice = random.choice(extra_hit_candidates)
        success_cols.append(choice)
        extra_hit_candidates = sorted(set(_spaced_candidates(extra_hit_candidates, success_cols)))

    if len(success_cols) < SUCCESS_COUNT_RANGE[0]:
        return None
    return success_cols[:target_count]


def _pick_gap_columns(width: int, success_cols: list[int]) -> tuple[list[int], list[int]] | None:
    min_gap_sources = GAP_SOURCE_COUNT_RANGE[0]
    max_gap_sources = min(GAP_SOURCE_COUNT_RANGE[1], SOURCE_COUNT_RANGE[1] - len(success_cols))
    if max_gap_sources < min_gap_sources:
        return None

    total_gap_count = random.randint(
        max(min_gap_sources, GUIDE_GAP_COUNT_RANGE[0]),
        GUIDE_GAP_COUNT_RANGE[1],
    )
    columns = list(range(GUIDE_MARGIN, width - GUIDE_MARGIN))
    gap_pool = _spaced_candidates(columns, success_cols)
    if len(gap_pool) < total_gap_count:
        return None

    chosen_gaps: list[int] = []
    while len(chosen_gaps) < total_gap_count:
        candidates = _spaced_candidates(gap_pool, chosen_gaps)
        if not candidates:
            return None
        chosen_gaps.append(random.choice(candidates))

    gap_source_count = random.randint(min_gap_sources, min(max_gap_sources, len(chosen_gaps)))
    random.shuffle(chosen_gaps)
    gap_source_cols = sorted(chosen_gaps[:gap_source_count])
    return sorted(chosen_gaps), gap_source_cols


def _assign_source_rows(
    guide_row: int, first_wall_row: int, source_cols: list[int]
) -> dict[int, int] | None:
    row_lo = guide_row + 2
    row_hi = first_wall_row - 2
    if row_lo > row_hi:
        return None

    rows = list(range(row_lo, row_hi + 1))
    assignments: dict[int, int] = {}
    far_cols = random.sample(source_cols, 1)
    for col in source_cols:
        if col in far_cols and row_hi >= guide_row + 3:
            assignments[col] = random.randint(guide_row + 3, row_hi)
        else:
            assignments[col] = random.choice(rows)
    return assignments


def _build_grid(
    height: int,
    width: int,
    guide_row: int,
    wall_segments: list[tuple[int, int, int]],
    success_cols: list[int],
    gap_cols: list[int],
    gap_source_cols: list[int],
    source_rows: dict[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[guide_row, GUIDE_MARGIN : width - GUIDE_MARGIN] = ORANGE
    for c in gap_cols:
        grid[guide_row, c] = BG

    for row, c0, c1 in wall_segments:
        grid[row, c0 : c1 + 1] = GREY

    for c in success_cols + gap_source_cols:
        grid[source_rows[c], c] = RED

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        full, full_stats = _solve_core(
            grid,
            filter_projection=True,
            start_mode="guide",
            enable_branching=True,
            emit_rays=True,
        )
        no_branch = _solve_core(
            grid,
            filter_projection=True,
            start_mode="guide",
            enable_branching=False,
            emit_rays=True,
        )[0]
        no_filter = _solve_core(
            grid,
            filter_projection=False,
            start_mode="guide",
            enable_branching=True,
            emit_rays=True,
        )[0]
        source_start = _solve_core(
            grid,
            filter_projection=True,
            start_mode="source",
            enable_branching=True,
            emit_rays=True,
        )[0]
        projection_only = _solve_core(
            grid,
            filter_projection=True,
            start_mode="guide",
            enable_branching=True,
            emit_rays=False,
        )[0]
    except AssertionError:
        return False

    guide_row = int(full_stats["guide_row"])
    successful = list(full_stats["successful"])
    failed = list(full_stats["failed"])

    if len(successful) < 2 or len(failed) < 1:
        return False
    if full_stats["wall_hits"] < 1 or full_stats["branch_count"] < 1:
        return False
    if full_stats["max_depth"] < 1:
        return False
    if not any(r >= guide_row + 3 for r, _c in successful):
        return False

    if np.array_equal(full, grid):
        return False
    if np.array_equal(full, no_branch):
        return False
    if np.array_equal(full, no_filter):
        return False
    if np.array_equal(full, source_start):
        return False
    if np.array_equal(full, projection_only):
        return False

    if len(full_stats["seen_starts"]) <= len(full_stats["starts"]):
        return False
    if not full_stats["painted"]:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    guide_row = random.randint(GUIDE_ROW_RANGE[0], min(GUIDE_ROW_RANGE[1], height - 12))

    layout = _pick_chain_layout(height, width, guide_row)
    if layout is None:
        return None
    wall_segments, chain_cols = layout
    wall_segments.sort()

    success_cols = _pick_success_columns(width, wall_segments, chain_cols)
    if success_cols is None:
        return None

    gap_info = _pick_gap_columns(width, success_cols)
    if gap_info is None:
        return None
    gap_cols, gap_source_cols = gap_info

    all_source_cols = sorted(success_cols + gap_source_cols)
    if not SOURCE_COUNT_RANGE[0] <= len(all_source_cols) <= SOURCE_COUNT_RANGE[1]:
        return None

    source_rows = _assign_source_rows(guide_row, wall_segments[0][0], all_source_cols)
    if source_rows is None:
        return None

    grid = _build_grid(
        height,
        width,
        guide_row,
        wall_segments,
        success_cols,
        gap_cols,
        gap_source_cols,
        source_rows,
    )

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
