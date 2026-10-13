from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _full_separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]


def _full_separator_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _intervals(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *cuts, length]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def _locate_marker(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == RED)
    border = [
        (int(r), int(c))
        for r, c in red
        if r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1
    ]
    assert len(border) == 1, "expected exactly one border marker"
    return border[0]


def _incoming_direction(shape: tuple[int, int], marker: tuple[int, int]) -> tuple[int, int]:
    r, c = marker
    h, w = shape
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise AssertionError("marker must lie on the border")


def _segment_through(
    base: np.ndarray, hit: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = hit
    dr, dc = direction
    if dr != 0:
        c0 = c
        while c0 - 1 >= 0 and base[r, c0 - 1] == BLUE:
            c0 -= 1
        c1 = c
        while c1 + 1 < base.shape[1] and base[r, c1 + 1] == BLUE:
            c1 += 1
        return [(r, cc) for cc in range(c0, c1 + 1)]

    r0 = r
    while r0 - 1 >= 0 and base[r0 - 1, c] == BLUE:
        r0 -= 1
    r1 = r
    while r1 + 1 < base.shape[0] and base[r1 + 1, c] == BLUE:
        r1 += 1
    return [(rr, c) for rr in range(r0, r1 + 1)]


def _near_endpoint(
    segment: list[tuple[int, int]], hit: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    dr, dc = direction
    if dr != 0:
        hit_c = hit[1]
        return min(segment[0], segment[-1], key=lambda p: (abs(p[1] - hit_c), p[1]))
    hit_r = hit[0]
    return min(segment[0], segment[-1], key=lambda p: (abs(p[0] - hit_r), p[0]))


def _segment_openings(
    base: np.ndarray, segment: list[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    out: list[tuple[int, int]] = []
    if dr != 0:
        left = (segment[0][0], segment[0][1] - 1)
        right = (segment[-1][0], segment[-1][1] + 1)
        for rr, cc in (left, right):
            if _in_bounds(base.shape, rr, cc) and base[rr, cc] not in (BLUE, GREY):
                out.append((rr, cc))
        return out

    top = (segment[0][0] - 1, segment[0][1])
    bottom = (segment[-1][0] + 1, segment[-1][1])
    for rr, cc in (top, bottom):
        if _in_bounds(base.shape, rr, cc) and base[rr, cc] not in (BLUE, GREY):
            out.append((rr, cc))
    return out


def _count_regions(
    grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    row_ranges = _intervals(grid.shape[0], sep_rows)
    col_ranges = _intervals(grid.shape[1], sep_cols)
    counts: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            counts.append(int(np.count_nonzero(patch == RED)))
            boxes.append((r0, c0, r1, c1))
    return counts, boxes


def _fill_color(count: int) -> int:
    if count == 0:
        return BLACK
    if count <= 3:
        return GREEN
    if count <= 6:
        return YELLOW
    return ORANGE


def _transform(grid: np.ndarray, *, with_meta: bool = False) -> Any:
    base = np.asarray(grid, dtype=int)
    assert base.ndim == 2, "expected a 2D grid"

    sep_rows = _full_separator_rows(base)
    sep_cols = _full_separator_cols(base)
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected 2 full grey rows and 2 full grey cols"

    marker = _locate_marker(base)
    direction = _incoming_direction(base.shape, marker)

    out = base.copy()
    stack = [marker]
    started = {marker}
    wall_hits = 0
    split_events = 0
    blocked_events = 0
    spawned_total = 0

    while stack:
        r, c = stack.pop()
        if base[r, c] not in (BLUE, GREY):
            out[r, c] = RED

        while True:
            nr, nc = r + direction[0], c + direction[1]
            if not _in_bounds(base.shape, nr, nc):
                break

            cell = int(base[nr, nc])
            if cell == GREY:
                r, c = nr, nc
                continue
            if cell == BLUE:
                segment = _segment_through(base, (nr, nc), direction)
                near = _near_endpoint(segment, (nr, nc), direction)
                out[near] = RED
                openings = _segment_openings(base, segment, direction)
                spawned_here = 0
                for start in openings:
                    if start not in started:
                        started.add(start)
                        stack.append(start)
                        if base[start] not in (BLUE, GREY):
                            out[start] = RED
                        spawned_here += 1
                wall_hits += 1
                spawned_total += spawned_here
                if spawned_here >= 2:
                    split_events += 1
                elif spawned_here == 0:
                    blocked_events += 1
                break

            out[nr, nc] = RED
            r, c = nr, nc

    counts, boxes = _count_regions(out, sep_rows, sep_cols)
    final = out.copy()
    for count, (r0, c0, r1, c1) in zip(counts, boxes):
        fill = _fill_color(count)
        if fill == BLACK:
            continue
        patch = final[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = fill

    if with_meta:
        return final, {
            "counts": counts,
            "wall_hits": wall_hits,
            "split_events": split_events,
            "blocked_events": blocked_events,
            "spawned_total": spawned_total,
            "sep_rows": sep_rows,
            "sep_cols": sep_cols,
        }
    return final


def main(grid: np.ndarray) -> np.ndarray:
    """
    Branch a border-launched ray through transparent partitions and recolor each region by local ray coverage.

    Input structure:
    - The grid has a black background, exactly two full grey horizontal separators,
      and exactly two full grey vertical separators, forming a 3x3 partition.
    - Exactly one red border marker sits on a non-separator edge cell.
    - Blue wall segments occupy non-separator cells inside the regions.

    Rule:
    - Launch a ray inward from the border marker along the perpendicular-to-edge direction.
    - The ray paints traversed non-grey cells red; full grey separator lines are transparent and keep their grey color.
    - When the ray meets a blue wall segment, paint the wall's near endpoint red, then spawn one new ray from each open segment end, continuing in the same direction.
    - After all rays terminate, count red pixels independently inside each of the 9 regions, ignoring separator cells, and recolor only black cells in that region:
      0 stays black, 1-3 becomes green, 4-6 becomes yellow, and 7+ becomes orange.

    Key insight:
    - The grey lines do not block the ray but do split the counting into independent regions, so wall-triggered branching changes later region colors without altering the partition structure.
    """
    return _transform(grid, with_meta=False)


def _sample_partition_sizes() -> list[int]:
    return [random.randint(3, 5) for _ in range(3)]


def _segment_candidates_for_column(
    col: int, left: int, right: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for length in range(1, 4):
        start_lo = max(left, col - length + 1)
        start_hi = min(col, right - length + 1)
        for c0 in range(start_lo, start_hi + 1):
            c1 = c0 + length - 1
            openings: list[int] = []
            if c0 - 1 >= left:
                openings.append(c0 - 1)
            if c1 + 1 <= right:
                openings.append(c1 + 1)
            kind = {0: "block", 1: "split1", 2: "split2"}[len(openings)]
            candidates.append({"c0": c0, "c1": c1, "openings": openings, "kind": kind})
    return candidates


def _region_bounds_for_col(col: int, sep_cols: list[int], width: int) -> tuple[int, int]:
    cuts = [-1, *sep_cols, width]
    for left_cut, right_cut in zip(cuts, cuts[1:]):
        if left_cut < col < right_cut:
            return left_cut + 1, right_cut - 1
    raise AssertionError("column must be inside a region")


def _choose_marker_col(sep_cols: list[int], width: int) -> int:
    usable = [c for c in range(width) if c not in sep_cols]
    middle_left = sep_cols[0] + 1
    middle_right = sep_cols[1] - 1
    middle = [c for c in usable if middle_left <= c <= middle_right]
    if middle and random.random() < 0.65:
        return random.choice(middle)
    return random.choice(usable)


def _choose_event_row(rows: list[int], used: set[int]) -> int | None:
    options = [r for r in rows if r not in used]
    if not options:
        return None
    return random.choice(options)


def _pick_candidate(
    active: set[int],
    branch_col: int,
    sep_cols: list[int],
    width: int,
    preferred: list[str],
) -> dict[str, Any] | None:
    left, right = _region_bounds_for_col(branch_col, sep_cols, width)
    candidates = _segment_candidates_for_column(branch_col, left, right)
    by_kind = {kind: [] for kind in ("split2", "split1", "block")}
    for cand in candidates:
        new_cols = set(cand["openings"])
        if new_cols & (active - {branch_col}):
            continue
        score = (len(new_cols - active), -abs((cand["c0"] + cand["c1"]) / 2 - branch_col))
        by_kind[cand["kind"]].append((score, cand))

    for kind in preferred:
        if by_kind[kind]:
            by_kind[kind].sort(reverse=True, key=lambda item: item[0])
            top = [cand for _score, cand in by_kind[kind][:4]]
            return random.choice(top)
    return None


def _build_canonical_downward() -> np.ndarray | None:
    row_sizes = _sample_partition_sizes()
    col_sizes = _sample_partition_sizes()
    h = sum(row_sizes) + 2
    w = sum(col_sizes) + 2

    sep_rows = [row_sizes[0], row_sizes[0] + 1 + row_sizes[1]]
    sep_cols = [col_sizes[0], col_sizes[0] + 1 + col_sizes[1]]
    row_bands = _intervals(h, sep_rows)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[sep_rows[0], :] = GREY
    grid[sep_rows[1], :] = GREY
    grid[:, sep_cols[0]] = GREY
    grid[:, sep_cols[1]] = GREY

    marker_col = _choose_marker_col(sep_cols, w)
    grid[0, marker_col] = RED

    active = {marker_col}
    used_rows: set[int] = set()

    # Mandatory early branch if possible.
    row0_choices = list(range(row_bands[0][0] + 1, row_bands[0][1] + 1)) or [row_bands[0][0]]
    first_row = _choose_event_row(row0_choices, used_rows)
    if first_row is None:
        return None
    first = _pick_candidate(active, marker_col, sep_cols, w, ["split2", "split1", "block"])
    if first is None:
        return None
    grid[first_row, first["c0"] : first["c1"] + 1] = BLUE
    used_rows.add(first_row)
    active.remove(marker_col)
    active.update(first["openings"])

    # Middle-band event, biased toward another split.
    if active:
        second_row = _choose_event_row(list(range(row_bands[1][0], row_bands[1][1] + 1)), used_rows)
        if second_row is not None:
            branch_col = random.choice(sorted(active))
            second = _pick_candidate(
                active,
                branch_col,
                sep_cols,
                w,
                ["split2", "split1", "block"] if len(active) > 1 else ["split2", "split1"],
            )
            if second is not None:
                grid[second_row, second["c0"] : second["c1"] + 1] = BLUE
                used_rows.add(second_row)
                active.remove(branch_col)
                active.update(second["openings"])

    # Bottom-band event, often used to block or trim one branch.
    if active and random.random() < 0.85:
        third_row = _choose_event_row(list(range(row_bands[2][0], row_bands[2][1] + 1)), used_rows)
        if third_row is not None:
            branch_col = random.choice(sorted(active))
            preferred = ["block", "split1", "split2"] if len(active) > 1 else ["split1", "split2", "block"]
            third = _pick_candidate(active, branch_col, sep_cols, w, preferred)
            if third is not None:
                grid[third_row, third["c0"] : third["c1"] + 1] = BLUE
                active.remove(branch_col)
                active.update(third["openings"])

    # Optional extra wall for variety.
    if active and random.random() < 0.55:
        band_index = random.choice([1, 2])
        extra_row = _choose_event_row(list(range(row_bands[band_index][0], row_bands[band_index][1] + 1)), used_rows)
        if extra_row is not None:
            branch_col = random.choice(sorted(active))
            extra = _pick_candidate(active, branch_col, sep_cols, w, ["split1", "split2", "block"])
            if extra is not None:
                grid[extra_row, extra["c0"] : extra["c1"] + 1] = BLUE

    return grid


def _accept_example(inp: np.ndarray) -> bool:
    try:
        out, meta = _transform(inp, with_meta=True)
    except AssertionError:
        return False

    counts = meta["counts"]
    nonzero_regions = sum(count > 0 for count in counts)
    bucket_count = len({_fill_color(count) for count in counts})
    non_separator_cells = inp.size - len(meta["sep_rows"]) * inp.shape[1] - len(meta["sep_cols"]) * inp.shape[0] + len(meta["sep_rows"]) * len(meta["sep_cols"])
    red_total = int(np.count_nonzero(out == RED))
    fill_colors = {int(v) for v in np.unique(out)} & {GREEN, YELLOW, ORANGE}

    if meta["wall_hits"] < 2:
        return False
    if meta["split_events"] < 1:
        return False
    if nonzero_regions < 3:
        return False
    if bucket_count < 3:
        return False
    if not fill_colors:
        return False
    if red_total < 8 or red_total > int(non_separator_cells * 0.6):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        base = _build_canonical_downward()
        if base is None:
            continue
        rotated = np.rot90(base, k=random.randrange(4)).copy()
        if _accept_example(rotated):
            return rotated
    raise RuntimeError("failed to generate a valid branching-ray partition input")
