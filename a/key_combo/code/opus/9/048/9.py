from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MARKER_PALETTE = (1, 2, 3, 6, 7, 8, 9)
GEN_RETRIES = 500


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _snake_positions(side: int) -> list[tuple[int, int]]:
    pos: list[tuple[int, int]] = []
    for rr in range(int(side)):
        cols = range(int(side)) if rr % 2 == 0 else range(int(side) - 1, -1, -1)
        for cc in cols:
            pos.append((int(rr), int(cc)))
    return pos


def _marker_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != YELLOW) & (grid != GREY)


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in np.argwhere(_marker_mask(grid))]
    coords.sort(key=lambda item: item[1])
    return coords


def _snake_fill(side: int, sequence: list[int]) -> np.ndarray:
    patch = np.zeros((int(side), int(side)), dtype=int)
    for color, (rr, cc) in zip(sequence, _snake_positions(side)):
        patch[int(rr), int(cc)] = int(color)
    return patch


def _row_major_fill(side: int, sequence: list[int]) -> np.ndarray:
    patch = np.zeros((int(side), int(side)), dtype=int)
    for idx, color in enumerate(sequence):
        rr, cc = divmod(int(idx), int(side))
        patch[int(rr), int(cc)] = int(color)
    return patch


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(g.min()) >= 0 and int(g.max()) <= 9

    centers = np.argwhere(g == YELLOW)
    blockers = np.argwhere(g == GREY)
    markers = np.argwhere(_marker_mask(g))

    assert len(centers) == 1, "expected exactly one yellow center"
    assert len(blockers) >= 1, "expected at least one grey blocker"
    assert 4 <= len(markers) <= 9, "expected 4..9 scattered markers"
    assert len({int(c) for _r, c in markers}) == len(markers), "markers must occupy distinct columns"

    center = tuple(map(int, centers[0]))
    blocker_list = [tuple(map(int, p)) for p in blockers]
    d = min(_chebyshev(center, p) for p in blocker_list)
    radius = int(d - 1)
    side = int(2 * radius + 1)
    assert radius >= 1

    r0 = center[0] - radius
    c0 = center[1] - radius
    r1 = center[0] + radius
    c1 = center[1] + radius
    assert 0 <= r0 <= r1 < g.shape[0]
    assert 0 <= c0 <= c1 < g.shape[1]

    square_mask = np.zeros_like(g, dtype=bool)
    square_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not bool(np.any((g == GREY) & square_mask)), "grey blockers cannot lie inside the fill square"
    assert not bool(np.any(_marker_mask(g) & square_mask)), "scattered markers must stay outside the fill square"

    ordered_markers = _sorted_markers(g)
    sequence = [color for _r, _c, color in ordered_markers]
    assert len(sequence) <= side * side, "square must have enough capacity"
    assert len(sequence) > side, "sequence must reach at least the second snake row"

    return {
        "center": center,
        "blockers": blocker_list,
        "radius": radius,
        "side": side,
        "square_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "ordered_markers": ordered_markers,
        "sequence": sequence,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Compress column-sorted scattered markers into a snake-filled square sized by the nearest grey blocker.

    Input structure:
    - A black background contains exactly one yellow center pixel, one or more grey blocker pixels, and 4 to 9
      scattered singleton markers of other colors.
    - Each scattered marker occupies a distinct input column, and the centered target square fits fully inside
      the grid.

    Rule:
    - Compute the minimum Chebyshev distance from the yellow center to any grey blocker; the target square radius
      is one less than that distance.
    - Sort the scattered markers by column from left to right, erase their original positions, and write that
      color sequence into the centered square in boustrophedon order: top row left-to-right, next row right-to-left,
      then alternating until the sequence ends.
    - Leave the remaining square cells black, preserve every grey blocker, and allow the square contents to cover
      the original yellow center.

    Key insight:
    - Two independent clues must be combined: the nearest-blocker Chebyshev distance fixes the square size, while
      the markers' left-to-right column order fixes the snake-filled color pattern placed inside that square.
    """
    info = _analyze_input(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    marker_coords = np.argwhere(_marker_mask(g))
    out[marker_coords[:, 0], marker_coords[:, 1]] = BG

    r0, c0, r1, c1 = info["square_bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for color, (dr, dc) in zip(info["sequence"], _snake_positions(info["side"])):
        out[r0 + dr, c0 + dc] = int(color)

    for br, bc in info["blockers"]:
        out[int(br), int(bc)] = GREY
    return out


def _ring_cells(shape: tuple[int, int], center: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    cells: list[tuple[int, int]] = []
    for r in range(cr - dist, cr + dist + 1):
        for c in range(cc - dist, cc + dist + 1):
            if 0 <= r < h and 0 <= c < w and _chebyshev(center, (r, c)) == int(dist):
                cells.append((int(r), int(c)))
    return cells


def _generate_sequence(side: int, n_markers: int) -> list[int] | None:
    second_row_start = side
    second_row_stop = min(2 * side, n_markers)
    for _ in range(200):
        if n_markers <= len(MARKER_PALETTE) and random.random() < 0.7:
            sequence = random.sample(list(MARKER_PALETTE), n_markers)
        else:
            distinct = random.randint(3, min(len(MARKER_PALETTE), n_markers))
            base = random.sample(list(MARKER_PALETTE), distinct)
            sequence = [random.choice(base) for _ in range(n_markers)]
            if len(set(sequence)) < 3:
                continue

        if len(set(sequence[second_row_start:second_row_stop])) < 2:
            continue
        if np.array_equal(_snake_fill(side, sequence), _row_major_fill(side, sequence)):
            continue
        return [int(x) for x in sequence]
    return None


def _place_blockers(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]] | None:
    dist = int(radius + 1)
    ring = _ring_cells(shape, center, dist)
    if not ring:
        return None

    axial = [p for p in ring if p[0] == center[0] or p[1] == center[1]]
    diagonal = [p for p in ring if p[0] != center[0] and p[1] != center[1]]
    mode = random.choices(
        ["diagonal_only", "axial_only", "mixed"],
        weights=[4, 3, 3],
        k=1,
    )[0]

    if mode == "diagonal_only" and diagonal:
        exact_pool = diagonal
    elif mode == "axial_only" and axial:
        exact_pool = axial
    else:
        exact_pool = ring

    exact_count = random.randint(1, min(3, len(exact_pool)))
    blockers = set(random.sample(exact_pool, exact_count))

    far_cells = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if _chebyshev(center, (r, c)) > dist and (r, c) not in blockers
    ]
    extra_count = random.randint(0, min(4, len(far_cells)))
    if extra_count:
        blockers.update(random.sample(far_cells, extra_count))
    return sorted(blockers)


def _candidate_marker_columns(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    occupied: set[tuple[int, int]],
) -> list[int]:
    h, w = map(int, shape)
    dist = int(radius + 1)
    cols: list[int] = []
    for c in range(w):
        if any((r, c) not in occupied and _chebyshev(center, (r, c)) > dist for r in range(h)):
            cols.append(int(c))
    return cols


def _sample_marker_rows(
    columns: list[int],
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    occupied: set[tuple[int, int]],
) -> list[int] | None:
    h, _w = map(int, shape)
    dist = int(radius + 1)
    top_rows = [r for r in range(h) if r < center[0] - dist]
    bottom_rows = [r for r in range(h) if r > center[0] + dist]
    valid_rows_by_col = {
        c: [r for r in range(h) if (r, c) not in occupied and _chebyshev(center, (r, c)) > dist]
        for c in columns
    }
    if any(not rows for rows in valid_rows_by_col.values()):
        return None

    for _ in range(300):
        mode = random.choices(["split", "free", "mostly_top", "mostly_bottom"], weights=[4, 3, 2, 2], k=1)[0]
        rows: list[int] = []
        for idx, c in enumerate(columns):
            choices = valid_rows_by_col[c]
            filtered = choices
            if mode == "split":
                if idx < len(columns) // 2 and top_rows:
                    pool = [r for r in choices if r in top_rows]
                    if pool:
                        filtered = pool
                elif idx >= len(columns) // 2 and bottom_rows:
                    pool = [r for r in choices if r in bottom_rows]
                    if pool:
                        filtered = pool
            elif mode == "mostly_top" and top_rows and random.random() < 0.75:
                pool = [r for r in choices if r in top_rows]
                if pool:
                    filtered = pool
            elif mode == "mostly_bottom" and bottom_rows and random.random() < 0.75:
                pool = [r for r in choices if r in bottom_rows]
                if pool:
                    filtered = pool
            rows.append(int(random.choice(filtered)))

        if len(set(rows)) < 2:
            continue
        deltas = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        if not any(delta > 0 for delta in deltas):
            continue
        if not any(delta < 0 for delta in deltas):
            continue
        return rows
    return None


def _place_markers(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    blockers: list[tuple[int, int]],
    n_markers: int,
    sequence: list[int],
) -> list[tuple[int, int, int]] | None:
    occupied = {center, *blockers}
    columns = _candidate_marker_columns(shape, center, radius, occupied)
    if len(columns) < n_markers:
        return None

    dist = int(radius + 1)
    left_cols = [c for c in columns if c < center[1] - dist]
    right_cols = [c for c in columns if c > center[1] + dist]

    for _ in range(300):
        if left_cols and right_cols and n_markers >= 5 and random.random() < 0.7:
            left_take = random.randint(1, min(len(left_cols), n_markers - 1))
            right_take = n_markers - left_take
            if right_take > len(right_cols):
                continue
            chosen_cols = sorted(random.sample(left_cols, left_take) + random.sample(right_cols, right_take))
        else:
            chosen_cols = sorted(random.sample(columns, n_markers))

        rows = _sample_marker_rows(chosen_cols, shape, center, radius, occupied)
        if rows is None:
            continue

        triplets = [(int(r), int(c), int(color)) for r, c, color in zip(rows, chosen_cols, sequence)]
        row_sorted_sequence = [color for _r, _c, color in sorted(triplets)]
        dist_sorted_sequence = [
            color for _r, _c, color in sorted(triplets, key=lambda item: (_chebyshev(center, (item[0], item[1])), item[1], item[0]))
        ]
        if row_sorted_sequence == sequence:
            continue
        if dist_sorted_sequence == sequence:
            continue
        return triplets
    return None


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return False

    g = np.asarray(grid, dtype=int)
    center = info["center"]
    radius = info["radius"]
    side = info["side"]
    sequence = info["sequence"]
    ordered_markers = info["ordered_markers"]
    blockers = info["blockers"]

    d = radius + 1
    if min(_chebyshev(center, p) for p in blockers) != d:
        return False
    if not any(_chebyshev(center, p) == d for p in blockers):
        return False
    if any(_chebyshev(center, p) <= radius for p in blockers):
        return False

    row_sorted_sequence = [color for _r, _c, color in sorted(ordered_markers)]
    dist_sorted_sequence = [
        color
        for _r, _c, color in sorted(
            ordered_markers,
            key=lambda item: (_chebyshev(center, (item[0], item[1])), item[1], item[0]),
        )
    ]
    if row_sorted_sequence == sequence or dist_sorted_sequence == sequence:
        return False

    snake_patch = _snake_fill(side, sequence)
    row_major_patch = _row_major_fill(side, sequence)
    if np.array_equal(snake_patch, row_major_patch):
        return False

    out = main(g)
    r0, c0, r1, c1 = info["square_bbox"]
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], snake_patch):
        return False

    marker_cells = np.zeros_like(g, dtype=bool)
    for r, c, _color in ordered_markers:
        marker_cells[int(r), int(c)] = True
    if np.any(out[marker_cells] != BG):
        return False

    if np.count_nonzero(out == GREY) != len(blockers):
        return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(13, 19)
    width = random.randint(13, 19)
    radius = random.choices([1, 2, 3], weights=[4, 4, 2], k=1)[0]
    side = int(2 * radius + 1)
    n_markers = random.randint(max(4, side + 1), min(9, side * side))

    dist = radius + 1
    center_rows = list(range(dist, height - dist))
    center_cols = list(range(dist, width - dist))
    if not center_rows or not center_cols:
        return None
    center = (random.choice(center_rows), random.choice(center_cols))

    blockers = _place_blockers((height, width), center, radius)
    if blockers is None:
        return None

    sequence = _generate_sequence(side, n_markers)
    if sequence is None:
        return None

    triplets = _place_markers((height, width), center, radius, blockers, n_markers, sequence)
    if triplets is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[int(center[0]), int(center[1])] = YELLOW
    for r, c in blockers:
        grid[int(r), int(c)] = GREY
    for r, c, color in triplets:
        grid[int(r), int(c)] = int(color)

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
