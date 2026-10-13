from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MASK = 6
RECT_COLORS = (1, 2, 3, 4, 5, 7)
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_valid_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if int(coords.shape[0]) != height * width:
                continue
            rectangles.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return rectangles


def _overlap_groups(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    group_items: list[list[int]] = []
    common_los: list[int] = []
    common_his: list[int] = []
    center_sums: list[float] = []
    counts: list[int] = []

    for idx, lo, hi in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        matches = [
            group_idx
            for group_idx in range(len(group_items))
            if max(common_los[group_idx], lo) <= min(common_his[group_idx], hi)
        ]
        assert len(matches) <= 1, "each interval should fit at most one band"
        if matches:
            group_idx = matches[0]
            group_items[group_idx].append(idx)
            common_los[group_idx] = max(common_los[group_idx], lo)
            common_his[group_idx] = min(common_his[group_idx], hi)
            center_sums[group_idx] += 0.5 * (lo + hi)
            counts[group_idx] += 1
        else:
            group_items.append([idx])
            common_los.append(lo)
            common_his.append(hi)
            center_sums.append(0.5 * (lo + hi))
            counts.append(1)

    assert group_items, "expected at least one interval group"
    ordered = sorted(range(len(group_items)), key=lambda idx: center_sums[idx] / counts[idx])
    mapping: dict[int, int] = {}
    for rank, group_idx in enumerate(ordered):
        for idx in group_items[group_idx]:
            mapping[idx] = rank
    return mapping, len(group_items)


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(pts))


def _orbit_partition(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            orbit = _d4_orbit(r, c, n)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter a hidden rectangle lattice, compress it to a square color table, and fill pink slots from D4 symmetry.

    Input structure:
    - The background is black, with many solid monochrome rectangles plus a few tiny clutter components.
    - Every meaningful object is a 4-connected rectangle at least 3x3 that exactly fills its bounding box.
    - These valid rectangles form a hidden square lattice: same-row rectangles have overlapping row intervals, same-column rectangles have overlapping column intervals, and some lattice cells are pink masks.

    Rule:
    - Keep only the filled 3x3-or-larger rectangles and ignore every other component.
    - Group rectangle row-intervals and column-intervals by overlap to compress the scattered layout into an NxN color grid.
    - For each pink compressed cell, collect the cells in its orbit under the square's D4 symmetries and replace pink with the unique visible orbit color.
    - Write the recovered color back into the corresponding pink rectangle in the original grid, leaving every other pixel unchanged.

    Key insight:
    - The symmetry is invisible in the raw scene and only appears after filtering away clutter and collapsing the uneven rectangle layout into a compact square table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    rectangles = _find_valid_rectangles(g, bg)
    assert rectangles, "expected at least one valid rectangle"

    row_by_rect, row_count = _overlap_groups(
        [(idx, rect["bbox"][0], rect["bbox"][2]) for idx, rect in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, rect["bbox"][1], rect["bbox"][3]) for idx, rect in enumerate(rectangles)]
    )
    assert row_count == col_count, "compressed table must be square"
    assert len(rectangles) == row_count * col_count, "expected a complete square lattice"

    compact = np.full((row_count, col_count), -1, dtype=int)
    rect_by_cell: dict[tuple[int, int], dict] = {}
    for idx, rect in enumerate(rectangles):
        cell = (row_by_rect[idx], col_by_rect[idx])
        assert compact[cell] == -1, "expected one rectangle per compressed cell"
        compact[cell] = rect["color"]
        rect_by_cell[cell] = rect
    assert np.all(compact >= 0), "compressed lattice should be complete"

    filled = compact.copy()
    for r, c in zip(*np.where(compact == MASK)):
        orbit_colors = {
            int(compact[rr, cc])
            for rr, cc in _d4_orbit(int(r), int(c), row_count)
            if int(compact[rr, cc]) != MASK
        }
        assert len(orbit_colors) == 1, "each pink cell must have a unique visible D4 orbit color"
        filled[r, c] = orbit_colors.pop()

    out = g.copy()
    for (r, c), rect in rect_by_cell.items():
        if rect["color"] != MASK:
            continue
        coords = rect["coords"]
        out[coords[:, 0], coords[:, 1]] = int(filled[r, c])
    return out


def _generate_symmetric_table(rng: np.random.Generator, n: int) -> np.ndarray:
    orbits = _orbit_partition(n)
    for _ in range(100):
        active_count = int(rng.integers(3, len(RECT_COLORS) + 1))
        active_colors = rng.choice(RECT_COLORS, size=active_count, replace=False)
        table = np.zeros((n, n), dtype=int)
        orbit_colors = rng.choice(active_colors, size=len(orbits), replace=True)
        for color, orbit in zip(orbit_colors, orbits):
            for r, c in orbit:
                table[r, c] = int(color)
        if len(set(int(v) for v in table.flat)) >= 3:
            return table
    raise RuntimeError("failed to build a diverse symmetric table")


def _mask_table(
    rng: np.random.Generator, table: np.ndarray
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    n = table.shape[0]
    orbits = _orbit_partition(n)
    orbit_ids: dict[tuple[int, int], int] = {}
    for orbit_idx, orbit in enumerate(orbits):
        for cell in orbit:
            orbit_ids[cell] = orbit_idx

    for _ in range(100):
        masked = table.copy()
        target = int(rng.integers(3, 6))
        selected: set[tuple[int, int]] = set()
        counts = {idx: 0 for idx in range(len(orbits))}
        prefer_repeat = bool(rng.integers(0, 2))
        while len(selected) < target:
            candidates = [idx for idx, orbit in enumerate(orbits) if len(orbit) - counts[idx] >= 2]
            if not candidates:
                break
            reused = [idx for idx in candidates if counts[idx] > 0]
            fresh = [idx for idx in candidates if counts[idx] == 0]
            if prefer_repeat and reused and rng.random() < 0.5:
                orbit_idx = int(rng.choice(reused))
            elif fresh:
                orbit_idx = int(rng.choice(fresh))
            else:
                orbit_idx = int(rng.choice(candidates))

            choices = [cell for cell in orbits[orbit_idx] if cell not in selected]
            cell = choices[int(rng.integers(len(choices)))]
            selected.add(cell)
            counts[orbit_idx] += 1

        if len(selected) != target:
            continue
        if len({orbit_ids[cell] for cell in selected}) < 2:
            continue
        if len({int(table[cell]) for cell in selected}) < 2:
            continue

        for r, c in selected:
            masked[r, c] = MASK
        return masked, selected
    return None


def _sample_extents(
    rng: np.random.Generator, n: int
) -> tuple[int, list[tuple[int, int]]] | None:
    if n == 5:
        size_lo, size_hi = 3, 4
        gap_lo, gap_hi = 1, 2
        margin_hi = 1
    else:
        size_lo, size_hi = 3, 6
        gap_lo, gap_hi = 1, 3
        margin_hi = 2

    for _ in range(200):
        lengths = rng.integers(size_lo, size_hi + 1, size=n)
        gaps = rng.integers(gap_lo, gap_hi + 1, size=n - 1) if n > 1 else np.zeros(0, dtype=int)
        lead = int(rng.integers(0, margin_hi + 1))
        tail = int(rng.integers(0, margin_hi + 1))
        total = int(lengths.sum() + gaps.sum() + lead + tail)
        if total > 30:
            continue

        pos = lead
        extents: list[tuple[int, int]] = []
        for idx, length in enumerate(lengths):
            r0 = pos
            r1 = pos + int(length) - 1
            extents.append((r0, r1))
            pos = r1 + 1
            if idx < n - 1:
                pos += int(gaps[idx])
        return total, extents
    return None


def _sample_intervals_in_extent(
    rng: np.random.Generator, extent: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    lo, hi = extent
    span = hi - lo + 1
    common = int(rng.integers(lo, hi + 1))
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(3, span + 1))
        start_lo = max(lo, common - length + 1)
        start_hi = min(common, hi - length + 1)
        start = int(rng.integers(start_lo, start_hi + 1))
        intervals.append((start, start + length - 1))
    return intervals


def _can_place_single(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    h, w = grid.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
            return False
    return True


def _can_place_domino(grid: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _add_clutter(rng: np.random.Generator, grid: np.ndarray) -> bool:
    clutter_colors = rng.choice(RECT_COLORS, size=2, replace=False)
    target = int(rng.integers(8, 17))
    placed = 0
    for _ in range(target * 40):
        if placed >= target:
            break
        use_domino = bool(rng.integers(0, 2))
        color = int(rng.choice(clutter_colors))
        if use_domino:
            pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    for dr, dc in ((1, 0), (0, 1)):
                        pair = ((r, c), (r + dr, c + dc))
                        if _can_place_domino(grid, pair):
                            pairs.append(pair)
            if pairs:
                (r0, c0), (r1, c1) = pairs[int(rng.integers(len(pairs)))]
                grid[r0, c0] = color
                grid[r1, c1] = color
                placed += 1
                continue

        singles = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if _can_place_single(grid, r, c)]
        if not singles:
            continue
        r, c = singles[int(rng.integers(len(singles)))]
        grid[r, c] = color
        placed += 1
    return placed >= 5


def _expected_output(
    grid: np.ndarray,
    masked_table: np.ndarray,
    full_table: np.ndarray,
    row_intervals: list[list[tuple[int, int]]],
    col_intervals: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    n = masked_table.shape[0]
    for r in range(n):
        for c in range(n):
            if int(masked_table[r, c]) != MASK:
                continue
            r0, r1 = row_intervals[r][c]
            c0, c1 = col_intervals[c][r]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(full_table[r, c])
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice([4, 5]))
    full_table = _generate_symmetric_table(rng, n)
    masked_result = _mask_table(rng, full_table)
    if masked_result is None:
        return None
    masked_table, _masked_cells = masked_result

    row_spec = _sample_extents(rng, n)
    col_spec = _sample_extents(rng, n)
    if row_spec is None or col_spec is None:
        return None
    grid_h, row_extents = row_spec
    grid_w, col_extents = col_spec

    row_intervals = [_sample_intervals_in_extent(rng, row_extents[r], n) for r in range(n)]
    col_intervals = [_sample_intervals_in_extent(rng, col_extents[c], n) for c in range(n)]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r in range(n):
        for c in range(n):
            r0, r1 = row_intervals[r][c]
            c0, c1 = col_intervals[c][r]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(masked_table[r, c])

    if not _add_clutter(rng, grid):
        return None

    rectangles = _find_valid_rectangles(grid, BG)
    if len(rectangles) != n * n:
        return None
    expected = _expected_output(grid, masked_table, full_table, row_intervals, col_intervals)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
