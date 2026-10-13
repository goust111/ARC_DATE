from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

LEGEND_PALETTE = (BLUE, RED, GREEN, YELLOW, MAGENTA, CYAN, MAROON)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _parse_legend_line(line: np.ndarray) -> list[int] | None:
    non_black = np.flatnonzero(line != BLACK)
    if non_black.size == 0:
        return None

    start = int(non_black[0])
    end = int(non_black[-1])
    segment = np.asarray(line[start : end + 1], dtype=int)
    if segment.size % 2 == 0:
        return None
    if np.any(segment == ORANGE):
        return None

    colors: list[int] = []
    for idx, value in enumerate(segment):
        value = int(value)
        if idx % 2 == 0:
            if value in (BLACK, GRAY, ORANGE):
                return None
            colors.append(value)
        elif value != GRAY:
            return None

    if len(set(colors)) != len(colors):
        return None
    return colors


def _find_legend_colors(grid: np.ndarray) -> list[int]:
    candidates: list[list[int]] = []
    for line in (grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]):
        parsed = _parse_legend_line(np.asarray(line, dtype=int))
        if parsed is not None:
            candidates.append(parsed)
    assert len(candidates) == 1, "expected exactly one edge legend"
    return candidates[0]


def _find_towers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    towers: list[dict[str, object]] = []
    for coords in _components(g == ORANGE):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        assert coords.shape[0] == 9, "each tower must have area 9"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each tower must be a 3x3 square"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ORANGE), "tower must be solid orange"
        towers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 1, c0 + 1),
                "coords": coords,
            }
        )

    assert towers, "expected at least one orange tower"
    centers = [tower["center"] for tower in towers]
    for i, (r0, c0) in enumerate(centers):
        for r1, c1 in centers[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) >= 4, "towers must not touch or overlap"
    return towers


def _tower_index_grid(towers: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    idx_grid = np.full(shape, -1, dtype=int)
    for idx, tower in enumerate(towers):
        coords = tower["coords"]
        idx_grid[coords[:, 0], coords[:, 1]] = idx
    return idx_grid


def _ray_hit_counts(towers: list[dict[str, object]], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    idx_grid = _tower_index_grid(towers, shape)
    counts = [0] * len(towers)
    for src_idx, tower in enumerate(towers):
        r0, c0 = tower["center"]
        for dr, dc in DIRS4:
            r, c = r0, c0
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                hit_idx = int(idx_grid[r, c])
                if hit_idx == src_idx:
                    continue
                if hit_idx >= 0:
                    counts[hit_idx] += 1
                    break
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each signal tower by the number of unobstructed tower-to-tower rays that hit it.

    Input structure:
    - The background is black.
    - The main scene contains several non-touching solid orange 3x3 squares that act as signal towers.
    - Exactly one outer edge row or column contains a legend made of distinct single-pixel colors separated by gray pixels.

    Rule:
    - Treat the center of every orange 3x3 square as a ray source and cast rays up, down, left, and right.
    - Each ray travels cell by cell until it reaches another tower pixel or leaves the grid; only the first tower it meets is hit.
    - Count how many incoming rays hit each tower.
    - Towers with zero hits stay fully orange; otherwise recolor the 8-cell outer ring of the tower with the legend color for that hit count, while keeping the center orange.
    - The legend itself stays unchanged.

    Key insight:
    - The towers interact through line-of-sight blocking, so the visible answer depends on counting only the first aligned tower reached in each direction and then mapping that count through the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors = _find_legend_colors(g)
    towers = _find_towers(g)
    hit_counts = _ray_hit_counts(towers, g.shape)

    out = g.copy()
    for tower, hits in zip(towers, hit_counts):
        if hits == 0:
            continue
        assert hits <= len(legend_colors), "legend must cover every observed hit count"
        outer_color = int(legend_colors[hits - 1])
        r0, c0, r1, c1 = tower["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = outer_color
        cr, cc = tower["center"]
        out[cr, cc] = ORANGE
    return out


# --------------- generator ---------------

MAX_ATTEMPTS = 500
CENTER_STEP = 4
COUNT_RANGE = (4, 6)
GAP_RANGE = (1, 3)
EXTRA_TOWERS_MAX = 3


def _sample_centers(reserved_before: int, reserved_after: int, count: int) -> tuple[list[int], int] | None:
    gap_before = random.randint(*GAP_RANGE)
    gap_after = random.randint(*GAP_RANGE)
    first = reserved_before + gap_before + 1
    centers = [first + CENTER_STEP * i for i in range(count)]
    size = centers[-1] + 2 + gap_after + reserved_after
    if size > 30:
        return None
    return centers, size


def _build_lattice(side: str) -> tuple[list[int], list[int], tuple[int, int]] | None:
    row_count = random.randint(*COUNT_RANGE)
    col_count = random.randint(*COUNT_RANGE)

    top_reserved = 1 if side == "top" else 0
    bottom_reserved = 1 if side == "bottom" else 0
    left_reserved = 1 if side == "left" else 0
    right_reserved = 1 if side == "right" else 0

    row_result = _sample_centers(top_reserved, bottom_reserved, row_count)
    col_result = _sample_centers(left_reserved, right_reserved, col_count)
    if row_result is None or col_result is None:
        return None
    rows, h = row_result
    cols, w = col_result
    return rows, cols, (h, w)


def _visibility_counts(points: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    counts = {point: 0 for point in points}
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in points:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    for r, cols in by_row.items():
        cols = sorted(cols)
        for c0, c1 in zip(cols, cols[1:]):
            counts[(r, c0)] += 1
            counts[(r, c1)] += 1

    for c, rows in by_col.items():
        rows = sorted(rows)
        for r0, r1 in zip(rows, rows[1:]):
            counts[(r0, c)] += 1
            counts[(r1, c)] += 1
    return counts


def _peak_four(row_count: int, col_count: int) -> set[tuple[int, int]]:
    ri = random.randint(1, row_count - 2)
    ci = random.randint(1, col_count - 2)
    return {(ri, ci), (ri - 1, ci), (ri + 1, ci), (ri, ci - 1), (ri, ci + 1)}


def _peak_three(row_count: int, col_count: int) -> set[tuple[int, int]]:
    ri = random.randint(1, row_count - 2)
    ci = random.randint(1, col_count - 2)
    orientation = random.choice(("up", "down", "left", "right"))
    points = {(ri, ci)}
    if orientation == "up":
        points |= {(ri - 1, ci), (ri, ci - 1), (ri, ci + 1)}
    elif orientation == "down":
        points |= {(ri + 1, ci), (ri, ci - 1), (ri, ci + 1)}
    elif orientation == "left":
        points |= {(ri, ci - 1), (ri - 1, ci), (ri + 1, ci)}
    else:
        points |= {(ri, ci + 1), (ri - 1, ci), (ri + 1, ci)}
    return points


def _peak_two(row_count: int, col_count: int) -> set[tuple[int, int]]:
    if random.random() < 0.5:
        if random.random() < 0.5:
            ri = random.randint(0, row_count - 1)
            ci = random.randint(1, col_count - 2)
            return {(ri, ci - 1), (ri, ci), (ri, ci + 1)}
        ri = random.randint(1, row_count - 2)
        ci = random.randint(0, col_count - 1)
        return {(ri - 1, ci), (ri, ci), (ri + 1, ci)}

    ri = random.randint(1, row_count - 2)
    ci = random.randint(1, col_count - 2)
    orientation = random.choice(("ul", "ur", "dl", "dr"))
    if orientation == "ul":
        return {(ri, ci), (ri - 1, ci), (ri, ci - 1)}
    if orientation == "ur":
        return {(ri, ci), (ri - 1, ci), (ri, ci + 1)}
    if orientation == "dl":
        return {(ri, ci), (ri + 1, ci), (ri, ci - 1)}
    return {(ri, ci), (ri + 1, ci), (ri, ci + 1)}


def _peak_one(row_count: int, col_count: int) -> set[tuple[int, int]]:
    if random.random() < 0.5:
        ri = random.randint(0, row_count - 1)
        ci = random.randint(0, col_count - 2)
        return {(ri, ci), (ri, ci + 1)}
    ri = random.randint(0, row_count - 2)
    ci = random.randint(0, col_count - 1)
    return {(ri, ci), (ri + 1, ci)}


def _base_points(row_count: int, col_count: int, required_peak: int) -> set[tuple[int, int]]:
    if required_peak == 4:
        return _peak_four(row_count, col_count)
    if required_peak == 3:
        return _peak_three(row_count, col_count)
    if required_peak == 2:
        return _peak_two(row_count, col_count)
    return _peak_one(row_count, col_count)


def _isolated_candidates(
    points: set[tuple[int, int]],
    row_count: int,
    col_count: int,
) -> list[tuple[int, int]]:
    used_rows = {r for r, _ in points}
    used_cols = {c for _, c in points}
    return [
        (r, c)
        for r in range(row_count)
        for c in range(col_count)
        if (r, c) not in points and r not in used_rows and c not in used_cols
    ]


def _counts_good(counts: list[int], required_peak: int, legend_len: int) -> bool:
    if not counts:
        return False
    return (
        any(value == 0 for value in counts)
        and sum(value > 0 for value in counts) >= 2
        and required_peak <= max(counts) <= legend_len
    )


def _add_extra_towers(
    points: set[tuple[int, int]],
    row_count: int,
    col_count: int,
    required_peak: int,
    legend_len: int,
) -> set[tuple[int, int]]:
    current = set(points)
    all_cells = [(r, c) for r in range(row_count) for c in range(col_count) if (r, c) not in current]
    random.shuffle(all_cells)
    target_extra = random.randint(0, min(EXTRA_TOWERS_MAX, len(all_cells)))
    added = 0
    for cell in all_cells:
        if added >= target_extra:
            break
        trial = set(current)
        trial.add(cell)
        trial_counts = list(_visibility_counts(trial).values())
        if _counts_good(trial_counts, required_peak, legend_len):
            current = trial
            added += 1
    return current


def _place_legend(grid: np.ndarray, side: str, colors: list[int]) -> None:
    sequence: list[int] = []
    for idx, color in enumerate(colors):
        if idx > 0:
            sequence.append(GRAY)
        sequence.append(int(color))

    length = len(sequence)
    if side in ("top", "bottom"):
        start = random.randint(0, grid.shape[1] - length)
        row = 0 if side == "top" else grid.shape[0] - 1
        grid[row, start : start + length] = np.asarray(sequence, dtype=int)
    else:
        start = random.randint(0, grid.shape[0] - length)
        col = 0 if side == "left" else grid.shape[1] - 1
        grid[start : start + length, col] = np.asarray(sequence, dtype=int)


def _render_input(
    rows: list[int],
    cols: list[int],
    shape: tuple[int, int],
    lattice_points: set[tuple[int, int]],
    side: str,
    legend_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for ri, ci in lattice_points:
        r = rows[ri]
        c = cols[ci]
        grid[r - 1 : r + 2, c - 1 : c + 2] = ORANGE
    _place_legend(grid, side, legend_colors)
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(("top", "bottom", "left", "right"))
    required_peak = random.randint(1, 4)
    legend_len = random.randint(max(2, required_peak), 4)

    lattice = _build_lattice(side)
    if lattice is None:
        return None
    rows, cols, shape = lattice
    row_count = len(rows)
    col_count = len(cols)

    points = _base_points(row_count, col_count, required_peak)
    isolated = _isolated_candidates(points, row_count, col_count)
    if not isolated:
        return None
    points.add(random.choice(isolated))
    points = _add_extra_towers(points, row_count, col_count, required_peak, legend_len)

    legend_colors = random.sample(list(LEGEND_PALETTE), legend_len)
    grid = _render_input(rows, cols, shape, points, side, legend_colors)

    towers = _find_towers(grid)
    counts = _ray_hit_counts(towers, grid.shape)
    if not _counts_good(counts, required_peak, legend_len):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
