from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8
BOUNDARY_COLORS = (RED, TEAL)

GRID_MIN = 15
GRID_MAX = 20
SEED_SIZES = (3, 4, 5)
GEN_ATTEMPTS = 500
CORRIDOR_COUNT_OPTIONS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)
CORNER_NAMES = ("tl", "tr", "bl", "br")


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_horizontal_corridors(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    width = int(grid.shape[1])
    for r in range(1, int(grid.shape[0]) - 1):
        if (
            int(grid[r, 0]) != BLACK
            and int(grid[r, width - 1]) != BLACK
            and np.all(grid[r, 1 : width - 1] == BLACK)
        ):
            rows.append(int(r))
    return rows


def _detect_vertical_corridors(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    height = int(grid.shape[0])
    for c in range(1, int(grid.shape[1]) - 1):
        if (
            int(grid[0, c]) != BLACK
            and int(grid[height - 1, c]) != BLACK
            and np.all(grid[1 : height - 1, c] == BLACK)
        ):
            cols.append(int(c))
    return cols


def _extract_seed(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    seed_mask = grid == YELLOW
    bbox = _bbox_from_mask(seed_mask)
    if bbox is None:
        raise AssertionError("expected a yellow seed sprite")
    r0, c0, r1, c1 = bbox
    seed = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    if seed.shape[0] != seed.shape[1]:
        raise AssertionError("expected the seed bbox to be square")
    return seed, bbox


def _stamp_seed(
    out: np.ndarray,
    seed: np.ndarray,
    center_r: int,
    center_c: int,
) -> None:
    height, width = map(int, seed.shape)
    top = int(center_r) - height // 2
    left = int(center_c) - width // 2
    for sr in range(height):
        for sc in range(width):
            if int(seed[sr, sc]) != YELLOW:
                continue
            rr = top + sr
            cc = left + sc
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                continue
            if rr in (0, out.shape[0] - 1) or cc in (0, out.shape[1] - 1):
                continue
            out[rr, cc] = YELLOW


def _build_expected_output(grid: np.ndarray, rows: Iterable[int], cols: Iterable[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    seed, _bbox = _extract_seed(out)
    out[out == YELLOW] = BLACK
    for r in rows:
        for c in cols:
            _stamp_seed(out, seed, int(r), int(c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect border-to-border corridors and stamp the yellow seed at every corridor intersection.

    Input structure:
    - A mostly black grid contains sparse red or teal boundary pixels on the outer border.
    - Some interior rows have non-black pixels at both side borders and only black cells between them, and some interior columns satisfy the analogous top-bottom condition.
    - One corner region contains a small yellow seed sprite on black background, away from the valid corridors.

    Rule:
    - Treat each interior row with colored pixels at both border ends and an all-black interior as a horizontal corridor.
    - Treat each interior column with colored pixels at both border ends and an all-black interior as a vertical corridor.
    - Extract the yellow seed by taking the minimal bounding box of all yellow pixels, erase the original seed, then stamp that template centered at every horizontal/vertical corridor intersection.
    - Yellow seed pixels overwrite interior cells, black seed pixels are transparent, border pixels stay unchanged, and any out-of-bounds part is clipped.

    Key insight:
    - The stamp locations are not given explicitly; they are the intersections of independently inferred horizontal and vertical clear corridors.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    corridor_rows = _detect_horizontal_corridors(g)
    corridor_cols = _detect_vertical_corridors(g)
    return _build_expected_output(g, corridor_rows, corridor_cols)


def _side_point(rng: np.random.Generator, size: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, int(rng.integers(0, size))
    if side == "bottom":
        return size - 1, int(rng.integers(0, size))
    if side == "left":
        return int(rng.integers(0, size)), 0
    return int(rng.integers(0, size)), size - 1


def _manhattan_path(
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = [(int(r), int(c))]
    row_first = bool(rng.integers(0, 2))
    for axis in (0, 1) if row_first else (1, 0):
        if axis == 0:
            step = 1 if tr >= r else -1
            while r != tr:
                r += step
                cells.append((int(r), int(c)))
        else:
            step = 1 if tc >= c else -1
            while c != tc:
                c += step
                cells.append((int(r), int(c)))
    return cells


def _make_seed_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
    cells = {center}
    for side in ("top", "bottom", "left", "right"):
        cells.update(_manhattan_path(center, _side_point(rng, size, side), rng))

    target_area = int(rng.integers(max(size + 1, len(cells)), size * size))
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                    frontier.append((int(rr), int(cc)))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    bbox = _bbox_from_mask(mask)
    if bbox != (0, 0, size - 1, size - 1):
        raise AssertionError("seed mask must use the full square bbox")
    return mask


def _choose_seed_bbox(
    height: int,
    width: int,
    size: int,
    corner: str,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    offset_r = int(rng.integers(1, 3))
    offset_c = int(rng.integers(1, 3))
    if corner == "tl":
        r0 = offset_r
        c0 = offset_c
    elif corner == "tr":
        r0 = offset_r
        c0 = width - size - offset_c
    elif corner == "bl":
        r0 = height - size - offset_r
        c0 = offset_c
    else:
        r0 = height - size - offset_r
        c0 = width - size - offset_c
    return int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)


def _subsets_with_spacing(
    candidates: list[int],
    count: int,
    min_gap: int,
    prefer_irregular: bool,
) -> list[tuple[int, ...]]:
    subsets = [
        combo
        for combo in itertools.combinations(candidates, count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
    ]
    if prefer_irregular and count >= 3:
        irregular = []
        for combo in subsets:
            gaps = [combo[i + 1] - combo[i] for i in range(len(combo) - 1)]
            if len(set(gaps)) > 1:
                irregular.append(combo)
        if irregular:
            return irregular
    return subsets


def _classify_row(grid: np.ndarray, r: int) -> str:
    left = int(grid[r, 0]) != BLACK
    right = int(grid[r, grid.shape[1] - 1]) != BLACK
    interior_has_fg = bool(np.any(grid[r, 1 : grid.shape[1] - 1] != BLACK))
    if left and right and not interior_has_fg:
        return "corridor"
    if left and right:
        return "blocked"
    if left ^ right:
        return "single"
    return "empty"


def _classify_col(grid: np.ndarray, c: int) -> str:
    top = int(grid[0, c]) != BLACK
    bottom = int(grid[grid.shape[0] - 1, c]) != BLACK
    interior_has_fg = bool(np.any(grid[1 : grid.shape[0] - 1, c] != BLACK))
    if top and bottom and not interior_has_fg:
        return "corridor"
    if top and bottom:
        return "blocked"
    if top ^ bottom:
        return "single"
    return "empty"


def _random_boundary_color(rng: np.random.Generator) -> int:
    return int(BOUNDARY_COLORS[int(rng.integers(0, len(BOUNDARY_COLORS)))])


def _force_both_boundary_colors(grid: np.ndarray, rng: np.random.Generator) -> None:
    boundary_cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)) and int(grid[r, c]) != BLACK
    ]
    present = {int(grid[r, c]) for r, c in boundary_cells}
    if RED in present and TEAL in present:
        return
    if not boundary_cells:
        raise AssertionError("expected some boundary clues")
    r, c = boundary_cells[int(rng.integers(0, len(boundary_cells)))]
    grid[r, c] = TEAL if RED in present else RED


def _allowed_blocker(
    r: int,
    c: int,
    corridor_rows: set[int],
    corridor_cols: set[int],
    forbidden_cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    return (
        1 <= r < height - 1
        and 1 <= c < width - 1
        and r not in corridor_rows
        and c not in corridor_cols
        and (r, c) not in forbidden_cells
    )


def _place_blockers(
    grid: np.ndarray,
    corridor_rows: set[int],
    corridor_cols: set[int],
    blocked_rows: list[int],
    blocked_cols: list[int],
    forbidden_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    height, width = map(int, grid.shape)
    blockers: set[tuple[int, int]] = set()
    row_needs = set(blocked_rows)
    col_needs = set(blocked_cols)

    joint = [
        (r, c)
        for r in blocked_rows
        for c in blocked_cols
        if _allowed_blocker(r, c, corridor_rows, corridor_cols, forbidden_cells, height, width)
    ]
    rng.shuffle(joint)
    for r, c in joint:
        if r in row_needs or c in col_needs:
            blockers.add((int(r), int(c)))
            row_needs.discard(int(r))
            col_needs.discard(int(c))

    for r in list(row_needs):
        candidates = [
            c
            for c in range(1, width - 1)
            if _allowed_blocker(r, c, corridor_rows, corridor_cols, forbidden_cells, height, width)
            and (r, c) not in blockers
        ]
        if not candidates:
            raise AssertionError("failed to place a required row blocker")
        c = int(candidates[int(rng.integers(0, len(candidates)))])
        blockers.add((int(r), int(c)))
        row_needs.discard(int(r))

    for c in list(col_needs):
        candidates = [
            r
            for r in range(1, height - 1)
            if _allowed_blocker(r, c, corridor_rows, corridor_cols, forbidden_cells, height, width)
            and (r, c) not in blockers
        ]
        if not candidates:
            raise AssertionError("failed to place a required column blocker")
        r = int(candidates[int(rng.integers(0, len(candidates)))])
        blockers.add((int(r), int(c)))
        col_needs.discard(int(c))

    return blockers


def _build_candidate_grid(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    size = int(SEED_SIZES[int(rng.integers(0, len(SEED_SIZES)))])
    radius = size // 2
    corner = CORNER_NAMES[int(rng.integers(0, len(CORNER_NAMES)))]
    seed_bbox = _choose_seed_bbox(height, width, size, corner, rng)
    seed_mask = _make_seed_mask(size, rng)

    safe_row_min = radius + 1
    safe_row_max = height - radius - 2
    safe_col_min = radius + 1
    safe_col_max = width - radius - 2
    if safe_row_min > safe_row_max or safe_col_min > safe_col_max:
        raise AssertionError("grid too small for safe corridor stamps")

    r0, c0, r1, c1 = seed_bbox
    excluded_rows = set(range(max(1, r0 - radius), min(height - 2, r1 + radius) + 1))
    excluded_cols = set(range(max(1, c0 - radius), min(width - 2, c1 + radius) + 1))
    row_candidates = [r for r in range(safe_row_min, safe_row_max + 1) if r not in excluded_rows]
    col_candidates = [c for c in range(safe_col_min, safe_col_max + 1) if c not in excluded_cols]

    min_gap = size + 1
    row_options = {
        count: _subsets_with_spacing(row_candidates, count, min_gap, prefer_irregular=True)
        for count in {count for count, _ in CORRIDOR_COUNT_OPTIONS}
    }
    col_options = {
        count: _subsets_with_spacing(col_candidates, count, min_gap, prefer_irregular=True)
        for count in {count for _, count in CORRIDOR_COUNT_OPTIONS}
    }
    feasible_counts = [
        (row_count, col_count)
        for row_count, col_count in CORRIDOR_COUNT_OPTIONS
        if row_options[row_count] and col_options[col_count]
    ]
    if not feasible_counts:
        raise AssertionError("no feasible corridor layout")

    row_count, col_count = feasible_counts[int(rng.integers(0, len(feasible_counts)))]
    corridor_rows = tuple(row_options[row_count][int(rng.integers(0, len(row_options[row_count])))])
    corridor_cols = tuple(col_options[col_count][int(rng.integers(0, len(col_options[col_count])))])

    grid = np.zeros((height, width), dtype=int)

    for rr in range(size):
        for cc in range(size):
            if seed_mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = YELLOW

    forbidden_cells = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }
    for cr in corridor_rows:
        for cc in corridor_cols:
            for rr in range(cr - radius, cr + radius + 1):
                for cx in range(cc - radius, cc + radius + 1):
                    forbidden_cells.add((int(rr), int(cx)))

    free_rows = [r for r in range(1, height - 1) if r not in corridor_rows and r not in range(r0, r1 + 1)]
    free_cols = [c for c in range(1, width - 1) if c not in corridor_cols and c not in range(c0, c1 + 1)]
    if len(free_rows) < 2 or len(free_cols) < 2:
        raise AssertionError("not enough non-corridor rows or columns")

    rng.shuffle(free_rows)
    rng.shuffle(free_cols)
    blocked_rows = free_rows[: int(rng.integers(1, min(3, len(free_rows))))]
    remaining_rows = [r for r in free_rows if r not in blocked_rows]
    if not remaining_rows:
        raise AssertionError("expected a single-end row")
    single_rows = remaining_rows[: int(rng.integers(1, min(3, len(remaining_rows))))]

    blocked_cols = free_cols[: int(rng.integers(1, min(3, len(free_cols))))]
    remaining_cols = [c for c in free_cols if c not in blocked_cols]
    if not remaining_cols:
        raise AssertionError("expected a single-end column")
    single_cols = remaining_cols[: int(rng.integers(1, min(3, len(remaining_cols))))]

    blockers = _place_blockers(
        grid,
        set(corridor_rows),
        set(corridor_cols),
        blocked_rows,
        blocked_cols,
        forbidden_cells,
        rng,
    )
    for r, c in blockers:
        grid[r, c] = _random_boundary_color(rng)

    for r in corridor_rows:
        grid[r, 0] = _random_boundary_color(rng)
        grid[r, width - 1] = _random_boundary_color(rng)
    for r in blocked_rows:
        grid[r, 0] = _random_boundary_color(rng)
        grid[r, width - 1] = _random_boundary_color(rng)
    for r in single_rows:
        if bool(rng.integers(0, 2)):
            grid[r, 0] = _random_boundary_color(rng)
        else:
            grid[r, width - 1] = _random_boundary_color(rng)

    for c in corridor_cols:
        grid[0, c] = _random_boundary_color(rng)
        grid[height - 1, c] = _random_boundary_color(rng)
    for c in blocked_cols:
        grid[0, c] = _random_boundary_color(rng)
        grid[height - 1, c] = _random_boundary_color(rng)
    for c in single_cols:
        if bool(rng.integers(0, 2)):
            grid[0, c] = _random_boundary_color(rng)
        else:
            grid[height - 1, c] = _random_boundary_color(rng)

    _force_both_boundary_colors(grid, rng)

    meta = {
        "corridor_rows": tuple(sorted(int(r) for r in corridor_rows)),
        "corridor_cols": tuple(sorted(int(c) for c in corridor_cols)),
        "blocked_rows": tuple(sorted(int(r) for r in blocked_rows)),
        "blocked_cols": tuple(sorted(int(c) for c in blocked_cols)),
        "single_rows": tuple(sorted(int(r) for r in single_rows)),
        "single_cols": tuple(sorted(int(c) for c in single_cols)),
        "seed_bbox": seed_bbox,
        "size": int(size),
        "radius": int(radius),
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    rows = tuple(_detect_horizontal_corridors(grid))
    cols = tuple(_detect_vertical_corridors(grid))
    if rows != meta["corridor_rows"] or cols != meta["corridor_cols"]:
        return False

    if not (2 <= len(rows) * len(cols) <= 6):
        return False

    seed, bbox = _extract_seed(grid)
    if bbox != meta["seed_bbox"]:
        return False
    if seed.shape[0] != meta["size"] or seed.shape[1] != meta["size"]:
        return False

    r0, c0, r1, c1 = bbox
    if any(r0 <= r <= r1 for r in rows):
        return False
    if any(c0 <= c <= c1 for c in cols):
        return False

    radius = int(meta["radius"])
    for r in rows:
        for c in cols:
            patch = grid[r - radius : r + radius + 1, c - radius : c + radius + 1]
            if patch.shape != (2 * radius + 1, 2 * radius + 1):
                return False
            if np.any(patch != BLACK):
                return False

    row_kinds = [_classify_row(grid, r) for r in range(1, grid.shape[0] - 1)]
    col_kinds = [_classify_col(grid, c) for c in range(1, grid.shape[1] - 1)]
    if "blocked" not in row_kinds or "single" not in row_kinds:
        return False
    if "blocked" not in col_kinds or "single" not in col_kinds:
        return False

    non_zero = {int(v) for v in np.unique(grid) if int(v) != BLACK}
    if non_zero != {RED, YELLOW, TEAL}:
        return False

    expected = _build_expected_output(grid, rows, cols)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False
    if np.array_equal(produced, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, meta = _build_candidate_grid(rng)
        except AssertionError:
            continue
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid corridor-stamping example")
