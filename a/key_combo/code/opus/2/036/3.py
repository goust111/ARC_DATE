from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, in_bounds


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DOMINANT_COLORS = (BLUE, RED, GREEN, YELLOW)
NOISE_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, TEAL)
QUADRANTS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 400
GRID_RANGE = (11, 17)
MARKER_COUNT_RANGE = (1, 2)
MARKER_STACK_PROB = 0.35


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty color multiset"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one solid grey separator cross"
    sr = int(rows[0])
    sc = int(cols[0])
    assert 0 < sr < grid.shape[0] - 1 and 0 < sc < grid.shape[1] - 1
    return sr, sc


def _quadrant_slices(sr: int, sc: int) -> dict[str, tuple[slice, slice]]:
    return {
        "TL": (slice(0, sr), slice(0, sc)),
        "TR": (slice(0, sr), slice(sc + 1, None)),
        "BL": (slice(sr + 1, None), slice(0, sc)),
        "BR": (slice(sr + 1, None), slice(sc + 1, None)),
    }


def _quadrant_of(r: int, c: int, sr: int, sc: int) -> str:
    assert r != sr and c != sc, "markers may not lie on the separator cross"
    if r < sr and c < sc:
        return "TL"
    if r < sr and c > sc:
        return "TR"
    if r > sr and c < sc:
        return "BL"
    return "BR"


def _quadrant_dominants(grid: np.ndarray, sr: int, sc: int) -> dict[str, int]:
    dominants: dict[str, int] = {}
    for name, region in _quadrant_slices(sr, sc).items():
        patch = grid[region]
        usable = patch[(patch != GREY) & (patch != MAROON)]
        dominants[name] = _unique_mode(usable)
    return dominants


def _orange_maps(grid: np.ndarray, sr: int, sc: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    orange_mask = grid == ORANGE
    orange_box = bbox(orange_mask)
    assert orange_box is not None, "expected a compact orange center blocker"
    r0, c0, r1, c1 = orange_box
    assert sr - 1 <= r0 <= r1 <= sr + 1 and sc - 1 <= c0 <= c1 <= sc + 1

    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in np.argwhere(orange_mask):
        row_to_cols[int(r)].append(int(c))
        col_to_rows[int(c)].append(int(r))
    for cols in row_to_cols.values():
        cols.sort()
    for rows in col_to_rows.values():
        rows.sort()
    return row_to_cols, col_to_rows


def _marker_instruction(
    pos: tuple[int, int],
    row_to_cols: dict[int, list[int]],
    col_to_rows: dict[int, list[int]],
) -> dict:
    r, c = pos
    row_match = False
    col_match = False
    direction: tuple[int, int] | None = None
    lane: int | None = None
    dist: int | None = None

    if r in row_to_cols:
        cols = row_to_cols[r]
        if c < cols[0]:
            row_match = True
            direction = (0, 1)
            lane = r
            dist = cols[0] - c
        elif c > cols[-1]:
            row_match = True
            direction = (0, -1)
            lane = r
            dist = c - cols[-1]

    if c in col_to_rows:
        rows = col_to_rows[c]
        if r < rows[0]:
            col_match = True
            direction = (1, 0)
            lane = c
            dist = rows[0] - r
        elif r > rows[-1]:
            col_match = True
            direction = (-1, 0)
            lane = c
            dist = r - rows[-1]

    assert row_match ^ col_match, "each marker must align with the orange blocker along exactly one axis"
    assert direction is not None and lane is not None and dist is not None
    return {"start": pos, "dir": direction, "lane": lane, "dist": int(dist)}


def _slide_until_blocked(
    start: tuple[int, int],
    direction: tuple[int, int],
    orange_mask: np.ndarray,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    dr, dc = direction
    while True:
        nr = r + dr
        nc = c + dc
        if not in_bounds(orange_mask.shape, nr, nc):
            return r, c
        if orange_mask[nr, nc] or (nr, nc) in occupied:
            return r, c
        r, c = nr, nc


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sr, sc = _find_separator_cross(g)
    orange_mask = g == ORANGE
    row_to_cols, col_to_rows = _orange_maps(g, sr, sc)
    dominants = _quadrant_dominants(g, sr, sc)

    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    assert marker_positions, "expected at least one marker"

    per_quadrant = Counter()
    instructions: list[dict] = []
    for pos in marker_positions:
        quad = _quadrant_of(pos[0], pos[1], sr, sc)
        per_quadrant[quad] += 1
        inst = _marker_instruction(pos, row_to_cols, col_to_rows)
        inst["quad"] = quad
        instructions.append(inst)
    assert all(1 <= per_quadrant[q] <= 2 for q in QUADRANTS), "expected 1-2 markers per quadrant"

    out = g.copy()
    for inst in instructions:
        r, c = inst["start"]
        out[r, c] = dominants[inst["quad"]]

    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for inst in instructions:
        dr, dc = inst["dir"]
        groups[(dr, dc, int(inst["lane"]))].append(inst)

    occupied: set[tuple[int, int]] = set()
    for key in sorted(groups):
        group = groups[key]
        group.sort(key=lambda item: int(item["dist"]))
        for inst in group:
            final = _slide_until_blocked(inst["start"], inst["dir"], orange_mask, occupied)
            occupied.add(final)
            out[final] = dominants[inst["quad"]]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each maroon marker to the orange center blocker and recolor it by its source quadrant's denoised dominant color.

    Input structure:
    - A single full grey row and a single full grey column split the grid into four nonempty noisy quadrants.
    - Each quadrant has one dominant color among blue, red, green, and yellow, plus sparse distractor colors.
    - A compact orange blocker sits against the separator cross, and each quadrant contains 1-2 maroon markers aligned with it by row or column.

    Rule:
    - Find the grey separator cross and compute each quadrant's dominant color while ignoring the maroon markers.
    - For every marker, remember its source quadrant, then slide it straight toward the orange blocker along its aligned row or column.
    - A marker ignores noise and the grey separator, but it stops just before the orange blocker or just before an earlier parked marker in the same lane.
    - Recolor each stopped marker with the dominant color of the quadrant where it started, and restore its source cell to that same quadrant color.

    Key insight:
    - The final position comes from the sliding geometry, but the final color comes from a separate denoising step on the marker's original quadrant.
    """
    return _solve(grid)


ORANGE_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "square": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    "hbar_top": ((-1, -1), (-1, 1)),
    "hbar_bottom": ((1, -1), (1, 1)),
    "vbar_left": ((-1, -1), (1, -1)),
    "vbar_right": ((-1, 1), (1, 1)),
}


def _shape_cells(sr: int, sc: int, shape_name: str) -> set[tuple[int, int]]:
    return {(sr + dr, sc + dc) for dr, dc in ORANGE_SHAPES[shape_name]}


def _quadrant_bounds(height: int, width: int, sr: int, sc: int) -> dict[str, tuple[range, range]]:
    return {
        "TL": (range(0, sr), range(0, sc)),
        "TR": (range(0, sr), range(sc + 1, width)),
        "BL": (range(sr + 1, height), range(0, sc)),
        "BR": (range(sr + 1, height), range(sc + 1, width)),
    }


def _marker_candidates(
    height: int,
    width: int,
    sr: int,
    sc: int,
    orange_cells: set[tuple[int, int]],
    quadrant: str,
) -> list[dict]:
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in orange_cells:
        row_to_cols[r].append(c)
        col_to_rows[c].append(r)
    for cols in row_to_cols.values():
        cols.sort()
    for rows in col_to_rows.values():
        rows.sort()

    row_range, col_range = _quadrant_bounds(height, width, sr, sc)[quadrant]
    top_half = quadrant in {"TL", "TR"}
    left_half = quadrant in {"TL", "BL"}
    candidates: list[dict] = []

    row_lane_rows = sorted(r for r in row_to_cols if (r < sr) == top_half)
    for lane_row in row_lane_rows:
        if left_half:
            target_col = row_to_cols[lane_row][0]
            possible_cols = [c for c in col_range if c <= target_col - 2]
            direction = (0, 1)
        else:
            target_col = row_to_cols[lane_row][-1]
            possible_cols = [c for c in col_range if c >= target_col + 2]
            direction = (0, -1)
        for c in possible_cols:
            pos = (lane_row, int(c))
            if pos in orange_cells:
                continue
            candidates.append(
                {
                    "pos": pos,
                    "lane_id": ("row", lane_row),
                    "dir": direction,
                    "dist": abs(int(c) - target_col),
                }
            )

    col_lane_cols = sorted(c for c in col_to_rows if (c < sc) == left_half)
    for lane_col in col_lane_cols:
        if top_half:
            target_row = col_to_rows[lane_col][0]
            possible_rows = [r for r in row_range if r <= target_row - 2]
            direction = (1, 0)
        else:
            target_row = col_to_rows[lane_col][-1]
            possible_rows = [r for r in row_range if r >= target_row + 2]
            direction = (-1, 0)
        for r in possible_rows:
            pos = (int(r), lane_col)
            if pos in orange_cells:
                continue
            candidates.append(
                {
                    "pos": pos,
                    "lane_id": ("col", lane_col),
                    "dir": direction,
                    "dist": abs(int(r) - target_row),
                }
            )

    assert candidates, f"expected at least one candidate marker lane for quadrant {quadrant}"
    return candidates


def _choose_markers_for_quadrant(candidates: list[dict], count: int) -> list[tuple[int, int]]:
    by_lane: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for cand in candidates:
        by_lane[cand["lane_id"]].append(cand)
    for lane_candidates in by_lane.values():
        lane_candidates.sort(key=lambda item: (int(item["dist"]), item["pos"]))

    if count == 2:
        stackable = [lane for lane, lane_candidates in by_lane.items() if len(lane_candidates) >= 2]
        if stackable and random.random() < MARKER_STACK_PROB:
            lane = random.choice(stackable)
            chosen = random.sample(by_lane[lane], 2)
            return [tuple(item["pos"]) for item in chosen]

    sample_pool = candidates[:]
    random.shuffle(sample_pool)
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    preferred_lanes = list(by_lane.keys())
    random.shuffle(preferred_lanes)
    for lane in preferred_lanes:
        lane_choices = by_lane[lane][:]
        random.shuffle(lane_choices)
        for cand in lane_choices:
            if cand["pos"] not in used:
                chosen.append(tuple(cand["pos"]))
                used.add(tuple(cand["pos"]))
                break
        if len(chosen) == count:
            return chosen
    for cand in sample_pool:
        if cand["pos"] in used:
            continue
        chosen.append(tuple(cand["pos"]))
        used.add(tuple(cand["pos"]))
        if len(chosen) == count:
            return chosen
    raise AssertionError("failed to choose enough marker positions")


def _fill_quadrant(
    grid: np.ndarray,
    sr: int,
    sc: int,
    quadrant: str,
    dominant: int,
    reserved: set[tuple[int, int]],
) -> None:
    region = _quadrant_bounds(*grid.shape, sr, sc)[quadrant]
    rows, cols = region
    coords = [(r, c) for r in rows for c in cols]
    for r, c in coords:
        grid[r, c] = dominant

    free = [pos for pos in coords if pos not in reserved]
    area = len(coords)
    noise_low = max(2, int(round(area * 0.12)))
    noise_high = max(noise_low, int(round(area * 0.18)))
    noise_count = min(len(free), random.randint(noise_low, noise_high))
    if noise_count:
        noise_positions = random.sample(free, noise_count)
        palette = [color for color in NOISE_COLORS if color != dominant]
        for r, c in noise_positions:
            grid[r, c] = random.choice(palette)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    sr = random.randint(4, height - 5)
    sc = random.randint(4, width - 5)
    shape_name = random.choice(tuple(ORANGE_SHAPES))
    orange_cells = _shape_cells(sr, sc, shape_name)

    markers_by_quadrant: dict[str, list[tuple[int, int]]] = {}
    all_markers: set[tuple[int, int]] = set()
    for quadrant in QUADRANTS:
        candidates = _marker_candidates(height, width, sr, sc, orange_cells, quadrant)
        marker_count = random.randint(*MARKER_COUNT_RANGE)
        chosen = _choose_markers_for_quadrant(candidates, marker_count)
        markers_by_quadrant[quadrant] = chosen
        all_markers.update(chosen)

    grid = np.zeros((height, width), dtype=int)
    grid[sr, :] = GREY
    grid[:, sc] = GREY

    dominant_assignment = dict(zip(QUADRANTS, random.sample(DOMINANT_COLORS, len(QUADRANTS))))
    for quadrant in QUADRANTS:
        reserved = {pos for pos in orange_cells if _quadrant_of(pos[0], pos[1], sr, sc) == quadrant}
        reserved.update(markers_by_quadrant[quadrant])
        _fill_quadrant(grid, sr, sc, quadrant, dominant_assignment[quadrant], reserved)

    for r, c in orange_cells:
        grid[r, c] = ORANGE
    for r, c in all_markers:
        grid[r, c] = MAROON

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if np.any(out == MAROON):
        return None
    if any(_quadrant_dominants(grid, sr, sc)[q] != dominant_assignment[q] for q in QUADRANTS):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
