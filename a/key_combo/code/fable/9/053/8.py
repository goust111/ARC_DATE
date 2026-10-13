from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
LANDING_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 21)
GRID_W_RANGE = (12, 23)
RAMP_COUNT_RANGE = (2, 5)
RAMP_LEN_RANGE = (2, 5)
LANDING_COUNT_WEIGHTS = ((2, 2), (3, 4), (4, 2))
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 500

NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ramp_direction(cells: list[tuple[int, int]]) -> int:
    ordered = sorted(cells)
    assert len(ordered) >= 2, "each grey ramp must have length >= 2"
    rows = [r for r, _ in ordered]
    cols = [c for _, c in ordered]
    assert len(set(rows)) == len(ordered), "ramp rows must be unique"
    assert len(set(cols)) == len(ordered), "ramp cols must be unique"
    row_diffs = [b - a for a, b in zip(rows, rows[1:])]
    col_diffs = [b - a for a, b in zip(cols, cols[1:])]
    assert all(diff == 1 for diff in row_diffs), "ramp rows must be consecutive"
    if all(diff == 1 for diff in col_diffs):
        return 1
    if all(diff == -1 for diff in col_diffs):
        return -1
    raise AssertionError("grey component must be a clean left-down or right-down diagonal")


def _build_ramp_dir_map(grid: np.ndarray) -> np.ndarray:
    grey_mask = np.asarray(grid == GREY, dtype=bool)
    dir_map = np.zeros_like(grid, dtype=int)
    for comp in _components8(grey_mask):
        direction = _ramp_direction(comp)
        for r, c in comp:
            dir_map[r, c] = direction
    return dir_map


def _trace_path(grey_mask: np.ndarray, dir_map: np.ndarray, source_c: int) -> dict:
    h, w = grey_mask.shape
    r = 0
    c = int(source_c)
    path = [(r, c)]
    directions: list[int] = []
    slide_steps = 0
    for _ in range(h * w * 4):
        if r == h - 1:
            return {
                "source_c": int(source_c),
                "end_c": int(c),
                "path": path,
                "cell_set": frozenset(path),
                "slide_steps": int(slide_steps),
                "dirs": frozenset(directions),
            }
        if grey_mask[r + 1, c]:
            direction = int(dir_map[r + 1, c])
            nc = c + direction
            assert 0 <= nc < w, "ramp deflection must stay in bounds"
            c = nc
            path.append((r, c))
            slide_steps += 1
            directions.append(direction)
        else:
            r += 1
            path.append((r, c))
            assert not grey_mask[r, c], "trajectory may not pass through a grey ramp cell"
    raise AssertionError("trajectory did not reach the bottom row")


def _paint_records(grid: np.ndarray, records: list[tuple[dict, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rec, color in records:
        for r, c in rec["path"]:
            if out[r, c] != GREY:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer each colored bottom landing's unique top source column and paint its full guided trajectory.

    Input structure:
    - A black background contains several single-pixel colored landing points on the bottom row, while the top row is empty.
    - The interior may contain several separated one-cell-thick grey diagonal ramps, each slanting down-left or down-right.
    - Grey ramps are obstacles: trajectories run through background cells and never overwrite the grey cells.

    Rule:
    - Consider every possible source column on the top row and simulate the falling trajectory forward: it normally drops one cell, but whenever the cell directly below is grey, it shifts one cell horizontally in that ramp's downhill direction and continues.
    - For each colored bottom landing, find the unique top source whose simulated trajectory ends exactly on that landing column.
    - Paint every visited trajectory cell, including the top source and the landing pixel, with that landing's color; keep the grey ramps unchanged.

    Key insight:
    - The visible colored pixels are effects, not causes: the correct source cannot be found by a naive vertical extension, because horizontal detours happen only when a trajectory is directly above a grey slope cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    assert int(bg) == BG, "the puzzle uses a black background"
    assert np.all(g[0] == bg), "the top row must be empty"
    assert not np.any(g[-1] == GREY), "grey ramps may not occupy the bottom row"

    grey_mask = g == GREY
    dir_map = _build_ramp_dir_map(g)
    landing_cols = [
        c for c, value in enumerate(g[-1])
        if int(value) not in {bg, GREY}
    ]
    assert landing_cols, "expected at least one colored landing on the bottom row"

    trajectories = [_trace_path(grey_mask, dir_map, source_c) for source_c in range(g.shape[1])]
    by_end: dict[int, list[dict]] = {}
    for rec in trajectories:
        by_end.setdefault(int(rec["end_c"]), []).append(rec)

    matched: list[tuple[dict, int]] = []
    used_cells: set[tuple[int, int]] = set()
    for landing_c in landing_cols:
        hits = by_end.get(int(landing_c), [])
        assert len(hits) == 1, "each landing must correspond to exactly one top source"
        rec = hits[0]
        cell_set = set(rec["cell_set"])
        assert used_cells.isdisjoint(cell_set), "landing trajectories may not cross"
        used_cells |= cell_set
        matched.append((rec, int(g[-1, landing_c])))

    return _paint_records(g, matched)


def _ramp_cells(h: int, w: int, start_r: int, start_c: int, length: int, direction: int) -> list[tuple[int, int]] | None:
    cells = [(start_r + i, start_c + direction * i) for i in range(length)]
    if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in cells):
        return None
    return cells


def _choose_landing_count() -> int:
    values = [value for value, _ in LANDING_COUNT_WEIGHTS]
    weights = [weight for _, weight in LANDING_COUNT_WEIGHTS]
    return int(random.choices(values, weights=weights, k=1)[0])


def _place_ramps(h: int, w: int) -> list[tuple[int, list[tuple[int, int]]]] | None:
    ramp_count = random.randint(*RAMP_COUNT_RANGE)
    directions = [random.choice((-1, 1)) for _ in range(ramp_count)]
    if ramp_count >= 2 and random.random() < 0.7:
        directions[0], directions[1] = -1, 1
        random.shuffle(directions)

    forbidden: set[tuple[int, int]] = set()
    ramps: list[tuple[int, list[tuple[int, int]]]] = []
    for direction in directions:
        placed = False
        for _ in range(MAX_LAYOUT_ATTEMPTS):
            length = random.randint(*RAMP_LEN_RANGE)
            if length > h - 2 or length > w:
                continue
            start_r = random.randint(1, h - length - 1)
            if direction == 1:
                if w - length - 1 < 1:
                    continue
                start_c = random.randint(1, w - length - 1)
            else:
                if w - 2 < length:
                    continue
                start_c = random.randint(length, w - 2)
            cells = _ramp_cells(h, w, start_r, start_c, length, direction)
            if cells is None:
                continue
            if any(cell in forbidden for cell in cells):
                continue
            ramps.append((direction, cells))
            for r, c in cells:
                for dr, dc in NEIGHBORS8 + ((0, 0),):
                    forbidden.add((r + dr, c + dc))
            placed = True
            break
        if not placed:
            return None
    return ramps


def _records_for_layout(h: int, w: int, ramps: list[tuple[int, list[tuple[int, int]]]]) -> list[dict]:
    grid = np.full((h, w), BG, dtype=int)
    for _direction, cells in ramps:
        for r, c in cells:
            grid[r, c] = GREY
    grey_mask = grid == GREY
    dir_map = _build_ramp_dir_map(grid)
    trajectories = [_trace_path(grey_mask, dir_map, source_c) for source_c in range(w)]
    end_counts = Counter(int(rec["end_c"]) for rec in trajectories)
    return [rec for rec in trajectories if end_counts[int(rec["end_c"])] == 1]


def _combo_valid(combo: tuple[dict, ...]) -> bool:
    used: set[tuple[int, int]] = set()
    any_slide = False
    any_offset = False
    for rec in combo:
        cell_set = set(rec["cell_set"])
        if not used.isdisjoint(cell_set):
            return False
        used |= cell_set
        any_slide |= int(rec["slide_steps"]) > 0
        any_offset |= int(rec["source_c"]) != int(rec["end_c"])
    return any_slide and any_offset


def _combo_score(combo: tuple[dict, ...]) -> int:
    slide_paths = sum(int(rec["slide_steps"]) > 0 for rec in combo)
    total_slides = sum(int(rec["slide_steps"]) for rec in combo)
    dirs = set().union(*(set(rec["dirs"]) for rec in combo))
    total_offset = sum(abs(int(rec["source_c"]) - int(rec["end_c"])) for rec in combo)
    end_cols = [int(rec["end_c"]) for rec in combo]
    spread = max(end_cols) - min(end_cols)
    return (
        1000 * len(combo)
        + 120 * slide_paths
        + 40 * (len(dirs) == 2)
        + 8 * total_slides
        + 3 * total_offset
        + spread
    )


def _choose_records(candidates: list[dict]) -> list[dict] | None:
    if len(candidates) < 2:
        return None
    preferred = _choose_landing_count()
    all_good: list[tuple[int, int, tuple[dict, ...]]] = []
    for k in range(2, min(4, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, k):
            if _combo_valid(combo):
                all_good.append((k, _combo_score(combo), combo))
    if not all_good:
        return None

    for k in (preferred, 4, 3, 2):
        pool = [item for item in all_good if item[0] == k]
        if not pool:
            continue
        pool.sort(key=lambda item: item[1], reverse=True)
        top = pool[: min(10, len(pool))]
        chosen = random.choice(top)[2]
        return sorted(chosen, key=lambda rec: int(rec["end_c"]))
    return None


def _build_input_and_output(
    h: int,
    w: int,
    ramps: list[tuple[int, list[tuple[int, int]]]],
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    for _direction, cells in ramps:
        for r, c in cells:
            inp[r, c] = GREY
    colors = random.sample(LANDING_COLORS, len(records))
    painted: list[tuple[dict, int]] = []
    for rec, color in zip(records, colors):
        inp[h - 1, int(rec["end_c"])] = int(color)
        painted.append((rec, int(color)))
    return inp, _paint_records(inp, painted)


def _semantic_validate(inp: np.ndarray, expected: np.ndarray) -> bool:
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if not np.all(inp[0] == BG):
        return False
    if np.any(inp[-1] == GREY):
        return False

    try:
        grey_mask = inp == GREY
        dir_map = _build_ramp_dir_map(inp)
        trajectories = [_trace_path(grey_mask, dir_map, source_c) for source_c in range(w)]
    except Exception:
        return False

    landing_cols = [c for c, value in enumerate(inp[-1]) if int(value) not in {BG, GREY}]
    if not (2 <= len(landing_cols) <= 4):
        return False

    counts = Counter(int(rec["end_c"]) for rec in trajectories)
    matched: list[tuple[dict, int]] = []
    used: set[tuple[int, int]] = set()
    any_slide = False
    any_offset = False
    for c in landing_cols:
        if counts[int(c)] != 1:
            return False
        rec = next(rec for rec in trajectories if int(rec["end_c"]) == int(c))
        cell_set = set(rec["cell_set"])
        if not used.isdisjoint(cell_set):
            return False
        used |= cell_set
        any_slide |= int(rec["slide_steps"]) > 0
        any_offset |= int(rec["source_c"]) != int(rec["end_c"])
        matched.append((rec, int(inp[-1, c])))
    if not any_slide or not any_offset:
        return False

    oracle = _paint_records(inp, matched)
    if not np.array_equal(oracle, expected):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    ramps = _place_ramps(h, w)
    if ramps is None:
        return None
    candidates = _records_for_layout(h, w, ramps)
    records = _choose_records(candidates)
    if records is None:
        return None
    inp, expected = _build_input_and_output(h, w, ramps, records)
    if not _semantic_validate(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid landing-trace puzzle")
