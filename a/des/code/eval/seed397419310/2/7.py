import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = np.array([1, 2, 3, 6, 7, 8, 9], dtype=int)

GEN_ATTEMPTS = 500
MIN_SIDE = 12
MAX_SIDE = 19
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
MAX_GAP = 6
MAX_OFFPATH_OBSTACLES = 5


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for rr, cc in _neighbors8(r, c, h, w):
            expanded.add((rr, cc))
    return expanded


def _horizontal_candidate(h: int, w: int, row: int, left_col: int, distance: int) -> dict | None:
    right_col = left_col + distance + 1
    if not (0 <= left_col < right_col < w):
        return None

    up_len = min(distance, row)
    down_len = min(distance, h - 1 - row)
    if up_len == 0 or down_len == 0:
        return None

    up_path = [(row - step, left_col) for step in range(1, up_len + 1)]
    down_path = [(row + step, right_col) for step in range(1, down_len + 1)]
    gap_cells = [(row, col) for col in range(left_col + 1, right_col)]
    seed_cells = [(row, left_col), (row, right_col)]
    footprint = set(seed_cells) | set(gap_cells) | set(up_path) | set(down_path)
    return {
        "orientation": "h",
        "distance": distance,
        "seed_cells": seed_cells,
        "gap_cells": gap_cells,
        "paths": [up_path, down_path],
        "footprint": footprint,
        "clipped": up_len < distance or down_len < distance,
    }


def _vertical_candidate(h: int, w: int, col: int, top_row: int, distance: int) -> dict | None:
    bottom_row = top_row + distance + 1
    if not (0 <= top_row < bottom_row < h):
        return None

    left_len = min(distance, col)
    right_len = min(distance, w - 1 - col)
    if left_len == 0 or right_len == 0:
        return None

    left_path = [(top_row, col - step) for step in range(1, left_len + 1)]
    right_path = [(bottom_row, col + step) for step in range(1, right_len + 1)]
    gap_cells = [(row, col) for row in range(top_row + 1, bottom_row)]
    seed_cells = [(top_row, col), (bottom_row, col)]
    footprint = set(seed_cells) | set(gap_cells) | set(left_path) | set(right_path)
    return {
        "orientation": "v",
        "distance": distance,
        "seed_cells": seed_cells,
        "gap_cells": gap_cells,
        "paths": [left_path, right_path],
        "footprint": footprint,
        "clipped": left_len < distance or right_len < distance,
    }


def _enumerate_candidates(
    h: int,
    w: int,
    orientation: str,
    occupied: set[tuple[int, int]],
    require_clipped: bool | None,
) -> list[dict]:
    candidates: list[dict] = []
    if orientation == "h":
        for row in range(1, h - 1):
            for left_col in range(w - 2):
                max_distance = min(MAX_GAP, w - left_col - 2)
                for distance in range(1, max_distance + 1):
                    candidate = _horizontal_candidate(h, w, row, left_col, distance)
                    if candidate is None:
                        continue
                    if require_clipped is not None and bool(candidate["clipped"]) != require_clipped:
                        continue
                    if candidate["footprint"] & occupied:
                        continue
                    candidates.append(candidate)
    else:
        for col in range(1, w - 1):
            for top_row in range(h - 2):
                max_distance = min(MAX_GAP, h - top_row - 2)
                for distance in range(1, max_distance + 1):
                    candidate = _vertical_candidate(h, w, col, top_row, distance)
                    if candidate is None:
                        continue
                    if require_clipped is not None and bool(candidate["clipped"]) != require_clipped:
                        continue
                    if candidate["footprint"] & occupied:
                        continue
                    candidates.append(candidate)
    return candidates


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pairs: list[dict] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"

        if int(coords[0, 0]) == int(coords[1, 0]):
            row = int(coords[0, 0])
            cols = sorted(int(v) for v in coords[:, 1])
            left_col, right_col = cols
            distance = right_col - left_col - 1
            assert distance >= 1, "horizontal seed pair must have black pixels between seeds"
            assert np.all(g[row, left_col + 1 : right_col] == BLACK), "horizontal seed gap must be black"
            up_len = min(distance, row)
            down_len = min(distance, h - 1 - row)
            pairs.append(
                {
                    "color": color,
                    "paths": [
                        {"start": (row, left_col), "delta": (-1, 0), "length": distance},
                        {"start": (row, right_col), "delta": (1, 0), "length": distance},
                    ],
                    "clipped": up_len < distance or down_len < distance,
                    "orientation": "h",
                }
            )
        else:
            cols = sorted({int(v) for v in coords[:, 1]})
            assert len(cols) == 1, f"color {color} must form a horizontal or vertical pair"
            col = cols[0]
            rows = sorted(int(v) for v in coords[:, 0])
            top_row, bottom_row = rows
            distance = bottom_row - top_row - 1
            assert distance >= 1, "vertical seed pair must have black pixels between seeds"
            assert np.all(g[top_row + 1 : bottom_row, col] == BLACK), "vertical seed gap must be black"
            left_len = min(distance, col)
            right_len = min(distance, w - 1 - col)
            pairs.append(
                {
                    "color": color,
                    "paths": [
                        {"start": (top_row, col), "delta": (0, -1), "length": distance},
                        {"start": (bottom_row, col), "delta": (0, 1), "length": distance},
                    ],
                    "clipped": left_len < distance or right_len < distance,
                    "orientation": "v",
                }
            )

    assert pairs, "expected at least one seed pair"
    return pairs


def _paint_path(
    src: np.ndarray,
    out: np.ndarray,
    painted: np.ndarray,
    color: int,
    start: tuple[int, int],
    delta: tuple[int, int],
    length: int,
) -> None:
    h, w = src.shape
    r0, c0 = start
    dr, dc = delta
    switched = False

    for step in range(1, length + 1):
        rr = r0 + dr * step
        cc = c0 + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break

        cell = int(src[rr, cc])
        assert cell in (BLACK, GREY), "stripes may only traverse black cells or grey obstacles"
        assert not painted[rr, cc], "stripe overlaps are ambiguous"
        if cell == GREY:
            switched = not switched

        out[rr, cc] = YELLOW if switched else color
        painted[rr, cc] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow perpendicular stripes from each same-color seed pair.

    Horizontal pair:
    - left seed grows upward
    - right seed grows downward

    Vertical pair:
    - top seed grows leftward
    - bottom seed grows rightward

    Stripe length is the number of black cells between the seeds.
    Each grey obstacle toggles the stripe color between the seed color and yellow,
    starting at the obstacle cell itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    painted = np.zeros(g.shape, dtype=bool)
    for pair in _extract_pairs(g):
        for path in pair["paths"]:
            _paint_path(
                g,
                out,
                painted,
                int(pair["color"]),
                tuple(path["start"]),
                tuple(path["delta"]),
                int(path["length"]),
            )

    return out


def _sample_orientations(pair_count: int, rng: np.random.Generator) -> list[str]:
    if pair_count == 2:
        orientations = ["h", "v"]
    else:
        horizontal_count = int(rng.integers(1, pair_count))
        orientations = ["h"] * horizontal_count + ["v"] * (pair_count - horizontal_count)
    rng.shuffle(orientations)
    return orientations


def _sample_clip_flags(pair_count: int, rng: np.random.Generator) -> list[bool]:
    flags = [False, True]
    while len(flags) < pair_count:
        flags.append(bool(rng.integers(0, 2)))
    rng.shuffle(flags)
    return flags


def _choose_spaced_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    assert 1 <= count <= length
    if count == 1:
        return np.array([int(rng.integers(0, length))], dtype=int)

    for _ in range(40):
        trial = np.sort(rng.choice(length, size=count, replace=False).astype(int))
        if np.all(np.diff(trial) >= 2):
            return trial
    return np.sort(rng.choice(length, size=count, replace=False).astype(int))


def _choose_on_path_obstacles(pairs: list[dict], rng: np.random.Generator) -> set[tuple[int, int]]:
    paths = [path for pair in pairs for path in pair["paths"] if path]
    assert len(paths) >= 2, "generator expects at least two visible paths"

    max_selected = min(len(paths) - 1, max(1, (len(paths) + 1) // 2))
    selected_count = int(rng.integers(1, max_selected + 1))
    selected = set(int(x) for x in rng.choice(len(paths), size=selected_count, replace=False))
    need_double_toggle = any(len(path) >= 2 for path in paths) and bool(rng.integers(0, 2))
    used_double_toggle = False

    obstacle_cells: set[tuple[int, int]] = set()
    for idx, path in enumerate(paths):
        if idx not in selected:
            continue

        max_count = min(3, len(path))
        if max_count == 1:
            count = 1
        elif need_double_toggle and not used_double_toggle and len(path) >= 2:
            count = int(rng.integers(2, max_count + 1))
            used_double_toggle = True
        else:
            count = int(rng.integers(1, max_count + 1))

        chosen = _choose_spaced_indices(len(path), count, rng)
        for pos in chosen:
            obstacle_cells.add(tuple(path[int(pos)]))

    assert obstacle_cells, "at least one obstacle must lie on a stripe"
    return obstacle_cells


def _choose_off_path_obstacles(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    h, w = grid.shape
    chosen: set[tuple[int, int]] = set()
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == BLACK and (r, c) not in blocked
    ]
    if not candidates:
        return chosen

    max_count = min(MAX_OFFPATH_OBSTACLES, len(candidates))
    if max_count == 0:
        return chosen
    target = int(rng.integers(0, max_count + 1))
    if target == 0:
        return chosen

    local_blocked = set(blocked)
    rng.shuffle(candidates)
    for r, c in candidates:
        if len(chosen) >= target:
            break
        if (r, c) in local_blocked:
            continue
        chosen.add((r, c))
        local_blocked.add((r, c))
        for rr, cc in _neighbors8(r, c, h, w):
            local_blocked.add((rr, cc))
    return chosen


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    colors = [int(c) for c in rng.choice(SEED_COLORS, size=pair_count, replace=False)]

    requests = list(zip(colors, _sample_orientations(pair_count, rng), _sample_clip_flags(pair_count, rng)))
    occupied: set[tuple[int, int]] = set()
    pairs: list[dict] = []

    for color, orientation, want_clipped in requests:
        candidates = _enumerate_candidates(h, w, orientation, occupied, want_clipped)
        if not candidates:
            candidates = _enumerate_candidates(h, w, orientation, occupied, None)
        if not candidates:
            return None

        candidate = dict(candidates[int(rng.integers(0, len(candidates)))])
        candidate["color"] = color
        pairs.append(candidate)
        occupied |= set(candidate["footprint"])

    orientations = {str(pair["orientation"]) for pair in pairs}
    clip_states = {bool(pair["clipped"]) for pair in pairs}
    if orientations != {"h", "v"}:
        return None
    if clip_states != {False, True}:
        return None

    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for r, c in pair["seed_cells"]:
            grid[r, c] = int(pair["color"])

    on_path_obstacles = _choose_on_path_obstacles(pairs, rng)
    for r, c in on_path_obstacles:
        assert int(grid[r, c]) == BLACK
        grid[r, c] = GREY

    blocked_for_offpath = _expand_cells(occupied | on_path_obstacles, h, w)
    off_path_obstacles = _choose_off_path_obstacles(grid, blocked_for_offpath, rng)
    for r, c in off_path_obstacles:
        grid[r, c] = GREY

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    if not np.any(solved == YELLOW):
        return None

    extracted = _extract_pairs(grid)
    if {pair["orientation"] for pair in extracted} != {"h", "v"}:
        return None
    if {bool(pair["clipped"]) for pair in extracted} != {False, True}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
