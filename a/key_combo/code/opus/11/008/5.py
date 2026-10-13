import itertools
import random
from collections import deque

import numpy as np


BG = 7
WALL = 6
MARKER = 2
BAR = 9

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bars: list[dict] = []
    used = np.zeros_like(grid, dtype=bool)
    for c in range(w):
        if int(grid[h - 1, c]) != BAR:
            continue
        r = h - 1
        rows: list[int] = []
        while r >= 0 and int(grid[r, c]) == BAR:
            rows.append(r)
            used[r, c] = True
            r -= 1
        rows.reverse()
        bars.append(
            {
                "col": int(c),
                "rows": np.array(rows, dtype=int),
                "height": int(len(rows)),
                "bottom": (int(h - 1), int(c)),
            }
        )
    assert not np.any((grid == BAR) & (~used)), "all maroon cells must belong to bottom-aligned vertical bars"
    return bars


def _bfs_distances(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    sr, sc = start
    dist = np.full((h, w), -1, dtype=int)
    dq = deque([(int(sr), int(sc))])
    dist[int(sr), int(sc)] = 0
    while dq:
        r, c = dq.popleft()
        nd = int(dist[r, c]) + 1
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                if int(grid[rr, cc]) in (BG, BAR, MARKER):
                    dist[rr, cc] = nd
                    dq.append((int(rr), int(cc)))
    return dist


def _render_output(grid: np.ndarray, chosen: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    marker_cells = np.argwhere(g == MARKER)
    assert len(marker_cells) == 1
    mr, mc = map(int, marker_cells[0])
    h, w = g.shape

    for bar in chosen:
        out[bar["rows"], int(bar["col"])] = BG

    cursor = int(mc) + 1
    for bar in chosen:
        end = min(w, cursor + int(bar["height"]))
        if cursor < w:
            out[mr, cursor:end] = BAR
        cursor += int(bar["height"]) + 1

    out[g == WALL] = WALL
    out[mr, mc] = MARKER
    return out


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker_cells = np.argwhere(g == MARKER)
    assert len(marker_cells) == 1, "grid must contain exactly one red marker"
    mr, mc = map(int, marker_cells[0])

    bars = _find_bars(g)
    dist = _bfs_distances(g, (mr, mc))

    reachable: list[dict] = []
    unreachable: list[dict] = []
    for bar in bars:
        bottom_r, bottom_c = bar["bottom"]
        if int(dist[bottom_r, bottom_c]) >= 0:
            item = dict(bar)
            item["distance"] = int(dist[bottom_r, bottom_c])
            reachable.append(item)
        else:
            unreachable.append(dict(bar))

    reachable.sort(key=lambda bar: (int(bar["distance"]), int(bar["col"])))
    out = _render_output(g, reachable)
    return {
        "grid": g,
        "output": out,
        "marker": (int(mr), int(mc)),
        "dist": dist,
        "bars": bars,
        "reachable": reachable,
        "unreachable": unreachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter bottom-aligned vertical bars by flood-fill reachability from the red marker, then rotate the reachable ones into a rightward chained assembly.

    Input structure:
    - The grid has an orange (7) background, magenta (6) wall segments, a unique red (2) marker pixel, and several maroon (9) bottom-aligned 1-cell-wide vertical bars.
    - Walls are 4-connected blockers; the marker sits in the lower-middle area; some maroon bars share the marker's passable region and some are cut off by walls.

    Rule:
    - Run 4-connected BFS from the red marker, allowing movement through orange background and maroon bar cells but not through magenta walls.
    - Mark a maroon bar as reachable iff its cells are touched by that BFS, and compute the BFS distance from the marker to the bar's bottom pixel.
    - Erase only the reachable bars from their original columns, keep unreachable bars and all walls in place, then place the reachable bars on the marker row starting immediately to the right of the marker.
    - Each placed bar is rotated clockwise into a horizontal segment whose length equals the original bar height; segments are chained in increasing BFS-distance order with a one-cell orange spacer between consecutive pieces, and the assembly is clipped at the right edge.

    Key insight:
    - The walls jointly determine both which bars participate and the order of the resulting segment lengths: a bar can be visually closer yet assemble later if its route to the marker must detour around a barrier.
    """
    return _solve_details(grid)["output"]


def _sample_reachable_heights(n_reachable: int, available: int, clip_goal: bool) -> list[int] | None:
    combos = [
        combo
        for combo in itertools.combinations((2, 3, 4, 5), n_reachable)
        if (sum(combo) + max(0, n_reachable - 1) > available) == clip_goal
    ]
    if not combos:
        return None
    heights = list(random.choice(combos))
    random.shuffle(heights)
    return heights


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    h = random.randint(13, 16)
    w = random.randint(16, 20)
    grid = np.full((h, w), BG, dtype=int)

    seal_col = random.randint(2, min(4, w - 12))
    grid[:, seal_col] = WALL

    max_wall_len = min(8, 21 - h, w - seal_col - 4)
    if max_wall_len < 5:
        return None
    wall_len = random.randint(5, max_wall_len)
    wall_start = seal_col
    wall_end = wall_start + wall_len - 1

    wall_row = h - random.choice((6, 7))
    if wall_row < 2:
        return None
    grid[wall_row, wall_start : wall_end + 1] = WALL

    marker_row = wall_row - random.randint(1, 2)
    marker_col = random.randint(seal_col + 2, min(seal_col + 4, wall_end - 1))
    if marker_row < 1 or marker_col <= seal_col + 1:
        return None

    total_bars = random.randint(4, 6)
    unreachable_count = random.randint(1, min(2, seal_col))
    reachable_count = total_bars - unreachable_count
    if not (2 <= reachable_count <= 4):
        return None

    available = w - (marker_col + 1)
    clip_goal = random.random() < 0.45
    reachable_heights = _sample_reachable_heights(reachable_count, available, clip_goal)
    if reachable_heights is None:
        return None

    candidate_reachable_cols = list(range(marker_col, wall_end + 1))
    if len(candidate_reachable_cols) < reachable_count:
        return None
    reachable_cols = sorted(random.sample(candidate_reachable_cols, reachable_count))
    unreachable_cols = sorted(random.sample(list(range(seal_col)), unreachable_count))

    for col, height in zip(reachable_cols, reachable_heights):
        top = h - height
        if top <= wall_row:
            return None
        grid[top:h, col] = BAR

    unreachable_heights = [random.randint(2, 5) for _ in range(unreachable_count)]
    for col, height in zip(unreachable_cols, unreachable_heights):
        top = h - height
        if top <= wall_row:
            return None
        grid[top:h, col] = BAR

    grid[marker_row, marker_col] = MARKER

    meta = {
        "shape": (h, w),
        "marker": (int(marker_row), int(marker_col)),
        "seal_col": int(seal_col),
        "wall_row": int(wall_row),
        "wall_span": (int(wall_start), int(wall_end)),
        "reachable_map": {int(c): int(hh) for c, hh in zip(reachable_cols, reachable_heights)},
        "unreachable_map": {int(c): int(hh) for c, hh in zip(unreachable_cols, unreachable_heights)},
        "clip_goal": bool(clip_goal),
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    try:
        details = _solve_details(grid)
    except Exception:
        return False

    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if int(np.count_nonzero(grid == WALL)) < 10 or int(np.count_nonzero(grid == WALL)) > 20:
        return False

    bars_by_col = {int(bar["col"]): int(bar["height"]) for bar in details["bars"]}
    expected_bars = dict(meta["reachable_map"])
    expected_bars.update(meta["unreachable_map"])
    if bars_by_col != expected_bars:
        return False

    reachable = details["reachable"]
    unreachable = details["unreachable"]
    if len(reachable) != len(meta["reachable_map"]) or len(unreachable) != len(meta["unreachable_map"]):
        return False

    reachable_cols = [int(bar["col"]) for bar in reachable]
    unreachable_cols = sorted(int(bar["col"]) for bar in unreachable)
    if sorted(reachable_cols) != sorted(meta["reachable_map"]):
        return False
    if unreachable_cols != sorted(meta["unreachable_map"]):
        return False

    distances = [int(bar["distance"]) for bar in reachable]
    if len(distances) != len(set(distances)):
        return False

    if reachable_cols != sorted(reachable_cols, reverse=True):
        return False
    if reachable_cols == sorted(reachable_cols):
        return False

    marker_col = int(meta["marker"][1])
    by_horizontal = [
        int(bar["col"])
        for bar in sorted(reachable, key=lambda bar: (abs(int(bar["col"]) - marker_col), int(bar["col"])))
    ]
    if by_horizontal == reachable_cols:
        return False

    distance_heights = [int(bar["height"]) for bar in reachable]
    column_heights = [meta["reachable_map"][col] for col in sorted(meta["reachable_map"])]
    if distance_heights == column_heights:
        return False

    clip_actual = sum(distance_heights) + max(0, len(distance_heights) - 1) > (w - (marker_col + 1))
    if clip_actual != bool(meta["clip_goal"]):
        return False

    out = details["output"]
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid == WALL], grid[grid == WALL]):
        return False
    mr, mc = meta["marker"]
    if int(out[mr, mc]) != MARKER:
        return False

    for col, height in meta["unreachable_map"].items():
        top = h - height
        if not np.all(out[top:h, col] == BAR):
            return False

    for col, height in meta["reachable_map"].items():
        top = h - height
        if not np.all(out[top:h, col] == BG):
            return False

    wrong_order = _render_output(grid, sorted(reachable, key=lambda bar: int(bar["col"])))
    if np.array_equal(wrong_order, out):
        return False

    wrong_filter = _render_output(
        grid,
        sorted(
            [dict(bar) for bar in details["bars"]],
            key=lambda bar: (-int(bar["col"]), int(bar["height"])),
        ),
    )
    if np.array_equal(wrong_filter, out):
        return False

    chain_row = out[mr, mc + 1 :]
    if np.count_nonzero(chain_row == BAR) == 0:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        grid, meta = built
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid reachable-bar assembly sample")
