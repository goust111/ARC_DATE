from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, in_bounds, neighbors4


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

GRID_H_CHOICES = tuple(range(16, 25))
GRID_W_CHOICES = tuple(range(16, 24))
CORRIDOR_COUNT_CHOICES = (4, 5, 5, 6, 6)
GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Bar:
    top: int
    bottom: int
    col: int
    height: int
    corridor: int
    full_height: bool


def _black_runs(row: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(row == BLACK)
    if cols.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for c in map(int, cols[1:]):
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return runs


def _wall_rows(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows: list[int] = []
    centers: list[int] = []
    for r in range(h):
        row = grid[r]
        if not np.all(np.isin(row, [BLACK, GREY])):
            continue
        runs = _black_runs(row)
        if len(runs) != 1:
            continue
        c0, c1 = runs[0]
        if c1 - c0 + 1 != 3:
            continue
        if int(np.count_nonzero(row == GREY)) != w - 3:
            continue
        rows.append(r)
        centers.append(c0 + 1)
    return rows, centers


def _corridors(height: int, wall_rows: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    prev = -1
    for wall in wall_rows:
        out.append((prev + 1, wall - 1))
        prev = wall
    out.append((prev + 1, height - 1))
    return out


def _corridor_index(row: int, corridors: list[tuple[int, int]]) -> int:
    for idx, (r0, r1) in enumerate(corridors):
        if r0 <= row <= r1:
            return idx
    raise AssertionError("row not inside any corridor")


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == BLUE)
    assert coords.shape == (1, 2), "expected exactly one blue seed"
    return tuple(map(int, coords[0]))


def _extract_bars(grid: np.ndarray, corridors: list[tuple[int, int]]) -> list[Bar]:
    bars: list[Bar] = []
    for coords in components_mask(grid == RED, connectivity=4):
        assert coords.shape[0] >= 2, "red bars must have height >= 2"
        rows = sorted(int(r) for r in coords[:, 0])
        cols = sorted(set(int(c) for c in coords[:, 1]))
        assert len(cols) == 1, "red components must be vertical one-cell bars"
        assert rows == list(range(rows[0], rows[-1] + 1)), "red bars must be contiguous"
        corridor = _corridor_index(rows[0], corridors)
        r0, r1 = corridors[corridor]
        assert rows[-1] <= r1, "red bar may not cross a wall row"
        height = rows[-1] - rows[0] + 1
        assert 2 <= height <= 5, "bar height must be 2..5"
        bars.append(
            Bar(
                top=rows[0],
                bottom=rows[-1],
                col=cols[0],
                height=height,
                corridor=corridor,
                full_height=(height == (r1 - r0 + 1)),
            )
        )
    return bars


def _reachable_mask(grid: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    passable = (grid == BLACK) | (grid == BLUE)
    seen = np.zeros_like(grid, dtype=bool)
    sr, sc = seed
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    seen[sr, sc] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    wall_rows, door_centers = _wall_rows(g)
    assert 3 <= len(wall_rows) <= 5, "expected 3..5 horizontal walls"
    assert wall_rows == sorted(wall_rows), "wall rows must be ordered"
    corridors = _corridors(g.shape[0], wall_rows)
    assert len(corridors) == len(wall_rows) + 1
    seed = _seed_coord(g)
    seed_corridor = _corridor_index(seed[0], corridors)
    bars = _extract_bars(g, corridors)
    reachable = _reachable_mask(g, seed)
    reachable_corridors = []
    for r0, r1 in corridors:
        reachable_corridors.append(bool(np.any(reachable[r0 : r1 + 1])))
    return {
        "grid": g,
        "wall_rows": wall_rows,
        "door_centers": door_centers,
        "corridors": corridors,
        "seed": seed,
        "seed_corridor": seed_corridor,
        "bars": bars,
        "reachable_mask": reachable,
        "reachable_corridors": reachable_corridors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the lower half of each red strip that lies in blue-reachable door-connected corridor space.

    Input structure:
    - Black is the background.
    - Three to five full-width gray horizontal walls each occupy one row and contain exactly one
      three-cell black door gap, splitting the canvas into horizontal corridor bands.
    - Corridors contain one-cell-wide vertical red strips, and some strips may span the full corridor
      height, acting as occupied dividers inside that corridor band.
    - A single blue seed pixel sits in open corridor space.

    Rule:
    - Flood through black corridor cells starting from the blue seed, crossing between adjacent corridor
      bands only via the wall door gaps; gray walls and red strips are treated as occupied obstacles.
    - Any corridor band containing reachable open space is activated.
    - For every red strip inside an activated corridor band, keep its upper ceil(H/2) cells red and
      recolor its lower floor(H/2) cells teal, where H is that strip's own height.
    - Red strips in non-activated corridor bands stay fully red, the blue seed is erased to black,
      and the gray walls plus black door gaps remain unchanged.

    Key insight:
    - The result depends on reachability through the wall-door topology rather than on the strips alone:
      visually identical red bars are distinguished only by whether their corridor space can be reached
      from the blue seed without crossing gray walls or other red strips.
    """
    info = _analyze(grid)
    out = info["grid"].copy()
    sr, sc = info["seed"]
    out[sr, sc] = BLACK
    for bar in info["bars"]:
        if not info["reachable_corridors"][bar.corridor]:
            continue
        keep = (bar.height + 1) // 2
        for r in range(bar.top + keep, bar.bottom + 1):
            out[r, bar.col] = TEAL
    return out


def _opposite(side: str) -> str:
    return "R" if side == "L" else "L"


def _sample_heights(n_corridors: int, seed_range: tuple[int, int]) -> list[int] | None:
    lo, hi = seed_range
    for _ in range(200):
        heights = [random.randint(2, 5) for _ in range(n_corridors)]
        total_h = sum(heights) + (n_corridors - 1)
        if total_h not in GRID_H_CHOICES:
            continue
        if sum(h >= 3 for h in heights) < n_corridors - 1:
            continue
        if max(heights[lo : hi + 1]) < 3:
            continue
        if max(heights[:lo] + heights[hi + 1 :] or [0]) < 3:
            continue
        return heights
    return None


def _build_side_sequence(length: int) -> list[str]:
    for _ in range(200):
        seq = [random.choice(["L", "R"]) for _ in range(length)]
        if length < 3:
            return seq
        has_equal = any(seq[i - 1] == seq[i] for i in range(1, length))
        has_switch = any(seq[i - 1] != seq[i] for i in range(1, length))
        if has_equal and has_switch:
            return seq
    return [random.choice(["L", "R"]) for _ in range(length)]


def _render_walls(grid: np.ndarray, wall_rows: list[int], door_centers: list[int]) -> None:
    for r, c in zip(wall_rows, door_centers):
        grid[r, :] = GREY
        grid[r, c - 1 : c + 2] = BLACK


def _bar_counts(grid: np.ndarray, wall_rows: list[int]) -> Counter:
    corridors = _corridors(grid.shape[0], wall_rows)
    return Counter(bar.corridor for bar in _extract_bars(grid, corridors))


def _bar_column_ok(grid: np.ndarray, r0: int, r1: int, col: int) -> bool:
    if col <= 0 or col >= grid.shape[1] - 1:
        return False
    return bool(np.all(grid[r0 : r1 + 1, col] == BLACK))


def _place_short_bar(
    grid: np.ndarray,
    corridor_bounds: tuple[int, int],
    col_forbidden: set[int],
) -> tuple[int, int, int] | None:
    r0, r1 = corridor_bounds
    height = r1 - r0 + 1
    if height < 3:
        return None
    max_h = min(5, height - 1)
    if max_h < 2:
        return None
    candidates = list(range(1, grid.shape[1] - 1))
    random.shuffle(candidates)
    for col in candidates:
        if col in col_forbidden:
            continue
        if any(abs(col - prev) <= 1 for prev in col_forbidden):
            continue
        if not _bar_column_ok(grid, r0, r1, col):
            continue
        bar_h = random.randint(2, max_h)
        top_choices = list(range(r0, r1 - bar_h + 2))
        random.shuffle(top_choices)
        for top in top_choices:
            if np.all(grid[top : top + bar_h, col] == BLACK):
                grid[top : top + bar_h, col] = RED
                return top, top + bar_h - 1, col
    return None


def _place_seed(
    grid: np.ndarray,
    corridor_bounds: tuple[int, int],
    split_col: int,
    side: str,
) -> tuple[int, int] | None:
    r0, r1 = corridor_bounds
    rows = [r for r in range(r0 + 1, r1) if 0 <= r < grid.shape[0]]
    if not rows:
        return None
    if side == "L":
        cols = range(1, split_col)
    else:
        cols = range(grid.shape[1] - 2, split_col, -1)
    center_r = (r0 + r1) / 2.0
    candidates: list[tuple[float, int, int]] = []
    for r in rows:
        for c in cols:
            if grid[r, c] != BLACK:
                continue
            if any(
                in_bounds(grid.shape, nr, nc) and grid[nr, nc] in (RED, GREY)
                for nr, nc in neighbors4(r, c)
            ):
                continue
            tie = c if side == "L" else -c
            candidates.append((abs(r - center_r), tie, c))
    if not candidates:
        return None
    _, _, c = min(candidates)
    best_rows = [r for r in rows if grid[r, c] == BLACK and not any(
        in_bounds(grid.shape, nr, nc) and grid[nr, nc] in (RED, GREY)
        for nr, nc in neighbors4(r, c)
    )]
    r = min(best_rows, key=lambda rr: abs(rr - center_r))
    grid[r, c] = BLUE
    return r, c


def _generator_analysis_ok(
    grid: np.ndarray,
    expected_reachable: set[int],
    barrier_corridors: set[int],
    need_barrier_pass: bool,
    need_open_switch: bool,
) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    bars = info["bars"]
    if not (4 <= len(bars) <= 7):
        return False
    counts = Counter(bar.corridor for bar in bars)
    if any(v > 2 for v in counts.values()):
        return False

    reached = {idx for idx, flag in enumerate(info["reachable_corridors"]) if flag}
    if reached != expected_reachable:
        return False
    if len(reached) == len(info["corridors"]):
        return False

    reachable_bars = [bar for bar in bars if bar.corridor in reached]
    unreachable_bars = [bar for bar in bars if bar.corridor not in reached]
    if not reachable_bars or not unreachable_bars:
        return False

    passable = (grid == BLACK) | (grid == BLUE)
    partial_corridors = set()
    for idx, (r0, r1) in enumerate(info["corridors"]):
        sub_passable = passable[r0 : r1 + 1]
        sub_reached = info["reachable_mask"][r0 : r1 + 1]
        if np.any(sub_passable & sub_reached) and np.any(sub_passable & ~sub_reached):
            partial_corridors.add(idx)
    if not partial_corridors.issubset(barrier_corridors & reached):
        return False
    for idx in partial_corridors:
        corridor_bars = [bar for bar in bars if bar.corridor == idx]
        if len(corridor_bars) != 1 or not corridor_bars[0].full_height:
            return False

    wall_rows = info["wall_rows"]
    door_centers = info["door_centers"]
    if need_barrier_pass:
        found = False
        for idx in reached:
            if idx not in barrier_corridors:
                continue
            if idx == 0 or idx == len(info["corridors"]) - 1:
                continue
            if idx not in partial_corridors:
                top_side = "L" if door_centers[idx - 1] < door_centers[idx] else "R"
                bottom_side = top_side
                found = True
                if top_side == bottom_side:
                    break
        if not found:
            return False

    if need_open_switch:
        found = False
        for idx in reached:
            if idx in barrier_corridors:
                continue
            if idx == 0 or idx == len(info["corridors"]) - 1:
                continue
            top_side = "L" if door_centers[idx - 1] < grid.shape[1] // 2 else "R"
            bottom_side = "L" if door_centers[idx] < grid.shape[1] // 2 else "R"
            if top_side != bottom_side:
                found = True
                break
        if not found:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(grid == BLUE)) != 1 or int(np.count_nonzero(out == BLUE)) != 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_corridors = random.choice(CORRIDOR_COUNT_CHOICES)
    interval_len_choices = [t for t in range(3, n_corridors) if t <= n_corridors - 1]
    if not interval_len_choices:
        return None
    if n_corridors >= 5:
        interval_len_choices = [t for t in interval_len_choices if t >= 4] or interval_len_choices
    interval_len = random.choice(interval_len_choices)
    left = random.randint(0, n_corridors - interval_len)
    right = left + interval_len - 1
    reachable_corridors = set(range(left, right + 1))

    heights = _sample_heights(n_corridors, (left, right))
    if heights is None:
        return None

    eligible_seed_corridors = [idx for idx in reachable_corridors if heights[idx] >= 3]
    unreachable_tall = [idx for idx in range(n_corridors) if idx not in reachable_corridors and heights[idx] >= 3]
    if not eligible_seed_corridors or not unreachable_tall:
        return None
    seed_corridor = random.choice(eligible_seed_corridors)

    total_h = sum(heights) + (n_corridors - 1)
    width = random.choice(GRID_W_CHOICES)
    split_col = random.randint(5, width - 6)

    wall_rows: list[int] = []
    cursor = heights[0]
    for idx in range(n_corridors - 1):
        wall_rows.append(cursor)
        cursor += 1 + heights[idx + 1]
    corridors = _corridors(total_h, wall_rows)

    wall_sides = [random.choice(["L", "R"]) for _ in range(n_corridors - 1)]
    inner_seq = _build_side_sequence(interval_len - 1)
    for wall_idx, side in enumerate(inner_seq, start=left):
        wall_sides[wall_idx] = side

    barrier_corridors: set[int] = set()
    if left > 0:
        barrier_corridors.add(left)
        wall_sides[left - 1] = _opposite(wall_sides[left])
    if right < n_corridors - 1:
        barrier_corridors.add(right)
        wall_sides[right] = _opposite(wall_sides[right - 1])

    equal_pair_corridors = [idx for idx in range(left + 1, right) if wall_sides[idx - 1] == wall_sides[idx]]
    switch_pair_corridors = [idx for idx in range(left + 1, right) if wall_sides[idx - 1] != wall_sides[idx]]
    if equal_pair_corridors:
        barrier_corridors.add(random.choice(equal_pair_corridors))
    for idx in equal_pair_corridors:
        if idx not in barrier_corridors and random.random() < 0.35:
            barrier_corridors.add(idx)

    grid = np.full((total_h, width), BLACK, dtype=int)
    door_centers: list[int] = []
    for side in wall_sides:
        if side == "L":
            center = random.randint(2, split_col - 3)
        else:
            center = random.randint(split_col + 3, width - 3)
        door_centers.append(center)
    _render_walls(grid, wall_rows, door_centers)

    bar_columns_by_corridor: dict[int, set[int]] = {idx: set() for idx in range(n_corridors)}
    for idx in barrier_corridors:
        r0, r1 = corridors[idx]
        grid[r0 : r1 + 1, split_col] = RED
        bar_columns_by_corridor[idx].add(split_col)

    counts = _bar_counts(grid, wall_rows)
    open_reachable_tall = [
        idx
        for idx in sorted(reachable_corridors)
        if idx not in barrier_corridors and heights[idx] >= 3 and counts[idx] < 2
    ]
    if not open_reachable_tall:
        return None

    forced = [random.choice(open_reachable_tall), random.choice(unreachable_tall)]
    random.shuffle(forced)
    for corridor_idx in forced:
        placed = _place_short_bar(grid, corridors[corridor_idx], bar_columns_by_corridor[corridor_idx])
        if placed is None:
            return None
        bar_columns_by_corridor[corridor_idx].add(placed[2])
        counts[corridor_idx] += 1

    target_bars_lo = max(4, int(sum(counts.values())))
    target_bars_hi = min(7, sum(2 - counts[idx] for idx in range(n_corridors)) + int(sum(counts.values())))
    if target_bars_lo > target_bars_hi:
        return None
    target_bars = random.randint(target_bars_lo, target_bars_hi)

    fill_candidates = [idx for idx in range(n_corridors) if heights[idx] >= 3 and counts[idx] < 2]
    random.shuffle(fill_candidates)
    while int(sum(counts.values())) < target_bars:
        options = [idx for idx in fill_candidates if counts[idx] < 2]
        if not options:
            return None
        corridor_idx = random.choice(options)
        if corridor_idx in barrier_corridors and corridor_idx in reachable_corridors:
            counts[corridor_idx] = 2
            continue
        placed = _place_short_bar(grid, corridors[corridor_idx], bar_columns_by_corridor[corridor_idx])
        if placed is None:
            counts[corridor_idx] = 2
            continue
        bar_columns_by_corridor[corridor_idx].add(placed[2])
        counts[corridor_idx] += 1

    if seed_corridor in barrier_corridors:
        if seed_corridor == 0:
            seed_side = wall_sides[0]
        elif seed_corridor == n_corridors - 1:
            seed_side = wall_sides[-1]
        elif seed_corridor in equal_pair_corridors:
            seed_side = wall_sides[seed_corridor]
        elif seed_corridor == left:
            seed_side = wall_sides[left]
        else:
            seed_side = wall_sides[seed_corridor - 1]
    else:
        if seed_corridor < n_corridors - 1 and seed_corridor in reachable_corridors and seed_corridor < right:
            seed_side = wall_sides[seed_corridor]
        elif seed_corridor > 0 and seed_corridor > left:
            seed_side = wall_sides[seed_corridor - 1]
        else:
            seed_side = "L"

    if _place_seed(grid, corridors[seed_corridor], split_col, seed_side) is None:
        return None

    need_barrier_pass = bool(equal_pair_corridors)
    need_open_switch = bool(switch_pair_corridors)
    if not _generator_analysis_ok(grid, reachable_corridors, barrier_corridors, need_barrier_pass, need_open_switch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
