import random

import numpy as np


BG = 0
WALL = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

MIN_SIZE = 12
MAX_SIZE = 24
MIN_SEGMENTS = 3
MAX_SEGMENTS = 7
MIN_SOURCES = 3
MAX_SOURCES = 7
GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _segment_cells(r0: int, c0: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "h":
        return [(r0, c0 + k) for k in range(length)]
    return [(r0 + k, c0) for k in range(length)]


def _source_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BG) & (grid != WALL))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _wall_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(grid == WALL)}


def _first_wall_hit(
    grid: np.ndarray,
    src_r: int,
    src_c: int,
    dr: int,
    dc: int,
) -> tuple[int, int, int] | None:
    h, w = grid.shape
    rr = src_r + dr
    cc = src_c + dc
    gap = 0
    while _in_bounds(h, w, rr, cc):
        val = int(grid[rr, cc])
        if val == WALL:
            return rr, cc, gap
        if val == BG:
            gap += 1
        rr += dr
        cc += dc
    return None


def _hit_candidates(grid: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
    candidates: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for src_r, src_c, color in _source_cells(grid):
        for dr, dc in DIRS4:
            hit = _first_wall_hit(grid, src_r, src_c, dr, dc)
            if hit is None:
                continue
            hit_r, hit_c, gap = hit
            candidates.setdefault((hit_r, hit_c), []).append((gap, src_r, src_c, color))
    return candidates


def _winning_hits(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    winners: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for cell, entries in _hit_candidates(grid).items():
        winners[cell] = min(entries, key=lambda item: (item[0], item[1], item[2]))
    return winners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey wall pixel that is first struck by a cardinal ray from a colored source.

    Input structure:
    - The background is black.
    - Grey pixels form short horizontal or vertical wall segments.
    - Every other non-black, non-grey pixel is a colored single-cell source.

    Rule:
    - Each source emits one ray upward, downward, leftward, and rightward.
    - A ray passes through background cells without marking them and stops at the first grey wall pixel it meets.
    - That struck wall pixel changes to the source color.
    - If multiple sources strike the same wall pixel, choose the source with the smallest ray distance, where distance is the number of background cells between source and wall; if distances tie, choose the source on the smaller row.
    - Grey wall pixels that are never struck stay grey.

    Key insight:
    - Only the first wall cell on each of the four source rays matters, so the task is a nearest-hit competition over wall pixels rather than drawing visible beams through the background.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for (r, c), (_gap, _src_r, _src_c, color) in _winning_hits(g).items():
        out[r, c] = int(color)
    return out


def _can_place_segment(
    cells: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if (r, c) in wall_cells or (r, c) in forbidden:
            return False
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in wall_cells and (nr, nc) not in cells:
                return False
    return True


def _can_place_source(
    pos: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    h: int,
    w: int,
) -> bool:
    r, c = pos
    if not _in_bounds(h, w, r, c):
        return False
    if pos in wall_cells or pos in source_cells:
        return False
    if r in used_rows or c in used_cols:
        return False
    for nr, nc in _neighbors8(r, c):
        if (nr, nc) in wall_cells or (nr, nc) in source_cells:
            return False
    return True


def _render_grid(
    h: int,
    w: int,
    wall_cells: set[tuple[int, int]],
    sources: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL
    for r, c, color in sources:
        grid[r, c] = int(color)
    return grid


def _contest_setup(h: int, w: int, tie_case: bool) -> dict | None:
    for _ in range(120):
        length = random.randint(3, 6)
        row = random.randint(3, h - 4)
        col0 = random.randint(2, w - length - 3)
        offset = random.randint(1, length - 2)
        target = (row, col0 + offset)

        top_max = row - 1
        bottom_max = h - row - 2
        if tie_case:
            max_gap = min(4, top_max, bottom_max)
            if max_gap < 1:
                continue
            top_gap = bottom_gap = random.randint(1, max_gap)
        else:
            top_choices = [g for g in range(1, min(4, top_max) + 1)]
            bottom_choices = [g for g in range(1, min(4, bottom_max) + 1)]
            pairs = [(a, b) for a in top_choices for b in bottom_choices if a != b]
            if not pairs:
                continue
            top_gap, bottom_gap = random.choice(pairs)

        top_src = (row - top_gap - 1, target[1])
        bottom_src = (row + bottom_gap + 1, target[1])

        segment = _segment_cells(row, col0, "h", length)
        forbidden = set()
        for rr in range(top_src[0] + 1, row):
            forbidden.add((rr, target[1]))
        for rr in range(row + 1, bottom_src[0]):
            forbidden.add((rr, target[1]))
        for sr, sc in [top_src, bottom_src]:
            forbidden.add((sr, sc))
            for nr, nc in _neighbors8(sr, sc):
                if _in_bounds(h, w, nr, nc):
                    forbidden.add((nr, nc))

        return {
            "segment": segment,
            "target": target,
            "top_src": top_src,
            "bottom_src": bottom_src,
            "top_gap": top_gap,
            "bottom_gap": bottom_gap,
            "forbidden": forbidden,
        }
    return None


def _add_random_segments(
    h: int,
    w: int,
    wall_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    segments = [sorted(wall_cells)]
    target_count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    tries = 0
    while len(segments) < target_count and tries < 600:
        tries += 1
        orient = random.choice(["h", "v"])
        length = random.randint(2, 6)
        if orient == "h":
            r0 = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
        else:
            r0 = random.randint(1, h - length - 1)
            c0 = random.randint(1, w - 2)
        cells = _segment_cells(r0, c0, orient, length)
        if not _can_place_segment(cells, wall_cells, forbidden, h, w):
            continue
        wall_cells.update(cells)
        segments.append(cells)
    if len(segments) < target_count:
        return None
    return segments


def _source_hit_summary(grid: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    summary: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for src_r, src_c, _color in _source_cells(grid):
        hits: list[tuple[int, int, int]] = []
        for dr, dc in DIRS4:
            hit = _first_wall_hit(grid, src_r, src_c, dr, dc)
            if hit is not None:
                hits.append(hit)
        summary[(src_r, src_c)] = hits
    return summary


def _hit_source_candidate(
    wall_cells: set[tuple[int, int]],
    source_positions: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    blocked_col: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    walls = list(wall_cells)
    random.shuffle(walls)
    choices = DIRS4[:]
    for target in walls:
        random.shuffle(choices)
        for dr, dc in choices:
            dists = list(range(1, 5))
            random.shuffle(dists)
            for gap in dists:
                sr = target[0] - dr * (gap + 1)
                sc = target[1] - dc * (gap + 1)
                if sc == blocked_col:
                    continue
                pos = (sr, sc)
                if not _can_place_source(pos, wall_cells, source_positions, used_rows, used_cols, h, w):
                    continue
                grid = _render_grid(h, w, wall_cells, [(r, c, 1) for r, c in source_positions | {pos}])
                hit = _first_wall_hit(grid, sr, sc, dr, dc)
                if hit is not None and hit[:2] == target and hit[2] == gap:
                    return pos
    return None


def _miss_source_candidate(
    wall_cells: set[tuple[int, int]],
    source_positions: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    blocked_col: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    for _ in range(250):
        pos = (random.randint(1, h - 2), random.randint(1, w - 2))
        if pos[1] == blocked_col:
            continue
        if not _can_place_source(pos, wall_cells, source_positions, used_rows, used_cols, h, w):
            continue
        grid = _render_grid(h, w, wall_cells, [(r, c, 1) for r, c in source_positions | {pos}])
        if all(_first_wall_hit(grid, pos[0], pos[1], dr, dc) is None for dr, dc in DIRS4):
            return pos
    return None


def _validate_generated(
    grid: np.ndarray,
    contest: dict,
    tie_case: bool,
) -> bool:
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    walls = _wall_cells(grid)
    winners = _winning_hits(grid)
    candidates = _hit_candidates(grid)
    if contest["target"] not in candidates or len(candidates[contest["target"]]) < 2:
        return False
    if len(walls - set(winners)) < 1:
        return False

    top_src = contest["top_src"]
    bottom_src = contest["bottom_src"]
    top_color = int(grid[top_src])
    bottom_color = int(grid[bottom_src])
    winner = winners.get(contest["target"])
    if winner is None:
        return False

    if tie_case:
        if contest["top_gap"] != contest["bottom_gap"]:
            return False
        if int(winner[3]) != top_color:
            return False
    else:
        expected = top_color if contest["top_gap"] < contest["bottom_gap"] else bottom_color
        if int(winner[3]) != expected:
            return False

    summary = _source_hit_summary(grid)
    if not any(len(hits) == 0 for hits in summary.values()):
        return False
    if not any(len(hits) > 0 for hits in summary.values()):
        return False
    return bool(out[contest["target"]] == winner[3])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    tie_case = bool(random.getrandbits(1))

    contest = _contest_setup(h, w, tie_case)
    if contest is None:
        return None

    wall_cells = set(contest["segment"])
    segments = _add_random_segments(h, w, wall_cells, set(contest["forbidden"]))
    if segments is None:
        return None

    total_sources = random.randint(MIN_SOURCES, min(MAX_SOURCES, len(SOURCE_COLORS)))
    extra_hit_goal = random.randint(0, min(3, total_sources - 3))
    extra_miss_goal = total_sources - 2 - extra_hit_goal
    if extra_miss_goal < 1:
        extra_miss_goal = 1
        extra_hit_goal = total_sources - 3

    source_positions = {contest["top_src"], contest["bottom_src"]}
    used_rows = {contest["top_src"][0], contest["bottom_src"][0]}
    used_cols = {contest["target"][1]}

    hit_positions: list[tuple[int, int]] = []
    miss_positions: list[tuple[int, int]] = []

    for _ in range(extra_hit_goal):
        pos = _hit_source_candidate(
            wall_cells,
            source_positions,
            used_rows,
            used_cols,
            contest["target"][1],
            h,
            w,
        )
        if pos is None:
            return None
        hit_positions.append(pos)
        source_positions.add(pos)
        used_rows.add(pos[0])
        used_cols.add(pos[1])

    for _ in range(extra_miss_goal):
        pos = _miss_source_candidate(
            wall_cells,
            source_positions,
            used_rows,
            used_cols,
            contest["target"][1],
            h,
            w,
        )
        if pos is None:
            return None
        miss_positions.append(pos)
        source_positions.add(pos)
        used_rows.add(pos[0])
        used_cols.add(pos[1])

    ordered_positions = [contest["top_src"], contest["bottom_src"]] + hit_positions + miss_positions
    colors = random.sample(SOURCE_COLORS, len(ordered_positions))
    sources = [(r, c, color) for (r, c), color in zip(ordered_positions, colors)]

    grid = _render_grid(h, w, wall_cells, sources)
    if not _validate_generated(grid, contest, tie_case):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid ARC input after {GEN_ATTEMPTS} attempts")
