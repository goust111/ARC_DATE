import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
GRAY = 5
PAIR_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _queen_step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "pair endpoints must be queen-aligned"
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    assert steps > 0, "expected two distinct endpoints"
    return int(sr), int(sc), int(steps)


def _queen_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc, steps = _queen_step(a, b)
    r0, c0 = a
    return [(int(r0 + i * sr), int(c0 + i * sc)) for i in range(steps + 1)]


def _settle_row(gray_mask: np.ndarray, start_row: int, col: int) -> int:
    h = gray_mask.shape[0]
    for r in range(int(start_row), h - 1):
        if bool(gray_mask[r + 1, int(col)]):
            return int(r)
    return int(h - 1)


def _pair_info(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [color for color in sorted(map(int, np.unique(g))) if color not in (BG, YELLOW, GRAY) and int(np.sum(g == color)) == 2]
    assert 2 <= len(colors) <= 3, "expected 2 or 3 marker pairs"

    gray_mask = g == GRAY
    pairs = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "every marker color must appear exactly twice"
        starts = [tuple(map(int, rc)) for rc in coords]
        dests = [(int(_settle_row(gray_mask, r, c)), int(c)) for r, c in starts]
        pairs.append(
            {
                "color": int(color),
                "starts": starts,
                "dests": dests,
                "priority": (min(c for _r, c in starts), max(c for _r, c in starts), int(color)),
            }
        )

    assert len({dest for pair in pairs for dest in pair["dests"]}) == 2 * len(pairs), "settled markers must be unique"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each same-color marker pair to its gravity-settled endpoints, then connect the settled endpoints while recoloring obstacle crossings.

    Input structure:
    - The grid has black background and several gray horizontal platform segments.
    - Two or three colors appear exactly twice each as singleton markers, with each pair representing one connection to draw.
    - Markers begin above their settled positions and fall vertically; platforms are static obstacles and may also lie on the later line paths.

    Rule:
    - Move every marker straight down until the next cell would be outside the grid or a gray platform cell, ignoring other markers while falling.
    - Remove the original marker pixels, keep the gray platforms, and treat the settled marker positions as the pair endpoints.
    - For each pair, walk the inclusive horizontal, vertical, or 45-degree diagonal segment between its settled endpoints.
    - Path cells that were gray in the original input become yellow; all other path cells become that pair's color. If multiple pairs want the same path cell, the pair whose input markers have the smaller column wins.

    Key insight:
    - The same gray platforms play two roles at once: they determine where the falling markers stop, and they also act as special recolor triggers when a later connection line passes through them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pairs = _pair_info(g)
    out = g.copy()
    claimed = np.zeros(g.shape, dtype=bool)

    for pair in pairs:
        for r, c in pair["starts"]:
            out[int(r), int(c)] = BG

    for pair in sorted(pairs, key=lambda item: item["priority"]):
        path = _queen_path(pair["dests"][0], pair["dests"][1])
        for r, c in path:
            if claimed[int(r), int(c)]:
                continue
            out[int(r), int(c)] = YELLOW if int(g[int(r), int(c)]) == GRAY else int(pair["color"])
            claimed[int(r), int(c)] = True

    return out


def _segment_options(width: int, required_col: int, forbidden_cols: Iterable[int], board_width: int) -> list[tuple[int, int]]:
    forbidden = {int(c) for c in forbidden_cols}
    options = []
    for c0 in range(max(0, int(required_col) - int(width) + 1), min(int(required_col), int(board_width) - int(width)) + 1):
        c1 = c0 + int(width) - 1
        cols = range(c0, c1 + 1)
        if any(col in forbidden for col in cols):
            continue
        options.append((int(c0), int(c1)))
    return options


def _choose_support_segment(required_col: int, forbidden_cols: Iterable[int], board_width: int) -> tuple[int, int] | None:
    choices = []
    widths = list(range(3, 7))
    random.shuffle(widths)
    for width in widths:
        choices.extend(_segment_options(width, required_col, forbidden_cols, board_width))
    return random.choice(choices) if choices else None


def _type_of_pair(a: tuple[int, int], b: tuple[int, int]) -> str:
    if a[0] == b[0]:
        return "horizontal"
    if a[1] == b[1]:
        return "vertical"
    return "diagonal"


def _platform_rows_ok(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        cols = np.flatnonzero(grid[r] == GRAY)
        if cols.size == 0:
            continue
        if r == h - 1:
            return False
        if int(cols[-1]) - int(cols[0]) + 1 != int(cols.size):
            return False
        if not 3 <= int(cols.size) <= 6:
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False
    if not _platform_rows_ok(grid):
        return False

    pair_colors = [color for color in sorted(map(int, np.unique(grid))) if color not in (BG, YELLOW, GRAY) and int(np.sum(grid == color)) == 2]
    if len(pair_colors) not in (2, 3):
        return False

    upper_half_limit = h // 2 - 1
    if upper_half_limit < 1:
        return False

    pairs = _pair_info(grid)
    if len({pair["priority"][0] for pair in pairs}) != len(pairs):
        return False
    if any(any(r > upper_half_limit for r, _c in pair["starts"]) for pair in pairs):
        return False
    if any(any(dest_r <= start_r for (start_r, _), (dest_r, _) in zip(pair["starts"], pair["dests"])) for pair in pairs):
        return False

    pair_types = {_type_of_pair(pair["dests"][0], pair["dests"][1]) for pair in pairs}
    if "horizontal" not in pair_types or "diagonal" not in pair_types:
        return False

    paths = []
    gray_hits = 0
    for pair in pairs:
        path = _queen_path(pair["dests"][0], pair["dests"][1])
        paths.append(path)
        gray_hits += sum(int(grid[r, c] == GRAY) for r, c in path)
    if gray_hits == 0:
        return False

    overlap_black = False
    for i in range(len(paths)):
        cells_i = set(paths[i])
        for j in range(i + 1, len(paths)):
            shared = cells_i & set(paths[j])
            if any(int(grid[r, c]) != GRAY for r, c in shared):
                overlap_black = True
                break
        if overlap_black:
            break
    if not overlap_black:
        return False

    covered = {cell for path in paths for cell in path}
    untouched_gray = any((r, c) not in covered for r, c in map(tuple, np.argwhere(grid == GRAY)))
    if not untouched_gray:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.sum(out == YELLOW)) == 0:
        return False
    if any(int(np.sum(out == pair["color"])) == 0 for pair in pairs):
        return False
    return True


def _free_fall_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if not np.any(grid[:, c] == GRAY)]


def _pick_start_rows(grid: np.ndarray, endpoints: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h = grid.shape[0]
    upper_half_limit = h // 2 - 1
    gray_mask = grid == GRAY
    used = set()
    starts = []
    ordered = sorted(endpoints, key=lambda rc: (rc[0], rc[1]))
    for dest_r, col in ordered:
        candidates = [
            int(r)
            for r in range(upper_half_limit + 1)
            if (int(r), int(col)) not in used
            and int(grid[int(r), int(col)]) == BG
            and int(_settle_row(gray_mask, int(r), int(col))) == int(dest_r)
        ]
        if not candidates:
            return None
        start_r = random.choice(candidates)
        used.add((int(start_r), int(col)))
        starts.append((int(start_r), int(col)))
    return starts


def _attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    upper_half_limit = h // 2 - 1

    if w < 10 or upper_half_limit < 2:
        return None

    grid = np.zeros((h, w), dtype=int)

    width_a = random.randint(4, 6)
    c0_a_min = 3
    c0_a_max = w - width_a - 4
    if c0_a_min > c0_a_max:
        return None
    c0_a = random.randint(c0_a_min, c0_a_max)
    c1_a = c0_a + width_a - 1
    p_a_min = upper_half_limit + 3
    p_a_max = h - 4
    if p_a_min > p_a_max:
        return None
    p_a = random.randint(p_a_min, p_a_max)
    r_a = p_a - 1
    x_cross = random.randint(c0_a + 1, c1_a - 1)

    sign = random.choice((-1, 1))
    possible_pb = []
    for p_b in range(3, r_a - 1):
        r_top = p_b - 1
        c_top = x_cross - sign * (r_a - r_top)
        c_bottom = x_cross + sign * ((h - 1) - r_a)
        if not (0 <= c_top < w and 0 <= c_bottom < w):
            continue
        if c0_a <= c_top <= c1_a or c0_a <= c_bottom <= c1_a:
            continue
        possible_pb.append((p_b, r_top, c_top, c_bottom))
    if not possible_pb:
        return None
    p_b, r_top_b, c_top_b, c_bottom_b = random.choice(possible_pb)

    seg_b = _choose_support_segment(c_top_b, {c0_a, c1_a, c_bottom_b}, w)
    if seg_b is None:
        return None
    c0_b, c1_b = seg_b

    grid[p_a, c0_a : c1_a + 1] = GRAY
    grid[p_b, c0_b : c1_b + 1] = GRAY

    pairs = [
        {"dests": [(r_a, c0_a), (r_a, c1_a)]},
        {"dests": [(r_top_b, c_top_b), (h - 1, c_bottom_b)]},
    ]

    if random.random() < 0.5:
        free_cols = _free_fall_columns(grid)
        blocked = {c_bottom_b}
        intervals = []
        for left in free_cols:
            for right in free_cols:
                if right - left < 3:
                    continue
                if any(left <= col <= right for col in blocked):
                    continue
                intervals.append((left, right))
        if intervals:
            left_c, right_c = random.choice(intervals)
            pairs.append({"dests": [(h - 1, left_c), (h - 1, right_c)]})

    pair_count = len(pairs)
    colors = random.sample(PAIR_COLORS, pair_count)
    all_endpoints = [dest for pair in pairs for dest in pair["dests"]]
    starts = _pick_start_rows(grid, all_endpoints)
    if starts is None:
        return None

    endpoint_to_start = {dest: start for dest, start in zip(sorted(all_endpoints), starts)}
    for color, pair in zip(colors, pairs):
        pair["color"] = int(color)
        pair_starts = [endpoint_to_start[dest] for dest in sorted(pair["dests"])]
        pair["starts"] = pair_starts
        for r, c in pair_starts:
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(color)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
