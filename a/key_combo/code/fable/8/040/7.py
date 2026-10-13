from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds, neighbors4


BG = int(Color.BLACK)
FRAME = int(Color.GREY)
YELLOW = int(Color.YELLOW)
RED = int(Color.RED)
CORE_PALETTE = (int(Color.BLUE), int(Color.GREEN), int(Color.PINK), int(Color.ORANGE), int(Color.TEAL), int(Color.MAROON))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _randint_or_none(lo: int, hi: int) -> int | None:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        return None
    return int(random.randint(lo, hi))


def _reachable(grid: np.ndarray, bg: int, start: tuple[int, int], moves: list[tuple[int, int]], bounds: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    seen = np.zeros_like(grid, dtype=bool)
    stack = [start]
    while stack:
        r, c = stack.pop()
        if seen[r, c]:
            continue
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        if int(grid[r, c]) != int(bg):
            continue
        seen[r, c] = True
        for dr, dc in moves:
            nr, nc = int(r + dr), int(c + dc)
            if in_bounds(grid.shape, nr, nc) and not seen[nr, nc]:
                stack.append((nr, nc))
    return seen


def _corridor_mask(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], bg: int) -> np.ndarray:
    if start == end:
        out = np.zeros_like(grid, dtype=bool)
        out[start] = True
        return out

    row_step = _sign(int(end[0]) - int(start[0]))
    col_step = _sign(int(end[1]) - int(start[1]))
    moves: list[tuple[int, int]] = []
    if row_step:
        moves.append((int(row_step), 0))
    if col_step:
        moves.append((0, int(col_step)))
    assert moves, "start and end must differ"

    bounds = (
        min(int(start[0]), int(end[0])),
        min(int(start[1]), int(end[1])),
        max(int(start[0]), int(end[0])),
        max(int(start[1]), int(end[1])),
    )
    forward = _reachable(grid, bg, start, moves, bounds)
    backward = _reachable(grid, bg, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start] and path[end], "door exteriors must be connected by at least one monotone shortest path"
    return path


def _detect_doors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    doors: list[dict] = []

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            core_color = int(g[r, c])
            if core_color == bg:
                continue

            bg_neighbors: list[tuple[int, int, int, int]] = []
            fg_neighbors: list[int] = []
            for nr, nc in neighbors4(r, c):
                value = int(g[nr, nc])
                if value == bg:
                    bg_neighbors.append((int(nr - r), int(nc - c), nr, nc))
                else:
                    fg_neighbors.append(value)

            if len(bg_neighbors) != 1 or len(fg_neighbors) != 3:
                continue
            if len(set(fg_neighbors)) != 1:
                continue

            frame_color = int(fg_neighbors[0])
            if frame_color == core_color:
                continue

            dr, dc, gap_r, gap_c = bg_neighbors[0]
            ext_r, ext_c = int(r + 2 * dr), int(c + 2 * dc)
            if not in_bounds(g.shape, ext_r, ext_c) or int(g[ext_r, ext_c]) != bg:
                continue

            ok = True
            frame_cells: list[tuple[int, int]] = []
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) == (r, c) or (rr, cc) == (gap_r, gap_c):
                        continue
                    if int(g[rr, cc]) != frame_color:
                        ok = False
                        break
                    frame_cells.append((int(rr), int(cc)))
                if not ok:
                    break
            if not ok:
                continue

            doors.append(
                {
                    "center": (int(r), int(c)),
                    "gap": (int(gap_r), int(gap_c)),
                    "ext": (int(ext_r), int(ext_c)),
                    "dir": (int(dr), int(dc)),
                    "core_color": int(core_color),
                    "frame_color": int(frame_color),
                    "frame_cells": tuple(frame_cells),
                }
            )

    assert doors, "expected at least one door"
    centers = [door["center"] for door in doors]
    assert len(set(centers)) == len(doors), "duplicate door detection"

    claimed = set()
    for door in doors:
        claimed.add(door["center"])
        claimed.update(door["frame_cells"])
    fg_cells = {tuple(map(int, coord)) for coord in np.argwhere(g != bg)}
    assert fg_cells == claimed, "all foreground pixels must belong to detected doors"
    return doors


def _pair_doors_by_color(doors: list[dict]) -> list[tuple[dict, dict]]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for door in doors:
        by_color[int(door["core_color"])].append(door)

    pairs: list[tuple[dict, dict]] = []
    for color, group in sorted(by_color.items()):
        assert len(group) == 2, f"core color {color} must appear on exactly two door centers"
        pairs.append((group[0], group[1]))
    assert pairs, "expected at least one color pair"
    return pairs


def _render_output(grid: np.ndarray, pairs: list[tuple[dict, dict]]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    counts = np.zeros_like(g, dtype=int)
    full_masks: list[np.ndarray] = []

    for door_a, door_b in pairs:
        path = _corridor_mask(g, door_a["ext"], door_b["ext"], bg)
        full = path.copy()
        full[door_a["gap"]] = True
        full[door_b["gap"]] = True
        counts[full] += 1
        full_masks.append(full)

    out = g.copy()
    out[(counts == 1) & (g == bg)] = YELLOW
    out[(counts >= 2) & (g == bg)] = RED
    return out, full_masks, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored door cores and paint the union of their shortest monotone doorway corridors, turning overlaps red.

    Input structure:
    - The grid has a dominant background plus several small doorway frames.
    - Each frame is a 3x3 ring with one orthogonal gap, and its center is a colored door core.
    - Every core color appears exactly twice; the background cell just outside each gap is that door's exterior cell.

    Rule:
    - Detect every doorway frame and pair the two doors whose center cores share the same color.
    - For each pair, treat all existing non-background pixels as blockers and collect every shortest 4-neighbor path from one exterior cell to the other that only steps toward the destination row and/or column.
    - Paint the union of those monotone shortest paths yellow, also painting the two doorway gaps yellow.
    - If a background cell belongs to corridor unions from two different color pairs, paint it red instead.

    Key insight:
    - Each pair contributes not one chosen route but the full shadow of all shortest monotone routes between the two exterior cells, so overlaps between different pairs are where the special red cells come from.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    doors = _detect_doors(g)
    pairs = _pair_doors_by_color(doors)
    out, _full_masks, _counts = _render_output(g, pairs)
    return out


def _stamp_door(grid: np.ndarray, ext: tuple[int, int], direction: tuple[int, int], frame_color: int, core_color: int) -> dict | None:
    ext_r, ext_c = map(int, ext)
    dr, dc = map(int, direction)
    center = (int(ext_r - 2 * dr), int(ext_c - 2 * dc))
    gap = (int(ext_r - dr), int(ext_c - dc))
    cr, cc = center
    H, W = grid.shape

    if not (1 <= cr < H - 1 and 1 <= cc < W - 1):
        return None
    if not in_bounds(grid.shape, gap[0], gap[1]) or not in_bounds(grid.shape, ext_r, ext_c):
        return None

    footprint: list[tuple[int, int]] = []
    for rr in range(cr - 1, cr + 2):
        for cc2 in range(cc - 1, cc + 2):
            if (rr, cc2) == center or (rr, cc2) == gap:
                continue
            footprint.append((int(rr), int(cc2)))

    cells_to_keep_bg = [center, gap, (ext_r, ext_c), *footprint]
    if any(int(grid[r, c]) != BG for r, c in cells_to_keep_bg):
        return None

    for r, c in footprint:
        grid[r, c] = int(frame_color)
    grid[center] = int(core_color)
    return {
        "center": center,
        "gap": gap,
        "ext": (int(ext_r), int(ext_c)),
        "dir": (int(dr), int(dc)),
        "core_color": int(core_color),
        "frame_color": int(frame_color),
        "frame_cells": tuple(footprint),
    }


def _scenario_rect_plus_line_cross(H: int, W: int) -> tuple[list[list[tuple[tuple[int, int], tuple[int, int]]]], bool] | None:
    if H < 16 or W < 16:
        return None

    r_top_line = _randint_or_none(3, max(3, H // 5))
    if r_top_line is None:
        return None
    r1 = _randint_or_none(r_top_line + 2, max(r_top_line + 2, H // 2 - 3))
    if r1 is None:
        return None
    r2 = _randint_or_none(r1 + 3, H - 7)
    if r2 is None:
        return None
    r_bottom_line = _randint_or_none(r2 + 2, H - 4)
    if r_bottom_line is None:
        return None

    c1 = _randint_or_none(3, max(3, W // 3))
    if c1 is None:
        return None
    c2 = _randint_or_none(c1 + 4, W - 4)
    if c2 is None:
        return None
    c_mid = _randint_or_none(c1 + 1, c2 - 1)
    if c_mid is None:
        return None

    pairs = [
        [((r1, c1), (1, 0)), ((r2, c2), (0, -1))],
        [((r_top_line, c_mid), (1, 0)), ((r_bottom_line, c_mid), (-1, 0))],
    ]
    return pairs, True


def _scenario_two_rectangles_overlap(H: int, W: int) -> tuple[list[list[tuple[tuple[int, int], tuple[int, int]]]], bool] | None:
    if H < 17 or W < 17:
        return None

    r1 = _randint_or_none(3, max(3, H // 4))
    if r1 is None:
        return None
    r2 = _randint_or_none(r1 + 2, H // 2)
    if r2 is None:
        return None
    r3 = _randint_or_none(r2 + 1, H - 7)
    if r3 is None:
        return None
    r4 = _randint_or_none(r3 + 2, H - 4)
    if r4 is None:
        return None

    c0 = _randint_or_none(3, max(3, W // 5))
    if c0 is None:
        return None
    c1 = _randint_or_none(c0 + 2, W // 2 - 1)
    if c1 is None:
        return None
    c2 = _randint_or_none(c1 + 1, W - 7)
    if c2 is None:
        return None
    c3 = _randint_or_none(c2 + 2, W - 4)
    if c3 is None:
        return None

    pairs = [
        [((r1, c1), (1, 0)), ((r3, c3), (0, -1))],
        [((r2, c0), (0, 1)), ((r4, c2), (-1, 0))],
    ]
    return pairs, True


def _scenario_rect_plus_line_separate(H: int, W: int) -> tuple[list[list[tuple[tuple[int, int], tuple[int, int]]]], bool] | None:
    if H < 16 or W < 16:
        return None

    r1 = _randint_or_none(3, H // 3)
    if r1 is None:
        return None
    r2 = _randint_or_none(r1 + 3, H // 2 + 1)
    if r2 is None:
        return None
    c1 = _randint_or_none(3, W // 3)
    if c1 is None:
        return None
    c2 = _randint_or_none(c1 + 4, W // 2 + 2)
    if c2 is None:
        return None

    r_line = _randint_or_none(r2 + 3, H - 4)
    if r_line is None:
        return None
    c_left = _randint_or_none(max(c2 + 2, W // 2), W - 8)
    if c_left is None:
        return None
    c_right = _randint_or_none(c_left + 4, W - 4)
    if c_right is None:
        return None

    pairs = [
        [((r1, c1), (1, 0)), ((r2, c2), (0, -1))],
        [((r_line, c_left), (0, 1)), ((r_line, c_right), (0, -1))],
    ]
    return pairs, False


SCENARIOS = (
    _scenario_rect_plus_line_cross,
    _scenario_two_rectangles_overlap,
    _scenario_rect_plus_line_separate,
)


def _sample_extra_pair(H: int, W: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if random.random() < 0.5:
        row = _randint_or_none(3, H - 4)
        c_left = _randint_or_none(3, W - 10)
        if row is None or c_left is None:
            return [((3, 3), (0, 1)), ((3, max(7, W - 4)), (0, -1))]
        c_right = _randint_or_none(c_left + 4, W - 4)
        if c_right is None:
            c_right = int(W - 4)
        return [((row, c_left), (0, 1)), ((row, c_right), (0, -1))]

    if random.random() < 0.5:
        col = _randint_or_none(3, W - 4)
        r_top = _randint_or_none(3, H - 10)
        if col is None or r_top is None:
            return [((3, 3), (1, 0)), ((max(7, H - 4), 3), (-1, 0))]
        r_bottom = _randint_or_none(r_top + 4, H - 4)
        if r_bottom is None:
            r_bottom = int(H - 4)
        return [((r_top, col), (1, 0)), ((r_bottom, col), (-1, 0))]

    r_left = _randint_or_none(4, H - 6)
    c_left = _randint_or_none(3, W - 9)
    if r_left is None or c_left is None:
        return [((4, 3), (0, 1)), ((max(7, H - 4), max(6, W - 4)), (-1, 0))]
    r_bottom = _randint_or_none(r_left + 2, H - 4)
    c_bottom = _randint_or_none(c_left + 3, W - 4)
    if r_bottom is None or c_bottom is None:
        return [((r_left, c_left), (0, 1)), ((max(r_left + 2, H - 4), max(c_left + 3, W - 4)), (-1, 0))]
    return [((r_left, c_left), (0, 1)), ((r_bottom, c_bottom), (-1, 0))]


def _pairings(indices: list[int]) -> list[list[tuple[int, int]]]:
    if not indices:
        return [[]]
    first = indices[0]
    out: list[list[tuple[int, int]]] = []
    for i in range(1, len(indices)):
        second = indices[i]
        rest = indices[1:i] + indices[i + 1 :]
        for tail in _pairings(rest):
            out.append([(int(first), int(second)), *tail])
    return out


def _render_matching(grid: np.ndarray, doors: list[dict], matching: list[tuple[int, int]]) -> np.ndarray | None:
    pairs = [(doors[i], doors[j]) for i, j in matching]
    try:
        out, _full_masks, _counts = _render_output(grid, pairs)
    except AssertionError:
        return None
    return out


def _has_unique_nearest_mismatch(doors: list[dict], intended_partner: dict[int, int]) -> bool:
    exteriors = [tuple(map(int, door["ext"])) for door in doors]
    for i, (r, c) in enumerate(exteriors):
        dists = []
        for j, (rr, cc) in enumerate(exteriors):
            if i == j:
                continue
            dists.append((abs(int(r - rr)) + abs(int(c - cc)), j))
        dists.sort()
        if len(dists) < 2 or dists[0][0] == dists[1][0]:
            continue
        if int(dists[0][1]) != int(intended_partner[i]):
            return True
    return False


def _build_grid_from_layout(H: int, W: int, pair_layouts: list[list[tuple[tuple[int, int], tuple[int, int]]]]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((H, W), BG, dtype=int)
    pair_colors = random.sample(CORE_PALETTE, len(pair_layouts))
    doors: list[dict] = []
    for pair_idx, layout in enumerate(pair_layouts):
        core_color = int(pair_colors[pair_idx])
        for ext, direction in layout:
            door = _stamp_door(grid, ext, direction, FRAME, core_color)
            if door is None:
                return None
            door["pair_idx"] = int(pair_idx)
            doors.append(door)
    return grid, doors


def _validate_instance(grid: np.ndarray, doors: list[dict], require_red: bool) -> bool:
    if max(grid.shape) > 30:
        return False

    try:
        detected = _detect_doors(grid)
        if len(detected) != len(doors):
            return False
        actual = main(grid)
    except AssertionError:
        return False

    by_pair: dict[int, list[dict]] = defaultdict(list)
    for door in doors:
        by_pair[int(door["pair_idx"])].append(door)
    intended_pairs = [tuple(group) for _, group in sorted(by_pair.items())]

    try:
        expected, full_masks, counts = _render_output(grid, intended_pairs)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    if require_red != bool(np.any(expected == RED)):
        return False

    if not any(abs(a["ext"][0] - b["ext"][0]) > 0 and abs(a["ext"][1] - b["ext"][1]) > 0 for a, b in intended_pairs):
        return False

    corridor_only = [mask.copy() for mask in full_masks]
    for idx, (door_a, door_b) in enumerate(intended_pairs):
        corridor_only[idx][door_a["gap"]] = False
        corridor_only[idx][door_b["gap"]] = False
        if not np.any(corridor_only[idx] & (counts == 1)):
            return False

    painted = int(np.count_nonzero(expected != grid))
    red_count = int(np.count_nonzero(expected == RED))
    if painted <= 4 or painted >= grid.size // 2:
        return False
    if red_count > painted // 2:
        return False

    intended_partner: dict[int, int] = {}
    for pair in intended_pairs:
        i = doors.index(pair[0])
        j = doors.index(pair[1])
        intended_partner[int(i)] = int(j)
        intended_partner[int(j)] = int(i)
    if not _has_unique_nearest_mismatch(doors, intended_partner):
        return False

    intended_matching = sorted(tuple(sorted((doors.index(a), doors.index(b)))) for a, b in intended_pairs)
    intended_output = expected
    for matching in _pairings(list(range(len(doors)))):
        normalized = sorted(tuple(sorted(pair)) for pair in matching)
        if normalized == intended_matching:
            continue
        other = _render_matching(grid, doors, matching)
        if other is not None and np.array_equal(other, intended_output):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 24)
    W = random.randint(16, 24)

    scenario = random.choices(SCENARIOS, weights=(4, 3, 2), k=1)[0]
    base = scenario(H, W)
    if base is None:
        return None
    pair_layouts, require_red = base

    want_third_pair = random.random() < 0.45
    if want_third_pair:
        pair_layouts = [*pair_layouts, _sample_extra_pair(H, W)]

    built = _build_grid_from_layout(H, W, pair_layouts)
    if built is None:
        return None
    grid, doors = built

    if not _validate_instance(grid, doors, require_red):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-corridor instance")
