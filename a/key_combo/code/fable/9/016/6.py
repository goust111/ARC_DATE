from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import random

import numpy as np


BG = 0
GHOST = 5
SIDES = ("top", "bottom", "left", "right")
DOMINO_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (8, 18)
TOTAL_DOMINO_RANGE = (4, 7)
LANE_COUNT_RANGE = (2, 4)
MAX_STACK = 4
MAX_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _edge_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[h - 1, :] = True
    elif side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, w - 1] = True
    else:
        raise ValueError(side)
    return mask


def _find_wall(grid: np.ndarray, bg: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    specs = [
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    ]
    for side, line in specs:
        color = int(line[0])
        if color == bg or not np.all(line == color):
            continue
        if np.array_equal(grid == color, _edge_mask(grid.shape, side)):
            candidates.append((side, color))
    assert len(candidates) == 1, "expected exactly one full edge wall"
    return candidates[0]


def _to_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.flipud(grid).copy()
    if side == "left":
        return np.rot90(grid, -1).copy()
    if side == "right":
        return np.rot90(grid, 1).copy()
    raise ValueError(side)


def _from_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.flipud(grid).copy()
    if side == "left":
        return np.rot90(grid, 1).copy()
    if side == "right":
        return np.rot90(grid, -1).copy()
    raise ValueError(side)


def _is_singleton_color(grid: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                return False
    return True


def _find_ghost_color(grid: np.ndarray, bg: int, wall_color: int) -> int:
    colors = sorted(set(map(int, grid.ravel())) - {bg, wall_color})
    candidates = [color for color in colors if _is_singleton_color(grid, color)]
    assert len(candidates) == 1, "expected exactly one singleton ghost color"
    return int(candidates[0])


def _find_dominoes_top(grid: np.ndarray, bg: int, wall_color: int, ghost_color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    dominoes: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, wall_color, ghost_color) or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            assert len(coords) == 2, "expected every moving object to be a domino"
            rows = sorted(r0 for r0, _ in coords)
            cols = {c0 for _, c0 in coords}
            assert len(cols) == 1 and rows[1] == rows[0] + 1, "top-normalized dominoes must be vertical"
            dominoes.append(
                {
                    "color": color,
                    "lane": int(next(iter(cols))),
                    "front": int(rows[0]),
                    "coords": np.array(sorted(coords), dtype=int),
                }
            )
    assert dominoes, "expected at least one domino"
    return dominoes


def _ghost_rows_by_lane_top(grid: np.ndarray, ghost_color: int) -> dict[int, list[int]]:
    by_lane: dict[int, list[int]] = defaultdict(list)
    for r, c in np.argwhere(grid == ghost_color):
        by_lane[int(c)].append(int(r))
    for lane in by_lane:
        by_lane[lane].sort()
    return dict(by_lane)


def _render_top_from_pairs(
    shape: tuple[int, int],
    wall_color: int,
    dominoes_by_lane: dict[int, list[dict]],
    ghosts_by_lane: dict[int, list[int]],
    *,
    reverse_order: bool = False,
    align_front: bool = True,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[0, :] = wall_color
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True

    for lane in sorted(dominoes_by_lane):
        dominoes = sorted(dominoes_by_lane[lane], key=lambda item: item["front"])
        ghosts = sorted(ghosts_by_lane[lane], reverse=reverse_order)
        assert len(dominoes) == len(ghosts)
        for domino, ghost_row in zip(dominoes, ghosts):
            anchor_row = domino["front"] if align_front else domino["front"] + 1
            delta = int(ghost_row - anchor_row)
            moved = domino["coords"] + np.array([delta, 0], dtype=int)
            assert np.all(0 <= moved[:, 0]) and np.all(moved[:, 0] < h)
            assert np.all(0 <= moved[:, 1]) and np.all(moved[:, 1] < w)
            assert not np.any(occupied[moved[:, 0], moved[:, 1]])
            out[moved[:, 0], moved[:, 1]] = int(domino["color"])
            occupied[moved[:, 0], moved[:, 1]] = True
    return out


def _solve_top(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    side, wall_color = _find_wall(g, bg)
    assert side == "top", "grid must already be normalized to the top-wall frame"
    ghost_color = _find_ghost_color(g, bg, wall_color)
    dominoes = _find_dominoes_top(g, bg, wall_color, ghost_color)

    dominoes_by_lane: dict[int, list[dict]] = defaultdict(list)
    for domino in dominoes:
        dominoes_by_lane[int(domino["lane"])].append(domino)
    ghosts_by_lane = _ghost_rows_by_lane_top(g, ghost_color)

    assert set(dominoes_by_lane) == set(ghosts_by_lane), "ghosts must match active lanes exactly"
    for lane, items in dominoes_by_lane.items():
        ordered = sorted(items, key=lambda item: item["front"])
        expected_fronts = list(range(1, 2 * len(ordered), 2))
        assert [int(item["front"]) for item in ordered] == expected_fronts, "dominoes must be tightly stacked against the wall"
        ghosts = ghosts_by_lane[lane]
        assert len(ghosts) == len(ordered), "each lane needs one ghost per domino"
        assert all(ghosts[i + 1] >= ghosts[i] + 2 for i in range(len(ghosts) - 1)), "ghost anchors in a lane must not overlap"
        assert ghosts[0] > expected_fronts[-1] + 1, "ghosts must sit in empty space away from the stack"

    return _render_top_from_pairs(g.shape, wall_color, dict(dominoes_by_lane), ghosts_by_lane)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each wall-packed domino back away from the wall so the wall-facing end lands on the same-lane ghost marker of matching rank.

    Input structure:
    - One non-background color forms a complete one-cell-thick edge wall.
    - Every other non-ghost foreground object is a monochrome domino already stacked flush against that wall.
    - Grey singleton ghost markers lie in the empty area, with each wall-perpendicular lane containing as many ghosts as stacked dominoes.

    Rule:
    - Keep the wall fixed and clear every domino from its stacked position.
    - In each lane, sort both the dominoes and the ghosts by distance from the wall; move the k-th closest domino to the k-th closest ghost.
    - Preserve each domino's shape and color, placing its wall-facing cell on the ghost and erasing all uncovered ghosts.

    Key insight:
    - The ghosts are not arbitrary targets: they encode the time-reversed pre-fall positions lane by lane, so the stack order must be undone in the same nearest-to-farthest order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    side, _wall_color = _find_wall(g, bg)
    top = _to_top(g, side)
    solved = _solve_top(top)
    return _from_top(solved, side)


def _sample_spaced_positions(start: int, end: int, count: int, gap: int = 2) -> list[int] | None:
    length = end - start + 1
    compressed = length - (count - 1) * (gap - 1)
    if count <= 0 or compressed < count:
        return None
    base = sorted(random.sample(range(compressed), count))
    return [start + b + i * (gap - 1) for i, b in enumerate(base)]


def _count_choices(total: int, lane_count: int) -> list[tuple[int, ...]]:
    return [
        counts
        for counts in product(range(1, MAX_STACK + 1), repeat=lane_count)
        if sum(counts) == total and 1 in counts and max(counts) > 1
    ]


def _choose_counts() -> tuple[int, ...] | None:
    total = random.randint(*TOTAL_DOMINO_RANGE)
    max_lanes = min(LANE_COUNT_RANGE[1], total)
    lane_count = random.randint(LANE_COUNT_RANGE[0], max_lanes)
    choices = _count_choices(total, lane_count)
    if not choices:
        return None
    return random.choice(choices)


def _choose_ghost_rows(h: int, count: int) -> list[int] | None:
    rows = _sample_spaced_positions(2 * count + 1, h - 2, count, gap=2)
    if rows is None:
        return None
    return rows


def _build_top_input(
    h: int,
    w: int,
    wall_color: int,
    lanes: list[int],
    counts: tuple[int, ...],
    colors: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = wall_color
    lane_data: list[dict] = []
    color_idx = 0

    for lane, count in zip(lanes, counts):
        ghosts = _choose_ghost_rows(h, count)
        if ghosts is None:
            return None
        dominoes: list[dict] = []
        for k in range(count):
            front = 1 + 2 * k
            cells = np.array([(front, lane), (front + 1, lane)], dtype=int)
            color = int(colors[color_idx])
            color_idx += 1
            grid[front, lane] = color
            grid[front + 1, lane] = color
            dominoes.append({"front": front, "lane": lane, "coords": cells, "color": color})
        for ghost in ghosts:
            grid[ghost, lane] = GHOST
        lane_data.append({"lane": lane, "ghosts": ghosts, "dominoes": dominoes})
    return grid, lane_data


def _lane_spacing_signature(lane_data: list[dict]) -> set[tuple[int, ...]]:
    sigs = set()
    for lane in lane_data:
        gaps = tuple(b - a for a, b in zip(lane["ghosts"], lane["ghosts"][1:]))
        sigs.add((lane["ghosts"][0],) + gaps)
    return sigs


def _semantic_check_top(inp: np.ndarray, lane_data: list[dict], wall_color: int) -> None:
    assert isinstance(inp, np.ndarray) and inp.ndim == 2
    assert max(inp.shape) <= 30

    bg = _get_bg(inp)
    side, found_wall = _find_wall(inp, bg)
    assert side == "top" and found_wall == wall_color
    assert _find_ghost_color(inp, bg, wall_color) == GHOST

    counts = [len(lane["dominoes"]) for lane in lane_data]
    assert len(lane_data) >= 2
    assert any(count == 1 for count in counts)
    assert any(count >= 2 for count in counts)
    assert len(_lane_spacing_signature(lane_data)) >= 2, "need varied ghost spacing so a uniform shift shortcut fails"

    solved = _solve_top(inp)
    assert solved.shape == inp.shape
    assert not np.array_equal(inp, solved), "transform must visibly change the grid"

    dominoes_by_lane = {lane["lane"]: lane["dominoes"] for lane in lane_data}
    ghosts_by_lane = {lane["lane"]: lane["ghosts"] for lane in lane_data}

    wrong_reversed = _render_top_from_pairs(
        inp.shape,
        wall_color,
        dominoes_by_lane,
        ghosts_by_lane,
        reverse_order=True,
        align_front=True,
    )
    wrong_back_aligned = _render_top_from_pairs(
        inp.shape,
        wall_color,
        dominoes_by_lane,
        ghosts_by_lane,
        reverse_order=False,
        align_front=False,
    )
    assert not np.array_equal(solved, wrong_reversed), "need at least one lane where order matters"
    assert not np.array_equal(solved, wrong_back_aligned), "ghosts must anchor the wall-facing end, not the far end"

    for lane in lane_data:
        ghosts = lane["ghosts"]
        dominoes = lane["dominoes"]
        assert len(ghosts) == len(dominoes)
        assert all(ghosts[i + 1] >= ghosts[i] + 2 for i in range(len(ghosts) - 1))
        assert ghosts[0] > dominoes[-1]["front"] + 1


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    wall_color = random.choice(DOMINO_COLORS)
    counts = _choose_counts()
    if counts is None:
        return None

    lane_count = len(counts)
    max_stack = max(counts)
    min_h = max(GRID_H_RANGE[0], 4 * max_stack + 1)
    min_w = max(GRID_W_RANGE[0], 2 * lane_count - 1)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    lanes = _sample_spaced_positions(0, w - 1, lane_count, gap=2)
    if lanes is None:
        return None

    colors = random.sample([c for c in DOMINO_COLORS if c != wall_color], sum(counts))
    built = _build_top_input(h, w, wall_color, lanes, counts, colors)
    if built is None:
        return None
    inp_top, lane_data = built

    _semantic_check_top(inp_top, lane_data, wall_color)
    inp = _from_top(inp_top, side)
    if not np.array_equal(main(inp), _from_top(_solve_top(inp_top), side)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ordered-unstacking input")
