from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRY_LIMIT = 600
MIN_BAND = 1
MAX_BAND = 2
MIN_WINDOW = 6
MAX_WINDOW = 14
MAX_SEARCH_SPACE = 5000


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _window_bbox(grid: np.ndarray) -> BBox:
    mask = grid == GRAY
    box = _bbox(mask)
    assert box is not None, "expected a gray occluding rectangle"
    r0, c0, r1, c1 = box
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "gray mask must be a filled rectangle"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "gray cells must form exactly one rectangle"
    return r0, c0, r1, c1


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _predict_cell(cell: Coord, seeds: dict[int, Coord]) -> tuple[int, str]:
    best_dist: int | None = None
    best_color = BLACK
    tie = False
    for color, pos in seeds.items():
        dist = _manhattan(cell, pos)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_color = int(color)
            tie = False
        elif dist == best_dist:
            tie = True
    assert best_dist is not None, "expected at least one seed"
    if tie:
        return BLACK, "tie"
    if best_dist % 2 == 1:
        return BLACK, "odd"
    return best_color, "color"


def _render_field(shape: tuple[int, int], seeds: dict[int, Coord]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            out[r, c] = _predict_cell((r, c), seeds)[0]
    return out


def _visible_cells(grid: np.ndarray, window: BBox) -> list[tuple[Coord, int]]:
    r0, c0, r1, c1 = window
    cells: list[tuple[Coord, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            cells.append(((r, c), int(grid[r, c])))
    return cells


def _candidate_seed_cells(grid: np.ndarray, window: BBox) -> tuple[list[int], dict[int, list[Coord]]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY))
    assert len(colors) >= 2, "expected at least two seed colors"

    r0, c0, r1, c1 = window
    candidates: dict[int, list[Coord]] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        cells = []
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            cells.append((rr, cc))
        assert cells, "every seed color must remain visible outside the window"
        candidates[color] = cells
    return colors, candidates


def _search_space_size(candidates: dict[int, list[Coord]], colors: list[int]) -> int:
    total = 1
    for color in colors:
        total *= len(candidates[color])
    return total


def _find_seed_solutions(grid: np.ndarray, max_solutions: int = 2) -> list[dict[int, Coord]]:
    g = np.asarray(grid, dtype=int)
    window = _window_bbox(g)
    colors, candidates = _candidate_seed_cells(g, window)
    visible = _visible_cells(g, window)
    color_cells = {color: candidates[color] for color in colors}
    order = sorted(colors, key=lambda color: (len(candidates[color]), color))

    assigned: dict[int, Coord] = {}
    solutions: list[dict[int, Coord]] = []

    def compatible_with_colored_cells(color: int, pos: Coord) -> bool:
        for cell in color_cells[color]:
            own_dist = _manhattan(pos, cell)
            if own_dist % 2 == 1:
                return False
            for other_color, other_pos in assigned.items():
                if _manhattan(other_pos, cell) <= own_dist:
                    return False

        for other_color, other_pos in assigned.items():
            for cell in color_cells[other_color]:
                other_dist = _manhattan(other_pos, cell)
                if _manhattan(pos, cell) <= other_dist:
                    return False
        return True

    def visible_matches() -> bool:
        for cell, expected in visible:
            predicted, _ = _predict_cell(cell, assigned)
            if predicted != expected:
                return False
        return True

    def backtrack(idx: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx == len(order):
            if visible_matches():
                solutions.append(dict(assigned))
            return

        color = order[idx]
        for pos in candidates[color]:
            if not compatible_with_colored_cells(color, pos):
                continue
            assigned[color] = pos
            backtrack(idx + 1)
            del assigned[color]
            if len(solutions) >= max_solutions:
                return

    backtrack(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the gray window's hidden nearest-seed ripple field and output only the window contents.

    Input structure:
    - A filled gray rectangle occludes part of a larger ripple field.
    - Outside the rectangle, several distinct-colored seed pixels and the visible ripple pattern remain.
    - Black cells are real field values, not empty background.

    Rule:
    - Treat every non-black non-gray color as belonging to exactly one visible seed outside the window.
    - Infer the unique seed position of each color that makes the entire visible region match this rule:
      each cell takes the nearest seed's color at even Manhattan distance, becomes black at odd distance,
      and also becomes black when two different-colored seeds tie for nearest distance.
    - Reconstruct the full field from those seeds and crop out the gray rectangle's true contents.

    Key insight:
    - The visible area is a partially revealed multi-source Manhattan Voronoi ripple pattern, so once the
      uniquely consistent seed positions are recovered, the hidden window can be recomputed exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    window = _window_bbox(g)
    solutions = _find_seed_solutions(g, max_solutions=2)
    assert len(solutions) == 1, "expected a unique seed placement"

    seeds = solutions[0]
    full = _render_field(g.shape, seeds)
    r0, c0, r1, c1 = window
    return full[r0 : r1 + 1, c0 : c1 + 1]


def _sample_canvas() -> tuple[tuple[int, int], BBox]:
    top = random.randint(MIN_BAND, MAX_BAND)
    bottom = random.randint(MIN_BAND, MAX_BAND)
    left = random.randint(MIN_BAND, MAX_BAND)
    right = random.randint(MIN_BAND, MAX_BAND)
    window_h = random.randint(MIN_WINDOW, MAX_WINDOW)
    window_w = random.randint(MIN_WINDOW, MAX_WINDOW)

    height = top + window_h + bottom
    width = left + window_w + right
    assert height <= 30 and width <= 30

    r0 = top
    c0 = left
    r1 = r0 + window_h - 1
    c1 = c0 + window_w - 1
    return (height, width), (r0, c0, r1, c1)


def _side_pool(side: str, shape: tuple[int, int], window: BBox) -> list[Coord]:
    height, width = shape
    r0, c0, r1, c1 = window
    coords: list[Coord] = []
    if side == "top":
        col_lo = max(0, c0 - 2)
        col_hi = min(width - 1, c1 + 2)
        for r in range(0, r0):
            for c in range(col_lo, col_hi + 1):
                coords.append((r, c))
    elif side == "bottom":
        col_lo = max(0, c0 - 2)
        col_hi = min(width - 1, c1 + 2)
        for r in range(r1 + 1, height):
            for c in range(col_lo, col_hi + 1):
                coords.append((r, c))
    elif side == "left":
        row_lo = max(0, r0 - 2)
        row_hi = min(height - 1, r1 + 2)
        for r in range(row_lo, row_hi + 1):
            for c in range(0, c0):
                coords.append((r, c))
    elif side == "right":
        row_lo = max(0, r0 - 2)
        row_hi = min(height - 1, r1 + 2)
        for r in range(row_lo, row_hi + 1):
            for c in range(c1 + 1, width):
                coords.append((r, c))
    else:
        raise ValueError(f"unknown side {side}")
    return coords


def _pick_seed_positions(shape: tuple[int, int], window: BBox, count: int) -> list[Coord] | None:
    side_options = [
        ("top", "bottom"),
        ("left", "right"),
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ]
    if count == 3:
        side_options = [
            ("top", "left", "right"),
            ("bottom", "left", "right"),
            ("top", "bottom", "left"),
            ("top", "bottom", "right"),
        ]

    sides = random.choice(side_options)
    pools = {side: _side_pool(side, shape, window) for side in sides}
    for _ in range(80):
        positions = [random.choice(pools[side]) for side in sides]
        if len(set(positions)) != len(positions):
            continue
        if min(_manhattan(a, b) for i, a in enumerate(positions) for b in positions[i + 1 :]) < 5:
            continue
        return positions
    return None


def _window_statistics(seeds: dict[int, Coord], window: BBox) -> tuple[int, int, int]:
    r0, c0, r1, c1 = window
    tie_black = 0
    odd_black = 0
    color_set: set[int] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            value, reason = _predict_cell((r, c), seeds)
            if value == BLACK and reason == "tie":
                tie_black += 1
            elif value == BLACK and reason == "odd":
                odd_black += 1
            elif value != BLACK:
                color_set.add(value)
    return tie_black, odd_black, len(color_set)


def _visible_nonseed_count(field: np.ndarray, seeds: dict[int, Coord], window: BBox) -> int:
    r0, c0, r1, c1 = window
    count = 0
    for color, (sr, sc) in seeds.items():
        coords = np.argwhere(field == color)
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            if (rr, cc) != (sr, sc):
                count += 1
    return count


def _semantic_checks(inp: np.ndarray, seeds: dict[int, Coord], field: np.ndarray, window: BBox) -> bool:
    colors, candidates = _candidate_seed_cells(inp, window)
    if _search_space_size(candidates, colors) > MAX_SEARCH_SPACE:
        return False
    if sum(len(candidates[color]) > 1 for color in colors) == 0:
        return False

    tie_black, odd_black, color_count = _window_statistics(seeds, window)
    if tie_black == 0 or odd_black == 0 or color_count < 2:
        return False

    if _visible_nonseed_count(field, seeds, window) < len(seeds):
        return False

    solutions = _find_seed_solutions(inp, max_solutions=2)
    if len(solutions) != 1:
        return False
    if solutions[0] != seeds:
        return False

    out = main(inp)
    expected = field[window[0] : window[2] + 1, window[1] : window[3] + 1]
    return np.array_equal(out, expected)


def _attempt_generate() -> np.ndarray | None:
    shape, window = _sample_canvas()
    seed_count = random.choice((2, 2, 3, 3))
    positions = _pick_seed_positions(shape, window, seed_count)
    if positions is None:
        return None

    colors = random.sample(PALETTE, seed_count)
    seeds = {color: pos for color, pos in zip(colors, positions)}
    field = _render_field(shape, seeds)

    inp = field.copy()
    r0, c0, r1, c1 = window
    inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY

    if not _semantic_checks(inp, seeds, field, window):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ripple-window puzzle")
