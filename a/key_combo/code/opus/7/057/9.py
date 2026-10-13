from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

BASE_COLORS = (BLUE, RED)
BLOCKER_COLORS = (GREY, ORANGE)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _is_base_color(value: int) -> bool:
    return int(value) in BASE_COLORS


def _is_blocker(value: int) -> bool:
    return int(value) in BLOCKER_COLORS


def _strict_majority(values: list[int]) -> int:
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked, "expected at least one visible base-color neighbor"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority"
    return int(ranked[0][0])


def _majority_base_neighbor_color(grid: np.ndarray, r: int, c: int) -> int:
    base_neighbors = [
        int(grid[nr, nc])
        for nr, nc in _neighbors4(r, c)
        if _in_bounds(grid.shape, nr, nc) and _is_base_color(int(grid[nr, nc]))
    ]
    return _strict_majority(base_neighbors)


def _base_color_at_or_near(grid: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    value = int(grid[r, c])
    if _is_base_color(value):
        return value
    return _majority_base_neighbor_color(grid, r, c)


def _reachable_monotone(
    passable: np.ndarray,
    start: tuple[int, int],
    moves: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds

    def valid(rr: int, cc: int) -> bool:
        return r0 <= rr <= r1 and c0 <= cc <= c1 and bool(passable[rr, cc])

    assert valid(*start)
    seen = np.zeros(passable.shape, dtype=bool)
    seen[start] = True
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if valid(nr, nc) and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _find_doorways(grid: np.ndarray) -> list[dict[str, tuple[int, int]]]:
    centers = np.argwhere(grid == ORANGE)
    assert len(centers) == 2, "expected exactly two doorway centers"

    doors: list[dict[str, tuple[int, int]]] = []
    for rr, cc in centers:
        r, c = int(rr), int(cc)
        gaps: list[tuple[int, int, int, int]] = []
        grey_neighbors = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            assert _in_bounds(grid.shape, nr, nc), "doorway center must be interior"
            value = int(grid[nr, nc])
            if value == BLACK:
                gaps.append((dr, dc, nr, nc))
            elif value == GREY:
                grey_neighbors += 1

        assert len(gaps) == 1 and grey_neighbors == 3, "expected a single gap in each 3x3 frame"
        dr, dc, gap_r, gap_c = gaps[0]
        ext = (r + 2 * dr, c + 2 * dc)
        assert _in_bounds(grid.shape, *ext), "exterior doorway cell must be in-bounds"
        assert not _is_blocker(int(grid[ext])), "exterior doorway cell must be passable"
        doors.append({"center": (r, c), "gap": (gap_r, gap_c), "exterior": ext})

    return doors


def _corridor_mask(grid: np.ndarray, doors: list[dict[str, tuple[int, int]]]) -> np.ndarray:
    start = doors[0]["exterior"]
    end = doors[1]["exterior"]
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    moves: list[tuple[int, int]] = []
    if row_step:
        moves.append((row_step, 0))
    if col_step:
        moves.append((0, col_step))
    assert moves, "exterior doorway cells must be distinct"

    passable = ~np.isin(grid, BLOCKER_COLORS)
    r0, r1 = sorted([start[0], end[0]])
    c0, c1 = sorted([start[1], end[1]])
    bounds = (r0, c0, r1, c1)
    forward = _reachable_monotone(passable, start, moves, bounds)
    backward = _reachable_monotone(passable, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start] and path[end], "expected a valid monotone corridor"
    return path


def _corridor_neighborhood(grid: np.ndarray, corridor: np.ndarray) -> np.ndarray:
    passable = ~np.isin(grid, BLOCKER_COLORS)
    neighborhood = corridor.copy()
    for rr, cc in np.argwhere(corridor):
        r, c = int(rr), int(cc)
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc]:
                neighborhood[nr, nc] = True
    return neighborhood


def _winning_color(
    grid: np.ndarray,
    neighborhood: np.ndarray,
    doors: list[dict[str, tuple[int, int]]],
) -> int:
    votes: Counter[int] = Counter()
    for rr, cc in np.argwhere((grid == YELLOW) & neighborhood):
        votes[_majority_base_neighbor_color(grid, int(rr), int(cc))] += 1

    assert votes, "expected at least one voting marker in the corridor neighborhood"
    blue_votes = int(votes[BLUE])
    red_votes = int(votes[RED])
    if red_votes > blue_votes:
        return RED
    if blue_votes > red_votes:
        return BLUE

    left_door = min(doors, key=lambda door: (door["center"][1], door["center"][0]))
    return _base_color_at_or_near(grid, left_door["exterior"])


def _paint_output(
    grid: np.ndarray,
    corridor: np.ndarray,
    doors: list[dict[str, tuple[int, int]]],
    fill_color: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[corridor & np.isin(grid, BASE_COLORS)] = int(fill_color)
    for door in doors:
        out[door["gap"]] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the monotone corridor between two doorway frames using votes from nearby embedded markers.

    Input structure:
    - The grid is tiled by large red and blue rectangular base regions, with two grey 3x3 doorway frames placed in different locations.
    - Each frame has one orange center, one black gap touching that center, and one exterior doorway cell just beyond the gap.
    - Yellow single-pixel markers are embedded inside the red or blue regions outside the frames.

    Rule:
    - Treat only the grey frame cells and orange centers as blockers, then compute the union of all monotone shortest paths between the two exterior doorway cells.
    - Form the corridor neighborhood from every corridor cell plus its passable 4-neighbors, and keep only the yellow markers inside that neighborhood as voters.
    - Classify each voter by the majority red/blue color in its 4-neighborhood, then fill the corridor cells and the two doorway gaps with the color whose nearby region gets more votes.

    Key insight:
    - The geometry of the doorway corridor selects which markers are allowed to vote, and those selected votes determine the final corridor color.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    doors = _find_doorways(grid)
    corridor = _corridor_mask(grid, doors)
    neighborhood = _corridor_neighborhood(grid, corridor)
    fill_color = _winning_color(grid, neighborhood, doors)
    return _paint_output(grid, corridor, doors, fill_color)


def _random_cuts(total: int, n_parts: int, min_size: int, rng: random.Random) -> list[int] | None:
    if total < n_parts * min_size:
        return None
    if n_parts == 1:
        return [0, total]
    for _ in range(80):
        cuts = sorted(rng.sample(range(min_size, total), n_parts - 1))
        points = [0, *cuts, total]
        sizes = [points[i + 1] - points[i] for i in range(n_parts)]
        if min(sizes) >= min_size:
            return points
    return None


def _make_base_grid(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    mode = rng.choice(["v2", "v3", "h2", "h3", "cross_a", "cross_b"])

    if mode == "v2":
        cuts = _random_cuts(width, 2, 5, rng)
        if cuts is None:
            return None
        left_color = rng.choice(BASE_COLORS)
        right_color = RED if left_color == BLUE else BLUE
        grid[:, cuts[0] : cuts[1]] = left_color
        grid[:, cuts[1] : cuts[2]] = right_color
        return grid

    if mode == "v3":
        cuts = _random_cuts(width, 3, 4, rng)
        if cuts is None:
            return None
        start_color = rng.choice(BASE_COLORS)
        colors = [start_color, RED if start_color == BLUE else BLUE, start_color]
        for i in range(3):
            grid[:, cuts[i] : cuts[i + 1]] = colors[i]
        return grid

    if mode == "h2":
        cuts = _random_cuts(height, 2, 5, rng)
        if cuts is None:
            return None
        top_color = rng.choice(BASE_COLORS)
        bottom_color = RED if top_color == BLUE else BLUE
        grid[cuts[0] : cuts[1], :] = top_color
        grid[cuts[1] : cuts[2], :] = bottom_color
        return grid

    if mode == "h3":
        cuts = _random_cuts(height, 3, 4, rng)
        if cuts is None:
            return None
        start_color = rng.choice(BASE_COLORS)
        colors = [start_color, RED if start_color == BLUE else BLUE, start_color]
        for i in range(3):
            grid[cuts[i] : cuts[i + 1], :] = colors[i]
        return grid

    row_cut = rng.randint(5, height - 5)
    col_cut = rng.randint(5, width - 5)
    if mode == "cross_a":
        tl_br = rng.choice(BASE_COLORS)
        tr_bl = RED if tl_br == BLUE else BLUE
        grid[:row_cut, :col_cut] = tl_br
        grid[:row_cut, col_cut:] = tr_bl
        grid[row_cut:, :col_cut] = tr_bl
        grid[row_cut:, col_cut:] = tl_br
    else:
        top_bottom = rng.choice(BASE_COLORS)
        other = RED if top_bottom == BLUE else BLUE
        grid[:row_cut, :col_cut] = top_bottom
        grid[:row_cut, col_cut:] = top_bottom
        grid[row_cut:, :col_cut] = other
        grid[row_cut:, col_cut:] = other
    return grid


def _choose_frame_centers(height: int, width: int, rng: random.Random) -> list[tuple[int, int]]:
    orient = rng.choice(["tl_br", "tr_bl"])
    top_rows = (2, max(2, height // 2 - 3))
    bottom_rows = (min(height // 2 + 2, height - 3), height - 3)
    left_cols = (2, max(2, width // 2 - 3))
    right_cols = (min(width // 2 + 2, width - 3), width - 3)

    r1 = rng.randint(*top_rows)
    r2 = rng.randint(*bottom_rows)
    if orient == "tl_br":
        c1 = rng.randint(*left_cols)
        c2 = rng.randint(*right_cols)
    else:
        c1 = rng.randint(*right_cols)
        c2 = rng.randint(*left_cols)
    return [(r1, c1), (r2, c2)]


def _valid_gap_directions(height: int, width: int, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (dr, dc)
        for dr, dc in DIR4
        if _in_bounds((height, width), r + dr, c + dc) and _in_bounds((height, width), r + 2 * dr, c + 2 * dc)
    ]


def _place_frame(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int]) -> None:
    r, c = center
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            grid[r + dr, c + dc] = GREY
    grid[r, c] = ORANGE
    gap = (r + direction[0], c + direction[1])
    grid[gap] = BLACK


def _marker_candidates(
    grid: np.ndarray,
    allowed: np.ndarray,
    color: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not allowed[r, c] or int(grid[r, c]) != int(color):
                continue
            try:
                majority = _majority_base_neighbor_color(grid, r, c)
            except AssertionError:
                continue
            if majority == int(color):
                candidates.append((r, c))
    return candidates


def _pick_spaced(
    cells: list[tuple[int, int]],
    count: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = cells[:]
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked = set(forbidden)
    for cell in pool:
        if cell in blocked:
            continue
        chosen.append(cell)
        blocked.add(cell)
        for nb in _neighbors4(*cell):
            blocked.add(nb)
        if len(chosen) == count:
            return chosen
    return None


def _vote_pairs_for_winner(winner: int) -> list[tuple[int, int]]:
    _ = winner
    return [(4, 2), (3, 2), (3, 1), (5, 2), (4, 1)]


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    doors = _find_doorways(grid)
    corridor = _corridor_mask(grid, doors)
    neighborhood = _corridor_neighborhood(grid, corridor)
    fill_color = _winning_color(grid, neighborhood, doors)
    return _paint_output(grid, corridor, doors, fill_color), fill_color, corridor, neighborhood


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 22)
    width = rng.randint(16, 22)
    grid = _make_base_grid(height, width, rng)
    if grid is None:
        return None

    centers = _choose_frame_centers(height, width, rng)
    directions = []
    for center in centers:
        options = _valid_gap_directions(height, width, center)
        directions.append(rng.choice(options))
    for center, direction in zip(centers, directions):
        _place_frame(grid, center, direction)

    try:
        doors = _find_doorways(grid)
        start = doors[0]["exterior"]
        end = doors[1]["exterior"]
        if start[0] == end[0] or start[1] == end[1]:
            return None

        corridor = _corridor_mask(grid, doors)
        if int(corridor.sum()) <= abs(start[0] - end[0]) + abs(start[1] - end[1]) + 1:
            return None
        neighborhood = _corridor_neighborhood(grid, corridor)
    except AssertionError:
        return None

    winner = rng.choice(BASE_COLORS)
    loser = RED if winner == BLUE else BLUE
    allowed_near = neighborhood & ~corridor & np.isin(grid, BASE_COLORS)
    allowed_any = neighborhood & np.isin(grid, BASE_COLORS)

    chosen: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int]] = []
    counts = _vote_pairs_for_winner(winner)
    rng.shuffle(counts)

    picked_counts = None
    for win_count, lose_count in counts:
        win_candidates = _marker_candidates(grid, allowed_near, winner)
        lose_candidates = _marker_candidates(grid, allowed_near, loser)
        if len(win_candidates) < win_count or len(lose_candidates) < lose_count:
            win_candidates = _marker_candidates(grid, allowed_any, winner)
            lose_candidates = _marker_candidates(grid, allowed_any, loser)

        win_cells = _pick_spaced(win_candidates, win_count, chosen, rng)
        if win_cells is None:
            continue
        chosen_with_win = chosen | set(win_cells)
        lose_cells = _pick_spaced(lose_candidates, lose_count, chosen_with_win, rng)
        if lose_cells is None:
            continue

        for r, c in win_cells:
            placements.append((r, c, YELLOW))
        for r, c in lose_cells:
            placements.append((r, c, YELLOW))
        chosen = chosen_with_win | set(lose_cells)
        picked_counts = (win_count, lose_count)
        break

    if picked_counts is None:
        return None

    outside_mask = (~neighborhood) & np.isin(grid, BASE_COLORS)
    extra_budget = rng.randint(0, 4)
    for _ in range(extra_budget):
        extra_color = rng.choice(BASE_COLORS)
        extra_candidates = _marker_candidates(grid, outside_mask, extra_color)
        extra_cells = _pick_spaced(extra_candidates, 1, chosen, rng)
        if extra_cells is None:
            continue
        cell = extra_cells[0]
        placements.append((cell[0], cell[1], YELLOW))
        chosen.add(cell)

    for r, c, color in placements:
        grid[r, c] = color

    try:
        expected, fill_color, corridor_after, neighborhood_after = _expected_output(grid)
    except AssertionError:
        return None

    if fill_color != winner:
        return None
    if int(((grid == YELLOW) & neighborhood_after).sum()) < 3:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.any(expected[corridor_after & np.isin(grid, BASE_COLORS)] == fill_color):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-voting ARC input")
