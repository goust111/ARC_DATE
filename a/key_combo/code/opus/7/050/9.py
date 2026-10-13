from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 5
MARKER = 6

Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _adjacent_to_color(grid: np.ndarray, pos: Coord, color: int) -> bool:
    return any(
        in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(color)
        for nr, nc in neighbors4(*pos)
    )


def _separator_index(grid: np.ndarray, axis: int) -> int:
    if axis == 0:
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] == SEPARATOR):
                return int(r)
    else:
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] == SEPARATOR):
                return int(c)
    raise AssertionError("expected one full grey separator row and column")


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    r_sep = _separator_index(grid, axis=0)
    c_sep = _separator_index(grid, axis=1)
    return [
        grid[:r_sep, :c_sep].copy(),
        grid[:r_sep, c_sep + 1 :].copy(),
        grid[r_sep + 1 :, :c_sep].copy(),
        grid[r_sep + 1 :, c_sep + 1 :].copy(),
    ]


def _marker_pos(panel: np.ndarray) -> Coord:
    coords = np.argwhere(panel == MARKER)
    assert coords.shape == (1, 2), "each quadrant must contain exactly one marker"
    return tuple(map(int, coords[0]))


def _obstacle_color(panel: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in np.unique(panel)}
    colors -= {int(bg), MARKER}
    assert len(colors) == 1, "each quadrant must contain exactly one obstacle color"
    return colors.pop()


def _sightline_to_obstacle(
    panel: np.ndarray, start: Coord, obstacle: int, bg: int
) -> tuple[Coord, Coord, int]:
    hits: list[tuple[Coord, Coord, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = start[0] + dr, start[1] + dc
        gap = 0
        previous_bg: Coord | None = None
        while in_bounds(panel.shape, r, c):
            value = int(panel[r, c])
            if value == obstacle:
                if previous_bg is not None:
                    hits.append(((dr, dc), previous_bg, gap))
                break
            if value != bg:
                break
            previous_bg = (r, c)
            gap += 1
            r += dr
            c += dc
    assert len(hits) == 1, "marker must have exactly one sightline to the obstacle"
    return hits[0]


def _walk_contour(
    panel: np.ndarray, start: Coord, direction: Coord, obstacle: int, bg: int
) -> tuple[Coord, int]:
    dr, dc = direction
    if dr != 0:
        turns = [(dr, -1), (dr, 1)]
    else:
        turns = [(-1, dc), (1, dc)]

    candidates: list[tuple[int, Coord]] = []
    for tr, tc in turns:
        r, c = start
        length = 0
        while True:
            nr, nc = r + tr, c + tc
            if not in_bounds(panel.shape, nr, nc):
                break
            if int(panel[nr, nc]) != bg:
                break
            if not _adjacent_to_color(panel, (nr, nc), obstacle):
                break
            r, c = nr, nc
            length += 1
        candidates.append((length, (r, c)))

    max_len = max(length for length, _ in candidates)
    best = [item for item in candidates if item[0] == max_len]
    assert len(best) == 1, "the contour walk must have a unique longest branch"
    walk_len, endpoint = best[0]
    return endpoint, int(walk_len)


def _evaluate_quadrant(panel: np.ndarray) -> dict:
    bg = get_bg(panel)
    marker_pos = _marker_pos(panel)
    obstacle = _obstacle_color(panel, bg)
    direction, contour_start, gap = _sightline_to_obstacle(panel, marker_pos, obstacle, bg)
    assert _adjacent_to_color(panel, contour_start, obstacle)
    endpoint, walk_len = _walk_contour(panel, contour_start, direction, obstacle, bg)
    return {
        "bg": int(bg),
        "marker": marker_pos,
        "obstacle": int(obstacle),
        "endpoint": endpoint,
        "score": int(gap + walk_len),
    }


def _apply_quadrant_result(panel: np.ndarray, info: dict) -> np.ndarray:
    out = panel.copy()
    out[info["marker"]] = info["bg"]
    out[info["endpoint"]] = info["obstacle"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the quadrant whose marker reaches the longest obstacle-adjacent slide path and recolor its endpoint to the obstacle color.

    Input structure:
    - The grid is divided into four quadrants by one full grey separator row and one full grey separator column.
    - Each quadrant has black background, one irregular connected obstacle of a unique non-grey color, and one magenta single-pixel marker.
    - The marker is not adjacent to the obstacle, but has exactly one horizontal or vertical sightline through background to that obstacle.

    Rule:
    - In each quadrant, slide from the marker along its unique sightline to the last background cell before the obstacle.
    - From that contour-start cell, test the two diagonal contour-following turns and keep the unique longer path that stays on background while remaining 4-neighbor-adjacent to the obstacle.
    - Compare the four total path lengths from marker start to contour endpoint; select the unique longest quadrant, remove its original marker, recolor the endpoint with the obstacle color, and output only that cropped quadrant.

    Key insight:
    - The contour slide is not the final output by itself; it is evaluated independently in all four quadrants, and only the quadrant with the longest reachable contour path survives the crop.
    """
    g = np.asarray(grid, dtype=int)
    quads = _quadrants(g)
    infos = [_evaluate_quadrant(panel) for panel in quads]
    scores = [info["score"] for info in infos]
    best_score = max(scores)
    assert scores.count(best_score) == 1, "quadrant scores must have a unique maximum"
    winner = scores.index(best_score)
    return _apply_quadrant_result(quads[winner], infos[winner])


GEN_RETRIES = 500
PANEL_RETRIES = 200
PANEL_SIDE_RANGE = (8, 12)
POSSIBLE_TOTALS = (3, 4, 5, 6, 7)
OBSTACLE_AREA_RANGE = (5, 10)
OBSTACLE_HEIGHT_RANGE = (4, 6)
OBSTACLE_WIDTH_RANGE = (3, 6)
GAP_RANGE = (2, 4)
WALK_RANGE = (1, 3)
OBSTACLE_COLORS = (1, 2, 3, 4, 7, 8, 9)


def _gen_boundary(height: int, start_row: int, walk_len: int, start_left: int) -> list[int]:
    lefts = [0] * height
    lefts[start_row] = start_left

    for r in range(start_row - 1, -1, -1):
        if lefts[r + 1] == 0:
            lefts[r] = 0
        else:
            lefts[r] = max(0, lefts[r + 1] - random.randint(0, 1))

    for i in range(1, walk_len + 1):
        lefts[start_row + i] = start_left + i

    stop_row = start_row + walk_len + 1
    if stop_row < height:
        lefts[stop_row] = lefts[stop_row - 1]
        for r in range(stop_row + 1, height):
            lefts[r] = lefts[r - 1] + random.randint(0, 1)
    return lefts


def _turn_pairs_for_total(total: int) -> list[tuple[int, int]]:
    pairs = []
    for gap in range(GAP_RANGE[0], GAP_RANGE[1] + 1):
        for walk_len in range(WALK_RANGE[0], WALK_RANGE[1] + 1):
            if gap + walk_len == total:
                pairs.append((gap, walk_len))
    return pairs


def _place_obstacle(panel: np.ndarray, color: int, top: int, left: int, lefts: list[int], width: int) -> None:
    for rr, inner_left in enumerate(lefts):
        panel[top + rr, left + inner_left : left + width] = color


def _gen_panel(side: int, obstacle_color: int, target_total: int) -> np.ndarray | None:
    pairs = _turn_pairs_for_total(target_total)
    if not pairs:
        return None

    for _ in range(PANEL_RETRIES):
        gap, walk_len = random.choice(pairs)
        min_height = max(OBSTACLE_HEIGHT_RANGE[0], walk_len + 3)
        if min_height > OBSTACLE_HEIGHT_RANGE[1]:
            continue
        height = random.randint(min_height, OBSTACLE_HEIGHT_RANGE[1])
        start_row = random.randint(1, height - walk_len - 2)
        max_start_left = min(gap, 4)
        if max_start_left < 1:
            continue
        start_left = random.randint(1, max_start_left)

        lefts = _gen_boundary(height, start_row, walk_len, start_left)
        max_left = max(lefts)
        base_left = gap - start_left + 1
        width_candidates = []
        for width in range(OBSTACLE_WIDTH_RANGE[0], OBSTACLE_WIDTH_RANGE[1] + 1):
            if width <= max_left:
                continue
            area = sum(width - inner_left for inner_left in lefts)
            if not (OBSTACLE_AREA_RANGE[0] <= area <= OBSTACLE_AREA_RANGE[1]):
                continue
            if base_left + width > side:
                continue
            width_candidates.append(width)
        if not width_candidates:
            continue

        width = random.choice(width_candidates)
        if height > side:
            continue
        top = random.randint(0, side - height)
        contour_row = top + start_row
        dest = (contour_row + walk_len, gap + walk_len)
        marker = (contour_row, 0)

        panel = np.full((side, side), BG, dtype=int)
        _place_obstacle(panel, obstacle_color, top, base_left, lefts, width)
        if panel[marker] != BG or panel[dest] != BG:
            continue
        panel[marker] = MARKER

        panel = np.rot90(panel, random.randrange(4)).copy()
        if random.random() < 0.5:
            panel = np.fliplr(panel).copy()

        info = _evaluate_quadrant(panel)
        if info["score"] != target_total:
            continue
        return panel
    return None


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    side = int(panels[0].shape[0])
    grid = np.full((side * 2 + 1, side * 2 + 1), BG, dtype=int)
    grid[side, :] = SEPARATOR
    grid[:, side] = SEPARATOR
    grid[:side, :side] = panels[0]
    grid[:side, side + 1 :] = panels[1]
    grid[side + 1 :, :side] = panels[2]
    grid[side + 1 :, side + 1 :] = panels[3]
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(*PANEL_SIDE_RANGE)
    totals = random.sample(list(POSSIBLE_TOTALS), 4)
    obstacle_colors = random.sample(list(OBSTACLE_COLORS), 4)

    panels = []
    for color, total in zip(obstacle_colors, totals):
        panel = _gen_panel(side, color, total)
        if panel is None:
            return None
        panels.append(panel)

    infos = [_evaluate_quadrant(panel) for panel in panels]
    scores = [info["score"] for info in infos]
    if len(set(scores)) != 4:
        return None

    grid = _assemble_grid(panels)
    winner = scores.index(max(scores))
    expected = _apply_quadrant_result(panels[winner], infos[winner])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
