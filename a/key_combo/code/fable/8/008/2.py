from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
EDGE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_EDGE = {value: key for key, value in EDGE_TO_DIR.items()}
ROTATE_CW = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r_raw, c_raw in np.argwhere(mask):
        start = (int(r_raw), int(c_raw))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in _neighbors4(*cur):
                if _in_bounds((h, w), nb) and mask[nb] and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _ordered_path(cells: set[tuple[int, int]], start: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    degrees = {
        cell: sum(nb in cells for nb in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    cur = endpoints[0] if start is None else start
    assert cur in endpoints, "ordered path must start from an endpoint"
    prev = None
    out = []
    while True:
        out.append(cur)
        nxt = [nb for nb in _neighbors4(*cur) if nb in cells and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]
    assert len(out) == len(cells), "path traversal must cover every cell"
    return out


def _extract_legends(grid: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    edge_cells = {
        "top": [(0, c) for c in range(1, w - 1) if int(grid[0, c]) != bg],
        "bottom": [(h - 1, c) for c in range(1, w - 1) if int(grid[h - 1, c]) != bg],
        "left": [(r, 0) for r in range(1, h - 1) if int(grid[r, 0]) != bg],
        "right": [(r, w - 1) for r in range(1, h - 1) if int(grid[r, w - 1]) != bg],
    }

    dir_by_color: dict[int, tuple[int, int]] = {}
    color_at_pos: dict[tuple[int, int], int] = {}
    for edge, coords in edge_cells.items():
        assert len(coords) == 1, "expected exactly one legend pixel on each edge"
        pos = coords[0]
        color = int(grid[pos])
        assert color != GREY and color != bg, "legend colors must differ from walker/background"
        dir_by_color[color] = EDGE_TO_DIR[edge]
        color_at_pos[pos] = color

    assert len(dir_by_color) == 4, "legend colors must be pairwise distinct"
    border_non_bg = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and int(grid[r, c]) != bg
    }
    assert border_non_bg == set(color_at_pos), "only the four legend pixels may occupy the border"

    for pos in color_at_pos:
        for nb in _neighbors8(*pos):
            if _in_bounds((h, w), nb) and nb != pos and int(grid[nb]) != bg:
                assert nb in color_at_pos, "legend pixels must stay isolated from interior objects"
    return dir_by_color, color_at_pos


def _simulate_positions(
    start: tuple[int, int],
    directions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    r, c = start
    trail = [start]
    for dr, dc in directions:
        r += dr
        c += dc
        if not _in_bounds(shape, (r, c)):
            return None
        trail.append((r, c))
    return trail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the colored path as a serial instruction tape and draw the walker trajectory decoded from border legends.

    Input structure:
    - The background is the dominant color.
    - There are exactly four isolated legend pixels, one on each border edge, and their colors are all distinct.
    - Inside the grid there is one 4-connected one-pixel-wide path using exactly those legend colors, plus one grey walker pixel adjacent to one path endpoint.

    Rule:
    - Each legend color means the direction of the edge it sits on: top=up, bottom=down, left=left, right=right.
    - Read the path colors starting from the endpoint touching the walker, then move the walker one cell per decoded direction.
    - Erase the whole colored path, keep the four legend pixels, and paint every visited walker cell, including start and end, grey.

    Key insight:
    - The path is not a geometric trajectory to copy; it is an instruction string whose color meanings are defined on the fly by the border legends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != GREY, "grey is reserved for the walker and output trace"
    dir_by_color, color_at_pos = _extract_legends(g, bg)
    legend_positions = set(color_at_pos)

    walker_coords = np.argwhere(g == GREY)
    assert walker_coords.shape[0] == 1, "expected exactly one walker pixel"
    walker = tuple(map(int, walker_coords[0]))
    assert walker not in legend_positions, "walker cannot overlap a legend"

    path_mask = np.isin(g, list(dir_by_color))
    for pos in legend_positions:
        path_mask[pos] = False
    comps = _mask_components(path_mask)
    assert len(comps) == 1, "expected exactly one interior path"
    path_cells = set(comps[0])

    path_neighbors_of_walker = [nb for nb in _neighbors4(*walker) if nb in path_cells]
    assert len(path_neighbors_of_walker) == 1, "walker must touch exactly one path cell"
    start_endpoint = path_neighbors_of_walker[0]

    ordered = _ordered_path(path_cells, start=start_endpoint)
    directions = [dir_by_color[int(g[r, c])] for r, c in ordered]
    trail = _simulate_positions(walker, directions, g.shape)
    assert trail is not None, "decoded walk must remain inside the grid"

    out = g.copy()
    for r, c in path_cells:
        out[r, c] = bg
    for r, c in trail:
        out[r, c] = GREY
    for pos, color in color_at_pos.items():
        out[pos] = color
    return out


def _candidate_steps(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cur = path[-1]
    candidates = []
    for nb in _neighbors4(*cur):
        if nb not in allowed or nb in used:
            continue
        if sum(adj in used for adj in _neighbors4(*nb)) != 1:
            continue
        temp_used = used | {nb}
        onward = 0
        for nxt in _neighbors4(*nb):
            if nxt in allowed and nxt not in temp_used and sum(adj in temp_used for adj in _neighbors4(*nxt)) == 1:
                onward += 1
        candidates.append((onward, random.random(), nb))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [nb for _, _, nb in candidates]


def _build_simple_path(
    allowed: set[tuple[int, int]],
    length: int,
    start: tuple[int, int],
) -> list[tuple[int, int]] | None:
    assert start in allowed

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return list(path)
        for nb in _candidate_steps(path, used, allowed):
            path.append(nb)
            used.add(nb)
            found = dfs(path, used)
            if found is not None:
                return found
            used.remove(nb)
            path.pop()
        return None

    return dfs([start], {start})


def _dirs_from_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]


def _turn_count(directions: list[tuple[int, int]]) -> int:
    return sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def _paint_expected(
    shape: tuple[int, int],
    bg: int,
    legends: dict[tuple[int, int], int],
    tape_path: list[tuple[int, int]],
    walker: tuple[int, int],
    trail: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for pos, color in legends.items():
        out[pos] = color
    for r, c in tape_path:
        out[r, c] = bg
    for r, c in trail:
        out[r, c] = GREY
    for pos, color in legends.items():
        out[pos] = color
    return out


def _legend_output_from_dirs(
    shape: tuple[int, int],
    bg: int,
    legends: dict[tuple[int, int], int],
    tape_path: list[tuple[int, int]],
    walker: tuple[int, int],
    directions: list[tuple[int, int]],
) -> np.ndarray | None:
    trail = _simulate_positions(walker, directions, shape)
    if trail is None:
        return None
    return _paint_expected(shape, bg, legends, tape_path, walker, trail)


def _rotate_positions(positions: list[tuple[int, int]], k: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = positions
    for _ in range(k % 4):
        out = [(c, h - 1 - r) for r, c in out]
        h, w = w, h
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    split = random.randint(7, h - 7)
    walker_col = random.randint(4, w - 5)
    walker = (split, walker_col)
    tape_start = (split - 1, walker_col)

    bottom_cells = {(r, c) for r in range(split, h - 2) for c in range(2, w - 2)}
    top_cells = {(r, c) for r in range(2, split) for c in range(2, w - 2)}

    trail_steps = random.randint(8, min(16, len(bottom_cells) - 1))
    trail_path = _build_simple_path(bottom_cells, trail_steps + 1, walker)
    if trail_path is None:
        return None
    trail_dirs = _dirs_from_path(trail_path)
    if set(trail_dirs) != set(EDGE_TO_DIR.values()):
        return None
    if _turn_count(trail_dirs) < 4:
        return None

    tape_path = _build_simple_path(top_cells, trail_steps, tape_start)
    if tape_path is None:
        return None
    if set(tape_path) & set(trail_path):
        return None
    if set(tape_path) == set(trail_path[1:]):
        return None

    tape_geom_dirs = [(walker[0] - tape_start[0], walker[1] - tape_start[1])] + _dirs_from_path(tape_path)
    if len(tape_geom_dirs) != len(trail_dirs):
        return None

    palette = [color for color in range(10) if color != GREY]
    bg = random.choice(palette)
    legend_colors = random.sample([color for color in palette if color != bg], 4)
    edge_colors = dict(zip(("top", "bottom", "left", "right"), legend_colors))
    color_by_dir = {EDGE_TO_DIR[edge]: color for edge, color in edge_colors.items()}

    top_pos = (0, random.randint(2, w - 3))
    bottom_pos = (h - 1, random.randint(2, w - 3))
    left_pos = (random.randint(2, h - 3), 0)
    right_pos = (random.randint(2, h - 3), w - 1)
    legends = {
        top_pos: edge_colors["top"],
        bottom_pos: edge_colors["bottom"],
        left_pos: edge_colors["left"],
        right_pos: edge_colors["right"],
    }

    command_colors = [color_by_dir[direction] for direction in trail_dirs]
    grid = np.full((h, w), bg, dtype=int)
    for pos, color in legends.items():
        grid[pos] = color
    for (r, c), color in zip(tape_path, command_colors):
        grid[r, c] = color
    grid[walker] = GREY

    expected = _paint_expected(grid.shape, bg, legends, tape_path, walker, trail_path)
    wrong_reverse = _legend_output_from_dirs(grid.shape, bg, legends, tape_path, walker, list(reversed(trail_dirs)))
    wrong_rotated = _legend_output_from_dirs(
        grid.shape,
        bg,
        legends,
        tape_path,
        walker,
        [ROTATE_CW[direction] for direction in trail_dirs],
    )
    wrong_geometry = _legend_output_from_dirs(grid.shape, bg, legends, tape_path, walker, tape_geom_dirs)

    if wrong_reverse is None or np.array_equal(wrong_reverse, expected):
        return None
    if wrong_rotated is None or np.array_equal(wrong_rotated, expected):
        return None
    if wrong_geometry is None or np.array_equal(wrong_geometry, expected):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    k = random.randint(0, 3)
    if k:
        rotated_grid = np.rot90(grid, k=k)
        rotated_expected = np.rot90(expected, k=k)
        if not np.array_equal(main(rotated_grid), rotated_expected):
            return None
        return rotated_grid
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ribbon-program input")
