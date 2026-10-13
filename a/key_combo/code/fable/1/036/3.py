from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

COLOR_TO_DELTA = {
    RED: (0, 1),
    YELLOW: (0, -1),
    BLUE: (-1, 0),
    GREEN: (1, 0),
}
INSTRUCTION_COLORS = tuple(COLOR_TO_DELTA.keys())
ALL_ALLOWED_COLORS = {BLACK, BLUE, RED, GREEN, YELLOW, GREY}

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _positions_from_moves(moves: list[int]) -> list[tuple[int, int]]:
    pos = (0, 0)
    positions = [pos]
    for color in moves:
        dr, dc = COLOR_TO_DELTA[color]
        pos = (pos[0] + dr, pos[1] + dc)
        positions.append(pos)
    return positions


def _turn_count(moves: list[int]) -> int:
    deltas = [COLOR_TO_DELTA[color] for color in moves]
    return sum(a != b for a, b in zip(deltas, deltas[1:]))


def _trace_snake_colors(grid: np.ndarray) -> list[int]:
    snake_mask = np.isin(grid, INSTRUCTION_COLORS)
    comps = _components_mask(snake_mask)
    assert len(comps) == 1, "expected one instruction snake"
    coords = comps[0]
    cells = {tuple(map(int, rc)) for rc in coords.tolist()}

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints = []
    for r, c in cells:
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in cells]
        neighbors[(r, c)] = nbrs
        if len(nbrs) == 1:
            endpoints.append((r, c))
        else:
            assert len(nbrs) == 2, "snake must be a branch-free 4-connected path"

    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    cur = min(endpoints)
    prev = None
    ordered = [cur]
    while True:
        nxt = [pos for pos in neighbors[cur] if pos != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "snake traversal must be unambiguous"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(cells), "snake traversal must visit every cell exactly once"
    return [int(grid[r, c]) for r, c in ordered]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colored snake as an ordered instruction tape and execute it from the grey start marker.

    Input structure:
    - The background is black.
    - One multicolored 4-connected snake uses only red, yellow, blue, and green cells, has no branches, and has exactly two endpoints.
    - A single isolated grey pixel marks the execution start away from the snake.

    Rule:
    - Pick the lexicographically top-left snake endpoint and traverse the snake cell by cell to the other endpoint.
    - Interpret the visited snake colors in that order as moves: red=right, yellow=left, blue=up, green=down.
    - Starting from the grey marker, execute one move per snake cell and color each landing cell with that move color, overwriting earlier trajectory colors if revisited.
    - Erase the entire snake from the output and keep only the start marker plus the executed trajectory.

    Key insight:
    - The snake is not a picture to preserve; it is a one-dimensional program whose meaning comes only from traversal order, while the visible output is drawn elsewhere at the grey start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))).issubset(ALL_ALLOWED_COLORS), "unexpected color present"

    bg = _get_bg(g)
    assert bg == BLACK, "background must be black"

    start_coords = np.argwhere(g == GREY)
    assert start_coords.shape[0] == 1, "expected one grey start marker"
    start = tuple(map(int, start_coords[0]))

    colors = _trace_snake_colors(g)

    out = np.full_like(g, bg)
    out[start] = GREY
    r, c = start
    for color in colors:
        dr, dc = COLOR_TO_DELTA[int(color)]
        r += dr
        c += dc
        assert _in_bounds(g.shape, r, c), "executed trajectory must stay inside the grid"
        out[r, c] = int(color)

    return out


def _search_tail(
    cur: tuple[int, int],
    visited: set[tuple[int, int]],
    steps_left: int,
    used_colors: set[int],
    max_h: int,
    max_w: int,
    rng: np.random.Generator,
    require_all_colors: bool,
    bbox: tuple[int, int, int, int],
    strict_path: bool = False,
) -> list[int] | None:
    if steps_left == 0:
        if require_all_colors and len(used_colors) < len(INSTRUCTION_COLORS):
            return None
        return []

    missing = [color for color in INSTRUCTION_COLORS if color not in used_colors]
    colors = list(INSTRUCTION_COLORS)
    rng.shuffle(colors)
    if steps_left <= len(missing):
        colors.sort(key=lambda color: 0 if color in missing else 1)

    candidates = []
    r0, c0, r1, c1 = bbox
    for color in colors:
        dr, dc = COLOR_TO_DELTA[color]
        nr = cur[0] + dr
        nc = cur[1] + dc
        nxt = (nr, nc)
        if nxt in visited:
            continue
        if strict_path:
            touched = {nbr for nbr in _neighbors4(nr, nc) if nbr in visited}
            if touched != {cur}:
                continue
        nr0 = min(r0, nr)
        nc0 = min(c0, nc)
        nr1 = max(r1, nr)
        nc1 = max(c1, nc)
        if nr1 - nr0 + 1 > max_h or nc1 - nc0 + 1 > max_w:
            continue
        new_used = used_colors | {color}
        missing_after = sum(color_id not in new_used for color_id in INSTRUCTION_COLORS)
        if require_all_colors and steps_left - 1 < missing_after:
            continue
        onward = 0
        for dr2, dc2 in COLOR_TO_DELTA.values():
            probe = (nr + dr2, nc + dc2)
            if probe in visited:
                continue
            pr0 = min(nr0, probe[0])
            pc0 = min(nc0, probe[1])
            pr1 = max(nr1, probe[0])
            pc1 = max(nc1, probe[1])
            if pr1 - pr0 + 1 <= max_h and pc1 - pc0 + 1 <= max_w:
                onward += 1
        candidates.append((onward, float(rng.random()), color, nxt, (nr0, nc0, nr1, nc1), new_used))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _onward, _noise, color, nxt, new_bbox, new_used in candidates:
        visited.add(nxt)
        tail = _search_tail(
            nxt,
            visited,
            steps_left - 1,
            new_used,
            max_h,
            max_w,
            rng,
            require_all_colors,
            new_bbox,
            strict_path,
        )
        visited.remove(nxt)
        if tail is not None:
            return [color] + tail
    return None


def _build_unique_trajectory(rng: np.random.Generator) -> list[int]:
    for _ in range(200):
        length = int(rng.integers(7, 13))
        max_h = int(rng.integers(4, 7))
        max_w = int(rng.integers(4, 7))
        moves = _search_tail(
            cur=(0, 0),
            visited={(0, 0)},
            steps_left=length,
            used_colors=set(),
            max_h=max_h,
            max_w=max_w,
            rng=rng,
            require_all_colors=True,
            bbox=(0, 0, 0, 0),
            strict_path=False,
        )
        if moves is None:
            continue
        if _turn_count(moves) < 3:
            continue
        return moves
    raise RuntimeError("failed to build a unique trajectory")


def _build_revisit_trajectory(rng: np.random.Generator) -> list[int]:
    motifs = [
        [RED, GREEN, YELLOW, RED, BLUE],
        [GREEN, RED, BLUE, GREEN, YELLOW],
        [YELLOW, BLUE, RED, YELLOW, GREEN],
        [BLUE, YELLOW, GREEN, BLUE, RED],
    ]
    for _ in range(200):
        base = list(motifs[int(rng.integers(0, len(motifs)))])
        max_h = int(rng.integers(4, 7))
        max_w = int(rng.integers(4, 7))
        extra = int(rng.integers(2, 8))
        positions = _positions_from_moves(base)
        bbox = _bbox_coords(list(set(positions)))
        tail = _search_tail(
            cur=positions[-1],
            visited=set(positions),
            steps_left=extra,
            used_colors=set(base),
            max_h=max_h,
            max_w=max_w,
            rng=rng,
            require_all_colors=False,
            bbox=bbox,
            strict_path=False,
        )
        if tail is None:
            continue
        moves = base + tail
        if len(set(_positions_from_moves(moves))) == len(moves) + 1:
            continue
        if _turn_count(moves) < 3:
            continue
        return moves
    raise RuntimeError("failed to build a revisiting trajectory")


def _build_snake_path(cell_count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    assert cell_count >= 2
    for _ in range(300):
        max_h = int(rng.integers(4, min(8, cell_count + 2)))
        max_w = int(rng.integers(4, min(8, cell_count + 2)))
        moves = _search_tail(
            cur=(0, 0),
            visited={(0, 0)},
            steps_left=cell_count - 1,
            used_colors=set(),
            max_h=max_h,
            max_w=max_w,
            rng=rng,
            require_all_colors=False,
            bbox=(0, 0, 0, 0),
            strict_path=True,
        )
        if moves is None:
            continue
        if _turn_count(moves) < 2:
            continue
        return _positions_from_moves(moves)
    raise RuntimeError("failed to build a snake path")


def _shift_positions(
    positions: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in positions]


def _place_boxes(
    snake_bbox: tuple[int, int, int, int],
    traj_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    snake_h = snake_bbox[2] - snake_bbox[0] + 1
    snake_w = snake_bbox[3] - snake_bbox[1] + 1
    traj_h = traj_bbox[2] - traj_bbox[0] + 1
    traj_w = traj_bbox[3] - traj_bbox[1] + 1

    snake_pad = tuple(int(rng.integers(0, 3)) for _ in range(4))
    traj_pad = tuple(int(rng.integers(0, 3)) for _ in range(4))
    outer = tuple(int(rng.integers(1, 3)) for _ in range(4))
    gap = int(rng.integers(3, 6))
    horizontal = bool(rng.integers(0, 2))
    snake_first = bool(rng.integers(0, 2))

    snake_box_h = snake_h + snake_pad[0] + snake_pad[2]
    snake_box_w = snake_w + snake_pad[1] + snake_pad[3]
    traj_box_h = traj_h + traj_pad[0] + traj_pad[2]
    traj_box_w = traj_w + traj_pad[1] + traj_pad[3]

    if horizontal:
        content_h = max(snake_box_h, traj_box_h)
        h = outer[0] + content_h + outer[2]
        w = outer[1] + snake_box_w + gap + traj_box_w + outer[3]
        if h > 30 or w > 30:
            raise RuntimeError("layout exceeds ARC bounds")
        snake_content_r = int(rng.integers(0, content_h - snake_box_h + 1))
        traj_content_r = int(rng.integers(0, content_h - traj_box_h + 1))
        if snake_first:
            left_first = outer[1]
            left_second = outer[1] + snake_box_w + gap
            snake_box = (outer[0] + snake_content_r, left_first)
            traj_box = (outer[0] + traj_content_r, left_second)
        else:
            left_first = outer[1]
            left_second = outer[1] + traj_box_w + gap
            traj_box = (outer[0] + traj_content_r, left_first)
            snake_box = (outer[0] + snake_content_r, left_second)
    else:
        content_w = max(snake_box_w, traj_box_w)
        h = outer[0] + snake_box_h + gap + traj_box_h + outer[2]
        w = outer[1] + content_w + outer[3]
        if h > 30 or w > 30:
            raise RuntimeError("layout exceeds ARC bounds")
        snake_content_c = int(rng.integers(0, content_w - snake_box_w + 1))
        traj_content_c = int(rng.integers(0, content_w - traj_box_w + 1))
        if snake_first:
            top_first = outer[0]
            top_second = outer[0] + snake_box_h + gap
            snake_box = (top_first, outer[1] + snake_content_c)
            traj_box = (top_second, outer[1] + traj_content_c)
        else:
            top_first = outer[0]
            top_second = outer[0] + traj_box_h + gap
            traj_box = (top_first, outer[1] + traj_content_c)
            snake_box = (top_second, outer[1] + snake_content_c)

    snake_anchor = (snake_box[0] + snake_pad[0], snake_box[1] + snake_pad[1])
    traj_anchor = (traj_box[0] + traj_pad[0], traj_box[1] + traj_pad[1])
    return (h, w), snake_anchor, traj_anchor


def _render_instance(
    snake_positions: list[tuple[int, int]],
    trajectory_moves: list[int],
    trajectory_positions: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    snake_bbox = _bbox_coords(snake_positions)
    traj_bbox = _bbox_coords(trajectory_positions)
    shape, snake_anchor, traj_anchor = _place_boxes(snake_bbox, traj_bbox, rng)

    snake_shifted = _shift_positions(
        snake_positions,
        snake_anchor[0] - snake_bbox[0],
        snake_anchor[1] - snake_bbox[1],
    )
    traj_shifted = _shift_positions(
        trajectory_positions,
        traj_anchor[0] - traj_bbox[0],
        traj_anchor[1] - traj_bbox[1],
    )

    start = traj_shifted[0]
    snake_cells = set(snake_shifted)
    traj_cells = set(traj_shifted)
    assert snake_cells.isdisjoint(traj_cells), "input objects must not overlap future trajectory area"

    grid = np.full(shape, BLACK, dtype=int)
    grid[start] = GREY

    ordered_snake = snake_shifted
    if ordered_snake[0] > ordered_snake[-1]:
        ordered_snake = list(reversed(ordered_snake))
    for (r, c), color in zip(ordered_snake, trajectory_moves):
        grid[r, c] = color

    assert np.count_nonzero(grid == GREY) == 1
    return grid


def _gen_attempt(rng: np.random.Generator, mode: str) -> np.ndarray | None:
    try:
        if mode == "revisit":
            trajectory_moves = _build_revisit_trajectory(rng)
        else:
            trajectory_moves = _build_unique_trajectory(rng)
        trajectory_positions = _positions_from_moves(trajectory_moves)
        snake_positions = _build_snake_path(len(trajectory_moves), rng)
        grid = _render_instance(snake_positions, trajectory_moves, trajectory_positions, rng)
    except RuntimeError:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if len(set(trajectory_moves)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    _GEN_CALL_COUNT += 1
    forced_mode = "revisit" if _GEN_CALL_COUNT % 3 == 2 else "unique"
    for attempt in range(400):
        mode = forced_mode if attempt < 150 else ("revisit" if _GEN_RNG.random() < 0.45 else "unique")
        grid = _gen_attempt(_GEN_RNG, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-instruction instance")
