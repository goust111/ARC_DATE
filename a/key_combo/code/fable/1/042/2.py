from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 400
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _normalize_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _coarse_bbox_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _max_coarse_dim(frame_size: int) -> int:
    return (30 - 1 - 4) // (frame_size - 1)


def _rotate_cw(local: tuple[int, int], inner_size: int) -> tuple[int, int]:
    r, c = local
    return c, inner_size - 1 - r


def _rotate_ccw(local: tuple[int, int], inner_size: int) -> tuple[int, int]:
    r, c = local
    return inner_size - 1 - c, r


def _find_frames(
    grid: np.ndarray, frame_color: int, bg: int, seed_color: int
) -> list[dict[str, object]]:
    h, w = grid.shape
    allowed_inside = {int(bg), int(seed_color)}
    by_size: dict[int, list[dict[str, object]]] = {}

    for size in range(3, min(h, w) + 1):
        found: list[dict[str, object]] = []
        for r0 in range(h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(w - size + 1):
                c1 = c0 + size - 1
                if not (
                    np.all(grid[r0, c0 : c1 + 1] == frame_color)
                    and np.all(grid[r1, c0 : c1 + 1] == frame_color)
                    and np.all(grid[r0 : r1 + 1, c0] == frame_color)
                    and np.all(grid[r0 : r1 + 1, c1] == frame_color)
                ):
                    continue
                interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                if interior.size == 0:
                    continue
                values = set(map(int, np.unique(interior)))
                if frame_color in values or not values <= allowed_inside:
                    continue
                found.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "top_left": (r0, c0),
                        "size": size,
                        "inner_size": size - 2,
                    }
                )
        if found:
            by_size[size] = found

    assert by_size, "expected at least one hollow square frame"
    size = max(by_size, key=lambda s: (len(by_size[s]), s))
    return by_size[size]


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    non_bg = [int(x) for x in np.unique(g) if int(x) != bg]
    assert len(non_bg) == 2, "expected one frame color and one seed color"

    counts = Counter(map(int, g[g != bg].ravel()))
    frame_color = max(non_bg, key=lambda color: counts[color])
    seed_color = min(non_bg, key=lambda color: counts[color])

    seed_pixels = np.argwhere(g == seed_color)
    assert len(seed_pixels) == 1, "expected exactly one seed pixel"
    seed_r, seed_c = map(int, seed_pixels[0])

    frames = _find_frames(g, frame_color, bg, seed_color)
    size = int(frames[0]["size"])
    inner_size = int(frames[0]["inner_size"])
    assert all(int(frame["size"]) == size for frame in frames), "all frames must share the same size"

    root = None
    root_local = None
    top_left_to_index: dict[tuple[int, int], int] = {}
    for index, frame in enumerate(frames):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        top_left_to_index[(r0, c0)] = index
        if r0 < seed_r < r1 and c0 < seed_c < c1:
            root = index
            root_local = (seed_r - r0 - 1, seed_c - c0 - 1)

    assert root is not None and root_local is not None, "seed must lie inside exactly one frame"

    step = size - 1
    graph: list[list[int]] = [[] for _ in frames]
    for index, frame in enumerate(frames):
        r0, c0 = map(int, frame["top_left"])
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            neighbor = top_left_to_index.get((r0 + dr, c0 + dc))
            if neighbor is not None:
                graph[index].append(neighbor)

    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "seed_color": seed_color,
        "frames": frames,
        "inner_size": inner_size,
        "root": root,
        "root_local": root_local,
        "graph": graph,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the seed through a tree of touching hollow square frames, rotating it one quarter-turn at each transfer.

    Input structure:
    - The background is black.
    - A set of same-sized grey hollow square frames touch along full edges, forming one connected chain or tree.
    - Exactly one frame cavity contains a single colored seed pixel; every other cavity is empty.

    Rule:
    - Keep the frames and the original seed color.
    - Start from the frame that already contains the seed and traverse the frame-adjacency tree.
    - Whenever the seed state moves to a horizontally adjacent frame, rotate its cavity-local position 90 degrees clockwise.
    - Whenever it moves to a vertically adjacent frame, rotate its cavity-local position 90 degrees counterclockwise.
    - Draw the resulting seed position in every reachable frame cavity, accumulating rotations independently along each branch.

    Key insight:
    - The seed's location is a propagated state on the frame tree; the axis of each single step, not the absolute distance,
      decides whether the next local position is a clockwise or counterclockwise quarter-turn.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    seed_color = int(scene["seed_color"])
    frames = scene["frames"]
    inner_size = int(scene["inner_size"])
    root = int(scene["root"])
    root_local = tuple(scene["root_local"])
    graph = scene["graph"]

    states: dict[int, tuple[int, int]] = {root: root_local}
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        r0, c0 = map(int, frames[current]["top_left"])
        for neighbor in graph[current]:
            if neighbor in states:
                continue
            nr0, nc0 = map(int, frames[neighbor]["top_left"])
            if r0 == nr0:
                states[neighbor] = _rotate_cw(states[current], inner_size)
            else:
                states[neighbor] = _rotate_ccw(states[current], inner_size)
            queue.append(neighbor)

    assert len(states) == len(frames), "expected one connected frame tree"

    out = g.copy()
    for index, frame in enumerate(frames):
        local_r, local_c = states[index]
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        out[r0 + 1 + local_r, c0 + 1 + local_c] = seed_color
    return out


def _build_chain(frame_count: int, need_turn: bool, frame_size: int) -> list[tuple[int, int]] | None:
    limit = _max_coarse_dim(frame_size)
    for _ in range(200):
        cells = [(0, 0)]
        while len(cells) < frame_count:
            tail_r, tail_c = cells[-1]
            options = []
            for dr, dc in DIR4:
                nxt = (tail_r + dr, tail_c + dc)
                if nxt in cells:
                    continue
                trial = _normalize_cells(cells + [nxt])
                box_h, box_w = _coarse_bbox_dims(trial)
                if box_h <= limit and box_w <= limit:
                    options.append(nxt)
            if not options:
                break
            if need_turn and len(cells) >= 2:
                prev_r, prev_c = cells[-2]
                cur_dir = (tail_r - prev_r, tail_c - prev_c)
                turners = [
                    nxt
                    for nxt in options
                    if (nxt[0] - tail_r, nxt[1] - tail_c) != cur_dir
                ]
                if turners and random.random() < 0.6:
                    options = turners
            cells.append(random.choice(options))
        if len(cells) != frame_count:
            continue
        steps = [
            "h" if cells[i][0] == cells[i - 1][0] else "v"
            for i in range(1, len(cells))
        ]
        if need_turn and len(set(steps)) < 2:
            continue
        if not need_turn and len(set(steps)) != 1:
            continue
        return _normalize_cells(cells)
    return None


def _build_branch_tree(frame_count: int, frame_size: int) -> tuple[list[tuple[int, int]], int] | None:
    limit = _max_coarse_dim(frame_size)
    arm_sets = [
        [(-1, 0), (1, 0), (0, 1)],
        [(-1, 0), (1, 0), (0, -1)],
        [(0, -1), (0, 1), (1, 0)],
        [(0, -1), (0, 1), (-1, 0)],
    ]
    for _ in range(200):
        arms = random.choice(arm_sets)
        cells = [(0, 0)] + arms[:]
        cell_set = set(cells)
        while len(cells) < frame_count:
            options = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt in cell_set:
                        continue
                    degree = sum((nxt[0] + rr, nxt[1] + cc) in cell_set for rr, cc in DIR4)
                    if degree != 1:
                        continue
                    trial = _normalize_cells(cells + [nxt])
                    box_h, box_w = _coarse_bbox_dims(trial)
                    if box_h <= limit and box_w <= limit:
                        options.append(nxt)
            if not options:
                break
            random.shuffle(options)
            cells.append(options[0])
            cell_set.add(options[0])
        if len(cells) != frame_count:
            continue
        normalized = _normalize_cells(cells)
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        center = normalized.index((-min_r, -min_c))
        return normalized, center
    return None


def _oracle_states(
    cells: list[tuple[int, int]], root: int, seed_local: tuple[int, int], inner_size: int
) -> dict[int, tuple[int, int]]:
    coord_to_index = {coord: index for index, coord in enumerate(cells)}
    graph: list[list[int]] = [[] for _ in cells]
    for index, (r, c) in enumerate(cells):
        for dr, dc in DIR4:
            neighbor = coord_to_index.get((r + dr, c + dc))
            if neighbor is not None:
                graph[index].append(neighbor)

    states: dict[int, tuple[int, int]] = {root: seed_local}
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        r, c = cells[current]
        for neighbor in graph[current]:
            if neighbor in states:
                continue
            nr, nc = cells[neighbor]
            if r == nr:
                states[neighbor] = _rotate_cw(states[current], inner_size)
            else:
                states[neighbor] = _rotate_ccw(states[current], inner_size)
            queue.append(neighbor)
    return states


def _render_input(
    cells: list[tuple[int, int]],
    frame_size: int,
    seed_index: int,
    seed_local: tuple[int, int],
    seed_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    cells = _normalize_cells(cells)
    box_h, box_w = _coarse_bbox_dims(cells)
    used_h = box_h * (frame_size - 1) + 1
    used_w = box_w * (frame_size - 1) + 1

    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    grid = np.full((used_h + top + bottom, used_w + left + right), BLACK, dtype=int)

    for cell_r, cell_c in cells:
        r0 = top + cell_r * (frame_size - 1)
        c0 = left + cell_c * (frame_size - 1)
        r1 = r0 + frame_size - 1
        c1 = c0 + frame_size - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

    root_r, root_c = cells[seed_index]
    seed_r = top + root_r * (frame_size - 1) + 1 + seed_local[0]
    seed_c = left + root_c * (frame_size - 1) + 1 + seed_local[1]
    grid[seed_r, seed_c] = int(seed_color)
    return grid, (top, left)


def _render_expected(
    input_grid: np.ndarray,
    cells: list[tuple[int, int]],
    frame_size: int,
    seed_index: int,
    seed_local: tuple[int, int],
    seed_color: int,
    offsets: tuple[int, int],
) -> np.ndarray:
    top, left = offsets
    inner_size = frame_size - 2
    states = _oracle_states(cells, seed_index, seed_local, inner_size)
    out = input_grid.copy()
    for index, (cell_r, cell_c) in enumerate(cells):
        local_r, local_c = states[index]
        pixel_r = top + cell_r * (frame_size - 1) + 1 + local_r
        pixel_c = left + cell_c * (frame_size - 1) + 1 + local_c
        out[pixel_r, pixel_c] = int(seed_color)
    return out


def _pick_seed_local(inner_size: int) -> tuple[int, int]:
    center = (inner_size // 2, inner_size // 2)
    options = [
        (r, c)
        for r in range(inner_size)
        for c in range(inner_size)
        if (r, c) != center
    ]
    return random.choice(options)


def _make_layout(mode: str) -> tuple[list[tuple[int, int]], int, int] | None:
    if mode == "branch":
        frame_size = 5 if random.random() < 0.7 else 7
        max_frames = 7 if frame_size == 5 else 5
        frame_count = random.randint(4, max_frames)
        built = _build_branch_tree(frame_count, frame_size)
        if built is None:
            return None
        cells, center_index = built
        if random.random() < 0.65:
            seed_index = center_index
        else:
            seed_index = random.randrange(len(cells))
        return cells, frame_size, seed_index

    if mode == "turn":
        frame_size = random.choice((5, 7))
        max_frames = 6 if frame_size == 5 else 4
        frame_count = random.randint(3, max_frames)
        cells = _build_chain(frame_count, True, frame_size)
        if cells is None:
            return None
        if frame_count >= 4 and random.random() < 0.6:
            seed_index = random.randint(1, frame_count - 2)
        else:
            seed_index = random.randrange(frame_count)
        return cells, frame_size, seed_index

    frame_size = random.choice((5, 7))
    max_frames = 6 if frame_size == 5 else 4
    frame_count = random.randint(2, max_frames)
    cells = _build_chain(frame_count, False, frame_size)
    if cells is None:
        return None
    seed_index = random.randrange(frame_count)
    return cells, frame_size, seed_index


def _gen_attempt(mode: str) -> np.ndarray | None:
    layout = _make_layout(mode)
    if layout is None:
        return None

    cells, frame_size, seed_index = layout
    seed_color = random.choice(SEED_COLORS)
    seed_local = _pick_seed_local(frame_size - 2)
    input_grid, offsets = _render_input(cells, frame_size, seed_index, seed_local, seed_color)
    expected = _render_expected(input_grid, cells, frame_size, seed_index, seed_local, seed_color, offsets)

    if max(input_grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.count_nonzero(input_grid == seed_color) != 1:
        return None
    if np.count_nonzero(expected == seed_color) != len(cells):
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    modes = ("branch", "turn", "straight")
    preferred_mode = modes[_GEN_CALL_COUNT % len(modes)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(preferred_mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transmission-chain input")
