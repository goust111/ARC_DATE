from __future__ import annotations

import random
from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation

from arc_utils import Color, STRUCT8, get_bg, in_bounds, label_mask, neighbors4


PATH_COLOR = Color.GREY
PLUG_COLOR = Color.TEAL
TOKEN_COLORS = [c for c in Color.NOT_BLACK if c not in {PATH_COLOR, PLUG_COLOR}]


def _adjacent_path_labels(labels: np.ndarray, r: int, c: int) -> set[int]:
    adjacent = set()
    for rr, cc in neighbors4(r, c):
        if in_bounds(labels.shape, rr, cc) and labels[rr, cc] > 0:
            adjacent.add(int(labels[rr, cc]))
    return adjacent


def _corridor_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path_labels, n_paths = label_mask(grid == PATH_COLOR, connectivity=4)
    assert n_paths == 2, "expected exactly two grey path components"

    corridor = grid != PATH_COLOR
    for label_id in range(1, n_paths + 1):
        corridor &= binary_dilation(path_labels == label_id, structure=STRUCT8)
    assert np.any(corridor), "expected a non-empty corridor"
    return corridor, path_labels


def _find_plug(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == PLUG_COLOR)
    assert len(coords) == 1, "expected exactly one plug pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _token_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    return (grid != bg) & (grid != PATH_COLOR) & (grid != PLUG_COLOR)


def _token_edge(grid: np.ndarray, token_coords: list[tuple[int, int]]) -> str:
    assert token_coords, "expected at least one token"
    if all(r == 0 for r, _ in token_coords):
        return "top"
    if all(r == grid.shape[0] - 1 for r, _ in token_coords):
        return "bottom"
    if all(c == 0 for _, c in token_coords):
        return "left"
    if all(c == grid.shape[1] - 1 for _, c in token_coords):
        return "right"
    raise AssertionError("tokens must all lie on one grid edge")


def _opening_on_edge(corridor: np.ndarray, edge: str) -> tuple[int, int]:
    if edge == "top":
        coords = [(0, int(c)) for c in np.flatnonzero(corridor[0, :])]
    elif edge == "bottom":
        row = corridor.shape[0] - 1
        coords = [(row, int(c)) for c in np.flatnonzero(corridor[row, :])]
    elif edge == "left":
        coords = [(int(r), 0) for r in np.flatnonzero(corridor[:, 0])]
    else:
        col = corridor.shape[1] - 1
        coords = [(int(r), col) for r in np.flatnonzero(corridor[:, col])]
    assert len(coords) == 1, "expected exactly one corridor opening on the token edge"
    return coords[0]


def _shortest_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nxt in neighbors4(*cur):
            if not in_bounds(mask.shape, *nxt) or not mask[nxt] or nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    assert goal in parent, "could not trace corridor from opening to plug"
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _path_to_plug(corridor: np.ndarray, plug: tuple[int, int], opening: tuple[int, int]) -> list[tuple[int, int]]:
    passable = corridor.copy()
    passable[plug] = False
    comp_labels, _ = label_mask(passable, connectivity=4)
    opening_label = int(comp_labels[opening])
    assert opening_label > 0, "opening must remain passable"

    targets = []
    for rr, cc in neighbors4(*plug):
        if in_bounds(passable.shape, rr, cc) and int(comp_labels[rr, cc]) == opening_label:
            targets.append((rr, cc))
    assert len(targets) == 1, "plug should split the corridor into two sides"
    return _shortest_path(comp_labels == opening_label, opening, targets[0])


def _sorted_token_sources(
    token_coords: list[tuple[int, int]], edge: str, opening: tuple[int, int]
) -> list[tuple[int, int]]:
    if edge in {"top", "bottom"}:
        return sorted(token_coords, key=lambda rc: (abs(rc[1] - opening[1]), rc[1]))
    return sorted(token_coords, key=lambda rc: (abs(rc[0] - opening[0]), rc[0]))


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Move edge tokens into the hidden corridor formed by the overlap of the two grey paths'
    8-neighborhood halos, packing them from the plug back toward the entrance.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    bg = get_bg(grid)
    assert bg == Color.BLACK, "expected black background"

    corridor, _ = _corridor_mask(grid)
    plug = _find_plug(grid)
    assert corridor[plug], "plug must lie inside the corridor"

    token_coords = [tuple(map(int, rc)) for rc in np.argwhere(_token_mask(grid, bg))]
    edge = _token_edge(grid, token_coords)
    opening = _opening_on_edge(corridor, edge)
    path_to_plug = _path_to_plug(corridor, plug, opening)

    ordered_tokens = _sorted_token_sources(token_coords, edge, opening)
    move_count = min(len(ordered_tokens), len(path_to_plug))
    destinations = list(reversed(path_to_plug))[:move_count]

    out = grid.copy()
    for src in ordered_tokens[:move_count]:
        out[src] = bg
    for src, dst in zip(ordered_tokens[:move_count], destinations):
        out[dst] = grid[src]
    return out


GEN_SHAPE = (12, 12)
GEN_TOP_ROW = 0
GEN_BOTTOM_ROW = GEN_SHAPE[0] - 1
GEN_LEFT_COL = 0
GEN_RIGHT_COL = GEN_SHAPE[1] - 1
GEN_MIN_CHANNEL_COL = 2
GEN_MAX_CHANNEL_COL = GEN_SHAPE[1] - 3
GEN_MIN_HORIZONTALS = 1
GEN_MAX_HORIZONTALS = 3
GEN_HORIZONTAL_MIN = 1
GEN_HORIZONTAL_MAX = 3
GEN_VERTICAL_MIN = 1
GEN_VERTICAL_MAX = 6
GEN_RETRY_LIMIT = 400


def _gen_walk_segment(path: list[tuple[int, int]], dr: int, dc: int, steps: int) -> None:
    r, c = path[-1]
    for _ in range(steps):
        r += dr
        c += dc
        path.append((r, c))


def _gen_partition_total(total: int, count: int, minimum: int, maximum: int) -> list[int] | None:
    parts = []
    remaining = total
    for i in range(count - 1):
        rest = count - i - 1
        lo = max(minimum, remaining - maximum * rest)
        hi = min(maximum, remaining - minimum * rest)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        parts.append(value)
        remaining -= value
    if not (minimum <= remaining <= maximum):
        return None
    parts.append(remaining)
    return parts


def _gen_channel_path() -> list[tuple[int, int]] | None:
    hdir = random.choice([-1, 1])
    h_count = random.randint(GEN_MIN_HORIZONTALS, GEN_MAX_HORIZONTALS)
    directions = [(1, 0)]
    for _ in range(h_count):
        directions.append((0, hdir))
        directions.append((1, 0))

    vertical_steps = _gen_partition_total(
        GEN_BOTTOM_ROW, h_count + 1, GEN_VERTICAL_MIN, GEN_VERTICAL_MAX
    )
    if vertical_steps is None:
        return None

    while True:
        horizontal_steps = [random.randint(GEN_HORIZONTAL_MIN, GEN_HORIZONTAL_MAX) for _ in range(h_count)]
        total_h = sum(horizontal_steps)
        if total_h <= GEN_MAX_CHANNEL_COL - GEN_MIN_CHANNEL_COL:
            break

    if hdir == 1:
        start_c = random.randint(GEN_MIN_CHANNEL_COL, GEN_MAX_CHANNEL_COL - total_h)
    else:
        start_c = random.randint(GEN_MIN_CHANNEL_COL + total_h, GEN_MAX_CHANNEL_COL)

    path = [(GEN_TOP_ROW, start_c)]
    vi = 0
    hi = 0
    for dr, dc in directions:
        if dr:
            steps = vertical_steps[vi]
            vi += 1
        else:
            steps = horizontal_steps[hi]
            hi += 1
        _gen_walk_segment(path, dr, dc, steps)

    if path[-1][0] != GEN_BOTTOM_ROW:
        return None
    if any(
        r < GEN_TOP_ROW or r > GEN_BOTTOM_ROW or c < GEN_MIN_CHANNEL_COL or c > GEN_MAX_CHANNEL_COL
        for r, c in path
    ):
        return None
    return path


def _gen_left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _gen_wall_sets(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left = set()
    right = set()
    channel = set(path)
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))

        for dr, dc in dirs:
            lr, lc = _gen_left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))

        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _gen_left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr, c + dc) for dr, dc in offsets]
                if all(cell not in channel for cell in direct):
                    er = r + offsets[0][0] + offsets[1][0]
                    ec = c + offsets[0][1] + offsets[1][1]
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _gen_mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(GEN_SHAPE, dtype=bool)
    if cells:
        rows, cols = zip(*cells)
        mask[rows, cols] = True
    return mask


def _gen_is_thin_component(mask: np.ndarray) -> bool:
    labels, count = label_mask(mask, connectivity=4)
    if count != 1:
        return False
    for r_raw, c_raw in np.argwhere(labels == 1):
        r = int(r_raw)
        c = int(c_raw)
        degree = 0
        for rr, cc in neighbors4(r, c):
            if in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
                degree += 1
        if degree > 2:
            return False
    return True


def _gen_render_valid_paths(
    channel_path: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    left, right = _gen_wall_sets(channel_path)
    if left & right or left & set(channel_path) or right & set(channel_path):
        return None, None
    if any(
        r < GEN_TOP_ROW or r > GEN_BOTTOM_ROW or c < GEN_LEFT_COL or c > GEN_RIGHT_COL
        for r, c in left | right
    ):
        return None, None

    left_mask = _gen_mask_from_cells(left)
    right_mask = _gen_mask_from_cells(right)
    if not _gen_is_thin_component(left_mask) or not _gen_is_thin_component(right_mask):
        return None, None

    grid = np.zeros(GEN_SHAPE, dtype=int)
    grid[left_mask | right_mask] = PATH_COLOR
    corridor, path_labels = _corridor_mask(grid)
    expected = _gen_mask_from_cells(set(channel_path))
    if not np.array_equal(corridor, expected):
        return None, None
    return grid, path_labels


def _gen_plug_index(path_labels: np.ndarray, channel_path: list[tuple[int, int]]) -> int | None:
    weak = []
    fallback = []
    for idx in range(2, len(channel_path) - 2):
        r, c = channel_path[idx]
        adjacent = _adjacent_path_labels(path_labels, r, c)
        if len(adjacent) == 1:
            weak.append(idx)
        elif len(adjacent) == 2:
            fallback.append(idx)
    candidates = weak if weak else fallback
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_token_positions(grid: np.ndarray, opening_col: int, capacity: int) -> list[tuple[int, int]] | None:
    left_slots = [c for c in range(opening_col - 1, 0, -1) if grid[GEN_TOP_ROW, c] == Color.BLACK]
    right_slots = [
        c for c in range(opening_col + 1, GEN_RIGHT_COL) if grid[GEN_TOP_ROW, c] == Color.BLACK
    ]
    options = [slots for slots in (left_slots, right_slots) if slots]
    if not options:
        return None

    slots = random.choice(options)
    max_tokens = min(len(slots), capacity + 3)
    if max_tokens < 1:
        return None
    count = random.randint(1, max_tokens)
    cols = random.sample(slots, count)
    return [(GEN_TOP_ROW, c) for c in cols]


def _gen_token_colors(count: int) -> list[int]:
    if count <= len(TOKEN_COLORS):
        return random.sample(TOKEN_COLORS, count)
    return [random.choice(TOKEN_COLORS) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    channel_path = _gen_channel_path()
    if channel_path is None:
        return None

    grid, path_labels = _gen_render_valid_paths(channel_path)
    if grid is None:
        return None

    plug_idx = _gen_plug_index(path_labels, channel_path)
    if plug_idx is None:
        return None
    grid[channel_path[plug_idx]] = PLUG_COLOR

    token_positions = _gen_token_positions(grid, channel_path[0][1], plug_idx)
    if token_positions is None:
        return None
    for (r, c), color in zip(token_positions, _gen_token_colors(len(token_positions))):
        grid[r, c] = color

    grid = np.rot90(grid, k=random.randint(0, 3))
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-corridor input")
