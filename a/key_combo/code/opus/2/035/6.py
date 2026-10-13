from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
PERPENDICULAR = {
    "right": ("up", "down"),
    "left": ("up", "down"),
    "up": ("left", "right"),
    "down": ("left", "right"),
}
PASSABLE_COLORS = {Color.ORANGE, Color.PINK, Color.YELLOW, Color.MAROON}
SPRITE_COLORS = (Color.YELLOW, Color.MAROON)
BAR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.TEAL]
REMOTE_SHAPES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_sprite_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color not in SPRITE_COLORS or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    if int(grid[r, c]) not in PASSABLE_COLORS:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in PASSABLE_COLORS


def trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == Color.PINK)
    assert starts.shape == (1, 2), "expected one magenta particle"
    r, c = map(int, starts[0])
    direction = "right"
    path = [(r, c)]

    for _ in range(g.size * 8):
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break
        if int(g[nr, nc]) in PASSABLE_COLORS:
            r, c = nr, nc
            path.append((r, c))
            continue

        options = []
        for turn_dir in PERPENDICULAR[direction]:
            tdr, tdc = DIRS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(g, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))
    return path


def _first_approach_index(
    path: list[tuple[int, int]], sprite_coords: list[tuple[int, int]]
) -> int | None:
    sprite_set = set(sprite_coords)
    frontier = set()
    for r, c in sprite_coords:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in sprite_set:
                frontier.add((nr, nc))
    for idx, coord in enumerate(path):
        if coord in frontier:
            return idx
    return None


def _sprite_crop(sprite: dict) -> np.ndarray:
    r0, c0, r1, c1 = sprite["bbox"]
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in sprite["coords"]:
        out[r - r0, c - c0] = int(sprite["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the hidden particle route, keep only the approached sprites, and concatenate their crops in visit order.

    Input structure:
    - The grid has orange background, one magenta particle on the left edge, colored corridor bars, and several small yellow or maroon sprites.
    - Corridor turns are encoded by same-color parallel bars flanking the next legal cell.
    - Sprites are 4-connected monochrome objects and do not block movement.

    Rule:
    - Trace the particle from the magenta start, moving right through orange or sprite cells.
    - When the next cell is a colored bar, turn 90 degrees into the unique perpendicular passable cell whose two side neighbors are equal non-passable bar colors.
    - For each sprite, find the first path step that is 4-neighbor-adjacent to any sprite pixel.
    - Keep only sprites that are ever approached, crop each kept sprite to its bbox with non-sprite cells turned black, sort by first approach time, and pack the crops left-to-right with top alignment.

    Key insight:
    - The path is never drawn in the output; it only acts as an invisible filter and ordering key for which sprites survive and how they are arranged.
    """
    g = np.asarray(grid, dtype=int)
    path = trace_path(g)
    sprites = find_sprite_components(g)

    chosen = []
    for sprite in sprites:
        first = _first_approach_index(path, sprite["coords"])
        if first is None:
            continue
        chosen.append((first, _sprite_crop(sprite)))

    if not chosen:
        return np.zeros((1, 1), dtype=int)

    chosen.sort(key=lambda item: item[0])
    height = max(crop.shape[0] for _, crop in chosen)
    width = sum(crop.shape[1] for _, crop in chosen)
    out = np.zeros((height, width), dtype=int)

    c0 = 0
    for _, crop in chosen:
        h, w = crop.shape
        out[:h, c0 : c0 + w] = crop
        c0 += w
    return out


def _rand_odd(low: int, high: int) -> int:
    values = [v for v in range(low, high + 1) if v % 2 == 1]
    return random.choice(values)


def _choose_rows(H: int, mode: str) -> tuple[int, int, int]:
    odds = list(range(1, H - 1, 2))
    for _ in range(200):
        a, b, c = sorted(random.sample(odds, 3))
        if b - a < 4 or c - b < 4:
            continue
        return (a, b, c) if mode == "down" else (c, b, a)
    raise RuntimeError("failed to choose rows")


def _choose_turn_cols(W: int) -> tuple[int, int]:
    odds = [c for c in range(3, W - 2, 2)]
    for _ in range(200):
        c2, c1 = sorted(random.sample(odds, 2))
        if c1 - c2 < 4:
            continue
        if c1 > W - 4 or c2 < 3:
            continue
        if c2 > W // 2:
            continue
        if c1 < W // 2:
            continue
        return c1, c2
    raise RuntimeError("failed to choose turn columns")


def _build_segments(
    H: int, W: int, mode: str
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    r1, r2, r3 = _choose_rows(H, mode)
    c1, c2 = _choose_turn_cols(W)

    if mode == "down":
        seg2_rows = range(r1 + 1, r2 + 1)
        seg4_rows = range(r2 + 1, r3 + 1)
    else:
        seg2_rows = range(r1 - 1, r2 - 1, -1)
        seg4_rows = range(r2 - 1, r3 - 1, -1)

    segments = [
        [(r1, c) for c in range(0, c1 + 1)],
        [(r, c1) for r in seg2_rows],
        [(r2, c) for c in range(c1 - 1, c2 - 1, -1)],
        [(r, c2) for r in seg4_rows],
        [(r3, c) for c in range(c2 + 1, W)],
    ]
    path = [coord for segment in segments for coord in segment]
    return segments, path


def _paint_wall(
    grid: np.ndarray,
    path_set: set[tuple[int, int]],
    r: int,
    c: int,
    color: int,
) -> bool:
    if not in_bounds(grid.shape, r, c) or (r, c) in path_set:
        return True
    cur = int(grid[r, c])
    if cur != Color.ORANGE and cur != color:
        return False
    grid[r, c] = color
    return True


def _render_corridors(
    grid: np.ndarray, segments: list[list[tuple[int, int]]], colors: list[int]
) -> bool:
    path_set = {coord for segment in segments for coord in segment}
    for idx, (segment, color) in enumerate(zip(segments, colors)):
        if len(segment) < 2:
            return False
        step_r = segment[1][0] - segment[0][0]
        step_c = segment[1][1] - segment[0][1]
        if step_r == 0:
            side_deltas = [(-1, 0), (1, 0)]
        else:
            side_deltas = [(0, -1), (0, 1)]

        suppressed = set()
        if idx + 1 < len(segments):
            turn_r = segments[idx + 1][0][0] - segment[-1][0]
            turn_c = segments[idx + 1][0][1] - segment[-1][1]
            prev_r, prev_c = segment[-2]
            suppressed.add((prev_r + turn_r, prev_c + turn_c))

        for r, c in segment:
            for dr, dc in side_deltas:
                wall = (r + dr, c + dc)
                if wall in suppressed:
                    continue
                if not _paint_wall(grid, path_set, wall[0], wall[1], color):
                    return False

        if idx + 1 < len(segments):
            r, c = segment[-1]
            if not _paint_wall(grid, path_set, r + step_r, c + step_c, color):
                return False
    return True


def _intervals_for_visited_sprites(
    segments: list[list[tuple[int, int]]], path_len: int
) -> dict[str, tuple[int, int]]:
    lens = [len(seg) for seg in segments]
    if min(lens[1], lens[3]) < 4 or min(lens[0], lens[2], lens[4]) < 5:
        raise RuntimeError("segments too short for sprite placement")

    p1 = lens[0]
    p2 = p1 + lens[1]
    p3 = p2 + lens[2]
    p4 = p3 + lens[3]

    a0 = max(2, lens[0] // 3)
    c0 = p2 + max(1, lens[2] // 2 - 1)
    e0 = p4 + max(1, lens[4] // 2 - 1)
    return {
        "A": (a0, 3),
        "B": (p1 - 2, 4),
        "C": (c0, 3),
        "D": (p3 - 2, 4),
        "E": (e0, 3),
    }


def _apply_path_sprite(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start: int,
    length: int,
    color: int,
) -> set[tuple[int, int]]:
    coords = set(path[start : start + length])
    for r, c in coords:
        grid[r, c] = color
    return coords


def _normalize_shape(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return {(r - min_r, c - min_c) for r, c in shape}


def _rotate_shape(shape: set[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    cur = set(shape)
    for _ in range(k % 4):
        cur = {(-c, r) for r, c in cur}
        cur = _normalize_shape(cur)
    return _normalize_shape(cur)


def _shape_dims(shape: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_remote_sprite(
    grid: np.ndarray,
    path_forbidden: set[tuple[int, int]],
    sprite_cells: set[tuple[int, int]],
) -> bool:
    shape = _rotate_shape(random.choice(REMOTE_SHAPES), random.randrange(4))
    color = random.choice(SPRITE_COLORS)
    H, W = _shape_dims(shape)
    positions = [
        (r0, c0)
        for r0 in range(grid.shape[0] - H + 1)
        for c0 in range(grid.shape[1] - W + 1)
    ]
    random.shuffle(positions)

    for r0, c0 in positions:
        coords = {(r0 + r, c0 + c) for r, c in shape}
        if any(int(grid[r, c]) != Color.ORANGE for r, c in coords):
            continue
        if any((r, c) in path_forbidden for r, c in coords):
            continue
        bad = False
        for r, c in coords:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in sprite_cells:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        for r, c in coords:
            grid[r, c] = color
        sprite_cells |= coords
        return True
    return False


def _generator_ok(
    grid: np.ndarray,
    expected_path: list[tuple[int, int]],
) -> bool:
    if max(grid.shape) > 30:
        return False
    traced = trace_path(grid)
    if traced != expected_path:
        return False

    sprites = find_sprite_components(grid)
    if not (4 <= len(sprites) <= 6):
        return False

    visited = []
    for sprite in sprites:
        first = _first_approach_index(traced, sprite["coords"])
        if first is not None:
            visited.append((first, sprite["bbox"]))

    if not (3 <= len(visited) < len(sprites)):
        return False

    visited.sort(key=lambda item: item[0])
    left_order = [bbox[1] for _, bbox in visited]
    if left_order == sorted(left_order):
        return False

    out = main(grid)
    return out.ndim == 2 and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    H = _rand_odd(17, 25)
    W = _rand_odd(19, 27)
    mode = random.choice(["down", "up"])

    try:
        segments, path = _build_segments(H, W, mode)
        intervals = _intervals_for_visited_sprites(segments, len(path))
    except RuntimeError:
        return None

    grid = np.full((H, W), Color.ORANGE, dtype=int)
    grid[path[0]] = Color.PINK

    wall_colors = random.sample(BAR_COLORS, k=len(segments))
    if not _render_corridors(grid, segments, wall_colors):
        return None

    sprite_cells: set[tuple[int, int]] = set()
    visited_keys = ["B", "C"] + random.sample(["A", "D", "E"], k=random.choice([1, 2]))
    for key in visited_keys:
        start, length = intervals[key]
        cells = _apply_path_sprite(grid, path, start, length, random.choice(SPRITE_COLORS))
        if sprite_cells & cells:
            return None
        sprite_cells |= cells

    path_forbidden = set(path)
    for r, c in path:
        path_forbidden.update(neighbors4(r, c))

    target_sprite_count = random.randint(4, 6)
    while len(find_sprite_components(grid)) < target_sprite_count:
        if not _place_remote_sprite(grid, path_forbidden, sprite_cells):
            return None

    if not _generator_ok(grid, path):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
