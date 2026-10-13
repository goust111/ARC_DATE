from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
PINK = 6
ORANGE = 7

DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
PERPENDICULAR = {
    "right": ("up", "down"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "up": ("left", "right"),
}
ROT_K = {
    "right": 0,
    "down": 3,  # clockwise 90
    "left": 2,
    "up": 1,  # clockwise 270
}

MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return ps.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _largest_black_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    nonblack = grid != BLACK
    ps = _prefix_sum(nonblack)
    h, w = grid.shape
    best: tuple[int, int, int, int] | None = None
    best_area = -1
    ties = 0

    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0 + 2, h):
                for c1 in range(c0 + 2, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if _rect_sum(ps, r0, c0, r1, c1) != 0:
                        continue
                    if area > best_area:
                        best = (r0, c0, r1, c1)
                        best_area = area
                        ties = 1
                    else:
                        ties += 1

    assert best is not None, "expected at least one 3x3 all-black rectangle"
    assert ties == 1, "expected a unique maximum black rectangle"
    return best


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    h, w = grid.shape
    if not _in_bounds((h, w), r, c):
        return False
    if int(grid[r, c]) not in {BLACK, PINK}:
        return False

    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {BLACK, PINK}


def _trace_particle(grid: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    starts = np.argwhere(grid == PINK)
    assert starts.shape == (1, 2), "expected exactly one magenta particle"
    r, c = map(int, starts[0])
    direction = "right"
    path = [(r, c)]

    for _ in range(grid.size * 4):
        dr, dc = DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break
        if int(grid[nr, nc]) in {BLACK, PINK}:
            r, c = nr, nc
            path.append((r, c))
            continue

        options: list[tuple[int, int, str]] = []
        for turn_dir in PERPENDICULAR[direction]:
            tdr, tdc = DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))

        assert len(options) <= 1, "ambiguous turn detected"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))

    return path, direction


def _path_cells(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i, (a, b) in enumerate(zip(vertices, vertices[1:])):
        r0, c0 = a
        r1, c1 = b
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            cells = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 > r0 else -1
            cells = [(r, c0) for r in range(r0, r1 + step, step)]
        if i:
            cells = cells[1:]
        out.extend(cells)
    return out


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    if a[0] == b[0]:
        return "right" if b[1] > a[1] else "left"
    return "down" if b[0] > a[0] else "up"


def _paint_path(grid: np.ndarray, vertices: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    dirs = [_direction(a, b) for a, b in zip(vertices, vertices[1:])]

    for i, (a, b) in enumerate(zip(vertices, vertices[1:])):
        prev_dir = None if i == 0 else dirs[i - 1]
        next_dir = None if i == len(dirs) - 1 else dirs[i + 1]
        cur_dir = dirs[i]

        if cur_dir in {"right", "left"}:
            r = a[0]
            c0, c1 = a[1], b[1]
            cols = range(min(c0, c1), max(c0, c1) + 1)
            open_top: set[int] = set()
            open_bottom: set[int] = set()
            if prev_dir == "down":
                open_top.add(c0)
            if prev_dir == "up":
                open_bottom.add(c0)
            if next_dir == "down":
                open_bottom.add(c1)
            if next_dir == "up":
                open_top.add(c1)
            for c in cols:
                if r - 1 >= 0 and c not in open_top:
                    grid[r - 1, c] = ORANGE
                if r + 1 < h and c not in open_bottom:
                    grid[r + 1, c] = ORANGE
            step = 1 if cur_dir == "right" else -1
            if next_dir in {"up", "down"}:
                ahead_c = c1 + step
                if 0 <= ahead_c < w:
                    grid[r, ahead_c] = ORANGE
        else:
            c = a[1]
            r0, r1 = a[0], b[0]
            rows = range(min(r0, r1), max(r0, r1) + 1)
            open_left: set[int] = set()
            open_right: set[int] = set()
            if prev_dir == "right":
                open_left.add(r0)
            if prev_dir == "left":
                open_right.add(r0)
            if next_dir == "right":
                open_right.add(r1)
            if next_dir == "left":
                open_left.add(r1)
            for r in rows:
                if c - 1 >= 0 and r not in open_left:
                    grid[r, c - 1] = ORANGE
                if c + 1 < w and r not in open_right:
                    grid[r, c + 1] = ORANGE
            step = 1 if cur_dir == "down" else -1
            if next_dir in {"left", "right"}:
                ahead_r = r1 + step
                if 0 <= ahead_r < h:
                    grid[ahead_r, c] = ORANGE

    cells = _path_cells(vertices)
    for r, c in cells:
        grid[r, c] = BLACK
    sr, sc = vertices[0]
    grid[sr, sc] = PINK


def _draw_room(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    grid[r0 : r0 + height, c0 : c0 + width] = BLACK
    grid[r0 - 1 : r0 + height + 1, c0 - 1] = ORANGE
    grid[r0 - 1 : r0 + height + 1, c0 + width] = ORANGE
    grid[r0 - 1, c0 - 1 : c0 + width + 1] = ORANGE
    grid[r0 + height, c0 - 1 : c0 + width + 1] = ORANGE


def _random_tile() -> np.ndarray:
    for _ in range(200):
        patch = np.zeros((3, 3), dtype=int)
        count = random.randint(4, 6)
        picks = random.sample(range(9), count)
        for idx in picks:
            patch[idx // 3, idx % 3] = GREEN
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    raise RuntimeError("failed to generate asymmetric tile")


def _draw_stripes(grid: np.ndarray) -> None:
    h, w = grid.shape
    for r in range(0, h, 4):
        grid[r, :] = ORANGE
    for c in range(0, w, 4):
        grid[:, c] = ORANGE


def _path_bbox(vertices: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = _path_cells(vertices)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _odd_range(start: int, stop: int) -> list[int]:
    return [x for x in range(start, stop + 1) if x % 2 == 1]


def _choose_vertices(h: int, w: int) -> tuple[list[tuple[int, int]], str]:
    final_dir = random.choice(["right", "down", "left", "up"])

    if final_dir == "right":
        start_r = random.choice(_odd_range(1, min(h - 2, 5)))
        vertices = [(start_r, 0), (start_r, w - 1)]
    elif final_dir == "down":
        start_r = random.choice(_odd_range(1, max(1, h // 3)))
        turn_c = random.choice(_odd_range(3, min(w - 4, 7)))
        vertices = [(start_r, 0), (start_r, turn_c), (h - 1, turn_c)]
    elif final_dir == "up":
        start_r = random.choice(_odd_range(3, min(h - 2, 9)))
        turn_c = random.choice(_odd_range(3, min(w - 4, 7)))
        vertices = [(start_r, 0), (start_r, turn_c), (0, turn_c)]
    else:
        start_r = random.choice(_odd_range(1, min(h - 4, 5)))
        turn_c = random.choice(_odd_range(3, min(w - 6, 7)))
        candidates = [r for r in _odd_range(start_r + 2, min(h - 2, 11))] + [r for r in _odd_range(1, start_r - 2)]
        turn_r = random.choice(candidates)
        vertices = [(start_r, 0), (start_r, turn_c), (turn_r, turn_c), (turn_r, 0)]

    cells = _path_cells(vertices)
    assert len(cells) == len(set(cells)), "path must not self-intersect"
    return vertices, final_dir


def _room_candidates(
    h: int,
    w: int,
    sizes: Iterable[tuple[int, int]],
    forbidden: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    tile_box = (h - 3, w - 3, h - 1, w - 1)

    for rh, rw in sizes:
        for r0 in _odd_range(1, h - rh - 1):
            for c0 in _odd_range(1, w - rw - 1):
                interior = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                border = (r0 - 1, c0 - 1, r0 + rh, c0 + rw)
                if _rects_intersect(interior, tile_box, margin=0):
                    continue
                if any(_rects_intersect(border, box, margin=1) for box in forbidden):
                    continue
                out.append((r0, c0, rh, rw))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the particle, use its final direction to rotate the tile, and tile the unique largest empty rectangle.

    Input structure:
    - The grid uses black background, orange bar segments, one magenta particle on the left edge, and one green 3x3 tile pattern in the bottom-right corner.
    - Orange bars lie on even rows and even columns, forming narrow corridors and enclosing several all-black rectangles.
    - Exactly one all-black rectangle of height and width at least 3 is the unique maximum-area empty region, and its side lengths are multiples of 3.

    Rule:
    - Starting from the magenta particle, move straight through black cells until blocked; when blocked by orange, turn 90 degrees into the unique perpendicular black corridor whose sides are flanked by same-color bars.
    - Recolor every traversed black cell magenta and record the final travel direction at the stopping point.
    - Independently search the original input for the unique largest all-black rectangle, rotate the bottom-right 3x3 tile according to the final direction, and tile that rectangle in row-major 3x3 blocks, giving every odd-indexed block an extra 180-degree rotation.
    - Finally erase the original bottom-right tile back to black.

    Key insight:
    - Two independent cues must be combined: path tracing determines tile orientation, while empty-rectangle search determines where the tiled fill goes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 7 and g.shape[1] >= 7, "grid too small for this task"

    original = g.copy()
    path, final_dir = _trace_particle(original)
    rect = _largest_black_rectangle(original)

    out = original.copy()
    for r, c in path[1:]:
        if int(out[r, c]) == BLACK:
            out[r, c] = PINK

    tile = original[-3:, -3:].copy()
    out[-3:, -3:] = BLACK
    base = np.rot90(tile, ROT_K[final_dir]).copy()

    r0, c0, r1, c1 = rect
    idx = 0
    for rr in range(r0, r1 + 1, 3):
        for cc in range(c0, c1 + 1, 3):
            block = base if idx % 2 == 0 else np.rot90(base, 2)
            out[rr : rr + 3, cc : cc + 3] = block
            idx += 1

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.choice([19, 21, 23, 25])
        w = random.choice([19, 21, 23, 25])
        grid = np.zeros((h, w), dtype=int)
        _draw_stripes(grid)

        vertices, planned_dir = _choose_vertices(h, w)
        path_box = _path_bbox(vertices)
        forbidden = [path_box]

        large_sizes = [(3, 9), (9, 3)]
        if h >= 21 and w >= 21:
            large_sizes.append((9, 9))
        target_candidates = _room_candidates(h, w, large_sizes, forbidden)
        if not target_candidates:
            continue
        target = random.choice(target_candidates)
        _draw_room(grid, *target)
        target_box = (target[0], target[1], target[0] + target[2] - 1, target[1] + target[3] - 1)
        forbidden.append(target_box)

        smaller_sizes = [(3, 3)]
        if target[2] * target[3] > 27:
            smaller_sizes.extend([(3, 9), (9, 3)])
        small_candidates = _room_candidates(h, w, smaller_sizes, forbidden)
        random.shuffle(small_candidates)
        for room in small_candidates[: random.randint(1, 2)]:
            _draw_room(grid, *room)
            forbidden.append((room[0], room[1], room[0] + room[2] - 1, room[1] + room[3] - 1))

        _paint_path(grid, vertices)
        tile = _random_tile()
        grid[-3:, -3:] = tile

        try:
            traced_path, actual_dir = _trace_particle(grid)
            rect = _largest_black_rectangle(grid)
            out = main(grid)
        except AssertionError:
            continue

        if actual_dir != planned_dir:
            continue
        if rect != (target[0], target[1], target[0] + target[2] - 1, target[1] + target[3] - 1):
            continue
        if len(traced_path) < 6:
            continue
        if np.array_equal(grid, out):
            continue
        if np.count_nonzero(grid[-3:, -3:] == GREEN) < 4:
            continue
        if np.array_equal(np.rot90(tile, 2), tile):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
