from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBSTACLE = 7
TRACK = 8
ENDPOINT = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPENDICULAR = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

GEN_HEIGHT_RANGE = (18, 26)
GEN_WIDTH_RANGE = (18, 26)
GEN_SEGMENT_LENGTHS = range(3, 7)
GEN_MAX_ATTEMPTS = 700


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while queue:
            cell = queue.popleft()
            coords.append(cell)
            for nr, nc in _neighbors4(cell):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        out.append(coords)
    return out


def _find_sprite(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates: list[tuple[int, int, np.ndarray]] = []
    for comp in _components(grid != BG):
        if len(comp) != 9:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if max(rows) - min(rows) != 2 or max(cols) - min(cols) != 2:
            continue
        r0 = min(rows)
        c0 = min(cols)
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        if patch.shape != (3, 3):
            continue
        if np.all(patch != BG) and len(set(map(int, patch.flat))) == 2:
            candidates.append((r0, c0, patch.copy()))
    assert len(candidates) == 1, "expected one filled 3x3 sprite"
    return candidates[0]


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _enumerate_paths(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    limit: int,
) -> list[list[tuple[int, int]]]:
    cells = set(map(tuple, np.argwhere(mask)))
    found: list[list[tuple[int, int]]] = []
    path = [start]
    seen = {start}

    def dfs(cell: tuple[int, int]) -> None:
        if len(found) >= limit:
            return
        if cell == end:
            found.append(path.copy())
            return
        for nxt in _neighbors4(cell):
            if nxt not in cells or nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            seen.remove(nxt)

    dfs(start)
    return found


def _count_corners(path: list[tuple[int, int]]) -> int:
    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def _rotate_clockwise(sprite: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(sprite, k=(-turns) % 4)


def _extract_turns_and_sprite(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _sprite_r0, _sprite_c0, sprite = _find_sprite(g)
    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(g == ENDPOINT)]
    assert len(endpoints) == 2, "expected exactly two maroon endpoints"

    track_mask = (g == TRACK) | (g == ENDPOINT)
    forbidden = np.zeros(g.shape, dtype=bool)
    for r, c in np.argwhere(g == OBSTACLE):
        rr = int(r)
        cc = int(c)
        forbidden[rr, cc] = True
        for nr, nc in _neighbors8((rr, cc)):
            if _in_bounds(g.shape, nr, nc):
                forbidden[nr, nc] = True

    assert all(not forbidden[r, c] for r, c in endpoints), "endpoint may not be blocked"
    safe_mask = track_mask & (~forbidden)
    paths = _enumerate_paths(safe_mask, endpoints[0], endpoints[1], limit=2)
    assert len(paths) == 1, "expected a unique safe path"
    return _count_corners(paths[0]), sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the remote 3x3 sprite by the quarter-turn count encoded by the safe track path.

    Input structure:
    - A black background contains one teal 4-connected track network with exactly two maroon endpoint pixels.
    - One or more orange obstacle pixels sit near parts of that track, and a separate filled 3x3 two-color sprite sits elsewhere.
    - The sprite is the only isolated 3x3 all-nonblack block in the grid.

    Rule:
    - Treat every orange obstacle and its full 8-neighborhood as forbidden.
    - In the remaining teal-plus-maroon track graph, find the unique 4-connected path between the two maroon endpoints.
    - Count how many times that path changes direction, then rotate the 3x3 sprite clockwise by that count modulo 4.
    - Output only the rotated 3x3 sprite.

    Key insight:
    - The track itself is only an instruction source: obstacle neighborhoods delete the tempting visible detours, and only the surviving path's corner count determines the sprite rotation.
    """
    turns, sprite = _extract_turns_and_sprite(np.asarray(grid, dtype=int))
    return _rotate_clockwise(sprite, turns % 4)


def _build_safe_path(
    height: int,
    width: int,
    corner_count: int,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]] | tuple[None, None]:
    num_segments = corner_count + 1
    margin = 3
    for _ in range(800):
        start = (
            random.randint(margin, height - 1 - margin),
            random.randint(margin, width - 1 - margin),
        )
        occupied = {start}
        path = [start]
        segments: list[dict[str, object]] = []

        def dfs(seg_idx: int, pos: tuple[int, int], prev_dir: tuple[int, int] | None) -> bool:
            if seg_idx == num_segments:
                return True
            directions = DIRS4 if prev_dir is None else PERPENDICULAR[prev_dir]
            directions = random.sample(directions, len(directions))
            lengths = list(GEN_SEGMENT_LENGTHS)
            random.shuffle(lengths)
            for direction in directions:
                for length in lengths:
                    new_cells: list[tuple[int, int]] = []
                    cur = pos
                    ok = True
                    for _ in range(length):
                        cur = (cur[0] + direction[0], cur[1] + direction[1])
                        if not (margin <= cur[0] < height - margin and margin <= cur[1] < width - margin):
                            ok = False
                            break
                        if cur in occupied:
                            ok = False
                            break
                        prev_cell = (cur[0] - direction[0], cur[1] - direction[1])
                        for nbr in _neighbors4(cur):
                            if nbr in occupied and nbr != prev_cell:
                                ok = False
                                break
                        if not ok:
                            break
                        new_cells.append(cur)
                        occupied.add(cur)
                    if not ok:
                        for cell in new_cells:
                            occupied.remove(cell)
                        continue
                    start_idx = len(path) - 1
                    path.extend(new_cells)
                    segments.append(
                        {
                            "start_idx": start_idx,
                            "end_idx": len(path) - 1,
                            "dir": direction,
                            "length": length,
                        }
                    )
                    if dfs(seg_idx + 1, cur, direction):
                        return True
                    segments.pop()
                    del path[-len(new_cells) :]
                    for cell in new_cells:
                        occupied.remove(cell)
            return False

        if dfs(0, start, None):
            return path, segments
    return None, None


def _add_detour(
    path: list[tuple[int, int]],
    segments: list[dict[str, object]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | tuple[None, None]:
    safe = set(path)
    corner_indices = list(range(len(segments) - 1))
    random.shuffle(corner_indices)
    for k in corner_indices:
        seg1 = segments[k]
        seg2 = segments[k + 1]
        length1 = int(seg1["length"])
        length2 = int(seg2["length"])
        if length1 < 3 or length2 < 3:
            continue
        d1 = tuple(seg1["dir"])
        d2 = tuple(seg2["dir"])
        a_values = list(range(2, length1))
        b_values = list(range(2, length2))
        random.shuffle(a_values)
        random.shuffle(b_values)
        for a in a_values:
            start = path[int(seg1["end_idx"]) - a]
            for b in b_values:
                end = path[int(seg2["start_idx"]) + b]
                detour = [start]
                cur = start
                for _ in range(b):
                    cur = (cur[0] + d2[0], cur[1] + d2[1])
                    detour.append(cur)
                for _ in range(a):
                    cur = (cur[0] + d1[0], cur[1] + d1[1])
                    detour.append(cur)
                if cur != end:
                    continue
                interior = detour[1:-1]
                interior_set = set(interior)
                if any((not _in_bounds(shape, r, c)) or (r, c) in safe for r, c in interior):
                    continue
                if any(
                    any(nbr in safe and nbr not in {start, end} for nbr in _neighbors4(cell))
                    for cell in interior
                ):
                    continue

                visible = safe | interior_set
                visible_paths = _enumerate_paths(
                    _mask_from_cells(shape, visible),
                    path[0],
                    path[-1],
                    limit=6,
                )
                mods = {_count_corners(candidate) % 4 for candidate in visible_paths}
                if len(visible_paths) < 2 or len(mods) < 2:
                    continue

                bend = detour[-2]
                obstacle_candidates = [
                    (bend[0] + dr, bend[1] + dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if dr or dc
                ]
                random.shuffle(obstacle_candidates)
                for obstacle in obstacle_candidates:
                    if not _in_bounds(shape, obstacle[0], obstacle[1]):
                        continue
                    if obstacle in visible:
                        continue
                    forbidden = {obstacle, *_neighbors8(obstacle)}
                    if any(cell in safe for cell in forbidden):
                        continue
                    return interior, [obstacle]
    return None, None


def _random_sprite() -> np.ndarray:
    palette = [color for color in range(1, 10) if color not in {OBSTACLE, TRACK, ENDPOINT}]
    for _ in range(200):
        c1, c2 = random.sample(palette, 2)
        bits = np.random.randint(0, 2, size=(3, 3), dtype=int)
        if bits.min() == bits.max():
            continue
        sprite = np.where(bits == 0, c1, c2)
        if len({_rotate_clockwise(sprite, k).tobytes() for k in range(4)}) == 4:
            return sprite
    raise RuntimeError("failed to sample a non-symmetric sprite")


def _place_sprite(grid: np.ndarray, sprite: np.ndarray) -> np.ndarray | None:
    occupied = set(map(tuple, np.argwhere(grid != BG)))
    blocked = set(occupied)
    for cell in list(occupied):
        for nbr in _neighbors8(cell):
            if _in_bounds(grid.shape, nbr[0], nbr[1]):
                blocked.add(nbr)
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid.shape[0] - 2):
        for c0 in range(grid.shape[1] - 2):
            cells = [(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)]
            if any(cell in blocked for cell in cells):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    out = grid.copy()
    out[r0 : r0 + 3, c0 : c0 + 3] = sprite
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        rotation = random.randint(0, 3)
        corner_count = 4 if rotation == 0 else rotation + 4 * random.randint(0, 1)

        path, segments = _build_safe_path(height, width, corner_count)
        if path is None or segments is None:
            continue

        detour_cells, obstacles = _add_detour(path, segments, (height, width))
        if detour_cells is None or obstacles is None:
            continue

        sprite = _random_sprite()
        expected = _rotate_clockwise(sprite, corner_count % 4)

        grid = np.zeros((height, width), dtype=int)
        for r, c in set(path) | set(detour_cells):
            grid[r, c] = TRACK
        for r, c in obstacles:
            grid[r, c] = OBSTACLE

        sr, sc = path[0]
        er, ec = path[-1]
        grid[sr, sc] = ENDPOINT
        grid[er, ec] = ENDPOINT

        placed = _place_sprite(grid, sprite)
        if placed is None:
            continue
        if not np.array_equal(main(placed), expected):
            continue
        return placed

    raise RuntimeError("failed to generate a valid input")
