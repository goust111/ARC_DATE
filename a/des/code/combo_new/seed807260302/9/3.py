import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
VALVE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for dr, dc in DIRS4)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _adjacent_valve_colors(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[int]:
    h, w = grid.shape
    valves: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                color = int(grid[nr, nc])
                if color not in (BG, PATH):
                    valves[(nr, nc)] = color
    return list(valves.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey path segment from the valve colors at its exposed ends.

    Input structure:
    - Background is black.
    - One or more thin orthogonal grey paths appear on the grid.
    - Some path cells are replaced by single-pixel colored valves; every valve is non-black and non-grey.
    - Removing those valves splits the remaining grey pixels into separate 4-connected path segments.

    Rule:
    - Keep every valve pixel unchanged.
    - For each grey segment, collect the distinct valve pixels that touch it by 4-neighborhood.
    - If the touching valves all have the same color and at least one valve exists, recolor the whole segment to that color.
    - Otherwise leave the segment grey.

    Key insight:
    - The colored pixels are separators rather than decorations: once the grey network is cut at the valve positions, each resulting segment is colored solely from the set of valve colors touching that segment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g == PATH):
        valve_colors = _adjacent_valve_colors(g, comp)
        if valve_colors and len(set(valve_colors)) == 1:
            color = int(valve_colors[0])
            for r, c in comp:
                out[r, c] = color

    return out


def _valve_indices(segment_lengths: list[int]) -> list[int]:
    indices: list[int] = []
    pos = int(segment_lengths[0])
    for seg_len in segment_lengths[1:]:
        indices.append(pos)
        pos += 1 + int(seg_len)
    return indices


def _sample_covering_spec() -> dict[str, list[int]]:
    a, b = random.sample(VALVE_COLORS, 2)
    segment_lengths = [random.randint(2, 5) for _ in range(4)]
    if random.random() < 0.5:
        valve_colors = [a, a, b]
    else:
        valve_colors = [a, b, b]
    return {"segment_lengths": segment_lengths, "valve_colors": valve_colors}


def _sample_extra_spec() -> dict[str, list[int]]:
    valve_count = random.randint(1, 4)
    segment_lengths = [random.randint(1, 4) for _ in range(valve_count + 1)]
    if valve_count == 1:
        valve_colors = [random.choice(VALVE_COLORS)]
    else:
        base_colors = random.sample(VALVE_COLORS, random.randint(2, min(4, len(VALVE_COLORS))))
        valve_colors = [random.choice(base_colors) for _ in range(valve_count)]
    return {"segment_lengths": segment_lengths, "valve_colors": valve_colors}


def _can_start(occupied: np.ndarray, r: int, c: int) -> bool:
    h, w = occupied.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if occupied[r, c]:
        return False
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
            return False
    return True


def _can_step(
    occupied: np.ndarray,
    used: set[tuple[int, int]],
    prev: tuple[int, int],
    nxt: tuple[int, int],
) -> bool:
    h, w = occupied.shape
    r, c = nxt
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if occupied[r, c] or nxt in used:
        return False

    for nr, nc in _neighbors4(r, c):
        if (nr, nc) == prev:
            continue
        if 0 <= nr < h and 0 <= nc < w and (occupied[nr, nc] or (nr, nc) in used):
            return False

    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
            return False

    return True


def _candidate_moves(
    occupied: np.ndarray,
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    target_len: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    current = path[-1]
    prev_dir = None
    if len(path) >= 2:
        prev = path[-2]
        prev_dir = (current[0] - prev[0], current[1] - prev[1])

    scored: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    for nr, nc in _neighbors4(*current):
        nxt = (nr, nc)
        if not _can_step(occupied, used, current, nxt):
            continue

        used.add(nxt)
        onward = 0
        for rr, cc in _neighbors4(nr, nc):
            if _can_step(occupied, used, nxt, (rr, cc)):
                onward += 1
        used.remove(nxt)

        if len(path) + 1 < target_len and onward == 0:
            continue

        direction = (nr - current[0], nc - current[1])
        turn_bonus = 0.2 if prev_dir is not None and direction != prev_dir else 0.0
        straight_bonus = 0.1 if prev_dir is not None and direction == prev_dir else 0.0
        score = 2.0 * onward + turn_bonus + straight_bonus + random.random()
        scored.append((score, nxt, direction))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(nxt, direction) for _score, nxt, direction in scored]


def _build_path(occupied: np.ndarray, target_len: int) -> list[tuple[int, int]] | None:
    h, w = occupied.shape
    starts = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _can_start(occupied, r, c)
    ]
    random.shuffle(starts)
    starts = starts[: min(len(starts), PATH_BUILD_ATTEMPTS)]

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == target_len:
            return path.copy()
        for nxt, _direction in _candidate_moves(occupied, path, used, target_len):
            path.append(nxt)
            used.add(nxt)
            result = dfs(path, used)
            if result is not None:
                return result
            used.remove(nxt)
            path.pop()
        return None

    for start in starts:
        path = [start]
        used = {start}
        result = dfs(path, used)
        if result is not None:
            return result
    return None


def _paint_path(grid: np.ndarray, cells: list[tuple[int, int]], spec: dict[str, list[int]]) -> None:
    for r, c in cells:
        grid[r, c] = PATH
    for idx, color in zip(_valve_indices(spec["segment_lengths"]), spec["valve_colors"], strict=True):
        r, c = cells[int(idx)]
        grid[r, c] = int(color)


def _segment_kinds(grid: np.ndarray) -> set[str]:
    kinds: set[str] = set()
    for comp in _components(grid == PATH):
        valve_colors = _adjacent_valve_colors(grid, comp)
        unique = set(valve_colors)
        if not valve_colors:
            continue
        if len(unique) == 1:
            if len(valve_colors) == 1:
                kinds.add("single")
            else:
                kinds.add("same")
        else:
            kinds.add("diff")
    return kinds


def _path_degrees_ok(grid: np.ndarray) -> bool:
    h, w = grid.shape
    mask = grid != BG
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            degree = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    degree += 1
            if degree > 2:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(1, 3)
    specs = [_sample_covering_spec()] + [_sample_extra_spec() for _ in range(path_count - 1)]
    total_cells = sum(sum(spec["segment_lengths"]) + len(spec["valve_colors"]) for spec in specs)

    side_min = max(12, int(np.ceil(np.sqrt(total_cells * 5.0))), 7 + 3 * path_count)
    side_max = min(30, side_min + 8)
    if side_min > side_max:
        return None
    h = random.randint(side_min, side_max)
    w = random.randint(side_min, side_max)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for spec in specs:
        length = int(sum(spec["segment_lengths"]) + len(spec["valve_colors"]))
        cells = _build_path(occupied, length)
        if cells is None:
            return None
        for r, c in cells:
            occupied[r, c] = True
        _paint_path(grid, cells, spec)

    if not _path_degrees_ok(grid):
        return None

    kinds = _segment_kinds(grid)
    if not {"single", "same", "diff"} <= kinds:
        return None

    counts = {int(color): int(np.count_nonzero(grid == color)) for color in np.unique(grid)}
    if counts.get(PATH, 0) <= max(count for color, count in counts.items() if color not in (BG, PATH)):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid valve-path input")
