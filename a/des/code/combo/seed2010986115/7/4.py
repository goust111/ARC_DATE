import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
WINDOW = 4
LEGEND = 5
PATTERN_COLORS = (2, 3, 6, 7, 8, 9)

MAX_GRID_H = 30
MAX_GRID_W = 30
GEN_ATTEMPTS = 400


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _tile(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps_h = (height + ph - 1) // ph
    reps_w = (width + pw - 1) // pw
    tiled = np.tile(pattern, (reps_h, reps_w))
    return tiled[:height, :width]


def _legend_patterns(grid: np.ndarray) -> list[np.ndarray]:
    patterns: list[tuple[int, int, np.ndarray]] = []
    for comp in _components_mask(grid == LEGEND):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(sub.shape)
        if not np.all(sub[border] == LEGEND):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or np.any(inner == LEGEND):
            continue
        patterns.append((c0, r0, inner.copy()))
    patterns.sort(key=lambda item: (item[0], item[1]))
    return [pattern for _, _, pattern in patterns]


def _rooms(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    rooms: list[tuple[int, int, int, int, int]] = []
    for comp in _components_mask(grid == BG):
        if _touches_edge(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        sub = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
        border = _border_mask(sub.shape)
        if not np.all(sub[1:-1, 1:-1] == BG):
            continue
        if not np.all(np.isin(sub[border], (WALL, WINDOW))):
            continue
        window_count = int(np.count_nonzero(sub[border] == WINDOW))
        rooms.append((r0, c0, r1, c1, window_count))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the legend pattern indexed by its window count.
    """
    g = np.asarray(grid, dtype=int)
    patterns = _legend_patterns(g)
    out = g.copy()

    for r0, c0, r1, c1, count in _rooms(g):
        if not (1 <= count <= len(patterns)):
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = _tile(patterns[count - 1], r1 - r0 + 1, c1 - c0 + 1)

    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    prev = 0
    result = []
    for cut in cuts + [total]:
        result.append(cut - prev)
        prev = cut
    return result


def _make_pattern(size: int, rng: random.Random) -> np.ndarray:
    colors = rng.sample(PATTERN_COLORS, rng.randint(2, min(4, len(PATTERN_COLORS))))
    mode = rng.choice(["checker", "vstripe", "hstripe", "diag", "quad", "offset"])
    arr = np.zeros((size, size), dtype=int)

    if mode == "checker":
        for r in range(size):
            for c in range(size):
                arr[r, c] = colors[(r + c) % len(colors)]
    elif mode == "vstripe":
        for c in range(size):
            arr[:, c] = colors[c % len(colors)]
    elif mode == "hstripe":
        for r in range(size):
            arr[r, :] = colors[r % len(colors)]
    elif mode == "diag":
        for r in range(size):
            for c in range(size):
                arr[r, c] = colors[(r + 2 * c) % len(colors)]
    elif mode == "quad":
        split_r = rng.randint(1, size - 1)
        split_c = rng.randint(1, size - 1)
        arr[:split_r, :split_c] = colors[0]
        arr[:split_r, split_c:] = colors[1 % len(colors)]
        arr[split_r:, :split_c] = colors[2 % len(colors)]
        arr[split_r:, split_c:] = colors[3 % len(colors)]
    else:
        offset = rng.randrange(len(colors))
        for r in range(size):
            for c in range(size):
                arr[r, c] = colors[(offset + 2 * r + c) % len(colors)]

    if len(np.unique(arr)) < 2:
        arr[0, 0] = colors[0]
        arr[-1, -1] = colors[1]
    return arr


def _sample_patterns(count: int, rng: random.Random) -> list[np.ndarray]:
    size = rng.randint(2, 4)
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(200):
        if len(patterns) == count:
            break
        pat = _make_pattern(size, rng)
        key = pat.tobytes()
        if key in seen:
            continue
        seen.add(key)
        patterns.append(pat)
    return patterns


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    window_count: int,
    rng: random.Random,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left : left + outer_w] = BG
    grid[top, left : left + outer_w] = WALL
    grid[top + outer_h - 1, left : left + outer_w] = WALL
    grid[top : top + outer_h, left] = WALL
    grid[top : top + outer_h, left + outer_w - 1] = WALL

    border_cells: list[tuple[int, int]] = []
    for c in range(left + 1, left + outer_w - 1):
        border_cells.append((top, c))
        border_cells.append((top + outer_h - 1, c))
    for r in range(top + 1, top + outer_h - 1):
        border_cells.append((r, left))
        border_cells.append((r, left + outer_w - 1))

    for r, c in rng.sample(border_cells, window_count):
        grid[r, c] = WINDOW


def _draw_legend(grid: np.ndarray, top: int, left: int, patterns: list[np.ndarray]) -> None:
    block_size = patterns[0].shape[0] + 2
    gap = 1
    for idx, pattern in enumerate(patterns):
        r0 = top
        c0 = left + idx * (block_size + gap)
        r1 = r0 + block_size
        c1 = c0 + block_size
        grid[r0:r1, c0:c1] = BG
        grid[r0, c0:c1] = LEGEND
        grid[r1 - 1, c0:c1] = LEGEND
        grid[r0:r1, c0] = LEGEND
        grid[r0:r1, c1 - 1] = LEGEND
        grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = pattern


def _room_layout(
    room_specs: list[tuple[int, int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int, int]], int, int]:
    room_count = len(room_specs)
    row_count = rng.randint(1, min(3, room_count))
    row_sizes = _random_composition(room_count, row_count, rng)

    specs = room_specs[:]
    rng.shuffle(specs)

    placements: list[tuple[int, int, int, int, int]] = []
    idx = 0
    y = rng.randint(1, 2)
    max_right = 0
    for row_size in row_sizes:
        row = specs[idx : idx + row_size]
        idx += row_size
        row_height = max(inner_h + 2 for inner_h, _inner_w, _count in row)
        x = rng.randint(1, 3)
        row_right = 0
        for inner_h, inner_w, count in row:
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            top = y + rng.randint(0, row_height - outer_h)
            placements.append((top, x, inner_h, inner_w, count))
            row_right = x + outer_w
            x = row_right + rng.randint(2, 4)
        max_right = max(max_right, row_right)
        y += row_height + rng.randint(2, 4)
    room_h = y + rng.randint(1, 2)
    room_w = max_right + rng.randint(1, 3)
    return placements, room_h, room_w


def _legend_span(patterns: list[np.ndarray]) -> tuple[int, int]:
    block_size = patterns[0].shape[0] + 2
    width = len(patterns) * block_size + (len(patterns) - 1)
    return block_size, width


def _ensure_count_variety(counts: list[int], max_count: int, rng: random.Random) -> list[int]:
    if all(count == 0 for count in counts):
        counts[rng.randrange(len(counts))] = rng.randint(1, max_count)
    if len(set(counts)) == 1:
        idx = rng.randrange(len(counts))
        counts[idx] = (counts[idx] + 1) % (max_count + 1)
        if counts[idx] == 0 and all(count == 0 for count in counts):
            counts[idx] = 1
    if rng.random() < 0.7 and 0 not in counts:
        counts[rng.randrange(len(counts))] = 0
    if rng.random() < 0.7 and max_count not in counts:
        counts[rng.randrange(len(counts))] = max_count
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pattern_count = rng.randint(2, 4)
    patterns = _sample_patterns(pattern_count, rng)
    if len(patterns) != pattern_count:
        return None

    room_count = rng.randint(2, 5)
    counts = [rng.randint(0, pattern_count) for _ in range(room_count)]
    counts = _ensure_count_variety(counts, pattern_count, rng)

    room_specs = []
    for count in counts:
        inner_h = rng.randint(2, 5)
        inner_w = rng.randint(2, 5)
        room_specs.append((inner_h, inner_w, count))

    placements, room_h, room_w = _room_layout(room_specs, rng)
    legend_h, legend_w = _legend_span(patterns)
    side = rng.choice(["bottom", "right"])
    gap = rng.randint(1, 2)

    if side == "bottom":
        final_h = room_h + gap + legend_h + 1
        final_w = max(room_w, legend_w + 2)
        if final_h > MAX_GRID_H or final_w > MAX_GRID_W:
            return None
        grid = np.full((final_h, final_w), BG, dtype=int)
        room_x = rng.randint(0, final_w - room_w)
        legend_x = rng.randint(1, final_w - legend_w - 1)
        legend_y = final_h - legend_h - 1
        room_y = 0
    else:
        final_h = max(room_h, legend_h + 2)
        final_w = room_w + gap + legend_w + 1
        if final_h > MAX_GRID_H or final_w > MAX_GRID_W:
            return None
        grid = np.full((final_h, final_w), BG, dtype=int)
        room_y = rng.randint(0, final_h - room_h)
        legend_y = rng.randint(1, final_h - legend_h - 1)
        room_x = 0
        legend_x = final_w - legend_w - 1

    for top, left, inner_h, inner_w, count in placements:
        _draw_room(grid, room_y + top, room_x + left, inner_h, inner_w, count, rng)

    _draw_legend(grid, legend_y, legend_x, patterns)

    expected = grid.copy()
    for top, left, inner_h, inner_w, count in placements:
        if count > 0:
            tiled = _tile(patterns[count - 1], inner_h, inner_w)
            expected[
                room_y + top + 1 : room_y + top + 1 + inner_h,
                room_x + left + 1 : room_x + left + 1 + inner_w,
            ] = tiled

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
