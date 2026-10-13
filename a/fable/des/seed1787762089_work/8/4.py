import numpy as np
from collections import Counter


BG = 0
MAP_STEP = 2
BLOCK_SIZE = 3
BLOCK_STEP = 4
GRID_SIZES = (3, 4)
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    points: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int]] = []
    for pts in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1
        color = next(iter(colors))
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(pts) == 1:
            points.append((r0, c0, color))
            continue
        if len(pts) == BLOCK_SIZE * BLOCK_SIZE and h == BLOCK_SIZE and w == BLOCK_SIZE:
            if all(grid[r, c] == color for r in range(r0, r0 + BLOCK_SIZE) for c in range(c0, c0 + BLOCK_SIZE)):
                blocks.append((r0, c0, color))
                continue
        raise AssertionError("unexpected foreground object")
    assert points and blocks
    return points, blocks


def _candidate_origins(values: list[int], step: int, size: int, limit: int) -> list[int]:
    candidates: set[int] | None = None
    for value in values:
        local = {
            value - step * idx
            for idx in range(size)
            if 0 <= value - step * idx and value - step * idx + step * (size - 1) < limit
        }
        candidates = local if candidates is None else candidates & local
    return sorted(candidates or [])


def _index_points(
    items: list[tuple[int, int, int]], origin_r: int, origin_c: int, rows: int, cols: int, step: int
) -> dict[tuple[int, int], int] | None:
    indexed: dict[tuple[int, int], int] = {}
    for r, c, color in items:
        dr = r - origin_r
        dc = c - origin_c
        if dr < 0 or dc < 0 or dr % step or dc % step:
            return None
        i = dr // step
        j = dc // step
        if i >= rows or j >= cols:
            return None
        key = (i, j)
        if key in indexed:
            return None
        indexed[key] = color
    return indexed


def _infer_alignment(
    grid: np.ndarray, points: list[tuple[int, int, int]], blocks: list[tuple[int, int, int]]
) -> tuple[int, int, int, int, int, int, dict[tuple[int, int], int], dict[tuple[int, int], int], dict[int, int]]:
    h, w = grid.shape
    point_rows = [r for r, _, _ in points]
    point_cols = [c for _, c, _ in points]
    block_rows = [r for r, _, _ in blocks]
    block_cols = [c for _, c, _ in blocks]
    candidates = []

    for rows in GRID_SIZES:
        for cols in GRID_SIZES:
            map_r0s = _candidate_origins(point_rows, MAP_STEP, rows, h)
            map_c0s = _candidate_origins(point_cols, MAP_STEP, cols, w)
            block_r0s = _candidate_origins(block_rows, BLOCK_STEP, rows, h)
            block_c0s = _candidate_origins(block_cols, BLOCK_STEP, cols, w)
            for map_r0 in map_r0s:
                for map_c0 in map_c0s:
                    point_map = _index_points(points, map_r0, map_c0, rows, cols, MAP_STEP)
                    if point_map is None:
                        continue
                    point_colors = set(point_map.values())
                    if not (3 <= len(point_colors) <= 4):
                        continue
                    map_r1 = map_r0 + MAP_STEP * (rows - 1)
                    map_c1 = map_c0 + MAP_STEP * (cols - 1)
                    for block_r0 in block_r0s:
                        for block_c0 in block_c0s:
                            if block_r0 < map_r0 or block_c0 < map_c0:
                                continue
                            if block_r0 < map_r1 + 3 and block_c0 < map_c1 + 3:
                                continue
                            block_map = _index_points(blocks, block_r0, block_c0, rows, cols, BLOCK_STEP)
                            if block_map is None:
                                continue
                            if not block_map or any(idx not in point_map for idx in block_map):
                                continue
                            color_map: dict[int, int] = {}
                            used_outputs: set[int] = set()
                            ok = True
                            for idx, block_color in block_map.items():
                                point_color = point_map[idx]
                                prev = color_map.get(point_color)
                                if prev is None:
                                    if block_color in used_outputs:
                                        ok = False
                                        break
                                    color_map[point_color] = block_color
                                    used_outputs.add(block_color)
                                elif prev != block_color:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            if set(color_map) != point_colors:
                                continue
                            if len(block_map) >= len(point_map):
                                continue
                            candidates.append(
                                (
                                    rows,
                                    cols,
                                    map_r0,
                                    map_c0,
                                    block_r0,
                                    block_c0,
                                    point_map,
                                    block_map,
                                    color_map,
                                )
                            )

    assert len(candidates) == 1, f"expected unique alignment, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the large 3x3 block array using the top-left point map after
    decoding the global point-color -> block-color substitution from witnesses.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    points, blocks = _classify_objects(g, bg)
    (
        rows,
        cols,
        _map_r0,
        _map_c0,
        block_r0,
        block_c0,
        point_map,
        block_map,
        color_map,
    ) = _infer_alignment(g, points, blocks)

    out = g.copy()
    for i in range(rows):
        for j in range(cols):
            key = (i, j)
            if key not in point_map or key in block_map:
                continue
            fill = color_map[point_map[key]]
            r0 = block_r0 + BLOCK_STEP * i
            c0 = block_c0 + BLOCK_STEP * j
            out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = fill
    return out


def _sample_layout(
    rng: np.random.Generator, rows: int, cols: int
) -> tuple[int, int, int, int, int, int]:
    map_h = MAP_STEP * (rows - 1) + 1
    map_w = MAP_STEP * (cols - 1) + 1
    main_h = BLOCK_STEP * (rows - 1) + BLOCK_SIZE
    main_w = BLOCK_STEP * (cols - 1) + BLOCK_SIZE

    for _ in range(100):
        map_r0 = int(rng.integers(0, 2))
        map_c0 = int(rng.integers(0, 2))
        layout = int(rng.integers(0, 3))
        if layout == 0:
            block_r0 = map_r0 + map_h + int(rng.integers(2, 5))
            block_c0 = map_c0 + map_w + int(rng.integers(2, 5))
        elif layout == 1:
            block_r0 = int(rng.integers(map_r0, map_r0 + 5))
            block_c0 = map_c0 + map_w + int(rng.integers(2, 5))
        else:
            block_r0 = map_r0 + map_h + int(rng.integers(2, 5))
            block_c0 = int(rng.integers(map_c0, map_c0 + 5))

        min_h = max(map_r0 + map_h, block_r0 + main_h)
        min_w = max(map_c0 + map_w, block_c0 + main_w)
        if min_h > 30 or min_w > 30:
            continue

        extra_h_lo = max(0, 18 - min_h)
        extra_h_hi = min(4, 30 - min_h)
        extra_w_lo = max(0, 20 - min_w)
        extra_w_hi = min(4, 30 - min_w)
        if extra_h_lo > extra_h_hi or extra_w_lo > extra_w_hi:
            continue

        height = min_h + int(rng.integers(extra_h_lo, extra_h_hi + 1))
        width = min_w + int(rng.integers(extra_w_lo, extra_w_hi + 1))
        return height, width, map_r0, map_c0, block_r0, block_c0

    raise RuntimeError("failed to place layout")


def _sample_mask(rng: np.random.Generator, rows: int, cols: int, color_count: int) -> np.ndarray:
    min_occ = max(color_count + 2, rows + cols - 1)
    for _ in range(200):
        p = float(rng.uniform(0.45, 0.8))
        mask = rng.random((rows, cols)) < p
        mask[0, 0] = True
        mask[rows - 1, cols - 1] = True
        if not mask.all(axis=1).all() or not mask.all(axis=0).all():
            continue
        if int(mask.sum()) < min_occ:
            continue
        return mask

    mask = np.zeros((rows, cols), dtype=bool)
    mask[0, 0] = True
    mask[rows - 1, cols - 1] = True
    for r in range(rows):
        mask[r, int(rng.integers(0, cols))] = True
    for c in range(cols):
        mask[int(rng.integers(0, rows)), c] = True

    remaining = [(r, c) for r in range(rows) for c in range(cols) if not mask[r, c]]
    rng.shuffle(remaining)
    need = max(0, min_occ - int(mask.sum()))
    for r, c in remaining[:need]:
        mask[r, c] = True
    return mask


def _assign_point_colors(
    rng: np.random.Generator, mask: np.ndarray, point_colors: list[int]
) -> dict[tuple[int, int], int]:
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(occupied)
    colored: dict[tuple[int, int], int] = {}
    for idx, color in enumerate(point_colors):
        colored[occupied[idx]] = color
    for pos in occupied[len(point_colors) :]:
        colored[pos] = int(point_colors[int(rng.integers(0, len(point_colors)))])
    return colored


def _choose_witnesses(
    rng: np.random.Generator, mask: np.ndarray, colors: dict[tuple[int, int], int]
) -> set[tuple[int, int]] | None:
    rows, cols = mask.shape
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    witnesses: set[tuple[int, int]] = {(0, 0), (rows - 1, cols - 1)}
    for point_color in sorted(set(colors.values())):
        options = [pos for pos in occupied if colors[pos] == point_color]
        witnesses.add(options[int(rng.integers(0, len(options)))])

    extras = [pos for pos in occupied if pos not in witnesses]
    rng.shuffle(extras)
    extra_count = int(rng.integers(0, len(extras) + 1))
    witnesses.update(extras[:extra_count])

    if len(witnesses) == len(occupied):
        removable = [
            pos
            for pos in witnesses
            if pos not in {(0, 0), (rows - 1, cols - 1)}
            and sum(1 for other in witnesses if colors[other] == colors[pos]) > 1
        ]
        if not removable:
            return None
        witnesses.remove(removable[int(rng.integers(0, len(removable)))])

    return witnesses if len(witnesses) < len(occupied) else None


def _render_case(
    height: int,
    width: int,
    map_r0: int,
    map_c0: int,
    block_r0: int,
    block_c0: int,
    rows: int,
    cols: int,
    colors: dict[tuple[int, int], int],
    color_map: dict[int, int],
    witnesses: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    for (i, j), point_color in colors.items():
        mr = map_r0 + MAP_STEP * i
        mc = map_c0 + MAP_STEP * j
        grid[mr, mc] = point_color
        expected[mr, mc] = point_color

        br = block_r0 + BLOCK_STEP * i
        bc = block_c0 + BLOCK_STEP * j
        expected[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE] = color_map[point_color]
        if (i, j) in witnesses:
            grid[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE] = color_map[point_color]
    return grid, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.choice(GRID_SIZES))
    cols = int(rng.choice(GRID_SIZES))
    color_count = int(rng.integers(3, min(4, rows * cols) + 1))

    try:
        height, width, map_r0, map_c0, block_r0, block_c0 = _sample_layout(rng, rows, cols)
    except RuntimeError:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    point_colors = [int(c) for c in palette[:color_count]]
    block_colors = [int(c) for c in palette[color_count : 2 * color_count]]
    rng.shuffle(block_colors)
    color_map = dict(zip(point_colors, block_colors))

    mask = _sample_mask(rng, rows, cols, color_count)
    colors = _assign_point_colors(rng, mask, point_colors)
    witnesses = _choose_witnesses(rng, mask, colors)
    if witnesses is None:
        return None

    grid, expected = _render_case(
        height, width, map_r0, map_c0, block_r0, block_c0, rows, cols, colors, color_map, witnesses
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
