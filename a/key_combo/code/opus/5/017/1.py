import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
OBSTACLE_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_start(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    comps = components_mask(grid == BLUE)
    assert len(comps) == 1, "expected one blue start bar"
    comp = comps[0]
    assert comp.shape[0] == 2, "expected a 2-pixel blue start bar"
    pts = [tuple(map(int, xy)) for xy in comp]
    boundary = [p for p in pts if p[0] in (0, grid.shape[0] - 1) or p[1] in (0, grid.shape[1] - 1)]
    assert len(boundary) == 1, "exactly one start pixel must lie on the boundary"
    edge = boundary[0]
    inner = pts[0] if pts[1] == edge else pts[1]
    delta = (inner[0] - edge[0], inner[1] - edge[1])
    assert delta in DIRS, "start bar must be axis-aligned"
    return edge, inner, DIRS.index(delta)


def _trace_walk(grid: np.ndarray) -> dict:
    _, inner, d = _find_start(grid)
    obstacle_mask = (grid != BLACK) & (grid != BLUE) & (grid != GREY)
    out = grid.copy()
    current = inner
    current_color = BLUE
    order = 0
    path_seq: list[tuple[int, int]] = []
    path_index: dict[tuple[int, int], int] = {}
    path_color: dict[tuple[int, int], int] = {}
    segment_ranges: list[tuple[int, int, int]] = []
    segment_start = 0
    segment_color = current_color
    turn_colors: list[int] = []

    while True:
        dr, dc = DIRS[d]
        nr, nc = current[0] + dr, current[1] + dc
        if not in_bounds(grid.shape, nr, nc):
            if order > segment_start:
                segment_ranges.append((segment_start, order - 1, segment_color))
            break
        nxt = int(grid[nr, nc])
        if nxt == BLACK:
            out[nr, nc] = current_color
            current = (nr, nc)
            path_seq.append(current)
            path_index[current] = order
            path_color[current] = current_color
            order += 1
            continue

        assert obstacle_mask[nr, nc], "next non-empty cell must be a colored obstacle"
        candidates = []
        for nd in ((d - 1) % 4, (d + 1) % 4):
            tr, tc = current[0] + DIRS[nd][0], current[1] + DIRS[nd][1]
            if in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) == BLACK and (tr, tc) not in path_index:
                candidates.append(nd)
        assert len(candidates) == 1, "each obstacle must expose exactly one perpendicular continuation"
        if order > segment_start:
            segment_ranges.append((segment_start, order - 1, segment_color))
        turn_colors.append(nxt)
        d = candidates[0]
        current_color = nxt
        segment_color = current_color
        segment_start = order

    return {
        "path_seq": path_seq,
        "path_index": path_index,
        "path_color": path_color,
        "path_mask": np.array([list(p) for p in path_seq], dtype=int) if path_seq else np.zeros((0, 2), dtype=int),
        "segment_ranges": segment_ranges,
        "turn_colors": turn_colors,
        "obstacle_mask": obstacle_mask,
        "painted_grid": out,
    }


def _pick_region_color(adjacent: set[tuple[int, int]], path_index: dict, path_color: dict) -> tuple[int, Counter]:
    counts = Counter(path_color[p] for p in adjacent)
    best_count = max(counts.values())
    tied = [color for color, cnt in counts.items() if cnt == best_count]
    if len(tied) == 1:
        return tied[0], counts
    best_color = min(tied, key=lambda color: min(path_index[p] for p in adjacent if path_color[p] == color))
    return best_color, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the corridor from the blue start bar, repaint gray regions by adjacent path-segment colors, then erase the obstacle markers.

    Input structure:
    - The background is black, with a single 2-pixel blue start bar touching the boundary.
    - Colored obstacle pixels sit just beyond turn points of an otherwise background corridor; gray 4-connected regions lie beside painted corridor segments.
    - The walkable continuation at each turn is the unique perpendicular background cell, and gray regions only use background cells in the input.

    Rule:
    - Start from the inner blue pixel, moving in from the boundary, and paint each visited background cell with the current brush color, initially blue.
    - If the next cell in the current direction is a colored obstacle C, do not paint that obstacle; instead turn into the unique perpendicular background continuation, and from the first cell after the turn onward switch the brush color to C.
    - After the walk stops at the grid edge, recolor each gray 4-connected region by the colors of its 4-neighbor path pixels: uniform adjacency keeps that color, otherwise use the most frequent adjacent color, breaking ties by the earliest adjacent path occurrence.
    - Finally erase every original colored obstacle pixel back to black and keep the blue start bar unchanged.

    Key insight:
    - The traversal does two jobs at once: it creates the geometry of the corridor path and also writes the segment colors that later serve as the only markers for recoloring each gray neutral region.
    """
    g = np.asarray(grid, dtype=int)
    info = _trace_walk(g)
    out = info["painted_grid"].copy()
    path_index = info["path_index"]
    path_color = info["path_color"]

    for comp in components_mask(g == GREY):
        cells = [tuple(map(int, xy)) for xy in comp]
        adjacent = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in path_index:
                    adjacent.add((nr, nc))
        if not adjacent:
            continue
        fill_color, _ = _pick_region_color(adjacent, path_index, path_color)
        rr, cc = np.array(cells, dtype=int).T
        out[rr, cc] = fill_color

    out[info["obstacle_mask"]] = BLACK
    return out


def _normalize_offsets(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rows = coords[:, 0] - coords[:, 0].min()
    cols = coords[:, 1] - coords[:, 1].min()
    return tuple(sorted((int(r), int(c)) for r, c in zip(rows, cols)))


def _template_variants(mask_rows: list[str]) -> list[tuple[tuple[int, int], ...]]:
    arr = np.array([[ch == "1" for ch in row] for row in mask_rows], dtype=bool)
    variants = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for candidate in (rot, np.fliplr(rot)):
            coords = np.argwhere(candidate)
            variants.add(_normalize_offsets(coords))
    return list(variants)


_REGION_VARIANTS: list[tuple[tuple[int, int], ...]] = []
for _mask in (
    ["111"],
    ["11", "10"],
    ["1111"],
    ["111", "100"],
    ["11", "11"],
    ["010", "111"],
    ["110", "011"],
    ["11111"],
    ["111", "010", "010"],
):
    _REGION_VARIANTS.extend(_template_variants(_mask))
_REGION_VARIANTS = sorted(set(_REGION_VARIANTS), key=lambda shape: (len(shape), shape))


def _sample_spaced_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(3, height - 2))
    for _ in range(200):
        rows = sorted(random.sample(choices, count))
        if rows[-1] > height - 3:
            continue
        if rows[0] < 3:
            continue
        if min(b - a for a, b in zip(rows, rows[1:])) >= 3:
            return rows
    return None


def _sample_columns(width: int, count: int) -> list[int] | None:
    pool = list(range(2, width - 2))
    for _ in range(300):
        cols = [random.choice(pool)]
        ok = True
        while len(cols) < count:
            candidates = [c for c in pool if c not in cols and abs(c - cols[-1]) >= 4]
            if not candidates:
                ok = False
                break
            cols.append(random.choice(candidates))
        if ok:
            return cols
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    if not cells:
        return
    rr, cc = np.array(cells, dtype=int).T
    grid[rr, cc] = color


def _build_canonical_base() -> np.ndarray | None:
    height = random.randint(15, 21)
    width = random.randint(17, 24)
    horizontal_runs = random.choice([2, 2, 3])
    rows = _sample_spaced_rows(height, horizontal_runs)
    cols = _sample_columns(width, horizontal_runs + 1)
    if rows is None or cols is None:
        return None

    turn_count = 2 * horizontal_runs
    turn_colors = random.sample(list(OBSTACLE_COLORS), turn_count)
    grid = np.zeros((height, width), dtype=int)
    grid[0, cols[0]] = BLUE
    grid[1, cols[0]] = BLUE

    occupied = {(0, cols[0]), (1, cols[0])}
    path_cells: list[tuple[int, int]] = []

    def reserve(cell: tuple[int, int], color: int) -> bool:
        if cell in occupied:
            return False
        if not in_bounds(grid.shape, cell[0], cell[1]):
            return False
        occupied.add(cell)
        grid[cell] = color
        return True

    current_row = 1
    current_col = cols[0]
    for i, turn_row in enumerate(rows):
        vertical = [(r, current_col) for r in range(current_row + 1, turn_row + 1)]
        if len(vertical) < 2:
            return None
        if any(cell in occupied for cell in vertical):
            return None
        for cell in vertical:
            occupied.add(cell)
        path_cells.extend(vertical)

        next_col = cols[i + 1]
        step = 1 if next_col > current_col else -1
        ahead = (turn_row + 1, current_col)
        side_block = (turn_row, current_col - step)
        if not reserve(ahead, turn_colors[2 * i]):
            return None
        if in_bounds(grid.shape, side_block[0], side_block[1]) and not reserve(side_block, turn_colors[2 * i]):
            return None

        horizontal = [(turn_row, c) for c in range(current_col + step, next_col + step, step)]
        if len(horizontal) < 4:
            return None
        if any(cell in occupied for cell in horizontal):
            return None
        for cell in horizontal:
            occupied.add(cell)
        path_cells.extend(horizontal)

        ahead = (turn_row, next_col + step)
        up_block = (turn_row - 1, next_col)
        if not reserve(ahead, turn_colors[2 * i + 1]):
            return None
        if in_bounds(grid.shape, up_block[0], up_block[1]) and not reserve(up_block, turn_colors[2 * i + 1]):
            return None

        current_row = turn_row
        current_col = next_col

    final_segment = [(r, current_col) for r in range(current_row + 1, height)]
    if len(final_segment) < 2 or any(cell in occupied for cell in final_segment):
        return None
    path_cells.extend(final_segment)
    return grid


def _region_info(
    cells: list[tuple[int, int]],
    path_index: dict[tuple[int, int], int],
    path_color: dict[tuple[int, int], int],
) -> dict | None:
    adjacent = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in path_index:
                adjacent.add((nr, nc))
    if not adjacent:
        return None
    counts = Counter(path_color[p] for p in adjacent)
    if len(counts) > 2:
        return None
    fill_color, _ = _pick_region_color(adjacent, path_index, path_color)
    return {
        "adjacent": adjacent,
        "counts": counts,
        "fill_color": fill_color,
        "size": len(cells),
    }


def _candidate_shapes(
    grid: np.ndarray,
    occupied: np.ndarray,
    gray_mask: np.ndarray,
    path_index: dict[tuple[int, int], int],
    path_color: dict[tuple[int, int], int],
    kind: str,
) -> list[tuple[list[tuple[int, int]], dict]]:
    h, w = grid.shape
    candidates = []
    for shape in _REGION_VARIANTS:
        sr = max(r for r, _ in shape) + 1
        sc = max(c for _, c in shape) + 1
        for r0 in range(h - sr + 1):
            for c0 in range(w - sc + 1):
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(occupied[r, c] for r, c in cells):
                    continue
                merge = False
                for r, c in cells:
                    for nr, nc in neighbors4(r, c):
                        if in_bounds(grid.shape, nr, nc) and gray_mask[nr, nc]:
                            merge = True
                            break
                    if merge:
                        break
                if merge:
                    continue
                info = _region_info(cells, path_index, path_color)
                if info is None:
                    continue
                counts = info["counts"]
                if kind == "same":
                    if len(counts) != 1:
                        continue
                elif kind == "majority":
                    if len(counts) != 2:
                        continue
                    values = sorted(counts.values(), reverse=True)
                    if values[0] == values[1]:
                        continue
                else:
                    if len(counts) != 2:
                        continue
                    values = sorted(counts.values(), reverse=True)
                    if values[0] != values[1]:
                        continue
                candidates.append((cells, info))
    random.shuffle(candidates)
    return candidates


def _add_regions(base: np.ndarray) -> np.ndarray | None:
    info = _trace_walk(base)
    path_index = info["path_index"]
    path_color = info["path_color"]
    occupied = (base != BLACK).copy()
    for r, c in path_index:
        occupied[r, c] = True

    grid = base.copy()
    gray_mask = np.zeros(base.shape, dtype=bool)
    placed_kinds = []
    for kind in ("tie", "majority", "same"):
        candidates = _candidate_shapes(grid, occupied, gray_mask, path_index, path_color, kind)
        if not candidates:
            return None
        cells, _ = candidates[0]
        for r, c in cells:
            grid[r, c] = GREY
            occupied[r, c] = True
            gray_mask[r, c] = True
        placed_kinds.append(kind)

    if random.random() < 0.6:
        extras = _candidate_shapes(grid, occupied, gray_mask, path_index, path_color, random.choice(["same", "majority"]))
        if extras:
            cells, _ = extras[0]
            for r, c in cells:
                grid[r, c] = GREY
                occupied[r, c] = True
                gray_mask[r, c] = True
    return grid


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _classify_region(adjacent: set[tuple[int, int]], path_index: dict, path_color: dict) -> tuple[str, int]:
    counts = Counter(path_color[p] for p in adjacent)
    fill_color, _ = _pick_region_color(adjacent, path_index, path_color)
    if len(counts) == 1:
        return "same", fill_color
    values = sorted(counts.values(), reverse=True)
    if values[0] == values[1]:
        return "tie", fill_color
    return "majority", fill_color


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        info = _trace_walk(inp)
        out = main(inp)
    except Exception:
        return False

    if len(info["turn_colors"]) < 4:
        return False
    if len(set(info["turn_colors"])) < 4:
        return False

    if np.any(out[info["obstacle_mask"]] != BLACK):
        return False

    region_types = set()
    gray_components = components_mask(inp == GREY)
    if len(gray_components) < 3:
        return False

    for comp in gray_components:
        if not 3 <= comp.shape[0] <= 8:
            return False
        cells = [tuple(map(int, xy)) for xy in comp]
        adjacent = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in info["path_index"]:
                    adjacent.add((nr, nc))
        if not adjacent:
            return False
        region_type, fill_color = _classify_region(adjacent, info["path_index"], info["path_color"])
        region_types.add(region_type)
        rr, cc = np.array(cells, dtype=int).T
        if not np.all(out[rr, cc] == fill_color):
            return False

    if region_types != {"same", "majority", "tie"}:
        return False

    # Make sure recoloring truly depends on the color-relay path, not just geometry.
    if not any(
        len(Counter(info["path_color"][p] for p in {
            q for r, c in [tuple(map(int, xy)) for xy in comp]
            for q in neighbors4(r, c)
            if q in info["path_index"]
        })) >= 2
        for comp in gray_components
    ):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base = _build_canonical_base()
    if base is None:
        return None
    with_regions = _add_regions(base)
    if with_regions is None:
        return None
    transformed = _apply_random_transform(with_regions)
    if not _validate_semantics(transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor color-relay input")
