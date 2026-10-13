from __future__ import annotations

import random

import numpy as np


BLACK = 0
GEN_GREY = 5
DIR_ORDER = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(sorted(coords), dtype=int))
    out.sort(key=lambda comp: (int(comp[:, 0].min()), int(comp[:, 1].min())))
    return out


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    shifted = arr - np.array([[r0, c0]], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _mask_from_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.asarray(shape, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _shape_forms(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    mask = _mask_from_shape(shape)
    forms: set[tuple[tuple[int, int], ...]] = set()
    cur = mask
    for _ in range(4):
        forms.add(_normalize_coords(np.argwhere(cur)))
        forms.add(_normalize_coords(np.argwhere(np.fliplr(cur))))
        cur = np.rot90(cur, 1)
    return forms


def _tip_info(coords: np.ndarray) -> dict[str, object] | None:
    arr = np.asarray(coords, dtype=int)
    rows = arr[:, 0]
    cols = arr[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    n = len(arr)
    row_shift_num = 2 * int(rows.sum()) - n * (r0 + r1)
    col_shift_num = 2 * int(cols.sum()) - n * (c0 + c1)

    edge_points = {
        "up": arr[rows == r0],
        "down": arr[rows == r1],
        "left": arr[cols == c0],
        "right": arr[cols == c1],
    }

    candidates: list[dict[str, object]] = []
    for name in DIR_ORDER:
        edge = edge_points[name]
        if len(edge) != 1:
            continue
        dr, dc = DIRS[name]
        score_num = -(row_shift_num * dr + col_shift_num * dc)
        candidates.append(
            {
                "direction": name,
                "tip": (int(edge[0, 0]), int(edge[0, 1])),
                "score_num": int(score_num),
                "opposite_width": int(len(edge_points[OPPOSITE[name]])),
            }
        )

    if not candidates:
        return None

    def _key(item: dict[str, object]) -> tuple[int, int]:
        return int(item["score_num"]), int(item["opposite_width"])

    best = max(candidates, key=_key)
    tied = [
        item
        for item in candidates
        if int(item["score_num"]) == int(best["score_num"])
        and int(item["opposite_width"]) == int(best["opposite_width"])
    ]
    if len(tied) > 1:
        tied.sort(key=lambda item: DIR_ORDER.index(str(item["direction"])))
        best = tied[0]

    return {
        "direction": str(best["direction"]),
        "tip": tuple(best["tip"]),
        "candidate_count": len(candidates),
        "unique_best": len(tied) == 1,
    }


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    row_hits: dict[int, list[int]] = {}
    col_hits: dict[int, list[int]] = {}

    for r in range(h):
        row = grid[r, :]
        if np.all(row == row[0]):
            row_hits.setdefault(int(row[0]), []).append(r)

    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]):
            col_hits.setdefault(int(col[0]), []).append(c)

    candidates = set(row_hits) & set(col_hits)
    assert candidates, "expected a separator color with full rows and columns"
    sep = max(
        candidates,
        key=lambda color: (
            len(row_hits[color]) + len(col_hits[color]),
            len(row_hits[color]) * len(col_hits[color]),
        ),
    )
    return int(sep), sorted(row_hits[sep]), sorted(col_hits[sep])


def _band_intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [size]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        start = a + 1
        end = b - 1
        if start <= end:
            out.append((start, end))
    return out


def _band_index(size: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(size, -1, dtype=int)
    for i, (a, b) in enumerate(intervals):
        out[a : b + 1] = i
    return out


def _region_backgrounds(
    grid: np.ndarray,
    sep_color: int,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(grid.shape, -1, dtype=int)
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(window))) - {BLACK, sep_color}
            assert len(colors) == 1, "each region must have one visible background color"
            out[r0 : r1 + 1, c0 : c1 + 1] = next(iter(colors))
    return out


def _trace_ray(
    shape_tip: tuple[int, int],
    direction: str,
    length: int,
    separator_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = separator_mask.shape
    dr, dc = DIRS[direction]
    r, c = shape_tip
    path: list[tuple[int, int]] = []
    while len(path) < length:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if separator_mask[r, c]:
            continue
        path.append((r, c))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each black object's tapered tip outward as a ray whose visible color is
    the background color of the object's home region.

    The grid is partitioned by one-cell-thick grey separator rows and columns into
    rectangular color bands. Each black 4-connected object has a tip direction:
    among its bounding-box edges, candidate tip edges are those occupied by exactly
    one object cell, and ties are broken by choosing the edge opposite the largest
    centroid shift. Starting just beyond that tip cell, paint a ray of length equal
    to the object's area, skipping grey separator cells without counting them.
    Black objects and grey separators stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, sep_rows, sep_cols = _find_separator_color(g)
    row_intervals = _band_intervals(g.shape[0], sep_rows)
    col_intervals = _band_intervals(g.shape[1], sep_cols)
    assert row_intervals and col_intervals, "expected at least one rectangular region"

    separator_mask = g == sep_color
    region_bg = _region_backgrounds(g, sep_color, row_intervals, col_intervals)
    objects = _components(g == BLACK)

    out = g.copy()
    for obj in objects:
        info = _tip_info(obj)
        assert info is not None, "every object must have a detectable tip"
        tip = tuple(info["tip"])
        direction = str(info["direction"])
        ray = _trace_ray(tip, direction, len(obj), separator_mask)
        assert ray is not None, "ray ran out of board before reaching object area"
        home_color = int(region_bg[int(obj[0, 0]), int(obj[0, 1])])
        assert home_color >= 0, "object must lie inside a colored region"
        for r, c in ray:
            if g[r, c] != BLACK:
                out[r, c] = home_color
    return out


def _connected(coords: list[tuple[int, int]]) -> bool:
    cells = set(coords)
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _build_shape_library() -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    unique_shapes: set[tuple[tuple[int, int], ...]] = set()
    for h in range(2, 5):
        for w in range(2, 5):
            n = h * w
            for bits in range(1 << n):
                area = bits.bit_count()
                if area < 3 or area > 8:
                    continue
                coords = [(i // w, i % w) for i in range(n) if (bits >> i) & 1]
                if _connected(coords):
                    unique_shapes.add(_normalize_coords(np.asarray(coords, dtype=int)))

    by_dir = {name: [] for name in DIR_ORDER}
    multi_by_dir = {name: [] for name in DIR_ORDER}
    for shape in unique_shapes:
        if len(_shape_forms(shape)) != 8:
            continue
        coords = np.asarray(shape, dtype=int)
        info = _tip_info(coords)
        if info is None or not bool(info["unique_best"]):
            continue
        entry = {
            "shape": shape,
            "height": int(coords[:, 0].max()) + 1,
            "width": int(coords[:, 1].max()) + 1,
            "area": len(shape),
            "tip": tuple(info["tip"]),
            "direction": str(info["direction"]),
            "candidate_count": int(info["candidate_count"]),
        }
        direction = str(info["direction"])
        by_dir[direction].append(entry)
        if int(info["candidate_count"]) > 1:
            multi_by_dir[direction].append(entry)

    for direction in DIR_ORDER:
        assert by_dir[direction], f"missing shapes for {direction}"
        assert multi_by_dir[direction], f"missing multi-edge shapes for {direction}"
        by_dir[direction].sort(
            key=lambda item: (int(item["area"]), int(item["height"]), int(item["width"]), item["shape"])
        )
        multi_by_dir[direction].sort(
            key=lambda item: (int(item["area"]), int(item["height"]), int(item["width"]), item["shape"])
        )
    return by_dir, multi_by_dir


SHAPES_BY_DIR, MULTI_SHAPES_BY_DIR = _build_shape_library()


GRID_LIMIT = 30
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
ROW_SIZE_RANGE = (5, 7)
COL_SIZE_RANGE = (5, 8)
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_OBJECTS = 4
OUTER_RETRIES = 400
PLACEMENT_ATTEMPTS = 140


def _sample_sizes(count: int, min_size: int, max_size: int, max_inner_total: int) -> list[int] | None:
    lo = count * min_size
    hi = min(count * max_size, max_inner_total)
    if lo > hi:
        return None
    total = random.randint(lo, hi)
    sizes = [min_size] * count
    for _ in range(total - lo):
        choices = [i for i, value in enumerate(sizes) if value < max_size]
        if not choices:
            break
        sizes[random.choice(choices)] += 1
    random.shuffle(sizes)
    return sizes


def _assign_region_colors(rows: int, cols: int) -> list[list[int]]:
    colors = [[-1] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            forbidden = set()
            if r > 0:
                forbidden.add(colors[r - 1][c])
            if c > 0:
                forbidden.add(colors[r][c - 1])
            choices = [color for color in PALETTE if color not in forbidden]
            colors[r][c] = random.choice(choices)
    return colors


def _build_partition() -> tuple[
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
] | None:
    row_count = random.randint(*ROW_COUNT_RANGE)
    col_count = random.randint(*COL_COUNT_RANGE)
    row_sizes = _sample_sizes(row_count, *ROW_SIZE_RANGE, GRID_LIMIT - (row_count + 1))
    col_sizes = _sample_sizes(col_count, *COL_SIZE_RANGE, GRID_LIMIT - (col_count + 1))
    if row_sizes is None or col_sizes is None:
        return None

    h = sum(row_sizes) + row_count + 1
    w = sum(col_sizes) + col_count + 1
    grid = np.full((h, w), GEN_GREY, dtype=int)

    row_intervals: list[tuple[int, int]] = []
    col_intervals: list[tuple[int, int]] = []

    cursor = 1
    for size in row_sizes:
        row_intervals.append((cursor, cursor + size - 1))
        cursor += size + 1

    cursor = 1
    for size in col_sizes:
        col_intervals.append((cursor, cursor + size - 1))
        cursor += size + 1

    region_colors = _assign_region_colors(row_count, col_count)
    for r_idx, (r0, r1) in enumerate(row_intervals):
        for c_idx, (c0, c1) in enumerate(col_intervals):
            grid[r0 : r1 + 1, c0 : c1 + 1] = region_colors[r_idx][c_idx]

    row_band = _band_index(h, row_intervals)
    col_band = _band_index(w, col_intervals)
    return grid, row_intervals, col_intervals, row_band, col_band


def _region_of(row_band: np.ndarray, col_band: np.ndarray, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return int(row_band[r]), int(col_band[c])


def _try_place_object(
    grid: np.ndarray,
    region: tuple[int, int],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    row_band: np.ndarray,
    col_band: np.ndarray,
    reserved: set[tuple[int, int]],
    require_multi: bool,
) -> dict[str, object] | None:
    region_r, region_c = region
    rr0, rr1 = row_intervals[region_r]
    cc0, cc1 = col_intervals[region_c]
    sep_mask = grid == GEN_GREY

    for _ in range(PLACEMENT_ATTEMPTS):
        direction = random.choice(DIR_ORDER)
        pool = MULTI_SHAPES_BY_DIR[direction] if require_multi else SHAPES_BY_DIR[direction]
        entry = random.choice(pool)
        shape_h = int(entry["height"])
        shape_w = int(entry["width"])

        top_lo = rr0 + 1
        top_hi = rr1 - shape_h
        left_lo = cc0 + 1
        left_hi = cc1 - shape_w
        if top_lo > top_hi or left_lo > left_hi:
            continue

        top = random.randint(top_lo, top_hi)
        left = random.randint(left_lo, left_hi)
        shape = entry["shape"]
        coords = [(top + r, left + c) for r, c in shape]
        if any(cell in reserved for cell in coords):
            continue

        tip_r, tip_c = entry["tip"]
        tip = (top + int(tip_r), left + int(tip_c))
        ray = _trace_ray(tip, direction, int(entry["area"]), sep_mask)
        if ray is None:
            continue
        if any(cell in reserved for cell in ray):
            continue

        home_region = (region_r, region_c)
        crossed = any(_region_of(row_band, col_band, cell) != home_region for cell in ray)
        if not crossed:
            continue

        return {
            "coords": coords,
            "ray": ray,
            "tip": tip,
            "direction": direction,
            "candidate_count": int(entry["candidate_count"]),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    built = _build_partition()
    if built is None:
        return None
    grid, row_intervals, col_intervals, row_band, col_band = built

    regions = [(r, c) for r in range(len(row_intervals)) for c in range(len(col_intervals))]
    random.shuffle(regions)
    target_objects = random.randint(1, min(MAX_OBJECTS, len(regions)))
    multi_index = random.randrange(target_objects) if random.random() < 0.5 else -1

    reserved: set[tuple[int, int]] = set()
    placed = []
    for obj_idx in range(target_objects):
        want_multi = obj_idx == multi_index
        region_order = random.sample(regions, len(regions))
        chosen = None
        for region in region_order:
            if region in {item["region"] for item in placed}:
                continue
            chosen = _try_place_object(
                grid,
                region,
                row_intervals,
                col_intervals,
                row_band,
                col_band,
                reserved,
                require_multi=want_multi,
            )
            if chosen is None:
                continue
            chosen["region"] = region
            for cell in chosen["coords"]:
                grid[cell] = BLACK
                reserved.add(cell)
            for cell in chosen["ray"]:
                reserved.add(cell)
            placed.append(chosen)
            break
        if chosen is None:
            return None

    if not placed:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[grid == GEN_GREY], grid[grid == GEN_GREY]):
        return None
    if int(np.sum(out == BLACK)) != int(np.sum(grid == BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
