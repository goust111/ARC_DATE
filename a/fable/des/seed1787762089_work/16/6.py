import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

PATTERN_COLORS = (BLUE, AZURE)
SPRITE_COLORS = (RED, YELLOW, GREEN)
PERIOD_OPTIONS = (3, 4)
HOLE_COUNT_OPTIONS = (2, 3)
MAX_RETRIES = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comp.sort()
            comps.append(comp)
    return comps


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return r0, c0, max(rows) - r0 + 1, max(cols) - c0 + 1


def _find_separator_row(grid: np.ndarray) -> int:
    h, _ = grid.shape
    candidates = [r for r in range(1, h - 1) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != BLACK]
    assert len(candidates) == 1, f"expected one separator row, got {candidates}"
    return int(candidates[0])


def _infer_period(top: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    h, w = top.shape
    visible = [(r, c, int(top[r, c])) for r in range(h) for c in range(w) if int(top[r, c]) != BLACK]
    assert visible, "top region must expose the pattern"

    best: tuple[int, int] | None = None
    best_phase: dict[tuple[int, int], int] | None = None
    for pr in range(1, min(6, h) + 1):
        for pc in range(1, min(6, w) + 1):
            phase: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, value in visible:
                key = (r % pr, c % pc)
                prev = phase.get(key)
                if prev is None:
                    phase[key] = value
                elif prev != value:
                    ok = False
                    break
            if not ok or len(phase) != pr * pc:
                continue
            if best is None or (pr * pc, pr, pc) < (best[0] * best[1], best[0], best[1]):
                best = (pr, pc)
                best_phase = phase
    assert best is not None and best_phase is not None, "failed to infer a complete period"
    return best[0], best[1], best_phase


def _repair_top(top: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], tuple[int, int]]:
    repaired = top.copy()
    holes = [
        comp
        for comp in _components4(top == BLACK)
        if all(0 < r < top.shape[0] - 1 and 0 < c < top.shape[1] - 1 for r, c in comp)
    ]
    pr, pc, phase = _infer_period(top)
    for comp in holes:
        for r, c in comp:
            repaired[r, c] = phase[(r % pr, c % pc)]
    return repaired, holes, (pr, pc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    top = g[:sep, :]
    warehouse = g[sep + 1 :, :]

    repaired_top, holes, _ = _repair_top(top)
    hole_by_shape: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for hole in holes:
        key = _normalize_shape(hole)
        assert key not in hole_by_shape, "hole shapes must be unique"
        hole_by_shape[key] = hole

    sprites = _components4(warehouse != BLACK)
    assert len(sprites) == len(hole_by_shape), "sprite count must match hole count"

    out = g.copy()
    out[:sep, :] = repaired_top
    out[sep + 1 :, :] = BLACK

    for sprite in sprites:
        key = _normalize_shape(sprite)
        assert key in hole_by_shape, "every sprite must match a hole shape"
        hole = hole_by_shape[key]
        hr0, hc0, _, _ = _shape_bbox(hole)
        sr0, sc0, _, _ = _shape_bbox(sprite)
        for r, c in sprite:
            dr = r - sr0
            dc = c - sc0
            out[sep + 1 + r, c] = repaired_top[hr0 + dr, hc0 + dc]

    return out


def _unit_period_rows(unit: np.ndarray) -> int:
    size = unit.shape[0]
    for p in range(1, size):
        ok = True
        for r in range(size):
            if not np.array_equal(unit[r], unit[(r + p) % size]):
                ok = False
                break
        if ok:
            return p
    return size


def _unit_period_cols(unit: np.ndarray) -> int:
    size = unit.shape[1]
    for p in range(1, size):
        ok = True
        for c in range(size):
            if not np.array_equal(unit[:, c], unit[:, (c + p) % size]):
                ok = False
                break
        if ok:
            return p
    return size


def _random_unit(rng: np.random.Generator, size: int) -> np.ndarray:
    for _ in range(300):
        unit = rng.choice(PATTERN_COLORS, size=(size, size))
        counts = [int(np.sum(unit == color)) for color in PATTERN_COLORS]
        if min(counts) < max(2, size - 1):
            continue
        if _unit_period_rows(unit) != size or _unit_period_cols(unit) != size:
            continue
        return unit.astype(int)
    raise RuntimeError("failed to build a non-degenerate periodic unit")


def _has_internal_hole(shape: tuple[tuple[int, int], ...]) -> bool:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    filled = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        filled[r, c] = True
    empties = _components4(~filled)
    for comp in empties:
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            return True
    return False


def _is_irregular(shape: tuple[tuple[int, int], ...], warehouse_h: int) -> bool:
    area = len(shape)
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    if not (5 <= area <= 9):
        return False
    if h < 2 or w < 2:
        return False
    if area == h * w:
        return False
    if h > warehouse_h or w > 6:
        return False
    if _has_internal_hole(shape):
        return False

    cell_set = set(shape)
    degrees = []
    for r, c in shape:
        deg = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
        degrees.append(deg)
    if max(degrees) < 2:
        return False
    if all(deg == 2 for deg in degrees):
        return False
    return True


def _random_shape(rng: np.random.Generator, warehouse_h: int) -> tuple[tuple[int, int], ...]:
    for _ in range(400):
        area = int(rng.integers(5, 10))
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            options = sorted(frontier)
            choice = options[int(rng.integers(len(options)))]
            cells.add(choice)
        shape = _normalize_shape(list(cells))
        if _is_irregular(shape, warehouse_h):
            return shape
    raise RuntimeError("failed to generate a valid hole shape")


def _shape_colors_at(pattern: np.ndarray, shape: tuple[tuple[int, int], ...], origin: tuple[int, int]) -> set[int]:
    r0, c0 = origin
    return {int(pattern[r0 + dr, c0 + dc]) for dr, dc in shape}


def _candidate_origins(
    shape: tuple[tuple[int, int], ...],
    canvas_h: int,
    canvas_w: int,
    occupied: np.ndarray,
    border_margin: int,
) -> list[tuple[int, int]]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    origins: list[tuple[int, int]] = []
    r_start = border_margin
    c_start = border_margin
    r_end = canvas_h - h - border_margin
    c_end = canvas_w - w - border_margin
    if r_end < r_start or c_end < c_start:
        return origins
    for r0 in range(r_start, r_end + 1):
        for c0 in range(c_start, c_end + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(canvas_h, r0 + h + 1)
            cc1 = min(canvas_w, c0 + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            origins.append((r0, c0))
    return origins


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _all_phases_visible(masked_top: np.ndarray, pr: int, pc: int) -> bool:
    seen = set()
    for r in range(masked_top.shape[0]):
        for c in range(masked_top.shape[1]):
            if int(masked_top[r, c]) != BLACK:
                seen.add((r % pr, c % pc))
    return len(seen) == pr * pc


def _build_expected(
    base_top: np.ndarray,
    sep: int,
    warehouse_h: int,
    width: int,
    hole_records: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]],
    sprite_records: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    out = np.zeros((sep + 1 + warehouse_h, width), dtype=int)
    out[:sep, :] = base_top
    out[sep, :] = GREY
    for hole_index, sprite_origin in sprite_records:
        shape, hole_origin = hole_records[hole_index]
        hr0, hc0 = hole_origin
        sr0, sc0 = sprite_origin
        for dr, dc in shape:
            out[sep + 1 + sr0 + dr, sc0 + dc] = int(base_top[hr0 + dr, hc0 + dc])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pr = pc = int(rng.choice(PERIOD_OPTIONS))
    warehouse_h = int(rng.integers(5, 8))
    main_h = int(rng.integers(max(16, pr * 4), 23))
    width = int(rng.integers(max(20, pc * 5), 29))
    hole_count = int(rng.choice(HOLE_COUNT_OPTIONS))

    unit = _random_unit(rng, pr)
    reps_r = (main_h + pr - 1) // pr
    reps_c = (width + pc - 1) // pc
    base_top = np.tile(unit, (reps_r, reps_c))[:main_h, :width]

    shapes: list[tuple[tuple[int, int], ...]] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(hole_count):
        for _ in range(200):
            shape = _random_shape(rng, warehouse_h)
            if shape not in used_shapes:
                used_shapes.add(shape)
                shapes.append(shape)
                break
        else:
            return None

    order = sorted(range(hole_count), key=lambda i: (len(shapes[i]), shapes[i][-1][1]), reverse=True)
    occupied_top = np.zeros((main_h, width), dtype=bool)
    hole_records: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = [None] * hole_count  # type: ignore[list-item]
    for idx in order:
        shape = shapes[idx]
        origins = _candidate_origins(shape, main_h, width, occupied_top, border_margin=1)
        origins = [origin for origin in origins if _shape_colors_at(base_top, shape, origin) == set(PATTERN_COLORS)]
        if not origins:
            return None
        origin = origins[int(rng.integers(len(origins)))]
        hole_records[idx] = (shape, origin)
        _paint_shape(occupied_top, shape, origin, True)

    top = base_top.copy()
    for shape, origin in hole_records:
        _paint_shape(top, shape, origin, BLACK)

    if not _all_phases_visible(top, pr, pc):
        return None
    inferred_pr, inferred_pc, _ = _infer_period(top)
    if (inferred_pr, inferred_pc) != (pr, pc):
        return None

    warehouse = np.zeros((warehouse_h, width), dtype=int)
    occupied_bottom = np.zeros((warehouse_h, width), dtype=bool)
    permuted_holes = list(rng.permutation(hole_count))
    sprite_colors = list(rng.permutation(SPRITE_COLORS))[:hole_count]
    sprite_records: list[tuple[int, tuple[int, int]]] = []
    for hole_index, color in zip(permuted_holes, sprite_colors):
        shape = hole_records[hole_index][0]
        origins = _candidate_origins(shape, warehouse_h, width, occupied_bottom, border_margin=0)
        if not origins:
            return None
        origin = origins[int(rng.integers(len(origins)))]
        sprite_records.append((hole_index, origin))
        _paint_shape(warehouse, shape, origin, int(color))
        _paint_shape(occupied_bottom, shape, origin, True)

    grid = np.zeros((main_h + 1 + warehouse_h, width), dtype=int)
    grid[:main_h, :] = top
    grid[main_h, :] = GREY
    grid[main_h + 1 :, :] = warehouse

    expected = _build_expected(base_top, main_h, warehouse_h, width, hole_records, sprite_records)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
