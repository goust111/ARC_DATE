from dataclasses import dataclass

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    r1: int
    c0: int
    c1: int


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[Rect]:
    rects: list[Rect] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert np.all(patch == color), "each foreground object must be one solid rectangle"
        rects.append(Rect(color, int(r0), int(r1), int(c0), int(c1)))
    rects.sort(key=lambda rect: (rect.r0, rect.c0, rect.color))
    return rects


def _blocked_directions(rect: Rect, rects: list[Rect]) -> set[str]:
    blocked: set[str] = set()
    for other in rects:
        if other == rect:
            continue
        if other.r1 < rect.r0 and _intervals_overlap(rect.c0, rect.c1, other.c0, other.c1):
            blocked.add("up")
        if other.r0 > rect.r1 and _intervals_overlap(rect.c0, rect.c1, other.c0, other.c1):
            blocked.add("down")
        if other.c1 < rect.c0 and _intervals_overlap(rect.r0, rect.r1, other.r0, other.r1):
            blocked.add("left")
        if other.c0 > rect.c1 and _intervals_overlap(rect.r0, rect.r1, other.r0, other.r1):
            blocked.add("right")
    return blocked


def _ring_cells(r0: int, r1: int, c0: int, c1: int, nr0: int, nr1: int, nc0: int, nc1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rr in range(nr0, nr1 + 1):
        for cc in range(nc0, nc1 + 1):
            if rr < r0 or rr > r1 or cc < c0 or cc > c1:
                cells.add((rr, cc))
    return cells


def _grow_shell_cells(rect: Rect, blocked: set[str], occupied: np.ndarray) -> set[tuple[int, int]]:
    h, w = occupied.shape
    k = len(blocked)
    if k == 0:
        return set()

    active = {direction for direction in DIRECTIONS if direction not in blocked}
    extents = {direction: 0 for direction in DIRECTIONS}
    shell: set[tuple[int, int]] = set()

    for _ in range(k):
        cur_r0 = rect.r0 - extents["up"]
        cur_r1 = rect.r1 + extents["down"]
        cur_c0 = rect.c0 - extents["left"]
        cur_c1 = rect.c1 + extents["right"]

        if "up" in active and cur_r0 == 0:
            active.remove("up")
        if "down" in active and cur_r1 == h - 1:
            active.remove("down")
        if "left" in active and cur_c0 == 0:
            active.remove("left")
        if "right" in active and cur_c1 == w - 1:
            active.remove("right")
        if not active:
            break

        grow = set(active)
        while grow:
            nr0 = cur_r0 - int("up" in grow)
            nr1 = cur_r1 + int("down" in grow)
            nc0 = cur_c0 - int("left" in grow)
            nc1 = cur_c1 + int("right" in grow)

            ring = _ring_cells(cur_r0, cur_r1, cur_c0, cur_c1, nr0, nr1, nc0, nc1)
            offenders: set[str] = set()
            for rr, cc in ring:
                if not occupied[rr, cc]:
                    continue
                if rr < cur_r0:
                    offenders.add("up")
                if rr > cur_r1:
                    offenders.add("down")
                if cc < cur_c0:
                    offenders.add("left")
                if cc > cur_c1:
                    offenders.add("right")

            reduced = grow - offenders
            if reduced == grow:
                break
            grow = reduced

        active = grow
        if not active:
            break

        nr0 = cur_r0 - int("up" in active)
        nr1 = cur_r1 + int("down" in active)
        nc0 = cur_c0 - int("left" in active)
        nc1 = cur_c1 + int("right" in active)
        shell |= _ring_cells(cur_r0, cur_r1, cur_c0, cur_c1, nr0, nr1, nc0, nc1)
        for direction in active:
            extents[direction] += 1

    return shell


def _collect_growth(grid: np.ndarray) -> tuple[int, list[Rect], list[dict[str, object]], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rects = _extract_rectangles(g, bg)
    occupied = g != bg

    claims = np.full(g.shape, -1, dtype=int)
    conflict = np.zeros(g.shape, dtype=bool)
    infos: list[dict[str, object]] = []

    for rect in rects:
        blocked = _blocked_directions(rect, rects)
        shell = _grow_shell_cells(rect, blocked, occupied)
        infos.append({"rect": rect, "blocked": blocked, "k": len(blocked), "shell": shell})
        for rr, cc in shell:
            prev = int(claims[rr, cc])
            if prev == -1 or prev == rect.color:
                claims[rr, cc] = rect.color
            else:
                conflict[rr, cc] = True

    return bg, rects, infos, claims, conflict


def _paint_claims(grid: np.ndarray, claims: np.ndarray, conflict: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mask = (claims != -1) & ~conflict
    out[mask] = claims[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each rectangle outward on its unblocked sides by the number of blocked sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _bg, _rects, _infos, claims, conflict = _collect_growth(g)
    return _paint_claims(g, claims, conflict)


GEN_ROWS_RANGE = (3, 4)
GEN_COLS_RANGE = (3, 4)
GEN_BAND_SIZE_RANGE = (2, 3)
GEN_LANE_SIZE_RANGE = (2, 3)
GEN_GAP_RANGE = (3, 4)
GEN_MARGIN_RANGE = (1, 2)
GEN_OBJECT_COUNT_RANGE = (4, 7)
GEN_MAX_ATTEMPTS = 400
GEN_MAX_SIDE = 30


def _randint(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _sample_strips(
    rng: np.random.Generator,
    count: int,
    size_range: tuple[int, int],
    gap_range: tuple[int, int],
    margin_range: tuple[int, int],
) -> tuple[list[tuple[int, int]], int]:
    strips: list[tuple[int, int]] = []
    pos = _randint(rng, margin_range)
    for idx in range(count):
        size = _randint(rng, size_range)
        start = pos
        end = start + size - 1
        strips.append((start, end))
        pos = end + 1
        if idx < count - 1:
            pos += _randint(rng, gap_range)
        else:
            pos += _randint(rng, margin_range)
    return strips, pos


def _pick_one(rng: np.random.Generator, cells: list[tuple[int, int]]) -> tuple[int, int]:
    return cells[int(rng.integers(0, len(cells)))]


def _focus_occupancy(rows: int, cols: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    focus_r = _randint(rng, (1, rows - 2))
    focus_c = _randint(rng, (1, cols - 2))
    occupied: set[tuple[int, int]] = {(focus_r, focus_c)}

    directions = list(DIRECTIONS)
    target_k = _randint(rng, (1, 4))
    chosen = rng.choice(len(directions), size=target_k, replace=False)
    blocked = {directions[int(idx)] for idx in np.asarray(chosen).tolist()}

    if "up" in blocked:
        occupied.add((_randint(rng, (0, focus_r - 1)), focus_c))
    if "down" in blocked:
        occupied.add((_randint(rng, (focus_r + 1, rows - 1)), focus_c))
    if "left" in blocked:
        occupied.add((focus_r, _randint(rng, (0, focus_c - 1))))
    if "right" in blocked:
        occupied.add((focus_r, _randint(rng, (focus_c + 1, cols - 1))))

    if rng.random() < 0.6:
        used_rows = {r for r, _c in occupied}
        used_cols = {c for _r, c in occupied}
        isolated = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in occupied and r not in used_rows and c not in used_cols
        ]
        if isolated:
            occupied.add(_pick_one(rng, isolated))

    extras = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in occupied and r != focus_r and c != focus_c
    ]
    rng.shuffle(extras)
    max_target = min(GEN_OBJECT_COUNT_RANGE[1], len(occupied) + len(extras))
    min_target = min(GEN_OBJECT_COUNT_RANGE[0], max_target)
    target_count = _randint(rng, (min_target, max_target))
    for cell in extras:
        if len(occupied) >= target_count:
            break
        occupied.add(cell)
    return occupied


def _random_occupancy(rows: int, cols: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    count = _randint(rng, (GEN_OBJECT_COUNT_RANGE[0], min(GEN_OBJECT_COUNT_RANGE[1], len(cells))))
    return set(cells[:count])


def _build_grid(
    rows: list[tuple[int, int]],
    cols: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    palette = np.arange(1, 10)
    rng.shuffle(palette)
    for color, (rr, cc) in zip(palette.tolist(), sorted(occupied_cells)):
        r0, r1 = rows[rr]
        c0, c1 = cols[cc]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _valid_sample(grid: np.ndarray) -> bool:
    bg, rects, infos, claims, conflict = _collect_growth(grid)
    if len(rects) < GEN_OBJECT_COUNT_RANGE[0]:
        return False
    if np.any(conflict):
        return False
    out = _paint_claims(grid, claims, conflict)
    if np.array_equal(out, grid):
        return False

    ks = [int(info["k"]) for info in infos]
    if len(set(ks)) < 2:
        return False
    if not any(0 < k < 4 for k in ks):
        return False
    if not any(len(info["shell"]) > 0 for info in infos):
        return False

    fg = int(np.count_nonzero(grid != bg))
    if fg * 2 >= grid.size:
        return False
    if int(np.count_nonzero(out != grid)) < 4:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count = _randint(rng, GEN_ROWS_RANGE)
    col_count = _randint(rng, GEN_COLS_RANGE)
    row_bands, h = _sample_strips(rng, row_count, GEN_BAND_SIZE_RANGE, GEN_GAP_RANGE, GEN_MARGIN_RANGE)
    col_lanes, w = _sample_strips(rng, col_count, GEN_LANE_SIZE_RANGE, GEN_GAP_RANGE, GEN_MARGIN_RANGE)
    if h > GEN_MAX_SIDE or w > GEN_MAX_SIDE:
        return None

    if rng.random() < 0.7:
        occupied_cells = _focus_occupancy(row_count, col_count, rng)
    else:
        occupied_cells = _random_occupancy(row_count, col_count, rng)

    if not (GEN_OBJECT_COUNT_RANGE[0] <= len(occupied_cells) <= 9):
        return None

    grid = _build_grid(row_bands, col_lanes, (h, w), occupied_cells, rng)
    if not _valid_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
