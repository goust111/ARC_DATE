import numpy as np


BACKGROUND = 8  # Azure
RED = 2
YELLOW = 4
ORANGE = 7
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground pixel by its number of same-color 4-neighbors.

    Same-color degree 4 -> background
    Same-color degree 3 -> keep original color
    Same-color degree 2 -> yellow
    Same-color degree 1 -> orange
    Same-color degree 0 -> red
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BACKGROUND:
                continue
            same = 0
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == color:
                    same += 1
            if same == 4:
                out[r, c] = BACKGROUND
            elif same == 3:
                out[r, c] = color
            elif same == 2:
                out[r, c] = YELLOW
            elif same == 1:
                out[r, c] = ORANGE
            else:
                out[r, c] = RED
    return out


# --------------- generator ---------------

GENERATION_RETRIES = 400
GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (2, 4)
PLACEMENT_TRIES = 250

INPUT_COLORS = (0, 1, 3, 5, 6, 9)

BLOB_ATTEMPTS = 100
SNAKE_ATTEMPTS = 120
BRANCH_ATTEMPTS = 40


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _shape_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    return int(max(rows) + 1), int(max(cols) + 1)


def _same_neighbor_counts(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {
        (int(r), int(c)): sum((int(r + dr), int(c + dc)) in cells for dr, dc in DIR4)
        for r, c in cells
    }


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (int(r + dr), int(c + dc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_transform(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(cells)
    if np.random.rand() < 0.5:
        pts = {(int(r), int(-c)) for r, c in pts}
    for _ in range(int(np.random.randint(4))):
        pts = {(int(c), int(-r)) for r, c in pts}
    return _normalize(pts)


def _weighted_choice(items: list[tuple[int, int]], weights: list[float]) -> tuple[int, int]:
    probs = np.asarray(weights, dtype=float)
    probs = probs / probs.sum()
    idx = int(np.random.choice(len(items), p=probs))
    return items[idx]


def _make_blob(area_low: int = 10, area_high: int = 16) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_ATTEMPTS):
        target = _randint(area_low, area_high)
        cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            weights: list[float] = []
            frontier_seen = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt in cells or nxt in frontier_seen:
                        continue
                    grown = set(cells)
                    grown.add(nxt)
                    norm = _normalize(grown)
                    hh, ww = _shape_size(norm)
                    if hh > 8 or ww > 8:
                        continue
                    frontier.append(nxt)
                    frontier_seen.add(nxt)
                    attach = sum((int(nxt[0] + rr), int(nxt[1] + cc)) in cells for rr, cc in DIR4)
                    weights.append(1.0 + 3.0 * attach)
            if not frontier:
                break
            cells.add(_weighted_choice(frontier, weights))

        cells = _normalize(cells)
        hh, ww = _shape_size(cells)
        degrees = set(_same_neighbor_counts(cells).values())
        if len(cells) < 8:
            continue
        if hh < 4 or ww < 4:
            continue
        if len(cells) == hh * ww:
            continue
        if not {2, 3, 4}.issubset(degrees):
            continue
        return _random_transform(cells)
    return None


def _make_snake() -> set[tuple[int, int]] | None:
    for _ in range(SNAKE_ATTEMPTS):
        target = _randint(4, 9)
        path = [(0, 0)]
        last_dir = None
        while len(path) < target:
            end_r, end_c = path[-1]
            occupied = set(path)
            options: list[tuple[int, int]] = []
            weights: list[float] = []
            for dr, dc in DIR4:
                nxt = (int(end_r + dr), int(end_c + dc))
                if nxt in occupied:
                    continue
                attach = sum((int(nxt[0] + rr), int(nxt[1] + cc)) in occupied for rr, cc in DIR4)
                if attach != 1:
                    continue
                grown = set(occupied)
                grown.add(nxt)
                norm = _normalize(grown)
                hh, ww = _shape_size(norm)
                if hh > 8 or ww > 8:
                    continue
                options.append(nxt)
                turn_bonus = 2.5 if last_dir is not None and (dr, dc) != last_dir else 1.0
                weights.append(turn_bonus)
            if not options:
                break
            nxt = _weighted_choice(options, weights)
            last_dir = (int(nxt[0] - end_r), int(nxt[1] - end_c))
            path.append(nxt)

        cells = _normalize(set(path))
        hh, ww = _shape_size(cells)
        degrees = list(_same_neighbor_counts(cells).values())
        if len(cells) != target:
            continue
        if hh < 2 or ww < 2:
            continue
        if degrees.count(1) != 2 or max(degrees) != 2:
            continue
        return _random_transform(cells)
    return None


def _make_branch() -> set[tuple[int, int]] | None:
    for _ in range(BRANCH_ATTEMPTS):
        kind = str(np.random.choice(["tee", "plus"]))
        if kind == "tee":
            arm = _randint(1, 2)
            stem = _randint(2, 4)
            cells = {(0, c) for c in range(2 * arm + 1)}
            for r in range(1, stem + 1):
                cells.add((r, arm))
        else:
            arm = _randint(1, 2)
            cells = {(arm, arm)}
            for k in range(1, arm + 1):
                cells.add((arm - k, arm))
                cells.add((arm + k, arm))
                cells.add((arm, arm - k))
                cells.add((arm, arm + k))
            if np.random.rand() < 0.7:
                tail = int(np.random.choice([0, 1, 2, 3]))
                extra = [(-arm - 1, 0), (arm + 1, 0), (0, -arm - 1), (0, arm + 1)][tail]
                cells.add((arm + extra[0], arm + extra[1]))
        cells = _normalize(cells)
        degrees = set(_same_neighbor_counts(cells).values())
        if len(cells) >= 5 and 3 in degrees:
            return _random_transform(cells)
    return None


def _shape_factory(kind: str) -> set[tuple[int, int]] | None:
    if kind == "blob":
        return _make_blob()
    if kind == "small_blob":
        return _make_blob(8, 12)
    if kind == "snake":
        return _make_snake()
    if kind == "branch":
        return _make_branch()
    raise ValueError(f"unknown shape kind: {kind}")


def _can_place(
    cells: set[tuple[int, int]],
    top: int,
    left: int,
    blocked: np.ndarray,
) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if bool(blocked[rr, cc]):
            return False
    return True


def _paint_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: set[tuple[int, int]],
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = grid.shape
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(color)
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _grid_neighbor_degrees(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    degrees = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND:
                continue
            same = 0
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color:
                    same += 1
            degrees.append(same)
    return degrees


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_HEIGHT_RANGE)
    w = _randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    target_objects = _randint(*OBJECT_COUNT_RANGE)
    kinds = ["blob", "snake"]
    extras = ["branch", "small_blob", "snake"]
    while len(kinds) < target_objects:
        kinds.append(str(np.random.choice(extras)))

    shapes = []
    for kind in kinds:
        shape = _shape_factory(kind)
        if shape is None:
            return None
        shapes.append(shape)

    shapes.sort(key=len, reverse=True)
    colors = list(np.random.choice(INPUT_COLORS, size=len(shapes), replace=False))

    for shape, color in zip(shapes, colors):
        sh, sw = _shape_size(shape)
        if sh >= h or sw >= w:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = _randint(0, h - sh)
            left = _randint(0, w - sw)
            if not _can_place(shape, top, left, blocked):
                continue
            _paint_shape(grid, blocked, shape, top, left, int(color))
            placed = True
            break
        if not placed:
            return None

    occupied = int(np.count_nonzero(grid != BACKGROUND))
    if occupied > (h * w) // 3:
        return None

    degrees = _grid_neighbor_degrees(grid)
    degree_set = set(degrees)

    # With 4-connected objects of area >= 4, degree 0 is unreachable.
    if not {1, 2, 3, 4}.issubset(degree_set):
        return None

    out = main(grid)
    if not np.any((grid != BACKGROUND) & (out == BACKGROUND)):
        return None
    if not np.any(out == YELLOW):
        return None
    if not np.any(out == ORANGE):
        return None
    if not np.any((grid != BACKGROUND) & (out == grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
