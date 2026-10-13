import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

OBJECT_COLORS = (RED, GREEN, YELLOW, ORANGE, AZURE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _legend_corner_coords(shape: tuple[int, int], corner: str, rows: int) -> list[tuple[int, int]]:
    h, w = shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - rows, 0
    else:
        r0, c0 = h - rows, w - 2
    return [(r0 + dr, c0 + dc) for dr in range(rows) for dc in range(2)]


def _legend_moat_ok(grid: np.ndarray, corner: str, rows: int) -> bool:
    h, w = grid.shape
    if corner in {"tl", "tr"} and rows < h:
        cols = range(0, 2) if corner == "tl" else range(w - 2, w)
        if any(int(grid[rows, c]) != BG for c in cols):
            return False
    if corner in {"bl", "br"} and h - rows - 1 >= 0:
        cols = range(0, 2) if corner == "bl" else range(w - 2, w)
        if any(int(grid[h - rows - 1, c]) != BG for c in cols):
            return False
    if corner in {"tl", "bl"} and 2 < w:
        rs = range(0, rows) if corner == "tl" else range(h - rows, h)
        if any(int(grid[r, 2]) != BG for r in rs):
            return False
    if corner in {"tr", "br"} and w - 3 >= 0:
        rs = range(0, rows) if corner == "tr" else range(h - rows, h)
        if any(int(grid[r, w - 3]) != BG for r in rs):
            return False
    return True


def _find_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, int]] = []
    for corner in ("tl", "tr", "bl", "br"):
        for rows in (3, 2):
            coords = _legend_corner_coords((h, w), corner, rows)
            values = [int(grid[r, c]) for r, c in coords]
            if not all(v in OBJECT_COLORS for v in values):
                continue
            if not _legend_moat_ok(grid, corner, rows):
                continue
            candidates.append((corner, rows))
            break
    assert len(candidates) == 1, "expected exactly one corner legend"

    corner, rows = candidates[0]
    coords = _legend_corner_coords((h, w), corner, rows)
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        mask[r, c] = True

    c0 = min(c for _r, c in coords)
    legend: list[tuple[int, int]] = []
    for r in sorted({r for r, _c in coords}):
        legend.append((int(grid[r, c0]), int(grid[r, c0 + 1])))
    return legend, mask


def _find_pointer(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    grey = np.argwhere(grid == GREY)
    magenta = np.argwhere(grid == MAGENTA)
    assert len(grey) == 3, "expected one 3-pixel grey bar"
    assert len(magenta) == 1, "expected one magenta control pixel"

    grey_mask = np.zeros_like(grid, dtype=bool)
    for r, c in grey:
        grey_mask[int(r), int(c)] = True
    mr, mc = map(int, magenta[0])

    rows = sorted({int(r) for r, _c in grey})
    cols = sorted({int(c) for _r, c in grey})
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[0] + 3))
        center = (r, cols[1])
    else:
        assert len(cols) == 1
        c = cols[0]
        assert rows == list(range(rows[0], rows[0] + 3))
        center = (rows[1], c)

    dr = center[0] - mr
    dc = center[1] - mc
    assert (dr, dc) in DIRECTIONS, "control must be orthogonally adjacent to the bar center"
    return center, (dr, dc), grey_mask


def _components(grid: np.ndarray, reserved: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, np.ndarray | int]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or reserved[r, c]:
                continue
            color = int(grid[r, c])
            if color not in OBJECT_COLORS:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or reserved[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append({"color": color, "coords": np.array(coords, dtype=int)})
    return objects


def _hits_ray(coords: np.ndarray, center: tuple[int, int], direction: tuple[int, int]) -> bool:
    r0, c0 = center
    dr, dc = direction
    if dr != 0:
        return bool(np.any((coords[:, 1] == c0) & ((coords[:, 0] - r0) * dr > 0)))
    return bool(np.any((coords[:, 0] == r0) & ((coords[:, 1] - c0) * dc > 0)))


def _apply_chain(color: int, legend: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in legend:
        if current == int(src):
            current = int(dst)
    return current


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend, legend_mask = _find_legend(g)
    center, direction, grey_mask = _find_pointer(g)
    reserved = legend_mask | grey_mask | (g == MAGENTA) | (g == BG)
    objects = _components(g, reserved)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        if _hits_ray(coords, center, direction):
            out[coords[:, 0], coords[:, 1]] = _apply_chain(int(obj["color"]), legend)
    return out


def _mark_blocked(blocked: np.ndarray, cells: np.ndarray, pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r = int(r)
        c = int(c)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _can_place(cells: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    return bool(
        np.all((0 <= cells[:, 0]) & (cells[:, 0] < h) & (0 <= cells[:, 1]) & (cells[:, 1] < w))
        and not np.any(blocked[cells[:, 0], cells[:, 1]])
    )


def _random_polyomino(rng: np.random.Generator, size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        cells.add(tuple(frontier.pop()) if len(frontier) == 1 else tuple(list(frontier)[int(rng.integers(len(frontier)))]))

    arr = np.array(sorted(cells), dtype=int)
    variant = int(rng.integers(8))
    r = arr[:, 0].copy()
    c = arr[:, 1].copy()
    if variant == 0:
        rr, cc = r, c
    elif variant == 1:
        rr, cc = r, -c
    elif variant == 2:
        rr, cc = -r, c
    elif variant == 3:
        rr, cc = -r, -c
    elif variant == 4:
        rr, cc = c, r
    elif variant == 5:
        rr, cc = c, -r
    elif variant == 6:
        rr, cc = -c, r
    else:
        rr, cc = -c, -r
    out = np.column_stack((rr, cc))
    out -= out.min(axis=0)
    return out.astype(int)


def _pointer_cells(center: tuple[int, int], direction: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    r, c = center
    dr, dc = direction
    control = (r - dr, c - dc)
    if dr != 0:
        bar = np.array([(r, c - 1), (r, c), (r, c + 1)], dtype=int)
    else:
        bar = np.array([(r - 1, c), (r, c), (r + 1, c)], dtype=int)
    return bar, control


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _try_place_hit_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    ray = _ray_cells(grid.shape, center, direction)
    if not ray:
        return None
    for _ in range(120):
        shape = _random_polyomino(rng, int(rng.integers(3, 9)))
        anchor_order = rng.permutation(len(shape))
        ray_order = rng.permutation(len(ray))
        for anchor_idx in anchor_order:
            ar, ac = map(int, shape[anchor_idx])
            for ray_idx in ray_order:
                rr, cc = ray[ray_idx]
                coords = shape + np.array([rr - ar, cc - ac], dtype=int)
                if not _can_place(coords, blocked):
                    continue
                grid[coords[:, 0], coords[:, 1]] = int(color)
                _mark_blocked(blocked, coords)
                return coords
    return None


def _try_place_miss_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    h, w = grid.shape
    for _ in range(200):
        shape = _random_polyomino(rng, int(rng.integers(3, 9)))
        max_r = h - int(shape[:, 0].max()) - 1
        max_c = w - int(shape[:, 1].max()) - 1
        if max_r < 0 or max_c < 0:
            continue
        top = int(rng.integers(0, max_r + 1))
        left = int(rng.integers(0, max_c + 1))
        coords = shape + np.array([top, left], dtype=int)
        if not _can_place(coords, blocked):
            continue
        if _hits_ray(coords, center, direction):
            continue
        grid[coords[:, 0], coords[:, 1]] = int(color)
        _mark_blocked(blocked, coords)
        return coords
    return None


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    corner: str,
    rows: list[tuple[int, int]],
) -> None:
    coords = _legend_corner_coords(grid.shape, corner, len(rows))
    cells = np.array(coords, dtype=int)
    assert _can_place(cells, blocked)
    c0 = min(c for _r, c in coords)
    by_row = sorted({r for r, _c in coords})
    for row_idx, r in enumerate(by_row):
        left, right = rows[row_idx]
        grid[r, c0] = int(left)
        grid[r, c0 + 1] = int(right)
    _mark_blocked(blocked, cells)


def _choose_pointer_center(
    shape: tuple[int, int],
    blocked: np.ndarray,
    direction: tuple[int, int],
    min_ray_len: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            bar, control = _pointer_cells((r, c), direction)
            feature = np.vstack((bar, np.array([control], dtype=int)))
            if not _can_place(feature, blocked):
                continue
            if len(_ray_cells(shape, (r, c), direction)) < min_ray_len:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _expected_output(
    grid: np.ndarray,
    legend: list[tuple[int, int]],
    selected: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    out = grid.copy()
    for coords, color in selected:
        out[coords[:, 0], coords[:, 1]] = _apply_chain(int(color), legend)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 23))
    w = int(rng.integers(16, 23))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_rows_n = int(rng.choice(np.array([2, 3], dtype=int)))
    chain = [int(x) for x in rng.choice(np.array(OBJECT_COLORS, dtype=int), size=legend_rows_n + 1, replace=False)]
    legend = list(zip(chain[:-1], chain[1:]))
    corner = str(rng.choice(np.array(["tl", "tr", "bl", "br"], dtype=object)))
    _place_legend(grid, blocked, corner, legend)

    direction = tuple(map(int, DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]))
    center = _choose_pointer_center(grid.shape, blocked, direction, 2 * legend_rows_n + 3, rng)
    if center is None:
        return None
    bar, control = _pointer_cells(center, direction)
    if not _can_place(np.vstack((bar, np.array([control], dtype=int))), blocked):
        return None
    grid[bar[:, 0], bar[:, 1]] = GREY
    grid[control[0], control[1]] = MAGENTA
    _mark_blocked(blocked, bar)
    _mark_blocked(blocked, np.array([control], dtype=int))

    selected_colors = chain[:-1]
    selected_specs: list[tuple[np.ndarray, int]] = []
    hit_order = list(range(len(selected_colors)))
    rng.shuffle(hit_order)
    for idx in hit_order:
        color = selected_colors[idx]
        coords = _try_place_hit_object(grid, blocked, center, direction, color, rng)
        if coords is None:
            return None
        selected_specs.append((coords, color))

    duplicate_color = int(selected_colors[int(rng.integers(len(selected_colors)))])
    duplicate_coords = _try_place_miss_object(grid, blocked, center, direction, duplicate_color, rng)
    if duplicate_coords is None:
        return None

    total_objects = int(rng.integers(max(4, legend_rows_n + 2), 7))
    extras = total_objects - len(selected_colors) - 1
    for _ in range(extras):
        color = int(rng.choice(np.array(OBJECT_COLORS, dtype=int)))
        if _try_place_miss_object(grid, blocked, center, direction, color, rng) is None:
            return None

    expected = _expected_output(grid, legend, selected_specs)
    if np.array_equal(expected, grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
