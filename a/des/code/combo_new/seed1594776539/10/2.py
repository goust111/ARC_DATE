import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 120
MAX_SHAPE_BOX = 5


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    rows = np.where(np.all(g == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one horizontal grey separator"
    sep = int(rows[0])

    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    visited[sep, :] = True
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color == BLACK:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            key = _shape_key(coords)
            top = max(r0 for r0, _ in coords) < sep
            bottom = min(r0 for r0, _ in coords) > sep
            assert top != bottom, "objects must lie entirely above or below the separator"
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "shape": key,
                    "is_top": top,
                }
            )

    assert objects and len(objects) % 2 == 0, "expected an even number of foreground objects"
    return sep, objects


def main(grid: np.ndarray) -> np.ndarray:
    """Pair translated-equal sprites across a horizontal divider and swap each pair's colors.

    Input structure:
    - The background is black.
    - A single full-width horizontal grey line divides the grid into upper and lower halves.
    - Each foreground object is a 4-connected monochrome sprite that lies fully in one half.
    - Every sprite shape appears exactly twice under translation only, once above the line and once below it.

    Rule:
    - Extract every foreground sprite except the grey divider line.
    - Group sprites by their translated shape, using exact 4-connected pixel sets with no rotation or reflection.
    - For each matched top/bottom pair, repaint the top sprite with the bottom sprite's original color and repaint the bottom sprite with the top sprite's original color.
    - Keep the background, divider line, and all sprite positions unchanged.

    Key insight:
    - The important relation is long-range shape matching across the grey divider; colors do not identify the pairs, but once the matching shapes are found the two colors are exchanged in place.
    """
    g = np.asarray(grid, dtype=int)
    _, objects = _extract_objects(g)

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for obj in objects:
        groups.setdefault(obj["shape"], []).append(obj)

    out = g.copy()
    for members in groups.values():
        assert len(members) == 2, "each shape must occur exactly twice"
        top_members = [obj for obj in members if obj["is_top"]]
        bottom_members = [obj for obj in members if not obj["is_top"]]
        assert len(top_members) == 1 and len(bottom_members) == 1, "each pair must span the divider"
        top_obj = top_members[0]
        bottom_obj = bottom_members[0]
        for r, c in top_obj["coords"]:
            out[r, c] = int(bottom_obj["color"])
        for r, c in bottom_obj["coords"]:
            out[r, c] = int(top_obj["color"])
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _shape_key(list(cells))


def _random_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    bbox_h = int(rng.integers(1, MAX_SHAPE_BOX + 1))
    bbox_w = int(rng.integers(1, MAX_SHAPE_BOX + 1))
    if bbox_h == 1 and bbox_w == 1:
        bbox_h = 1
        bbox_w = 2

    max_area = min(bbox_h * bbox_w, 9)
    min_area = 2 if max_area <= 3 else 3
    target_area = int(rng.integers(min_area, max_area + 1))

    seed = (int(rng.integers(bbox_h)), int(rng.integers(bbox_w)))
    cells = {seed}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        frontier.sort()
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)

    key = _normalize_shape(cells)
    h = 1 + max(r for r, _ in key)
    w = 1 + max(c for _, c in key)
    assert h <= MAX_SHAPE_BOX and w <= MAX_SHAPE_BOX
    return key


def _can_place(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
) -> bool:
    h, w = occupied.shape
    abs_cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
    for r, c in abs_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        rr0 = max(0, r - 1)
        rr1 = min(h, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + 2)
        if occupied[rr0:rr1, cc0:cc1].any():
            return False
    return True


def _paint_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
    color: int,
) -> None:
    for dr, dc in shape:
        r = r0 + dr
        c = c0 + dc
        grid[r, c] = color
        occupied[r, c] = True


def _place_in_band(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    row_lo: int,
    row_hi: int,
    color: int,
) -> bool:
    shape_h = 1 + max(r for r, _ in shape)
    shape_w = 1 + max(c for _, c in shape)
    col_hi = grid.shape[1] - shape_w
    if row_hi - row_lo < shape_h or col_hi < 0:
        return False

    for _ in range(PLACE_RETRIES):
        r0 = int(rng.integers(row_lo, row_hi - shape_h + 1))
        c0 = int(rng.integers(0, col_hi + 1))
        if _can_place(occupied, shape, r0, c0):
            _paint_shape(grid, occupied, shape, r0, c0, color)
            return True
    return False


def _make_sample(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(1, 5))
    top_h = int(rng.integers(5, 14))
    bottom_h = int(rng.integers(5, 14))
    height = top_h + 1 + bottom_h
    width = int(rng.integers(10, 27))
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    sep = top_h
    grid[sep, :] = GREY
    occupied = np.zeros_like(grid, dtype=bool)

    colors = list(rng.permutation(PALETTE))
    if 2 * pair_count > len(colors):
        return None

    shapes: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for _ in range(pair_count):
        for _ in range(80):
            shape = _random_shape(rng)
            if shape not in seen:
                seen.add(shape)
                shapes.append(shape)
                break
        else:
            return None

    order = list(rng.permutation(pair_count))
    for idx in order:
        top_color = int(colors.pop())
        bottom_color = int(colors.pop())
        shape = shapes[idx]
        top_ok = _place_in_band(rng, grid, occupied, shape, 0, sep, top_color)
        bottom_ok = _place_in_band(rng, grid, occupied, shape, sep + 1, height, bottom_color)
        if not (top_ok and bottom_ok):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _make_sample(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-shape color-swap sample")
