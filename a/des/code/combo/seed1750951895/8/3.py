import numpy as np


BG = 0
FRAME = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_RETRIES = 400
RNG = np.random.default_rng()


def _frame_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[grid != BG], return_counts=True)
    assert vals.size > 0, "expected a non-background frame"
    return int(vals[np.argmax(counts)])


def _diamond_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0, "empty frame mask"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _diamond_geometry(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], int, int, int]:
    frame = _frame_color(grid)
    mask = grid == frame
    r0, c0, r1, c1 = _diamond_bbox(mask)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h % 2 == 1, "diamond bbox must be an odd square"
    radius = h // 2
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    expected = np.zeros_like(mask, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            expected[r, c] = abs(r - cr) + abs(c - cc) == radius
    assert np.array_equal(mask, expected), "frame must be a complete one-pixel diamond border"
    return frame, (r0, c0, r1, c1), radius, cr, cc


def _inside_diamond(r: int, c: int, cr: int, cc: int, radius: int) -> bool:
    return abs(r - cr) + abs(c - cc) < radius


def _marker_direction(
    r: int, c: int, bbox: tuple[int, int, int, int], cr: int, cc: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0 and c0 < c < c1 and c != cc:
        sides.append((1, 0))
    if r == r1 and c0 < c < c1 and c != cc:
        sides.append((-1, 0))
    if c == c0 and r0 < r < r1 and r != cr:
        sides.append((0, 1))
    if c == c1 and r0 < r < r1 and r != cr:
        sides.append((0, -1))
    assert len(sides) == 1, "marker must lie on exactly one bbox side"
    return sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot horizontal/vertical rays from colored side markers into a diamond frame.

    - The grey one-pixel border defines a true diamond interior.
    - Each non-grey marker on the diamond bounding-box sides emits along the inward
      axis of that side.
    - Rays start painting only after they have crossed into the interior.
    - Rays can overwrite earlier ray paint, but they stop at original interior
      blockers or when they leave the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frame, bbox, radius, cr, cc = _diamond_geometry(g)
    r0, c0, r1, c1 = bbox

    out = g.copy()
    original_blockers = np.zeros_like(g, dtype=bool)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if _inside_diamond(r, c, cr, cc, radius) and g[r, c] != BG:
                original_blockers[r, c] = True

    markers: list[tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = int(g[r, c])
            if v in (BG, frame):
                continue
            if r in (r0, r1) or c in (c0, c1):
                markers.append((r, c, v))

    assert markers, "expected at least one side marker"
    markers.sort(key=lambda item: (item[0], item[1]))

    h, w = g.shape
    for mr, mc, color in markers:
        dr, dc = _marker_direction(mr, mc, bbox, cr, cc)
        r, c = mr + dr, mc + dc
        entered = False
        while 0 <= r < h and 0 <= c < w:
            if not entered:
                if _inside_diamond(r, c, cr, cc, radius):
                    entered = True
                else:
                    r += dr
                    c += dc
                    continue
            if not _inside_diamond(r, c, cr, cc, radius):
                break
            if original_blockers[r, c]:
                break
            out[r, c] = color
            r += dr
            c += dc

    return out


def _draw_frame(grid: np.ndarray, cr: int, cc: int, radius: int) -> None:
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if abs(r - cr) + abs(c - cc) == radius:
                grid[r, c] = FRAME


def _inside_cells(cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - radius + 1, cr + radius):
        for c in range(cc - radius + 1, cc + radius):
            if _inside_diamond(r, c, cr, cc, radius):
                cells.append((r, c))
    return cells


def _side_candidates(
    bbox: tuple[int, int, int, int], cr: int, cc: int
) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1) if c != cc],
        "bottom": [(r1, c) for c in range(c0 + 1, c1) if c != cc],
        "left": [(r, c0) for r in range(r0 + 1, r1) if r != cr],
        "right": [(r, c1) for r in range(r0 + 1, r1) if r != cr],
    }


def _pick_side_markers(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_take = min(3, len(candidates))
    count = int(RNG.integers(1, max_take + 1))
    idx = RNG.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in np.sort(idx)]


def _ray_cells_from_marker(
    marker: tuple[int, int],
    bbox: tuple[int, int, int, int],
    cr: int,
    cc: int,
    radius: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = _marker_direction(marker[0], marker[1], bbox, cr, cc)
    h, w = shape
    r, c = marker[0] + dr, marker[1] + dc
    cells = []
    entered = False
    while 0 <= r < h and 0 <= c < w:
        if not entered:
            if _inside_diamond(r, c, cr, cc, radius):
                entered = True
            else:
                r += dr
                c += dc
                continue
        if not _inside_diamond(r, c, cr, cc, radius):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _marker_map(markers: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    return {(r, c): color for r, c, color in markers}


def _place_obstacles(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    cr: int,
    cc: int,
    radius: int,
) -> None:
    inside = _inside_cells(cr, cc, radius)
    ray_sets = [
        _ray_cells_from_marker((r, c), bbox, cr, cc, radius, grid.shape)
        for r, c, _ in markers
    ]
    intersection_cells = []
    counts: dict[tuple[int, int], int] = {}
    for cells in ray_sets:
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1
    for cell, n in counts.items():
        if n >= 2:
            intersection_cells.append(cell)

    obstacle_total = int(RNG.integers(0, min(5, max(1, len(inside) // 4)) + 1))
    if obstacle_total == 0:
        return

    obstacle_color_pool = PALETTE[PALETTE != FRAME]
    chosen: set[tuple[int, int]] = set()

    if intersection_cells and RNG.integers(0, 2) == 0:
        cell = intersection_cells[int(RNG.integers(0, len(intersection_cells)))]
        chosen.add(cell)

    shuffled = list(inside)
    RNG.shuffle(shuffled)
    for cell in shuffled:
        if len(chosen) >= obstacle_total:
            break
        if cell in chosen:
            continue
        chosen.add(cell)

    for r, c in chosen:
        grid[r, c] = int(RNG.choice(obstacle_color_pool))


def _has_crossing(out: np.ndarray, inp: np.ndarray) -> bool:
    painted = (out != inp) & (inp == BG)
    if not np.any(painted):
        return False
    vals, counts = np.unique(out[painted], return_counts=True)
    return bool(np.any(counts >= 2) and vals.size >= 2)


def _gen_attempt() -> np.ndarray | None:
    radius = int(RNG.integers(3, 7))
    side = 2 * radius + 1
    h = side + int(RNG.integers(2, 8))
    w = side + int(RNG.integers(2, 8))
    if h < side or w < side:
        return None

    top_pad = int(RNG.integers(0, h - side + 1))
    left_pad = int(RNG.integers(0, w - side + 1))
    cr = top_pad + radius
    cc = left_pad + radius
    bbox = (cr - radius, cc - radius, cr + radius, cc + radius)

    grid = np.zeros((h, w), dtype=int)
    _draw_frame(grid, cr, cc, radius)

    side_positions = _side_candidates(bbox, cr, cc)
    markers: list[tuple[int, int, int]] = []
    for side_name in ("top", "left", "bottom", "right"):
        positions = _pick_side_markers(side_positions[side_name])
        colors = RNG.choice(PALETTE, size=len(positions), replace=True)
        for (r, c), color in zip(positions, colors):
            grid[r, c] = int(color)
            markers.append((r, c, int(color)))

    _place_obstacles(grid, markers, bbox, cr, cc, radius)
    out = main(grid)

    changed = (out != grid)
    painted_count = int(np.count_nonzero(changed))
    if painted_count < max(4, radius):
        return None
    if len(np.unique(grid)) < 4:
        return None

    has_vertical = any(r in (bbox[0], bbox[2]) for r, _, _ in markers)
    has_horizontal = any(c in (bbox[1], bbox[3]) for _, c, _ in markers)
    if not (has_vertical and has_horizontal):
        return None

    marker_rows = {r for r, _, _ in markers}
    marker_cols = {c for _, c, _ in markers}
    if len(marker_rows) < 3 or len(marker_cols) < 3:
        return None

    if RNG.integers(0, 2) == 0 and not _has_crossing(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diamond-ray input")
