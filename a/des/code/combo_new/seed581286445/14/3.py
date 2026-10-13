import numpy as np


BG = 0
OBJ = 5
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _perimeter_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height - 1):
        coords.append((r, width - 1))
    if height > 1:
        for c in range(width - 1, -1, -1):
            coords.append((height - 1, c))
    if width > 1:
        for r in range(height - 2, 0, -1):
            coords.append((r, 0))
    return coords


def _read_palette(grid: np.ndarray) -> list[int]:
    values = [int(grid[r, c]) for r, c in _perimeter_coords(*grid.shape)]
    runs: list[int] = []
    for value in values:
        if not runs or runs[-1] != value:
            runs.append(value)
    if len(runs) > 1 and runs[0] == runs[-1]:
        runs.pop()
    return runs


def _find_objects(grid: np.ndarray) -> list[np.ndarray]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[np.ndarray] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == BG:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append(np.array(cells, dtype=int))
    return objects


def _depth(coords: np.ndarray, height: int, width: int) -> int:
    return min(
        min(int(r), int(c), int(height - 1 - r), int(width - 1 - c))
        for r, c in coords
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Use the ordered border palette to recolor interior objects by their depth from the edge.

    Input structure:
    - Black background fills the interior space outside the objects.
    - The outermost ring is split into distinct contiguous color bands arranged clockwise from the top-left corner.
    - Inside the border sit several disjoint 4-connected monochrome objects.

    Rule:
    - Read the outer ring clockwise and collapse consecutive equal cells into an ordered palette.
    - For each interior object, compute the minimum orthogonal distance from any of its cells to the grid edge.
    - Recolor depth d with palette color number d, cycling when d exceeds the palette length.
    - Leave the border palette unchanged.

    Key insight:
    - The edge is not decoration: its clockwise band order encodes a cyclic palette, and each object's nearest-edge layer selects its output color.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    palette = _read_palette(g)
    if not palette:
        return g.copy()

    out = g.copy()
    for coords in _find_objects(g):
        color = palette[(_depth(coords, height, width) - 1) % len(palette)]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _partition_perimeter(total: int, bands: int) -> list[int]:
    lengths = np.full(bands, 2, dtype=int)
    extra = int(total - 2 * bands)
    if extra > 0:
        lengths += RNG.multinomial(extra, np.full(bands, 1.0 / bands))
    return [int(x) for x in lengths]


def _paint_border(grid: np.ndarray, colors: list[int], lengths: list[int]) -> None:
    coords = _perimeter_coords(*grid.shape)
    assert sum(lengths) == len(coords)
    idx = 0
    for color, length in zip(colors, lengths):
        for _ in range(length):
            r, c = coords[idx]
            grid[r, c] = int(color)
            idx += 1


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    stack = [tuple(map(int, points[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (int(nr), int(nc))
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _make_mask(height: int, width: int) -> np.ndarray | None:
    full = np.ones((height, width), dtype=bool)
    min_target = max(1, int(np.ceil(height * width * 0.45)))
    if height > 1 and width > 1:
        min_target = max(min_target, 4)
    target = int(RNG.integers(min_target, height * width + 1))
    best = full.copy()

    for _ in range(80):
        mask = full.copy()
        cells = [(r, c) for r in range(height) for c in range(width)]
        RNG.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _touches_all_sides(trial) and _is_connected(trial):
                mask = trial
        if int(mask.sum()) == target and _touches_all_sides(mask) and _is_connected(mask):
            return mask
        if int(mask.sum()) < int(best.sum()) and _touches_all_sides(mask) and _is_connected(mask):
            best = mask
    return best if _touches_all_sides(best) and _is_connected(best) else None


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def _try_place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    depth: int,
) -> tuple[list[tuple[int, int]], int] | None:
    height, width = grid.shape
    inner_h = height - 2 * depth
    inner_w = width - 2 * depth
    if inner_h <= 0 or inner_w <= 0:
        return None

    h_max = min(5, inner_h)
    w_max = min(5, inner_w)
    for _ in range(120):
        h = int(RNG.integers(1, h_max + 1))
        w = int(RNG.integers(1, w_max + 1))
        mask = _make_mask(h, w)
        if mask is None:
            continue

        side_specs: list[tuple[str, int, int, int, int]] = []
        if width - depth - w >= depth:
            side_specs.append(("top", depth, depth, depth, width - depth - w))
            side_specs.append(("bottom", height - depth - h, depth, depth, width - depth - w))
        if height - depth - h >= depth:
            side_specs.append(("left", depth, depth, height - depth - h, depth))
            side_specs.append(("right", depth, width - depth - w, height - depth - h, depth))
        if not side_specs:
            continue

        side = side_specs[int(RNG.integers(len(side_specs)))]
        name, r0, c0, r_hi, c_hi = side
        if name in ("top", "bottom"):
            c0 = int(RNG.integers(c0, c_hi + 1))
        else:
            r0 = int(RNG.integers(r0, r_hi + 1))

        coords = [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]
        if any(blocked[r, c] for r, c in coords):
            continue

        for r, c in coords:
            grid[r, c] = OBJ
        _mark_blocked(blocked, coords)
        return coords, depth
    return None


def _expected_output(grid: np.ndarray, objects: list[tuple[list[tuple[int, int]], int]], palette: list[int]) -> np.ndarray:
    out = grid.copy()
    for coords, depth in objects:
        color = palette[(depth - 1) % len(palette)]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        out[rows, cols] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(10, 26))
    width = int(RNG.integers(10, 26))
    max_depth = min((height - 1) // 2, (width - 1) // 2)
    if max_depth < 2:
        return None

    want_wrap = bool(max_depth >= 4 and RNG.integers(0, 2))
    palette_cap = min(len(PALETTE_POOL), max(2, min(6, max_depth + 1)))
    if want_wrap:
        palette_cap = min(palette_cap, max_depth - 1)
        if palette_cap < 2:
            want_wrap = False

    if want_wrap:
        band_count = int(RNG.integers(2, palette_cap + 1))
    else:
        band_count = int(RNG.integers(2, min(len(PALETTE_POOL), 6) + 1))

    colors = list(map(int, RNG.choice(PALETTE_POOL, size=band_count, replace=False)))
    lengths = _partition_perimeter(2 * height + 2 * width - 4, band_count)

    grid = np.full((height, width), BG, dtype=int)
    _paint_border(grid, colors, lengths)

    depths: list[int] = []
    if want_wrap:
        depths.append(int(RNG.integers(band_count + 1, max_depth + 1)))
        depths.append(int(RNG.integers(1, min(band_count, max_depth) + 1)))
    else:
        top_depth = min(band_count, max_depth)
        depths.append(int(RNG.integers(1, top_depth + 1)))
        depths.append(int(RNG.integers(1, top_depth + 1)))

    target_count = int(RNG.integers(2, min(7, max_depth + 3)))
    while len(depths) < target_count:
        if want_wrap and RNG.random() < 0.35:
            depths.append(int(RNG.integers(band_count + 1, max_depth + 1)))
        else:
            limit = max_depth if want_wrap else min(band_count, max_depth)
            depths.append(int(RNG.integers(1, limit + 1)))

    if len(set(depths)) < 2:
        alt = max_depth if want_wrap else min(band_count, max_depth)
        if alt >= 2:
            depths[-1] = 1 if depths[-1] != 1 else alt

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    objects: list[tuple[list[tuple[int, int]], int]] = []
    for depth in sorted(depths, reverse=True):
        placed = _try_place_object(grid, blocked, depth)
        if placed is not None:
            objects.append(placed)

    if len(objects) < 2:
        return None

    actual_depths = [depth for _, depth in objects]
    if len(set(actual_depths)) < 2:
        return None
    if want_wrap and max(actual_depths) <= band_count:
        return None
    if not want_wrap and max(actual_depths) > band_count:
        return None

    expected = _expected_output(grid, objects, colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
