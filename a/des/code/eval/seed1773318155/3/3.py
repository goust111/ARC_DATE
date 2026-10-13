import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

FRAME_SIDE_RANGE = (5, 10)
CANVAS_COUNT_RANGE = (2, 6)
GRID_MIN_DIM = 24
GRID_MAX_DIM = 35
LEGEND_SIZE = 3
LEGEND_GAP_CHOICES = (1, 2)
PLACEMENT_RETRIES = 250
GENERATION_RETRIES = 300
CLIP_PROBABILITY = 0.35


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rotation_from_quadrant(quadrant: int) -> int:
    return quadrant


def _rotate_offsets(offsets: list[tuple[int, int]], rotation: int) -> list[tuple[int, int]]:
    if rotation == 0:
        return [(dr, dc) for dr, dc in offsets]
    if rotation == 1:
        return [(dc, -dr) for dr, dc in offsets]
    if rotation == 2:
        return [(-dr, -dc) for dr, dc in offsets]
    if rotation == 3:
        return [(-dc, dr) for dr, dc in offsets]
    raise ValueError(f"unsupported rotation {rotation}")


def _seed_rotation(canvas: dict[str, int]) -> int:
    mid_r = canvas["r0"] + (canvas["side"] - 1) // 2
    mid_c = canvas["c0"] + (canvas["side"] - 1) // 2
    top = canvas["seed_r"] <= mid_r
    left = canvas["seed_c"] <= mid_c
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _find_canvases(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    canvases: list[dict[str, int]] = []
    for coords in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 5:
            continue
        if int(coords.shape[0]) != 4 * side_h - 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(patch[0, :] == GRAY)
            and np.all(patch[-1, :] == GRAY)
            and np.all(patch[:, 0] == GRAY)
            and np.all(patch[:, -1] == GRAY)
        ):
            continue
        if np.any(patch[1:-1, 1:-1] == GRAY):
            continue
        interior = patch[1:-1, 1:-1]
        seed_coords = np.argwhere((interior != bg) & (interior != GRAY))
        if seed_coords.shape[0] != 1:
            continue
        sr = int(seed_coords[0, 0] + r0 + 1)
        sc = int(seed_coords[0, 1] + c0 + 1)
        canvases.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "side": side_h,
                "seed_r": sr,
                "seed_c": sc,
                "color": int(grid[sr, sc]),
            }
        )
    assert canvases, "expected at least one framed canvas"
    return canvases


def _find_legends(grid: np.ndarray, bg: int, canvases: list[dict[str, int]]) -> dict[int, list[tuple[int, int]]]:
    in_canvas = np.zeros_like(grid, dtype=bool)
    for canvas in canvases:
        in_canvas[canvas["r0"] : canvas["r1"] + 1, canvas["c0"] : canvas["c1"] + 1] = True

    legends: dict[int, list[tuple[int, int]]] = {}
    color_mask = (grid != bg) & (grid != GRAY)
    for coords in _components(color_mask):
        if np.any(in_canvas[coords[:, 0], coords[:, 1]]):
            continue
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if int(coords.shape[0]) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 > LEGEND_SIZE or c1 - c0 + 1 > LEGEND_SIZE:
            continue
        offsets = sorted((int(r - r0), int(c - c0)) for r, c in coords)
        if (0, 0) not in offsets:
            continue
        assert color not in legends, "each legend color must be unique"
        legends[color] = offsets
    return legends


def _apply_stamp(
    out: np.ndarray,
    canvas: dict[str, int],
    color: int,
    offsets: list[tuple[int, int]],
    rotation: int,
) -> None:
    for dr, dc in _rotate_offsets(offsets, rotation):
        rr = canvas["seed_r"] + dr
        cc = canvas["seed_c"] + dc
        if canvas["r0"] < rr < canvas["r1"] and canvas["c0"] < cc < canvas["c1"]:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify multiple framed canvases, map each seed color to a same-colored legend stamp,
    rotate the stamp by the seed quadrant, and stamp it into the canvas using the legend's
    top-left colored cell as the fixed anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    canvases = _find_canvases(g, bg)
    legends = _find_legends(g, bg, canvases)

    out = g.copy()
    for canvas in canvases:
        color = canvas["color"]
        assert color in legends, "every seed color must have a matching legend"
        rotation = _seed_rotation(canvas)
        _apply_stamp(out, canvas, color, legends[color], rotation)
    return out


def _shape_array(offsets: list[tuple[int, int]]) -> np.ndarray:
    arr = np.zeros((LEGEND_SIZE, LEGEND_SIZE), dtype=bool)
    for r, c in offsets:
        arr[r, c] = True
    return arr


def _connected_cells(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    target = {tuple(map(int, p)) for p in coords}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == target


def _random_legend(rng: np.random.Generator) -> list[tuple[int, int]]:
    for _ in range(200):
        mask = np.zeros((LEGEND_SIZE, LEGEND_SIZE), dtype=bool)
        mask[0, 0] = True
        target_area = int(rng.integers(3, 7))
        while int(mask.sum()) < target_area:
            filled = np.argwhere(mask)
            fr, fc = map(int, filled[int(rng.integers(len(filled)))])
            neighbors = []
            for nr, nc in ((fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)):
                if 0 <= nr < LEGEND_SIZE and 0 <= nc < LEGEND_SIZE and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            nr, nc = neighbors[int(rng.integers(len(neighbors)))]
            mask[nr, nc] = True
        if int(mask.sum()) != target_area:
            continue
        if not _connected_cells(mask):
            continue
        if not mask[2, :].any() or not mask[:, 2].any():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return sorted((int(r), int(c)) for r, c in np.argwhere(mask))
    raise RuntimeError("failed to sample a valid legend shape")


def _balanced_quadrants(count: int, rng: np.random.Generator) -> list[int]:
    base = list(map(int, rng.permutation(4)))
    if count <= 4:
        return base[:count]
    extra = list(map(int, rng.integers(0, 4, size=count - 4)))
    quadrants = base + extra
    rng.shuffle(quadrants)
    return quadrants


def _legend_positions(
    height: int,
    width: int,
    count: int,
    orientation: str,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    gaps = [int(rng.choice(LEGEND_GAP_CHOICES)) for _ in range(max(0, count - 1))]
    span = LEGEND_SIZE * count + sum(gaps)
    if orientation in ("top", "bottom"):
        row = 1 if orientation == "top" else height - LEGEND_SIZE - 1
        max_start = width - span - 1
        if row < 0 or max_start < 1:
            return None
        col = int(rng.integers(1, max_start + 1))
        positions = []
        for i in range(count):
            positions.append((row, col))
            col += LEGEND_SIZE
            if i < len(gaps):
                col += gaps[i]
        return positions
    col = 1 if orientation == "left" else width - LEGEND_SIZE - 1
    max_start = height - span - 1
    if col < 0 or max_start < 1:
        return None
    row = int(rng.integers(1, max_start + 1))
    positions = []
    for i in range(count):
        positions.append((row, col))
        row += LEGEND_SIZE
        if i < len(gaps):
            row += gaps[i]
    return positions


def _rect_is_free(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _occupy(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _interior_positions(canvas_r: int, canvas_c: int, side: int, quadrant: int) -> list[tuple[int, int]]:
    mid_r = canvas_r + (side - 1) // 2
    mid_c = canvas_c + (side - 1) // 2
    if quadrant in (0, 1):
        rows = range(canvas_r + 1, mid_r + 1)
    else:
        rows = range(mid_r + 1, canvas_r + side - 1)
    if quadrant in (0, 3):
        cols = range(canvas_c + 1, mid_c + 1)
    else:
        cols = range(mid_c + 1, canvas_c + side - 1)
    return [(int(r), int(c)) for r in rows for c in cols]


def _would_clip(canvas_r: int, canvas_c: int, side: int, seed: tuple[int, int], offsets: list[tuple[int, int]]) -> bool:
    r0, c0 = canvas_r, canvas_c
    r1, c1 = canvas_r + side - 1, canvas_c + side - 1
    for dr, dc in offsets:
        rr = seed[0] + dr
        cc = seed[1] + dc
        if not (r0 < rr < r1 and c0 < cc < c1):
            return True
    return False


def _choose_seed(
    canvas_r: int,
    canvas_c: int,
    side: int,
    quadrant: int,
    legend_offsets: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    positions = _interior_positions(canvas_r, canvas_c, side, quadrant)
    rotated = _rotate_offsets(legend_offsets, _rotation_from_quadrant(quadrant))
    clip_positions = [pos for pos in positions if _would_clip(canvas_r, canvas_c, side, pos, rotated)]
    safe_positions = [pos for pos in positions if pos not in clip_positions]
    if clip_positions and rng.random() < CLIP_PROBABILITY:
        pool = clip_positions
    elif safe_positions:
        pool = safe_positions
    else:
        pool = positions
    return tuple(map(int, pool[int(rng.integers(len(pool)))]))


def _expected_output(grid: np.ndarray, canvases: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for canvas in canvases:
        _apply_stamp(
            out,
            {
                "r0": int(canvas["r0"]),
                "c0": int(canvas["c0"]),
                "r1": int(canvas["r1"]),
                "c1": int(canvas["c1"]),
                "side": int(canvas["side"]),
                "seed_r": int(canvas["seed_r"]),
                "seed_c": int(canvas["seed_c"]),
            },
            int(canvas["color"]),
            list(canvas["legend_offsets"]),
            int(canvas["quadrant"]),
        )
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(*CANVAS_COUNT_RANGE))
    sides = list(map(int, rng.integers(*FRAME_SIDE_RANGE, size=count)))
    colors = list(map(int, rng.choice(PALETTE, size=count, replace=False)))
    quadrants = _balanced_quadrants(count, rng)
    legends = [_random_legend(rng) for _ in range(count)]

    area_need = sum((side + 2) * (side + 2) for side in sides) + count * 25
    base_dim = max(GRID_MIN_DIM, int(np.ceil(np.sqrt(area_need))) - 1)
    if base_dim > GRID_MAX_DIM:
        return None

    orientation = str(rng.choice(("top", "bottom", "left", "right")))
    height = int(rng.integers(base_dim, GRID_MAX_DIM + 1))
    width = int(rng.integers(base_dim, GRID_MAX_DIM + 1))

    legend_positions = _legend_positions(height, width, count, orientation, rng)
    if legend_positions is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    items: list[dict[str, object]] = []
    for i in range(count):
        legend_r, legend_c = legend_positions[i]
        if not _rect_is_free(blocked, legend_r, legend_c, LEGEND_SIZE, LEGEND_SIZE, pad=1):
            return None
        _occupy(blocked, legend_r, legend_c, LEGEND_SIZE, LEGEND_SIZE, pad=1)
        for dr, dc in legends[i]:
            grid[legend_r + dr, legend_c + dc] = colors[i]
        items.append(
            {
                "color": colors[i],
                "legend_offsets": legends[i],
                "quadrant": quadrants[i],
                "side": sides[i],
            }
        )

    placement_order = sorted(range(count), key=lambda idx: -int(items[idx]["side"]))
    for idx in placement_order:
        side = int(items[idx]["side"])
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = int(rng.integers(1, height - side))
            c = int(rng.integers(1, width - side))
            if not _rect_is_free(blocked, r, c, side, side, pad=1):
                continue
            _occupy(blocked, r, c, side, side, pad=1)
            grid[r : r + side, c] = GRAY
            grid[r : r + side, c + side - 1] = GRAY
            grid[r, c : c + side] = GRAY
            grid[r + side - 1, c : c + side] = GRAY
            seed_r, seed_c = _choose_seed(
                r,
                c,
                side,
                int(items[idx]["quadrant"]),
                list(items[idx]["legend_offsets"]),
                rng,
            )
            grid[seed_r, seed_c] = int(items[idx]["color"])
            items[idx]["r0"] = r
            items[idx]["c0"] = c
            items[idx]["r1"] = r + side - 1
            items[idx]["c1"] = c + side - 1
            items[idx]["seed_r"] = seed_r
            items[idx]["seed_c"] = seed_c
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, items)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
