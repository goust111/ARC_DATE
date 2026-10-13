import numpy as np


BG = 0
FILL = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 300
DIRECTIONS = ("up", "down", "left", "right")

SHAPE_CATALOG = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _corner_majority(grid: np.ndarray) -> int:
    corners = np.array([grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]], dtype=int)
    values, counts = np.unique(corners, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(coords)


def _infer_bias_side(seed_mask: np.ndarray) -> str:
    coords = np.argwhere(seed_mask)
    center_r = (seed_mask.shape[0] - 1) / 2.0
    center_c = (seed_mask.shape[1] - 1) / 2.0
    centroid_r = float(coords[:, 0].mean())
    centroid_c = float(coords[:, 1].mean())
    dr = centroid_r - center_r
    dc = centroid_c - center_c
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"

    top_gap = int(coords[:, 0].min())
    bottom_gap = int(seed_mask.shape[0] - 1 - coords[:, 0].max())
    left_gap = int(coords[:, 1].min())
    right_gap = int(seed_mask.shape[1] - 1 - coords[:, 1].max())
    gaps = {
        "up": top_gap,
        "down": bottom_gap,
        "left": left_gap,
        "right": right_gap,
    }
    return min(gaps, key=gaps.get)


def _scaled_shadow(seed_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(seed_mask)
    r0, c0, r1, c1 = _bbox(coords)
    cropped = seed_mask[r0 : r1 + 1, c0 : c1 + 1]
    scale = int(coords.shape[0])
    block = np.ones((scale, scale), dtype=np.uint8)
    return np.kron(cropped.astype(np.uint8), block).astype(bool)


def _shadow_start(
    bbox: tuple[int, int, int, int], shadow_shape: tuple[int, int], direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    shadow_h, shadow_w = shadow_shape
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "right":
        return mid_r - shadow_h // 2, c1 + 1
    if direction == "left":
        return mid_r - shadow_h // 2, c0 - shadow_w
    if direction == "down":
        return r1 + 1, mid_c - shadow_w // 2
    return r0 - shadow_h, mid_c - shadow_w // 2


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _corner_majority(g)
    objects: list[dict] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if len(coords) < 2:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(*patch.shape)
        if not np.all(patch[border] == color):
            continue
        interior = patch[1:-1, 1:-1]
        if interior.size == 0:
            continue
        interior_values = set(map(int, np.unique(interior)))
        fill_values = interior_values - {color}
        if len(fill_values) != 1:
            continue
        fill = fill_values.pop()
        if np.any((interior != color) & (interior != fill)):
            continue
        seed_mask = interior == color
        seed_count = int(seed_mask.sum())
        if not 2 <= seed_count <= 4:
            continue
        if not _is_connected(seed_mask):
            continue
        objects.append(
            {
                "color": color,
                "fill": int(fill),
                "bbox": (r0, c0, r1, c1),
                "seed_mask": seed_mask,
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _corner_majority(g)
    out = g.copy()
    for obj in _extract_objects(g):
        seed_mask = obj["seed_mask"]
        direction = OPPOSITE[_infer_bias_side(seed_mask)]
        shadow = _scaled_shadow(seed_mask)
        start_r, start_c = _shadow_start(obj["bbox"], shadow.shape, direction)
        for rr0, cc0 in np.argwhere(shadow):
            rr = int(start_r + rr0)
            cc = int(start_c + cc0)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg and int(out[rr, cc]) == bg:
                out[rr, cc] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project an enlarged copy of each object's interior seed pattern out of the rectangle on the opposite side.

    Input structure:
    - The background is black.
    - Several separated rectangular objects appear on the grid.
    - Each object has a single-color rectangular border, a grey interior fill, and a small same-color seed pattern embedded inside the fill.
    - Each seed is a 2-4 pixel 4-connected shape and sits noticeably closer to one interior side than to the others.

    Rule:
    - For each rectangular object, isolate the interior same-color seed pattern.
    - Determine which side of the grey interior the seed is biased toward by comparing the seed's centroid with the rectangle center.
    - Project a shadow from the opposite outer side of the rectangle.
    - Scale the seed pattern by N in both axes, where N is the number of seed pixels, keep its orientation, and place it outside the rectangle so the shadow side is centered on that border's midpoint.
    - Paint the shadow in the object's border color, only onto background cells, and leave the original object unchanged.

    Key insight:
    - The inner colored pixels are not noise: their shape is copied, their count sets the scale factor, and their off-center placement reverses the shadow direction.
    """
    return _transform(np.asarray(grid, dtype=int))


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    mask = SHAPE_CATALOG[int(rng.integers(len(SHAPE_CATALOG)))]
    k = int(rng.integers(4))
    return np.rot90(mask, k=k).copy()


def _seed_offsets(limit: int, rng: np.random.Generator) -> list[int]:
    center = limit // 2
    candidates = [center - 1, center, center + 1]
    valid = sorted({x for x in candidates if 0 <= x <= limit})
    rng.shuffle(valid)
    return valid


def _build_seed_mask(
    interior_h: int, interior_w: int, shape: np.ndarray, bias: str, rng: np.random.Generator
) -> np.ndarray | None:
    sh, sw = shape.shape
    max_r = interior_h - sh
    max_c = interior_w - sw
    if max_r < 0 or max_c < 0:
        return None

    row_choices = _seed_offsets(max_r, rng)
    col_choices = _seed_offsets(max_c, rng)

    if bias == "left":
        side_choices = [c for c in range(min(2, max_c + 1)) if max_c - c >= 2]
        positions = [(r, c) for c in side_choices for r in row_choices]
    elif bias == "right":
        side_choices = [c for c in range(max(0, max_c - 1), max_c + 1) if c >= 2]
        positions = [(r, c) for c in reversed(side_choices) for r in row_choices]
    elif bias == "up":
        side_choices = [r for r in range(min(2, max_r + 1)) if max_r - r >= 2]
        positions = [(r, c) for r in side_choices for c in col_choices]
    else:
        side_choices = [r for r in range(max(0, max_r - 1), max_r + 1) if r >= 2]
        positions = [(r, c) for r in reversed(side_choices) for c in col_choices]

    rng.shuffle(positions)
    for top, left in positions:
        seed_mask = np.zeros((interior_h, interior_w), dtype=bool)
        seed_mask[top : top + sh, left : left + sw] = shape
        if _infer_bias_side(seed_mask) == bias:
            return seed_mask
    return None


def _make_object_patch(
    color: int, bias: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(100):
        shape = _random_shape(rng)
        sh, sw = shape.shape
        interior_h = int(rng.integers(max(5, sh + 3), min(9, sh + 6) + 1))
        interior_w = int(rng.integers(max(5, sw + 3), min(9, sw + 6) + 1))
        seed_mask = _build_seed_mask(interior_h, interior_w, shape, bias, rng)
        if seed_mask is None:
            continue
        patch = np.full((interior_h + 2, interior_w + 2), FILL, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        patch[1:-1, 1:-1][seed_mask] = color
        return patch, seed_mask
    return None


def _shadow_cells_for_patch(
    top: int, left: int, patch: np.ndarray, seed_mask: np.ndarray
) -> set[tuple[int, int]]:
    bbox = (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1)
    direction = OPPOSITE[_infer_bias_side(seed_mask)]
    shadow = _scaled_shadow(seed_mask)
    start_r, start_c = _shadow_start(bbox, shadow.shape, direction)
    cells = set()
    for rr0, cc0 in np.argwhere(shadow):
        cells.add((int(start_r + rr0), int(start_c + cc0)))
    return cells


def _actual_cells(top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in np.argwhere(patch != BG):
        cells.add((int(top + rr), int(left + cc)))
    return cells


def _touches_corner(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return any(cell in corners for cell in cells)


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_patch(
    grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, seed_mask: np.ndarray, rng: np.random.Generator
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_order = rng.permutation(h - ph + 1)
    col_order = rng.permutation(w - pw + 1)
    for top in row_order:
        for left in col_order:
            obj_cells = _actual_cells(int(top), int(left), patch)
            shadow_cells = _shadow_cells_for_patch(int(top), int(left), patch, seed_mask)
            if not shadow_cells:
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in shadow_cells):
                continue
            all_cells = obj_cells | shadow_cells
            if _touches_corner(all_cells, h, w):
                continue
            if any(bool(blocked[r, c]) for r, c in all_cells):
                continue
            for rr, cc in obj_cells:
                grid[rr, cc] = int(patch[rr - int(top), cc - int(left)])
            _mark_blocked(blocked, all_cells)
            return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.choice([2, 3]))
    h_low = 18 if count == 2 else 24
    w_low = 18 if count == 2 else 24
    h = int(rng.integers(h_low, 31))
    w = int(rng.integers(w_low, 31))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = [int(v) for v in rng.choice(PALETTE, size=count, replace=False)]
    biases = list(rng.choice(np.array(DIRECTIONS), size=count, replace=False))
    rng.shuffle(biases)

    for color, bias0 in zip(colors, biases):
        bias = str(bias0)
        made = _make_object_patch(color, bias, rng)
        if made is None:
            return None
        patch, seed_mask = made
        if not _place_patch(grid, blocked, patch, seed_mask, rng):
            return None

    objects = _extract_objects(grid)
    if len(objects) != count:
        return None
    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
