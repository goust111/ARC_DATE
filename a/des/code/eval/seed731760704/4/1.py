import numpy as np


BG = 0
WALL = 5
COLORS = tuple(c for c in range(1, 10) if c != WALL)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


TEXTURE_MASKS = (
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _has_hole(mask: np.ndarray) -> bool:
    padded = np.pad(mask, 1, constant_values=False)
    return len(_components(~padded)) > 1


def _local_component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _parse_legend_block(block: np.ndarray, wall_color: int):
    nz = block != BG
    if block.shape != (3, block.shape[1]) or not np.all(nz.any(axis=1)):
        return None

    full_cols = np.flatnonzero(np.count_nonzero(nz, axis=0) == 3)
    if len(full_cols) != 3 or not np.array_equal(full_cols, np.arange(full_cols[0], full_cols[0] + 3)):
        return None

    c0 = int(full_cols[0])
    texture = block[:, c0 : c0 + 3]
    if texture.shape != (3, 3) or np.any(texture == BG) or np.any(texture == wall_color):
        return None

    seed_mask = nz.copy()
    seed_mask[:, c0 : c0 + 3] = False
    seed_coords = np.argwhere(seed_mask)
    if len(seed_coords) != 1:
        return None

    sr, sc = map(int, seed_coords[0])
    seed_color = int(block[sr, sc])
    if seed_color in (BG, wall_color):
        return None
    return seed_color, texture.copy()


def _parse_legend(grid: np.ndarray, wall_color: int, bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if sub.shape[0] < 5 or sub.shape[1] < 6:
        return None
    if not (
        np.all(sub[0, :] == wall_color)
        and np.all(sub[-1, :] == wall_color)
        and np.all(sub[:, 0] == wall_color)
        and np.all(sub[:, -1] == wall_color)
    ):
        return None
    if np.any(sub[1:-1, 1:-1] == wall_color):
        return None

    inner = sub[1:-1, 1:-1]
    row_has_data = np.any(inner != BG, axis=1)
    groups = []
    start = None
    for idx, flag in enumerate(row_has_data):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            groups.append((start, idx))
            start = None
    if start is not None:
        groups.append((start, len(row_has_data)))
    if not groups:
        return None

    mapping: dict[int, np.ndarray] = {}
    for start, end in groups:
        if end - start != 3:
            return None
        parsed = _parse_legend_block(inner[start:end, :], wall_color)
        if parsed is None:
            return None
        seed_color, texture = parsed
        if seed_color in mapping:
            return None
        mapping[seed_color] = texture

    return mapping


def _find_legend(grid: np.ndarray):
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for comp in _components(grid == color):
            bbox = _bbox(comp)
            local = _local_component_mask(comp, bbox)
            if not (
                np.all(local[0, :])
                and np.all(local[-1, :])
                and np.all(local[:, 0])
                and np.all(local[:, -1])
                and not np.any(local[1:-1, 1:-1])
            ):
                continue
            mapping = _parse_legend(grid, color, bbox)
            if not mapping:
                continue
            r0, c0, r1, c1 = bbox
            score = (r1 + c1, r1, c1, len(mapping))
            candidates.append((score, color, bbox, mapping))

    assert candidates, "legend not found"
    _, wall_color, bbox, mapping = max(candidates, key=lambda item: item[0])
    return wall_color, bbox, mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    wall_color, legend_bbox, mapping = _find_legend(g)
    lr0, lc0, lr1, lc1 = legend_bbox

    passable = g != wall_color
    passable[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = g.copy()
    for comp in _components(passable):
        if _touches_border(comp, h, w):
            continue

        colors = {
            int(g[r, c])
            for r, c in comp
            if int(g[r, c]) not in (BG, wall_color)
        }
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        seed_color = next(iter(colors))
        assert seed_color in mapping, "seed color missing from legend"
        texture = mapping[seed_color]
        r0, c0, _, _ = _bbox(comp)
        for r, c in comp:
            out[r, c] = int(texture[(r - r0) % 3, (c - c0) % 3])

    return out


def _make_texture(rng: np.random.Generator) -> np.ndarray:
    mask = TEXTURE_MASKS[int(rng.integers(len(TEXTURE_MASKS)))]
    if rng.random() < 0.5:
        mask = ~mask
    a, b = map(int, rng.choice(COLORS, size=2, replace=False))
    return np.where(mask, a, b).astype(int)


def _make_arm_shape(rng: np.random.Generator) -> np.ndarray | None:
    core_h = int(rng.integers(3, 7))
    core_w = int(rng.integers(3, 7))
    ext = {side: int(rng.integers(0, 4)) for side in ("top", "bottom", "left", "right")}
    if not any(ext.values()):
        side = ("top", "bottom", "left", "right")[int(rng.integers(4))]
        ext[side] = int(rng.integers(1, 4))

    canvas = np.zeros((core_h + ext["top"] + ext["bottom"], core_w + ext["left"] + ext["right"]), dtype=bool)
    r0 = ext["top"]
    c0 = ext["left"]
    canvas[r0 : r0 + core_h, c0 : c0 + core_w] = True

    for side, length in ext.items():
        if length == 0:
            continue
        if side in ("top", "bottom"):
            max_span = core_w
            span = int(rng.integers(1, max_span + 1))
            if max_span >= 3 and span == max_span:
                span = int(rng.integers(1, max_span))
            start = int(rng.integers(0, core_w - span + 1))
            cc0 = c0 + start
            cc1 = cc0 + span
            if side == "top":
                canvas[r0 - length : r0, cc0:cc1] = True
            else:
                canvas[r0 + core_h : r0 + core_h + length, cc0:cc1] = True
        else:
            max_span = core_h
            span = int(rng.integers(1, max_span + 1))
            if max_span >= 3 and span == max_span:
                span = int(rng.integers(1, max_span))
            start = int(rng.integers(0, core_h - span + 1))
            rr0 = r0 + start
            rr1 = rr0 + span
            if side == "left":
                canvas[rr0:rr1, c0 - length : c0] = True
            else:
                canvas[rr0:rr1, c0 + core_w : c0 + core_w + length] = True

    shape = _trim(canvas)
    if shape.size == 0 or np.all(shape):
        return None
    return shape


def _make_shape_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        if rng.random() < 0.3:
            shape = np.ones((int(rng.integers(3, 7)), int(rng.integers(3, 7))), dtype=bool)
        else:
            shape = _make_arm_shape(rng)
            if shape is None:
                continue
        shape = _trim(shape)
        if shape.size == 0:
            continue
        area = int(shape.sum())
        if area < 8 or area > 28:
            continue
        if len(_components(shape)) != 1 or _has_hole(shape):
            continue
        return shape
    return None


def _make_region_patch(rng: np.random.Generator, seed_color: int) -> np.ndarray | None:
    shape = _make_shape_mask(rng)
    if shape is None:
        return None
    h, w = shape.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1:-1, 1:-1] = shape
    boundary = _dilate8(inner) & ~inner
    patch = np.zeros(inner.shape, dtype=int)
    patch[boundary] = WALL

    cells = np.argwhere(inner)
    sr, sc = map(int, cells[int(rng.integers(len(cells)))])
    patch[sr, sc] = seed_color
    return patch


def _legend_patch(entries: list[tuple[int, np.ndarray]]) -> np.ndarray:
    inner_h = 4 * len(entries) - 1
    inner = np.zeros((inner_h, 4), dtype=int)
    for idx, (seed_color, texture) in enumerate(entries):
        row0 = 4 * idx
        inner[row0 : row0 + 3, 1:4] = texture
        inner[row0 + 1, 0] = seed_color

    patch = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=int)
    patch[0, :] = WALL
    patch[-1, :] = WALL
    patch[:, 0] = WALL
    patch[:, -1] = WALL
    patch[1:-1, 1:-1] = inner
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = patch.shape
    return not blocked[top - 1 : top + h + 1, left - 1 : left + w + 1].any()


def _place_patch(grid: np.ndarray, blocked: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    mask = patch != BG
    region = grid[top : top + h, left : left + w]
    region[mask] = patch[mask]
    blocked[top - 1 : top + h + 1, left - 1 : left + w + 1] = True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_regions = int(rng.choice([2, 3, 4]))
    height = int(rng.integers(26, 35))
    width = int(rng.integers(28, 39))

    seed_colors = list(map(int, rng.choice(COLORS, size=n_regions, replace=False)))
    texture_ids = rng.choice(len(TEXTURE_MASKS), size=n_regions, replace=False)
    textures = []
    for idx in texture_ids:
        mask = TEXTURE_MASKS[int(idx)]
        if rng.random() < 0.5:
            mask = ~mask
        a, b = map(int, rng.choice(COLORS, size=2, replace=False))
        textures.append(np.where(mask, a, b).astype(int))

    entry_order = list(range(n_regions))
    rng.shuffle(entry_order)
    legend_entries = [(seed_colors[i], textures[i]) for i in entry_order]
    legend = _legend_patch(legend_entries)
    legend_h, legend_w = legend.shape
    if legend_h >= height - 2 or legend_w >= width - 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    lr0 = height - legend_h
    lc0 = width - legend_w
    grid[lr0:, lc0:] = legend

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    blocked[max(0, lr0 - 1) : min(height, height), max(0, lc0 - 1) : min(width, width)] = True

    patches = []
    for seed_color in seed_colors:
        patch = _make_region_patch(rng, seed_color)
        if patch is None:
            return None
        patches.append((seed_color, patch))
    patches.sort(key=lambda item: -int(np.count_nonzero(item[1])))

    for _, patch in patches:
        ph, pw = patch.shape
        row_max = height - ph - 1
        col_max = width - pw - 1
        if row_max <= 1 or col_max <= 1:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(rng.integers(1, row_max))
            left = int(rng.integers(1, col_max))
            if not _can_place(blocked, top, left, patch):
                continue
            _place_patch(grid, blocked, top, left, patch)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
