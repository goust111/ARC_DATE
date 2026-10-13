import numpy as np


BLACK = 0
AZURE = 8
OCCLUDER_COLOR = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _solid_rectangle_mask(grid: np.ndarray, color: int) -> np.ndarray | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch == color):
        return None
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _is_occluder_candidate(grid: np.ndarray, mask: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if mask[r, c]:
                if mask[r, mc]:
                    return False
                continue
            if mask[r, mc]:
                continue
            if int(grid[r, c]) != int(grid[r, mc]):
                return False
    return True


def _find_occluder_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    candidates: list[np.ndarray] = []
    for color in map(int, np.unique(g)):
        mask = _solid_rectangle_mask(g, color)
        if mask is None:
            continue
        if _is_occluder_candidate(g, mask):
            candidates.append(mask)
    assert len(candidates) == 1, "expected exactly one solid rectangular occluder"
    return candidates[0]


def _restore_pattern(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            mc = w - 1 - c
            assert not mask[r, mc], "occluder must not overlap its own mirror"
            out[r, c] = int(g[r, mc])
    return out


def _is_compact_plus(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    color = int(grid[r, c])
    if color == BLACK:
        return False
    orth = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    if any(int(grid[rr, cc]) != color for rr, cc in orth):
        return False
    corners = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
    if any(int(grid[rr, cc]) == color for rr, cc in corners):
        return False
    far = ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2))
    for rr, cc in far:
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color:
            return False
    return True


def _plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    return [(r, c) for r in range(h) for c in range(w) if _is_compact_plus(g, r, c)]


def _find_unique_plus(grid: np.ndarray) -> tuple[int, int]:
    centers = _plus_centers(grid)
    assert len(centers) == 1, "expected exactly one compact plus motif"
    return centers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a left-right symmetric pattern hidden behind one solid rectangle,
    then recolor the unique compact plus motif to azure.
    """
    g = np.asarray(grid, dtype=int)
    mask = _find_occluder_mask(g)
    restored = _restore_pattern(g, mask)
    center_r, center_c = _find_unique_plus(restored)
    out = restored.copy()
    for rr, cc in (
        (center_r, center_c),
        (center_r - 1, center_c),
        (center_r + 1, center_c),
        (center_r, center_c - 1),
        (center_r, center_c + 1),
    ):
        out[rr, cc] = AZURE
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_has_plus(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=int)
    for r, c in shape:
        mask[r, c] = 1
    return bool(_plus_centers(mask))


def _random_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        box_h = int(rng.integers(2, 5))
        box_w = int(rng.integers(2, 5))
        target = int(rng.integers(3, min(7, box_h * box_w) + 1))
        start = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
        cells = {start}
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in DIRS4
            if 0 <= start[0] + dr < box_h and 0 <= start[1] + dc < box_w
        }
        while frontier and len(cells) < target:
            choices = tuple(frontier)
            nxt = choices[int(rng.integers(0, len(choices)))]
            frontier.remove(nxt)
            cells.add(nxt)
            for dr, dc in DIRS4:
                rr = nxt[0] + dr
                cc = nxt[1] + dc
                if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if len(cells) != target:
            continue
        shape = _normalize_shape(cells)
        if _shape_has_plus(shape):
            continue
        return shape
    return None


def _mark_buffer(mask: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        mask[r0:r1, c0:c1] = True


def _try_place_pair(
    grid: np.ndarray,
    forbidden: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    mid = w // 2
    sh, sw = _shape_size(shape)
    if sw > mid:
        return False
    row_max = h - sh
    col_max = mid - sw
    if row_max < 0 or col_max < 0:
        return False

    for _ in range(150):
        top = int(rng.integers(0, row_max + 1))
        left = int(rng.integers(0, col_max + 1))
        left_cells = {(top + r, left + c) for r, c in shape}
        right_cells = {(rr, w - 1 - cc) for rr, cc in left_cells}
        cells = left_cells | right_cells
        if len(cells) != 2 * len(shape):
            continue
        if any(forbidden[r, c] for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        _mark_buffer(forbidden, cells, pad=1)
        return True
    return False


def _expected_output(restored: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    out = restored.copy()
    r, c = center
    for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        out[rr, cc] = AZURE
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(11, 20))
    w = int(rng.choice(np.arange(13, 24, 2)))
    mid = w // 2

    pair_count = int(rng.integers(2, 6))
    colors = list(map(int, rng.choice(PALETTE, size=pair_count + 1, replace=False)))
    plus_color = colors[0]
    pair_colors = colors[1:]

    full = np.full((h, w), BLACK, dtype=int)
    center_r = int(rng.integers(2, h - 2))
    plus_cells = {
        (center_r, mid),
        (center_r - 1, mid),
        (center_r + 1, mid),
        (center_r, mid - 1),
        (center_r, mid + 1),
    }

    forbidden = np.zeros((h, w), dtype=bool)
    row0 = max(0, center_r - 2)
    row1 = min(h, center_r + 3)
    col0 = max(0, mid - 4)
    col1 = min(w, mid + 5)
    forbidden[row0:row1, col0:col1] = True

    for color in pair_colors:
        shape = _random_shape(rng)
        if shape is None:
            return None
        if not _try_place_pair(full, forbidden, shape, color, rng):
            return None

    for r, c in plus_cells:
        full[r, c] = plus_color

    occ_h = int(rng.integers(1, min(4, h) + 1))
    top_min = max(0, center_r - occ_h + 1)
    top_max = min(center_r, h - occ_h)
    top = int(rng.integers(top_min, top_max + 1))
    occ_w = int(rng.integers(1, min(4, mid) + 1))
    left_side = bool(rng.integers(0, 2) == 0)
    if left_side:
        left = mid - occ_w
    else:
        left = mid + 1

    inp = full.copy()
    inp[top : top + occ_h, left : left + occ_w] = OCCLUDER_COLOR

    if _plus_centers(inp):
        return None

    try:
        mask = _find_occluder_mask(inp)
        restored = _restore_pattern(inp, mask)
        plus_center = _find_unique_plus(restored)
    except AssertionError:
        return None

    if not np.array_equal(restored, full):
        return None
    if plus_center != (center_r, mid):
        return None

    expected = _expected_output(full, plus_center)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
