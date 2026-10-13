import numpy as np
from collections import deque


BG = 0
SEP = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PANEL_SIZE_RANGE = (5, 10)
SHAPE_KINDS = ("growth", "rect", "el", "tee", "stair")
MAX_GENERATION_ATTEMPTS = 300
BLOCK_ATTEMPTS = 80
OUTPUT_MIN_VISIBLE_COLORS = 3
MAX_OVERLAP_FRACTION = 0.32


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _expand_run(arr: np.ndarray, idx: int, value: int) -> tuple[int, int]:
    lo = idx
    hi = idx
    while lo > 0 and int(arr[lo - 1]) == value:
        lo -= 1
    while hi + 1 < arr.shape[0] and int(arr[hi + 1]) == value:
        hi += 1
    return lo, hi


def _random_growth_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    area_lo = max(3, (h * w + 2) // 3)
    area_hi = max(area_lo + 1, h * w)
    target = int(rng.integers(area_lo, area_hi + 1))
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True
    frontier = {cell for cell in _neighbors4(start[0], start[1], h, w)}
    while int(mask.sum()) < target and frontier:
        cells = list(frontier)
        pick = cells[int(rng.integers(len(cells)))]
        frontier.remove(pick)
        if mask[pick]:
            continue
        mask[pick] = True
        for nxt in _neighbors4(pick[0], pick[1], h, w):
            if not mask[nxt]:
                frontier.add(nxt)
    return _crop_mask(mask)


def _random_rect_mask(rng: np.random.Generator, side: int) -> np.ndarray:
    h = int(rng.integers(1, min(side, 5) + 1))
    w = int(rng.integers(1, min(side, 5) + 1))
    if h == 1 and w == 1:
        w = 2
    return np.ones((h, w), dtype=bool)


def _random_el_mask(rng: np.random.Generator, side: int) -> np.ndarray:
    h = int(rng.integers(2, min(side, 6) + 1))
    w = int(rng.integers(2, min(side, 6) + 1))
    arm_h = int(rng.integers(1, min(2, h) + 1))
    arm_w = int(rng.integers(1, min(2, w) + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:arm_h, :] = True
    mask[:, :arm_w] = True
    return mask


def _random_tee_mask(rng: np.random.Generator, side: int) -> np.ndarray:
    h = int(rng.integers(2, min(side, 6) + 1))
    w = int(rng.integers(3, min(side, 6) + 1))
    stem_w = int(rng.integers(1, min(2, w) + 1))
    stem_left = int(rng.integers(0, w - stem_w + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, stem_left : stem_left + stem_w] = True
    return mask


def _random_stair_mask(rng: np.random.Generator, side: int) -> np.ndarray:
    steps = int(rng.integers(2, min(side, 6) + 1))
    thick = int(rng.integers(1, 3))
    mask = np.zeros((steps + thick - 1, steps + thick - 1), dtype=bool)
    for i in range(steps):
        mask[i : i + thick, i : i + thick] = True
        if i + 1 < steps:
            mask[i + thick - 1, i : i + 2] = True
    return mask


def _sample_shape(rng: np.random.Generator, side: int) -> np.ndarray:
    kind = SHAPE_KINDS[int(rng.integers(len(SHAPE_KINDS)))]
    if kind == "growth":
        h = int(rng.integers(2, min(side, 6) + 1))
        w = int(rng.integers(2, min(side, 6) + 1))
        return _random_growth_mask(rng, h, w)
    if kind == "rect":
        return _random_rect_mask(rng, side)
    if kind == "el":
        return _random_el_mask(rng, side)
    if kind == "tee":
        return _random_tee_mask(rng, side)
    return _random_stair_mask(rng, side)


def _make_block(rng: np.random.Generator, side: int, color: int, quarter_turns: int) -> np.ndarray | None:
    for _ in range(BLOCK_ATTEMPTS):
        shape = _sample_shape(rng, side - 1)
        sh, sw = shape.shape
        if sh > side or sw > side:
            continue
        top = int(rng.integers(0, side - sh + 1))
        left = int(rng.integers(0, side - sw + 1))
        block = np.full((side, side), BG, dtype=int)
        block[top : top + sh, left : left + sw][shape] = color
        if _component_count(block != BG) != 1:
            continue
        if quarter_turns and np.array_equal(block, np.rot90(block, -quarter_turns)):
            continue
        return block
    return None


def _stamp_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    side = blocks[0].shape[0]
    out = np.full((side, side), BG, dtype=int)
    coverage = np.zeros((side, side), dtype=int)
    for k, block in enumerate(blocks):
        rotated = np.rot90(block, -k)
        mask = rotated != BG
        coverage[mask] += 1
        out[mask] = rotated[mask]
    return out, coverage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant stamp and overlay the rotated prints onto the central canvas.

    Input structure:
    - Black background surrounds a centered blank canvas.
    - Four corner quadrants each contain exactly one monochrome 4-connected object, and the four object colors are distinct.
    - Gray separator arms sit above, below, left, and right of the blank canvas, so the canvas is enclosed by a gray cross-like frame.

    Rule:
    - Locate the central blank canvas and the four corner quadrants around it.
    - Treat each quadrant as a same-sized stamp mask in quadrant coordinates.
    - Rotate the stamps by quadrant: top-left 0 degrees, top-right 90 degrees clockwise, bottom-right 180 degrees, bottom-left 270 degrees clockwise.
    - Stamp the rotated masks onto the blank canvas in the order top-left, top-right, bottom-right, bottom-left; later stamps overwrite earlier ones on overlaps.
    - Return only the central canvas after all four stamps have been applied.

    Key insight:
    - The corner objects are not copied to separate outputs; each corner is a full-frame stamp whose quadrant determines its rotation before all four are merged on one shared canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    mid_r = h // 2
    mid_c = w // 2
    bg = int(g[mid_r, mid_c])

    c0, c1 = _expand_run(g[mid_r, :], mid_c, bg)
    r0, r1 = _expand_run(g[:, mid_c], mid_r, bg)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w
    side = side_h

    assert 0 < r0 <= r1 < h - 1
    assert 0 < c0 <= c1 < w - 1
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg)

    sep_neighbors = {
        int(g[mid_r, c0 - 1]),
        int(g[mid_r, c1 + 1]),
        int(g[r0 - 1, mid_c]),
        int(g[r1 + 1, mid_c]),
    }
    assert len(sep_neighbors) == 1
    (sep,) = tuple(sep_neighbors)
    assert sep != bg

    assert np.all(g[:r0, c0 : c1 + 1] == sep)
    assert np.all(g[r1 + 1 :, c0 : c1 + 1] == sep)
    assert np.all(g[r0 : r1 + 1, :c0] == sep)
    assert np.all(g[r0 : r1 + 1, c1 + 1 :] == sep)

    quadrants = [
        g[:r0, :c0],
        g[:r0, c1 + 1 :],
        g[r1 + 1 :, c1 + 1 :],
        g[r1 + 1 :, :c0],
    ]

    out = np.full((side, side), bg, dtype=int)
    seen_colors = set()
    for k, block in enumerate(quadrants):
        assert block.shape == (side, side)
        mask = block != bg
        assert np.any(mask)
        assert _component_count(mask) == 1
        colors = set(map(int, block[mask].ravel()))
        assert len(colors) == 1
        (color,) = tuple(colors)
        assert color != sep
        seen_colors.add(color)

        rotated = np.rot90(block, -k)
        rot_mask = rotated != bg
        out[rot_mask] = rotated[rot_mask]

    assert len(seen_colors) == 4
    return out


# --------------- generator ---------------


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    side = int(rng.integers(*PANEL_SIZE_RANGE))
    grid = np.full((3 * side, 3 * side), BG, dtype=int)

    grid[:side, side : 2 * side] = SEP
    grid[side : 2 * side, :side] = SEP
    grid[side : 2 * side, 2 * side :] = SEP
    grid[2 * side :, side : 2 * side] = SEP

    colors = rng.choice(COLOR_POOL, size=4, replace=False)
    blocks = []
    for k, color in enumerate(colors):
        block = _make_block(rng, side, int(color), k)
        if block is None:
            return None
        blocks.append(block)

    grid[:side, :side] = blocks[0]
    grid[:side, 2 * side :] = blocks[1]
    grid[2 * side :, 2 * side :] = blocks[2]
    grid[2 * side :, :side] = blocks[3]

    out, coverage = _stamp_blocks(blocks)
    overlap_cells = int(np.count_nonzero(coverage > 1))
    if overlap_cells > int(np.ceil(MAX_OVERLAP_FRACTION * side * side)):
        return None

    visible_colors = set(map(int, out[out != BG].ravel()))
    if len(visible_colors) < OUTPUT_MIN_VISIBLE_COLORS:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-quadrant input")
