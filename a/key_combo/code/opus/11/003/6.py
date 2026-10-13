import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6
ORANGE = 7

GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return cols[0]


def _extract_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 2, "expected a non-empty grid with separator"

    sep = _find_separator_col(g)
    assert 1 <= sep < w - 1, "separator must split the grid into non-empty left/right zones"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] >= 2, "left zone must have a source column and travel space"
    assert right.shape[1] >= 4, "right zone must have a panel and swatch space"

    assert np.all(left[:, 0] == RED), "expected a red source pixel in every row of the leftmost column"
    assert not np.any(left[:, 0] == BLUE), "blockers cannot cover the source pixels"

    other_left = left[:, 1:]
    allowed_left = {BLACK, BLUE, RED}
    assert set(map(int, np.unique(left))).issubset(allowed_left), "unexpected colors in the left zone"
    assert set(map(int, np.unique(right))).issubset({BLACK, GREEN, PINK, ORANGE}), "unexpected colors in the right zone"

    survivor = (~np.any(other_left == BLUE, axis=1)).astype(int)
    live_rows = int(survivor.sum())
    dead_rows = int(h - live_rows)
    assert live_rows >= 2, "need at least two surviving rays"
    assert dead_rows >= 1, "need at least one blocked ray"

    green_mask = right == GREEN
    panel_box = _bbox(green_mask)
    assert panel_box is not None, "expected a green inner-mask panel"
    pr0, pc0, pr1, pc1 = panel_box
    assert pc0 == 0, "inner mask panel must be flush to the separator"
    inner = green_mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
    p, q = inner.shape
    assert 3 <= p <= 4 and 3 <= q <= 4, "inner mask should be about 3x3 or 4x4"
    assert p <= h, "outer mask height must be at least the inner mask height"
    assert np.any(inner), "inner mask cannot be empty"
    assert not np.all(inner), "inner mask cannot be fully filled"
    assert np.all(inner.any(axis=1)), "every inner-mask row must contain foreground"
    assert np.all(inner.any(axis=0)), "every inner-mask column must contain foreground"

    orange = np.argwhere(right == ORANGE)
    pink = np.argwhere(right == PINK)
    assert orange.shape[0] == 1 and pink.shape[0] == 1, "expected exactly one orange and one magenta swatch"

    out_h = h * p
    assert out_h <= 30 and q <= 30, "output would exceed ARC size limits"

    tile_live = np.where(inner, ORANGE, PINK).astype(int)
    assert np.any(tile_live == ORANGE) and np.any(tile_live == PINK), "live tile must visibly encode the mask"

    return {
        "grid": g,
        "height": h,
        "separator": sep,
        "survivor": survivor,
        "inner": inner,
        "p": p,
        "q": q,
    }


def _render_output(case: dict) -> np.ndarray:
    inner = case["inner"]
    blocks = []
    live_tile = np.where(inner, ORANGE, PINK).astype(int)
    dead_tile = np.full(inner.shape, PINK, dtype=int)
    for bit in case["survivor"]:
        blocks.append(live_tile if int(bit) == 1 else dead_tile)
    return np.vstack(blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a vertical Kronecker stack whose outer mask is determined by which left-edge rays survive to the separator.

    Input structure:
    - A black-background grid is split by one full-height grey separator column into left and right zones.
    - In the left zone, every row starts with a red source pixel in column 0, and some rows contain blue blocker cells farther right.
    - In the right zone, a green-on-black binary mask panel is flush to the separator, and there is one orange swatch plus one magenta swatch.

    Rule:
    - For each row, fire a horizontal ray rightward from the red source until it either hits a blue blocker or reaches the separator.
    - Record surviving rows as 1 and blocked rows as 0, forming an outer N x 1 binary vector.
    - Extract the green panel as the inner binary mask. For each surviving row, emit one recolored copy of that mask using orange for mask foreground and magenta for mask background; for each blocked row, emit a full magenta block of the same size.
    - Stack those blocks from top to bottom and output only the stacked result.

    Key insight:
    - The outer Kronecker mask is not drawn explicitly: it must be inferred from selective ray survival in the left half, then combined with the right-side panel mask and swatches.
    """
    case = _extract_case(grid)
    return _render_output(case)


def _sample_inner_mask(rng: np.random.Generator, p: int, q: int) -> np.ndarray:
    while True:
        mask = rng.random((p, q)) < rng.uniform(0.35, 0.75)
        if not np.any(mask) or np.all(mask):
            continue
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue
        if np.all(mask == mask[0, 0]):
            continue
        return mask


def _sample_survivor_vector(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        vec = (rng.random(n) < rng.uniform(0.4, 0.7)).astype(int)
        ones = int(vec.sum())
        if ones < 2 or ones >= n:
            continue
        transitions = int(np.count_nonzero(vec[:-1] != vec[1:]))
        if transitions == 0:
            continue
        if np.all(vec == np.r_[np.ones(ones, dtype=int), np.zeros(n - ones, dtype=int)]):
            continue
        if np.all(vec == np.r_[np.zeros(n - ones, dtype=int), np.ones(ones, dtype=int)]):
            continue
        return vec


def _place_blockers(rng: np.random.Generator, left: np.ndarray, survivor: np.ndarray) -> None:
    h, w = left.shape
    blocked = survivor == 0
    used_rows = np.zeros(h, dtype=bool)
    row = 0
    while row < h:
        if not blocked[row] or used_rows[row]:
            row += 1
            continue
        if row + 1 < h and blocked[row + 1] and not used_rows[row + 1] and rng.random() < 0.55:
            col = int(rng.integers(1, w - 1))
            left[row : row + 2, col] = BLUE
            used_rows[row : row + 2] = True
            row += 2
            continue
        if rng.random() < 0.5 and w >= 4:
            col = int(rng.integers(1, w - 2))
            left[row, col : col + 2] = BLUE
        else:
            col = int(rng.integers(1, w))
            left[row, col] = BLUE
        used_rows[row] = True
        row += 1
    assert np.all(np.any(left[:, 1:] == BLUE, axis=1) == blocked)


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    p = int(rng.integers(3, 5))
    q = int(rng.integers(3, 5))
    n_max = min(8, 30 // p)
    n = int(rng.integers(max(5, p), n_max + 1))
    left_w = int(rng.integers(8, 13))
    extra_right = int(rng.integers(2, 5))
    right_w = q + extra_right

    panel_top = int(rng.integers(0, n - p + 1))
    inner = _sample_inner_mask(rng, p, q)
    survivor = _sample_survivor_vector(rng, n)

    grid = np.full((n, left_w + 1 + right_w), BLACK, dtype=int)
    left = grid[:, :left_w]
    left[:, 0] = RED
    _place_blockers(rng, left, survivor)
    grid[:, left_w] = GREY

    right = grid[:, left_w + 1 :]
    right[panel_top : panel_top + p, :q][inner] = GREEN

    swatch_cells = [(r, c) for r in range(n) for c in range(q, right_w)]
    assert len(swatch_cells) >= 2
    chosen = rng.choice(len(swatch_cells), size=2, replace=False)
    order = [ORANGE, PINK] if rng.random() < 0.5 else [PINK, ORANGE]
    for idx, color in zip(chosen, order):
        r, c = swatch_cells[int(idx)]
        right[r, c] = color

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        case = _extract_case(grid)
        out = _render_output(case)
    except AssertionError:
        return False

    g = case["grid"]
    sep = case["separator"]
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    survivor = case["survivor"]
    inner = case["inner"]
    p, q = inner.shape

    assert out.shape == (case["height"] * p, q)
    assert max(out.shape) <= 30
    assert np.array_equal(np.any(left[:, 1:] == BLUE, axis=1).astype(int), 1 - survivor)

    live_tile = np.where(inner, ORANGE, PINK).astype(int)
    dead_tile = np.full((p, q), PINK, dtype=int)
    assert not np.array_equal(live_tile, dead_tile)

    for row, bit in enumerate(survivor):
        block = out[row * p : (row + 1) * p]
        assert np.array_equal(block, live_tile if int(bit) == 1 else dead_tile)

    green_rows = np.argwhere(right == GREEN)[:, 0]
    assert green_rows.min() >= 0 and green_rows.max() < case["height"]

    orange_pos = tuple(map(int, np.argwhere(right == ORANGE)[0]))
    pink_pos = tuple(map(int, np.argwhere(right == PINK)[0]))
    assert orange_pos != pink_pos

    if np.any(survivor == 0):
        toggled = survivor.copy()
        toggled[np.flatnonzero(survivor == 0)[0]] = 1
        changed = np.vstack([live_tile if int(bit) == 1 else dead_tile for bit in toggled])
        assert not np.array_equal(changed, out), "outer mask must affect the result"

    flipped_inner = inner.copy()
    flip_r, flip_c = next((r, c) for r in range(p) for c in range(q) if inner[r, c] == 0)
    flipped_inner[flip_r, flip_c] = True
    alt_live = np.where(flipped_inner, ORANGE, PINK)
    alt_out = np.vstack([alt_live if int(bit) == 1 else dead_tile for bit in survivor])
    assert not np.array_equal(alt_out, out), "inner mask must affect the result"

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _build_candidate(rng)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
