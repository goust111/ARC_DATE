import numpy as np

from arc_utils import bbox


def _d4_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    last = n - 1
    return {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the missing rectangular patch by completing the square's full reflection symmetry.

    Input structure:
    - A square patterned grid has one solid rectangular region of black missing pixels.
    - Outside that missing rectangle, the pattern obeys the same value under all eight square symmetries.

    Rule:
    - Find the bounding box of the black missing region.
    - For each missing pixel, inspect its D4 symmetry orbit across horizontal, vertical, and diagonal reflections.
    - Copy the unique non-black value found elsewhere in that orbit into the corresponding output cell.

    Key insight:
    - The output is not the repaired full grid; it is exactly the hidden rectangle, and every hidden value is determined by its symmetric copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square grid"

    missing = g == 0
    missing_bbox = bbox(missing)
    assert missing_bbox is not None, "expected one black missing region"
    r0, c0, r1, c1 = missing_bbox
    patch_h = r1 - r0 + 1
    patch_w = c1 - c0 + 1
    assert int(missing.sum()) == patch_h * patch_w, "missing pixels must form a solid rectangle"

    out = np.zeros((patch_h, patch_w), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            values = {
                int(g[rr, cc])
                for rr, cc in _d4_orbit(r, c, height)
                if int(g[rr, cc]) != 0
            }
            assert len(values) == 1, "each missing pixel must have one symmetric value"
            out[r - r0, c - c0] = values.pop()

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and fundamental-domain coloring.
GRID_SIZE = 24
FUNDAMENTAL_SIZE = GRID_SIZE // 2
NONZERO_COLORS = np.arange(1, 10)
PALETTE_SIZE_RANGE = (7, 10)
NEIGHBOR_COLOR_PROB = 0.42
# Missing rectangle bounds and rejection retry budget.
RECT_SIZE_RANGE = (3, 9)
RECT_AREA_MAX = 42
MAX_GENERATION_ATTEMPTS = 300
RECT_DIMS = tuple(
    (h, w)
    for h in range(*RECT_SIZE_RANGE)
    for w in range(*RECT_SIZE_RANGE)
    if h * w <= RECT_AREA_MAX
)


def _gen_valid_rect_positions(h: int, w: int, n: int = GRID_SIZE) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(n - h + 1):
        for c0 in range(n - w + 1):
            rect = {
                (r, c)
                for r in range(r0, r0 + h)
                for c in range(c0, c0 + w)
            }
            if not any(_d4_orbit(r, c, n).issubset(rect) for r, c in rect):
                positions.append((r0, c0))
    return positions


def _gen_sample_missing_rect() -> tuple[int, int, int, int] | None:
    h, w = RECT_DIMS[int(np.random.randint(len(RECT_DIMS)))]
    positions = _gen_valid_rect_positions(h, w)
    if not positions:
        return None
    r0, c0 = positions[int(np.random.randint(len(positions)))]
    return r0, c0, h, w


def _gen_symmetric_pattern() -> np.ndarray:
    domain = [(r, c) for r in range(FUNDAMENTAL_SIZE) for c in range(r, FUNDAMENTAL_SIZE)]
    np.random.shuffle(domain)

    k = int(np.random.randint(*PALETTE_SIZE_RANGE))
    palette = np.random.choice(NONZERO_COLORS, size=k, replace=False)
    domain_color: dict[tuple[int, int], int] = {}

    seed_cells = domain[:k]
    seed_colors = palette.copy()
    np.random.shuffle(seed_colors)
    for cell, color in zip(seed_cells, seed_colors):
        domain_color[cell] = int(color)

    for r, c in sorted(domain):
        if (r, c) in domain_color:
            continue
        neighbor_colors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < FUNDAMENTAL_SIZE and nr <= nc < FUNDAMENTAL_SIZE and (nr, nc) in domain_color:
                    neighbor_colors.append(domain_color[(nr, nc)])
        if neighbor_colors and np.random.random() < NEIGHBOR_COLOR_PROB:
            color = int(np.random.choice(neighbor_colors))
        else:
            color = int(np.random.choice(palette))
        domain_color[(r, c)] = color

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for (r, c), color in domain_color.items():
        for rr, cc in _d4_orbit(r, c, GRID_SIZE):
            grid[rr, cc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    full = _gen_symmetric_pattern()
    rect = _gen_sample_missing_rect()
    if rect is None:
        return None
    r0, c0, h, w = rect

    out = full[r0 : r0 + h, c0 : c0 + w].copy()
    inp = full.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = 0

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e66aafb8 input")
