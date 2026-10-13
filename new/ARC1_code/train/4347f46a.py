import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow each solid foreground rectangle by replacing its non-border interior pixels with the background.

    Input structure
    - The background is the most common color.
    - Every foreground object is one monochrome 4-connected solid axis-aligned rectangle.
    - Each rectangle is at least three pixels tall and three pixels wide, so it has an interior.

    Rule
    - For each rectangle, preserve its perimeter pixels.
    - Set every pixel strictly inside that rectangle's bbox to the background color.

    Key insight
    - The output is not moving or matching objects; it converts each filled rectangle into its hollow frame in place.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    components = components_mask(g != bg, connectivity=4)
    assert components, "expected at least one foreground rectangle"

    for coords in components:
        values = g[coords[:, 0], coords[:, 1]]
        colors = np.unique(values)
        assert len(colors) == 1, "each foreground component must be monochrome"
        color = int(colors[0])

        r0, c0, r1, c1 = bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "rectangles must have a removable interior"
        assert len(coords) == height * width, "foreground component must fill its bbox"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"

        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    return out

# --------------- generator (o5x) ---------------

# Sampling ranges and colors.
CANVAS_ROWS, CANVAS_COLS = (8, 18), (7, 19)
RECT_COUNT, RECT_HEIGHT, RECT_WIDTH = (1, 5), (3, 7), (3, 8)
BG, RECT_COLORS = 0, np.arange(1, 10)
# Placement spacing and retry limit.
EDGE_BORDER, RECT_PADDING, RECT_PADDING_CONNECTIVITY, MAX_GENERATION_ATTEMPTS = 1, 1, 8, 300

def _gen_expected_output_4347f46a(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, h, w in rects:
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BG
    return out


def _place_rectangles(grid: np.ndarray, colors, sizes, order):
    rects: list[tuple[int, int, int, int]] = []
    for i in order:
        h, w = sizes[i]
        patch = np.full((h, w), colors[i], dtype=int)
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=EDGE_BORDER,
                padding=RECT_PADDING,
                padding_connectivity=RECT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(grid, patch, r0, c0, transparent=BG)
        rects.append((r0, c0, h, w))
    return rects


def _gen_attempt() -> np.ndarray | None:
    H = int(np.random.randint(CANVAS_ROWS[0], CANVAS_ROWS[1] + 1))
    W = int(np.random.randint(CANVAS_COLS[0], CANVAS_COLS[1] + 1))
    n_rects = int(np.random.randint(RECT_COUNT[0], RECT_COUNT[1] + 1))

    max_h = min(RECT_HEIGHT[1], H - 2 * EDGE_BORDER)
    max_w = min(RECT_WIDTH[1], W - 2 * EDGE_BORDER)
    if max_h < RECT_HEIGHT[0] or max_w < RECT_WIDTH[0]:
        return None

    colors = np.random.choice(RECT_COLORS, size=n_rects, replace=False).astype(int).tolist()
    sizes = [
        (int(np.random.randint(RECT_HEIGHT[0], max_h + 1)), int(np.random.randint(RECT_WIDTH[0], max_w + 1)))
        for _ in range(n_rects)
    ]
    order = sorted(range(n_rects), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    total_area = sum(h * w for h, w in sizes)
    max_area = max(h * w for h, w in sizes)
    if H * W - total_area <= max_area:
        return None

    grid = np.full((H, W), BG, dtype=int)
    rects = _place_rectangles(grid, colors, sizes, order)
    if rects is None:
        return None

    expected = _gen_expected_output_4347f46a(grid, rects)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 4347f46a input")
