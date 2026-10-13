import random

import numpy as np


ARC_GRID_MAX = 30
BG = 0
COLOR_POOL = tuple(range(1, 10))

RING_COUNT_RANGE = (2, 5)
INNER_SIDE_RANGE = (3, 9)
STEP_OPTIONS = (2, 4)
OUTER_SIDE_LIMITS = {2: 20, 3: 17, 4: 15, 5: 13}

PALETTE_COL_RANGE = (0, 3)
EXTRA_MARGIN_MAX = 3
PLACEMENT_GAP = 1
PLACEMENT_ATTEMPTS = 200
SIZE_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
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


def _is_rect_ring(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[bool, tuple[int, int] | None]:
    if coords.size == 0:
        return False, None

    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1 or bg in colors:
        return False, None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, None

    expected = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected:
        return False, None

    pts = {tuple(map(int, cell)) for cell in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != on_border:
                return False, None

    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return False, None

    return True, (h, w)


def _draw_ring(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _render_nested_rings(sizes: list[tuple[int, int]], colors: list[int], bg: int) -> np.ndarray:
    outer_h, outer_w = sizes[0]
    out = np.full((outer_h, outer_w), bg, dtype=int)
    for (h, w), color in zip(sizes, colors):
        top = (outer_h - h) // 2
        left = (outer_w - w) // 2
        assert (outer_h - h) % 2 == 0
        assert (outer_w - w) % 2 == 0
        _draw_ring(out, top, left, h, w, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered hollow rings into one concentric nest colored by the left palette strip.

    Input structure:
    - Background is black.
    - The left side contains a one-column vertical palette strip: one non-background pixel per row, stacked contiguously from top to bottom.
    - One full background column separates that palette strip from the main area.
    - The main area contains several same-colored hollow rectangular rings scattered apart; each ring is a one-pixel-thick closed border with black interior.

    Rule:
    - Read the palette colors from top to bottom.
    - Detect every hollow rectangular ring in the main area and sort the rings from largest to smallest by size.
    - Build an output canvas equal to the largest ring's bounding box, then center each smaller ring inside it to form perfect concentric nesting.
    - Recolor the outermost ring with the first palette color, the next ring with the second palette color, and continue inward in palette order.

    Key insight:
    - The scattered grey rings are size-coded parts of one nested structure, while the true layer colors are hidden in the separate left-edge palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _dominant_color(g)
    non_bg = np.argwhere(g != bg)
    assert non_bg.size > 0

    palette_col = int(non_bg[:, 1].min())
    assert palette_col + 1 < w
    assert np.all(g[:, palette_col + 1] == bg)

    palette_rows = np.flatnonzero(g[:, palette_col] != bg)
    assert palette_rows.size > 0
    assert np.array_equal(palette_rows, np.arange(int(palette_rows[0]), int(palette_rows[-1]) + 1))
    palette_colors = [int(g[r, palette_col]) for r in palette_rows]

    main_start = palette_col + 2
    assert main_start < w
    region = g[:, main_start:]
    comps = _components(region != bg)
    assert comps

    rings: list[tuple[int, int]] = []
    for comp in comps:
        ok, size = _is_rect_ring(region, comp, bg)
        assert ok and size is not None
        rings.append(size)

    assert len(rings) == len(palette_colors)
    rings.sort(key=lambda size: (size[0] * size[1], size[0], size[1]), reverse=True)

    outer_h, outer_w = rings[0]
    for prev, cur in zip(rings, rings[1:]):
        assert cur[0] <= prev[0] - 2 and cur[1] <= prev[1] - 2
        assert (outer_h - cur[0]) % 2 == 0
        assert (outer_w - cur[1]) % 2 == 0

    return _render_nested_rings(rings, palette_colors, bg)


def _sample_same_parity(lo: int, hi: int, parity: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == parity]
    return random.choice(choices)


def _sample_ring_sizes(n_rings: int) -> list[tuple[int, int]] | None:
    side_limit = OUTER_SIDE_LIMITS[n_rings]
    for _ in range(SIZE_ATTEMPTS):
        parity_h = random.randint(0, 1)
        parity_w = random.randint(0, 1)
        inner_h = _sample_same_parity(INNER_SIDE_RANGE[0], min(INNER_SIDE_RANGE[1], side_limit), parity_h)
        inner_w = _sample_same_parity(INNER_SIDE_RANGE[0], min(INNER_SIDE_RANGE[1], side_limit), parity_w)

        sizes = [(inner_h, inner_w)]
        for _ in range(n_rings - 1):
            prev_h, prev_w = sizes[-1]
            sizes.append((prev_h + random.choice(STEP_OPTIONS), prev_w + random.choice(STEP_OPTIONS)))

        sizes = list(reversed(sizes))
        if sizes[0][0] > side_limit or sizes[0][1] > side_limit:
            continue
        if len({h * w for h, w in sizes}) != n_rings:
            continue
        return sizes
    return None


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _place_rings(
    canvas: np.ndarray,
    sizes: list[tuple[int, int]],
    ring_color: int,
    start_col: int,
) -> bool:
    occupied_boxes: list[tuple[int, int, int, int]] = []
    for h, w in sizes:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, ARC_GRID_MAX - h)
            left = random.randint(start_col, ARC_GRID_MAX - w)
            box = (top, left, top + h - 1, left + w - 1)
            padded = _expand_box(box, PLACEMENT_GAP)
            if any(_boxes_overlap(padded, other) for other in occupied_boxes):
                continue
            _draw_ring(canvas, top, left, h, w, ring_color)
            occupied_boxes.append(padded)
            placed = True
            break
        if not placed:
            return False
    return True


def _crop_with_random_margin(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_extra = random.randint(0, min(EXTRA_MARGIN_MAX, int(r0)))
    left_extra = random.randint(0, min(EXTRA_MARGIN_MAX, int(c0)))
    bottom_extra = random.randint(0, min(EXTRA_MARGIN_MAX, int(grid.shape[0] - 1 - r1)))
    right_extra = random.randint(0, min(EXTRA_MARGIN_MAX, int(grid.shape[1] - 1 - c1)))
    return grid[
        int(r0) - top_extra : int(r1) + bottom_extra + 1,
        int(c0) - left_extra : int(c1) + right_extra + 1,
    ]


def _gen_attempt() -> np.ndarray | None:
    n_rings = random.randint(*RING_COUNT_RANGE)
    sizes = _sample_ring_sizes(n_rings)
    if sizes is None:
        return None

    palette_colors = random.sample(COLOR_POOL, n_rings)
    remaining = [c for c in COLOR_POOL if c not in palette_colors]
    ring_color = random.choice(remaining)

    canvas = np.full((ARC_GRID_MAX, ARC_GRID_MAX), BG, dtype=int)
    palette_col = random.randint(*PALETTE_COL_RANGE)
    palette_row0 = random.randint(0, ARC_GRID_MAX - n_rings)
    for i, color in enumerate(palette_colors):
        canvas[palette_row0 + i, palette_col] = int(color)

    if palette_col + 2 >= ARC_GRID_MAX:
        return None
    if not _place_rings(canvas, sizes, ring_color, palette_col + 2):
        return None

    grid = _crop_with_random_margin(canvas, BG)
    if grid.ndim != 2 or max(grid.shape) > ARC_GRID_MAX:
        return None

    expected = _render_nested_rings(sizes, palette_colors, BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
