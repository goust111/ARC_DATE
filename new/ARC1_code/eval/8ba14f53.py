import numpy as np

from arc_utils import Color, find_components, paste
import random
from itertools import combinations


def _enclosed_bg_count(g: np.ndarray, comp: dict, bg: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((crop == bg) | (crop == color))

    empty = crop == bg
    h, w = crop.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return int(np.count_nonzero(empty & ~outside))


def main(grid: np.ndarray) -> np.ndarray:
    """Count each colored object's enclosed black pixels and pack those counts into a 3x3 summary.

    Input structure
    - A black background contains exactly two 4-connected monochrome foreground objects arranged left-to-right.
    - Each foreground object encloses one or more black pixels inside its bounding box.

    Rule
    - Count, for each object, the black pixels inside its bounding box that are not 4-connected to the bounding-box border through black pixels.
    - In left-to-right object order, allocate enough 3-wide output rows for that count and fill that many cells left-to-right with the object's color.

    Key insight
    - The output is not the hole shapes themselves; it is a compact row-packed count summary of the enclosed background area of each object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = Color.BACKGROUND
    assert np.any(g == bg)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    comps = sorted(comps, key=lambda comp: comp["bbox"][1])
    assert len(comps) == 2
    assert comps[0]["bbox"][3] < comps[1]["bbox"][1]

    out = np.full((3, 3), bg, dtype=int)
    out_row = 0
    for comp in comps:
        count = _enclosed_bg_count(g, comp, bg)
        assert count > 0
        rows_needed = (count + 2) // 3
        assert out_row + rows_needed <= out.shape[0]

        for k in range(count):
            out[out_row + k // out.shape[1], k % out.shape[1]] = int(comp["color"])
        out_row += rows_needed

    return out

# --------------- generator (o5x) ---------------

# Canvas/output geometry.
GRID_ROWS, GRID_COLS = 4, 9
OUTPUT_ROWS, OUTPUT_COLS = 3, 3

# Enclosed-count and loop-shape constraints.
BG_COLOR = Color.BACKGROUND
ENCLOSED_COUNT_MIN, ENCLOSED_COUNT_MAX = 1, 6
TOTAL_ENCLOSED_MAX = 7
LOOP_HEIGHTS, LOOP_WIDTHS = (3, 4), (3, 4, 5)
MAX_FILLED_INTERIOR = 2

# Palette, layout, mutation, and retry sampling.
COLOR_CHOICES = range(Color.BLUE, Color.MAROON + 1)
MIN_GAP_COLS = 1
IRREGULAR_CORNER_PROB = 0.35
CORNER_REMOVAL_COUNTS = (1, 2)
GENERATION_ATTEMPTS = 300

COUNT_PAIRS = [
    (a, b)
    for a in range(ENCLOSED_COUNT_MIN, ENCLOSED_COUNT_MAX + 1)
    for b in range(ENCLOSED_COUNT_MIN, ENCLOSED_COUNT_MAX + 1)
    if (a + OUTPUT_COLS - 1) // OUTPUT_COLS + (b + OUTPUT_COLS - 1) // OUTPUT_COLS
    <= OUTPUT_ROWS
    and a + b <= TOTAL_ENCLOSED_MAX
]

def _gen_pack_output(colors: tuple[int, int], counts: tuple[int, int]) -> np.ndarray:
    out = np.full((OUTPUT_ROWS, OUTPUT_COLS), BG_COLOR, dtype=int)
    out_row = 0
    for color, count in zip(colors, counts):
        offsets = np.arange(count)
        out[out_row + offsets // OUTPUT_COLS, offsets % OUTPUT_COLS] = color
        out_row += (count + OUTPUT_COLS - 1) // OUTPUT_COLS
    return out


def _gen_dim_options(count: int) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in LOOP_HEIGHTS
        for w in LOOP_WIDTHS
        if 0 <= (h - 2) * (w - 2) - count <= MAX_FILLED_INTERIOR
    ]


def _gen_patch_valid(patch: np.ndarray, color: int, count: int) -> bool:
    comps = find_components(patch, bg=BG_COLOR, connectivity=4, monochromatic=True)
    if len(comps) != 1 or int(comps[0]["color"]) != int(color):
        return False
    return _enclosed_bg_count(patch, comps[0], BG_COLOR) == count


def _gen_loop_patch(count: int, color: int, dims: tuple[int, int]) -> np.ndarray | None:
    h, w = dims
    patch = np.full((h, w), BG_COLOR, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    fill = len(interior) - count
    for r, c in random.sample(interior, fill):
        patch[r, c] = color

    if not _gen_patch_valid(patch, color, count):
        return None

    if random.random() < IRREGULAR_CORNER_PROB:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        variants = []
        for k in CORNER_REMOVAL_COUNTS:
            for removed in combinations(corners, k):
                variant = patch.copy()
                for r, c in removed:
                    variant[r, c] = BG_COLOR
                if _gen_patch_valid(variant, color, count):
                    variants.append(variant)
        if variants:
            patch = random.choice(variants)

    return patch


def _gen_place_pair(patch1: np.ndarray, patch2: np.ndarray, dims1: tuple[int, int], dims2: tuple[int, int]) -> np.ndarray:
    total_extra = GRID_COLS - dims1[1] - dims2[1]
    gap = random.randint(MIN_GAP_COLS, total_extra)
    side_extra = total_extra - gap
    left_pad = random.randint(0, side_extra)
    c1 = left_pad
    c2 = c1 + dims1[1] + gap
    r1 = random.randint(0, GRID_ROWS - dims1[0])
    r2 = random.randint(0, GRID_ROWS - dims2[0])

    grid = np.full((GRID_ROWS, GRID_COLS), BG_COLOR, dtype=int)
    paste(grid, patch1, r1, c1)
    paste(grid, patch2, r2, c2)
    return grid


def _gen_attempt() -> np.ndarray | None:
    enc1, enc2 = random.choice(COUNT_PAIRS)
    colors = tuple(random.sample(COLOR_CHOICES, 2))
    expected = _gen_pack_output(colors, (enc1, enc2))

    option_pairs = [
        (d1, d2)
        for d1 in _gen_dim_options(enc1)
        for d2 in _gen_dim_options(enc2)
        if d1[1] + d2[1] <= GRID_COLS - MIN_GAP_COLS
    ]
    if not option_pairs:
        return None

    dims1, dims2 = random.choice(option_pairs)
    patch1 = _gen_loop_patch(enc1, colors[0], dims1)
    patch2 = _gen_loop_patch(enc2, colors[1], dims2)
    if patch1 is None or patch2 is None:
        return None

    grid = _gen_place_pair(patch1, patch2, dims1, dims2)
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
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
