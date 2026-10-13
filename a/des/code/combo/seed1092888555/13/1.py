import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
REGION_COUNT_RANGE = (2, 5)
REGION_HEIGHT_RANGE = (3, 7)
REGION_WIDTH_RANGE = (3, 7)
GAP = 1


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_horizontal_stripes(block: np.ndarray, colors: tuple[int, int]) -> tuple[bool, Counter]:
    row_colors = []
    for r in range(block.shape[0]):
        row = block[r, :]
        if np.unique(row).size != 1:
            return False, Counter()
        row_colors.append(int(row[0]))
    if any(color not in colors for color in row_colors):
        return False, Counter()
    if len(set(row_colors)) != 2:
        return False, Counter()
    if any(row_colors[i] == row_colors[i - 1] for i in range(1, len(row_colors))):
        return False, Counter()
    counts = Counter(row_colors)
    if counts[colors[0]] == counts[colors[1]]:
        return False, Counter()
    return True, counts


def _is_vertical_stripes(block: np.ndarray, colors: tuple[int, int]) -> tuple[bool, Counter]:
    col_colors = []
    for c in range(block.shape[1]):
        col = block[:, c]
        if np.unique(col).size != 1:
            return False, Counter()
        col_colors.append(int(col[0]))
    if any(color not in colors for color in col_colors):
        return False, Counter()
    if len(set(col_colors)) != 2:
        return False, Counter()
    if any(col_colors[i] == col_colors[i - 1] for i in range(1, len(col_colors))):
        return False, Counter()
    counts = Counter(col_colors)
    if counts[colors[0]] == counts[colors[1]]:
        return False, Counter()
    return True, counts


def _stripe_region_info(block: np.ndarray) -> tuple[str, int] | None:
    colors_arr = np.unique(block)
    if colors_arr.size != 2:
        return None
    colors = (int(colors_arr[0]), int(colors_arr[1]))
    ok_h, counts_h = _is_horizontal_stripes(block, colors)
    ok_v, counts_v = _is_vertical_stripes(block, colors)
    if ok_h == ok_v:
        return None
    if ok_h:
        dominant = max(colors, key=lambda color: counts_h[color])
        return "horizontal", int(dominant)
    dominant = max(colors, key=lambda color: counts_v[color])
    return "vertical", int(dominant)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find alternating two-color striped rectangles and repaint each one with its
    dominant stripe color. Horizontal stripes are judged by row majority;
    vertical stripes are judged by column majority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components(g != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == BLACK):
            continue
        if coords.shape[0] != block.size:
            continue
        info = _stripe_region_info(block)
        if info is None:
            continue
        _, dominant = info
        out[r0 : r1 + 1, c0 : c1 + 1] = dominant
    return out


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _odd_in_range(bounds: tuple[int, int]) -> int:
    choices = [x for x in range(bounds[0], bounds[1] + 1) if x % 2 == 1]
    return random.choice(choices)


def _sample_size(orientation: str) -> tuple[int, int]:
    if orientation == "horizontal":
        return _odd_in_range(REGION_HEIGHT_RANGE), _randint(REGION_WIDTH_RANGE)
    return _randint(REGION_HEIGHT_RANGE), _odd_in_range(REGION_WIDTH_RANGE)


def _make_stripes(height: int, width: int, orientation: str, colors: tuple[int, int]) -> np.ndarray:
    dominant, other = colors
    start_with_dominant = random.choice([True, False])
    block = np.zeros((height, width), dtype=int)
    if orientation == "horizontal":
        for r in range(height):
            use_dominant = (r % 2 == 0) == start_with_dominant
            block[r, :] = dominant if use_dominant else other
    else:
        for c in range(width):
            use_dominant = (c % 2 == 0) == start_with_dominant
            block[:, c] = dominant if use_dominant else other
    return block


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r1 = r0 + height
    c1 = c0 + width
    rr0 = max(0, r0 - GAP)
    cc0 = max(0, c0 - GAP)
    rr1 = min(h, r1 + GAP)
    cc1 = min(w, c1 + GAP)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_regions(grid: np.ndarray, specs: list[tuple[int, int, str, tuple[int, int]]]) -> bool:
    occupied = np.zeros_like(grid, dtype=bool)
    gh, gw = grid.shape
    random.shuffle(specs)
    for height, width, orientation, colors in specs:
        positions = [
            (r0, c0)
            for r0 in range(gh - height + 1)
            for c0 in range(gw - width + 1)
            if _can_place(occupied, r0, c0, height, width)
        ]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        grid[r0 : r0 + height, c0 : c0 + width] = _make_stripes(height, width, orientation, colors)
        occupied[r0 : r0 + height, c0 : c0 + width] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    height = _randint(GRID_HEIGHT_RANGE)
    width = _randint(GRID_WIDTH_RANGE)
    grid = np.zeros((height, width), dtype=int)

    region_count = _randint(REGION_COUNT_RANGE)
    orientations = [random.choice(["horizontal", "vertical"]) for _ in range(region_count)]
    if region_count >= 2 and len(set(orientations)) == 1:
        orientations[0] = "horizontal"
        orientations[1] = "vertical"

    specs = []
    for orientation in orientations:
        rh, rw = _sample_size(orientation)
        colors = tuple(random.sample(PALETTE, 2))
        dominant_first = random.choice([True, False])
        dominant = colors[0] if dominant_first else colors[1]
        other = colors[1] if dominant_first else colors[0]
        specs.append((rh, rw, orientation, (dominant, other)))

    total_area = sum(h * w for h, w, _, _ in specs)
    if total_area > height * width * 0.55:
        return None
    if not _place_regions(grid, specs):
        return None

    transformed = main(grid)
    if np.array_equal(grid, transformed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-region input")
