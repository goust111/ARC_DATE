import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _filled_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    return len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _ring_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if set(points) != expected:
        return None
    return r0, c0, r1, c1


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(cols) != 1:
        raise AssertionError("expected exactly one full-height grey separator column")
    return cols[0]


def _palette_colors(left: np.ndarray) -> list[int]:
    bands = []
    for comp in _components(left != BLACK):
        colors = {int(left[r, c]) for r, c in comp}
        if len(colors) != 1:
            raise AssertionError("palette band must be monochrome")
        if not _filled_rectangle(comp):
            raise AssertionError("palette band must be a solid rectangle")
        r0, c0, _, _ = _bbox(comp)
        bands.append((r0, c0, colors.pop()))
    if not bands:
        raise AssertionError("expected at least one palette band")
    bands.sort()
    return [color for _, _, color in bands]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each fenced region by looking up the palette stripe indexed by its hole count.

    Input structure:
    - The background is black, and one full-height grey column splits the grid into a left palette zone and a right puzzle zone.
    - The left zone contains several separated horizontal monochrome palette bars, ordered from top to bottom.
    - The right zone contains multiple one-pixel-thick maroon rectangular fences whose interiors are filled with one non-black color plus scattered black hole pixels.

    Rule:
    - Read the palette bar colors in top-to-bottom order so bar 1 is the top bar, bar 2 is the next bar, and so on.
    - For each maroon rectangular fence, count the number of black hole pixels strictly inside the fence.
    - Recolor every non-black interior pixel of that fence with the color of palette bar N, where N is the hole count; keep the black holes, maroon fence, grey separator, and left palette unchanged.

    Key insight:
    - The black holes inside each fenced region are not noise; their count is the visible index into the ordered palette on the left.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sep = _separator_col(g)
    palette = _palette_colors(g[:, :sep])
    out = g.copy()
    right = g[:, sep + 1 :]

    for comp in _components(right == MAROON):
        box = _ring_bbox(comp)
        if box is None:
            raise AssertionError("maroon component must be a rectangular ring")
        r0, c0, r1, c1 = box
        inner = g[r0 + 1 : r1, sep + 1 + c0 + 1 : sep + 1 + c1]
        hole_count = int(np.count_nonzero(inner == BLACK))
        if not (1 <= hole_count <= len(palette)):
            raise AssertionError("hole count must index an existing palette band")
        target = palette[hole_count - 1]
        repaint = inner != BLACK
        out_inner = out[r0 + 1 : r1, sep + 1 + c0 + 1 : sep + 1 + c1]
        out_inner[repaint] = target

    return out


def _sample_holes(inner_h: int, inner_w: int, hole_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    parity_classes = [
        [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == parity]
        for parity in (0, 1)
    ]
    rng.shuffle(parity_classes)
    for cells in parity_classes:
        if len(cells) >= hole_count:
            return rng.sample(cells, hole_count)
    return None


def _place_boxes(
    height: int, width: int, sizes: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        box_h, box_w = sizes[idx]
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(height, r0 + box_h + 1)
            pc1 = min(width, c0 + box_w + 1)
            if occupancy[pr0:pr1, pc0:pc1].any():
                continue
            occupancy[pr0:pr1, pc0:pc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [p for p in placements if p is not None]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    band_count = rng.randint(3, 4)
    left_width = rng.randint(3, 6)
    band_heights = [rng.randint(1, 2) for _ in range(band_count)]
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    palette_height = top_margin + sum(band_heights) + (band_count - 1) + bottom_margin

    grid_h = rng.randint(max(12, palette_height), 24)
    max_right_w = min(22, 30 - left_width - 1)
    if max_right_w < 10:
        return None
    right_width = rng.randint(10, max_right_w)

    color_order = list(PALETTE_POOL)
    rng.shuffle(color_order)
    palette_colors = color_order[:band_count]
    object_count = rng.randint(2, min(3, band_count))
    fill_colors = color_order[band_count : band_count + object_count]
    hole_counts = rng.sample(list(range(1, band_count + 1)), object_count)

    objects = []
    for fill_color, hole_count in zip(fill_colors, hole_counts):
        spec = None
        for _ in range(80):
            inner_h = rng.randint(3, 6)
            inner_w = rng.randint(3, 7)
            if inner_h + 2 > grid_h or inner_w + 2 > right_width:
                continue
            if inner_h * inner_w - hole_count < 2:
                continue
            holes = _sample_holes(inner_h, inner_w, hole_count, rng)
            if holes is None:
                continue
            spec = {
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "fill_color": fill_color,
                "hole_count": hole_count,
                "holes": holes,
            }
            break
        if spec is None:
            return None
        objects.append(spec)

    sizes = [(obj["outer_h"], obj["outer_w"]) for obj in objects]
    placements = _place_boxes(grid_h, right_width, sizes, rng)
    if placements is None:
        return None

    grid = np.zeros((grid_h, left_width + 1 + right_width), dtype=int)
    grid[:, left_width] = GREY

    row = top_margin
    for color, band_h in zip(palette_colors, band_heights):
        band_w = rng.randint(2, left_width)
        start_c = rng.randint(0, left_width - band_w)
        grid[row : row + band_h, start_c : start_c + band_w] = color
        row += band_h
        if row < grid_h:
            row += 1

    right_offset = left_width + 1
    for obj, (r0, c0) in zip(objects, placements):
        r1 = r0 + obj["outer_h"] - 1
        c1 = c0 + obj["outer_w"] - 1
        abs_c0 = right_offset + c0
        abs_c1 = right_offset + c1

        grid[r0, abs_c0 : abs_c1 + 1] = MAROON
        grid[r1, abs_c0 : abs_c1 + 1] = MAROON
        grid[r0 : r1 + 1, abs_c0] = MAROON
        grid[r0 : r1 + 1, abs_c1] = MAROON

        grid[r0 + 1 : r1, abs_c0 + 1 : abs_c1] = obj["fill_color"]
        for hr, hc in obj["holes"]:
            grid[r0 + 1 + hr, abs_c0 + 1 + hc] = BLACK

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
