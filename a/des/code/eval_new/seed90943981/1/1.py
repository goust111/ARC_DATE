import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _read_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    for r in range(h - 1, -1, -1):
        cols = [c for c in range(w) if int(grid[r, c]) not in (BLACK, GREY)]
        if not cols:
            continue
        if any(cols[i + 1] - cols[i] == 1 for i in range(len(cols) - 1)):
            continue
        return r, [int(grid[r, c]) for c in cols]
    raise ValueError("legend row not found")


def _frame_boxes(grid: np.ndarray, legend_row: int) -> list[tuple[int, int, int, int]]:
    mask = (grid == GREY).copy()
    if 0 <= legend_row < grid.shape[0]:
        mask[legend_row, :] = False

    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components(mask):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(coords) != 2 * (h + w) - 4:
            continue

        patch = mask[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(patch, expected):
            boxes.append((r0, c0, r1, c1))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame using the legend color indexed by its interior dot count.

    Input structure:
    - Background is black.
    - Several one-pixel-thick grey rectangular frames enclose black interiors.
    - Each frame interior contains scattered single-pixel colored dots.
    - Near the bottom of the grid, one row forms a left-to-right color legend:
      isolated non-black pixels separated by black gaps.

    Rule:
    - Read the bottom legend row from left to right so its 1st, 2nd, 3rd, ...
      colors define a count-to-color lookup table.
    - For every grey rectangular frame, count the interior pixels that are
      neither black nor grey.
    - Replace the entire grey border of that frame with the legend color at
      the matching count index.
    - Keep the interior dots and the legend row unchanged.

    Key insight:
    The border color is not determined by the dots' colors themselves; only the
    number of colored interior pixels matters, and that count must be looked up
    in the separate bottom legend.
    """
    g = np.asarray(grid, dtype=int)
    legend_row, legend = _read_legend(g)
    out = g.copy()

    for r0, c0, r1, c1 in _frame_boxes(g, legend_row):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero((interior != BLACK) & (interior != GREY)))
        if not 1 <= count <= len(legend):
            raise ValueError("frame count is outside legend range")
        color = legend[count - 1]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    return out


def _spaced_positions(inner_h: int, inner_w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    choices = []
    for r_parity in (0, 1):
        for c_parity in (0, 1):
            cells = [
                (r, c)
                for r in range(r_parity, inner_h, 2)
                for c in range(c_parity, inner_w, 2)
            ]
            if len(cells) >= count:
                choices.append(cells)
    if not choices:
        return None
    cells = list(rng.choice(choices))
    rng.shuffle(cells)
    return cells[:count]


def _region_shapes(max_count: int) -> dict[int, list[tuple[int, int]]]:
    shapes: dict[int, list[tuple[int, int]]] = {}
    for count in range(1, max_count + 1):
        options: list[tuple[int, int]] = []
        for inner_h in range(2, 8):
            for inner_w in range(2, 9):
                if count > ((inner_h + 1) // 2) * ((inner_w + 1) // 2):
                    continue
                if inner_h * inner_w < count + 2:
                    continue
                options.append((inner_h + 2, inner_w + 2))
        shapes[count] = options
    return shapes


def _sample_counts(num_regions: int, legend_len: int, rng: random.Random) -> list[int]:
    counts = [rng.randint(1, legend_len) for _ in range(num_regions)]
    if num_regions >= 2 and rng.random() < 0.5:
        counts[0] = 1
    if num_regions >= 2 and rng.random() < 0.5:
        counts[-1] = legend_len
    rng.shuffle(counts)
    return counts


def _place_rectangles(
    grid: np.ndarray,
    counts: list[int],
    legend_colors: list[int],
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray] | None:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    occ[h - 2 :, :] = True
    shapes = _region_shapes(max(counts))

    regions: list[tuple[int, int, int, int, int]] = []
    for count in counts:
        oh, ow = rng.choice(shapes[count])
        placements = []
        for r0 in range(1, h - 2 - oh + 1):
            for c0 in range(1, w - ow):
                r1 = r0 + oh
                c1 = c0 + ow
                if occ[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1].any():
                    continue
                placements.append((r0, c0, oh, ow))
        if not placements:
            return None

        r0, c0, oh, ow = rng.choice(placements)
        grid[r0, c0 : c0 + ow] = GREY
        grid[r0 + oh - 1, c0 : c0 + ow] = GREY
        grid[r0 : r0 + oh, c0] = GREY
        grid[r0 : r0 + oh, c0 + ow - 1] = GREY

        inner_h, inner_w = oh - 2, ow - 2
        dots = _spaced_positions(inner_h, inner_w, count, rng)
        if dots is None:
            return None
        dot_colors = rng.sample(PALETTE_COLORS, count)
        for (dr, dc), color in zip(dots, dot_colors):
            grid[r0 + 1 + dr, c0 + 1 + dc] = color

        occ[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True
        regions.append((r0, c0, oh, ow, count))

    out = grid.copy()
    for r0, c0, oh, ow, count in regions:
        color = legend_colors[count - 1]
        out[r0, c0 : c0 + ow] = color
        out[r0 + oh - 1, c0 : c0 + ow] = color
        out[r0 : r0 + oh, c0] = color
        out[r0 : r0 + oh, c0 + ow - 1] = color
    return regions, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    legend_len = rng.randint(3, 6)
    num_regions = rng.randint(2, 5)
    min_width = max(12, 2 * legend_len + 3)
    h = rng.randint(12, 28)
    w = rng.randint(min_width, 30)

    legend_colors = rng.sample(PALETTE_COLORS, legend_len)
    grid = np.full((h, w), BLACK, dtype=int)

    span = 2 * legend_len - 1
    start = rng.randint(0, w - span)
    for idx, color in enumerate(legend_colors):
        grid[h - 1, start + 2 * idx] = color

    counts = _sample_counts(num_regions, legend_len, rng)
    placed = _place_rectangles(grid, counts, legend_colors, rng)
    if placed is None:
        return None
    _, expected = placed

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
    raise RuntimeError("failed to generate a valid input grid")
