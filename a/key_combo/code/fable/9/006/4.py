import random
from collections import Counter

import numpy as np


AXIS_MIN_SEP = 4
GRID_HEIGHT_RANGE = (17, 30)
GRID_WIDTH_RANGE = (17, 30)
BLOCK_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": len(coords),
                }
            )

    return comps


def _is_solid_3x3_block(grid: np.ndarray, comp: dict) -> bool:
    if int(comp["area"]) != 9:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    color = int(comp["color"])
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _extract_blocks(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg)
    assert comps, "expected at least one foreground component"

    blocks: list[dict] = []
    for comp in comps:
        assert _is_solid_3x3_block(g, comp), "every foreground component must be a solid 3x3 block"
        r0, c0, r1, c1 = comp["bbox"]
        blocks.append(
            {
                "color": int(comp["color"]),
                "center": (r0 + 1, c0 + 1),
                "bbox": (r0, c0, r1, c1),
            }
        )

    assert len(blocks) >= 2, "expected at least two blocks"
    rows = [center_r for center_r, _ in (block["center"] for block in blocks)]
    cols = [center_c for _, center_c in (block["center"] for block in blocks)]
    assert len(set(rows)) == len(rows), "block center rows must be unique"
    assert len(set(cols)) == len(cols), "block center cols must be unique"
    return bg, blocks


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _paint_intersections(grid: np.ndarray, bg: int, blocks: list[dict]) -> np.ndarray:
    out = grid.copy()

    for i, block_a in enumerate(blocks):
        ra, ca = block_a["center"]
        for block_b in blocks[i + 1 :]:
            rb, cb = block_b["center"]
            for point in ((ra, cb), (rb, ca)):
                pr, pc = point
                assert int(grid[pr, pc]) == bg, "every invisible-cross intersection must start on background"
                dist_a = _chebyshev(point, block_a["center"])
                dist_b = _chebyshev(point, block_b["center"])
                assert dist_a != dist_b, "the nearer block must be unique at each intersection"
                out[pr, pc] = block_a["color"] if dist_a < dist_b else block_b["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the hidden row-column crossings induced by each pair of solid blocks.

    Input structure:
    - The grid has a dominant background color and several monochrome solid 3x3 blocks.
    - Block center rows are all distinct, and block center columns are all distinct.
    - The row of any block and the column of any other block cross on background cells.

    Rule:
    - Keep every 3x3 block unchanged and draw no visible lines.
    - For each unordered pair of blocks A and B, consider the two crossings
      `(row(A), col(B))` and `(row(B), col(A))`.
    - Recolor each crossing with the color of the block whose center is closer in
      chessboard distance; the puzzle guarantees the two distances are never equal.

    Key insight:
    - The missing horizontal and vertical arms are only implicit: each block
      contributes a center row and a center column, and only their pairwise
      crossings become visible through a nearest-block color competition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, blocks = _extract_blocks(g)
    return _paint_intersections(g, bg, blocks)


def _max_axis_count(length: int, min_sep: int) -> int:
    return 1 + (length - 3) // min_sep


def _sample_axis_positions(length: int, count: int, min_sep: int) -> list[int] | None:
    candidates = list(range(1, length - 1))
    for _ in range(200):
        positions = sorted(random.sample(candidates, count))
        if all(b - a >= min_sep for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _layout_is_valid(centers: list[tuple[int, int]]) -> bool:
    has_row_smaller = False
    has_col_smaller = False

    for i, (ra, ca) in enumerate(centers):
        for rb, cb in centers[i + 1 :]:
            dr = abs(ra - rb)
            dc = abs(ca - cb)
            if dr == dc:
                return False
            if dr < dc:
                has_row_smaller = True
            else:
                has_col_smaller = True

            for pr, pc in ((ra, cb), (rb, ca)):
                for rc, cc in centers:
                    if abs(pr - rc) <= 1 and abs(pc - cc) <= 1:
                        return False

    return has_row_smaller and has_col_smaller


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    max_count = min(_max_axis_count(height, AXIS_MIN_SEP), _max_axis_count(width, AXIS_MIN_SEP), BLOCK_COUNT_RANGE[1])
    if max_count < BLOCK_COUNT_RANGE[0]:
        return None
    block_count = random.randint(BLOCK_COUNT_RANGE[0], max_count)

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], block_count)

    for _ in range(200):
        rows = _sample_axis_positions(height, block_count, AXIS_MIN_SEP)
        cols = _sample_axis_positions(width, block_count, AXIS_MIN_SEP)
        if rows is None or cols is None:
            return None

        shuffled_cols = cols[:]
        random.shuffle(shuffled_cols)
        centers = list(zip(rows, shuffled_cols))
        if not _layout_is_valid(centers):
            continue

        grid = np.full((height, width), bg, dtype=int)
        for color, (r, c) in zip(colors, centers):
            grid[r - 1 : r + 2, c - 1 : c + 2] = color

        detected_bg, blocks = _extract_blocks(grid)
        if detected_bg != bg:
            continue
        if len(blocks) != block_count:
            continue
        if not _layout_is_valid([tuple(block["center"]) for block in blocks]):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
