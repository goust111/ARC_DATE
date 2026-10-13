import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
RING_COLORS = (BLUE, RED, GREEN, YELLOW)

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_ring_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    mask = (grid != BLACK) & (grid != ORANGE)
    candidates = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox_coords(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if int(coords.shape[0]) == area:
            candidates.append((area, (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected one filled concentric-layer block"

    _area, box = candidates[0]
    r0, c0, r1, c1 = box
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = panel.shape

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    depth = np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols))
    max_depth = int(depth.max())
    outer_to_inner = []
    for d in range(max_depth + 1):
        colors = set(map(int, np.unique(panel[depth == d])))
        assert len(colors) == 1 and BLACK not in colors and ORANGE not in colors
        outer_to_inner.append(next(iter(colors)))
    assert len(set(outer_to_inner)) == len(outer_to_inner)
    return box, list(reversed(outer_to_inner))


def _right_partitions(grid: np.ndarray, n_layers: int):
    orange = np.argwhere(grid == ORANGE)
    assert orange.size > 0, "expected orange data pixels on the right"
    r0, c0, r1, c1 = _bbox_coords(orange)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % n_layers == 0 and w % n_layers == 0
    part_h = h // n_layers
    part_w = w // n_layers
    return (r0, c0, r1, c1), part_h, part_w


def _paint_by_partition_order(
    grid: np.ndarray,
    *,
    reverse_colors: bool = False,
    scan_mode: str = "nonempty_row_major",
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    _ring_box, inner_to_outer = _extract_ring_info(out)
    if reverse_colors:
        inner_to_outer = list(reversed(inner_to_outer))

    n_layers = len(inner_to_outer)
    (r0, c0, _r1, _c1), part_h, part_w = _right_partitions(out, n_layers)

    partitions = []
    for qr in range(n_layers):
        for qc in range(n_layers):
            rs = slice(r0 + qr * part_h, r0 + (qr + 1) * part_h)
            cs = slice(c0 + qc * part_w, c0 + (qc + 1) * part_w)
            mask = out[rs, cs] == ORANGE
            if scan_mode == "all_row_major" or np.any(mask):
                partitions.append((qr, qc, rs, cs, mask))

    if scan_mode == "nonempty_col_major":
        partitions.sort(key=lambda item: (item[1], item[0]))

    for idx, (_qr, _qc, rs, cs, mask) in enumerate(partitions):
        if not np.any(mask):
            continue
        block = out[rs, cs]
        block[mask] = inner_to_outer[idx % n_layers]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor right-side orange pixels by projecting a left concentric layer order into an implicit partition grid.

    Input structure:
    - Black is the background.
    - The left side contains one filled rectangular block of concentric monochrome layers, each layer a distinct color.
    - After a black gap, the right side contains orange pixels scattered inside a larger black rectangle whose borders are only implied by the orange extent.

    Rule:
    - Detect the left concentric block and read its layer colors from inner to outer.
    - Let the number of layers be n. Use the orange bounding box on the right as an implicit canvas and split it into an n x n grid of equal rectangles.
    - Scan these rectangles in reading order, but count only the rectangles that actually contain orange pixels.
    - Recolor all orange pixels in the k-th nonempty rectangle to the k-th color of the inner-to-outer layer cycle, repeating cyclically.
    - Keep the left block, black gap, and black background unchanged.

    Key insight:
    - The left layers provide two hidden controls at once: their nesting order gives the legend, and their count determines the right-side partition lattice. Because only nonempty partitions are numbered, both horizontal and vertical partitioning affect the final colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _paint_by_partition_order(g)


def _build_left_panel(n_layers: int) -> tuple[np.ndarray, list[int]]:
    h = random.choice((2 * n_layers - 1, 2 * n_layers))
    w = random.choice((2 * n_layers - 1, 2 * n_layers, 2 * n_layers + 1))
    outer_to_inner = random.sample(RING_COLORS, n_layers)

    panel = np.zeros((h, w), dtype=int)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    depth = np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols))
    for d, color in enumerate(outer_to_inner):
        panel[depth == d] = color
    return panel, list(reversed(outer_to_inner))


def _occupied_mask(n_layers: int) -> np.ndarray | None:
    total = n_layers * n_layers
    target = random.randint(n_layers + 1, min(total - 1, 2 * n_layers + 2))
    mask = np.zeros((n_layers, n_layers), dtype=bool)
    corners = [(0, 0), (0, n_layers - 1), (n_layers - 1, 0), (n_layers - 1, n_layers - 1)]
    for r, c in corners:
        mask[r, c] = True

    cells = [(r, c) for r in range(n_layers) for c in range(n_layers) if not mask[r, c]]
    random.shuffle(cells)
    for r, c in cells:
        if int(mask.sum()) >= target:
            break
        mask[r, c] = True

    if int(mask.sum()) == total:
        return None
    if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
        return None
    if len({int(np.sum(row)) for row in mask}) == 1 and len({int(np.sum(col)) for col in mask.T}) == 1:
        return None
    return mask


def _random_cluster(part_h: int, part_w: int, forced: tuple[int, int] | None) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(part_h) for c in range(part_w)]
    if forced is not None:
        interior.remove(forced)

    start_choices = [(r, c) for r, c in interior if 0 < r < part_h - 1 and 0 < c < part_w - 1]
    if not start_choices:
        start_choices = interior[:]
    start = random.choice(start_choices)
    target = random.randint(2, min(5, part_h * part_w - (1 if forced is not None else 0) - 1))

    cluster = {start}
    frontier = [start]
    while len(cluster) < target and frontier:
        base = random.choice(frontier)
        grown = False
        nbrs = list(_neighbors4(*base))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            cell = (nr, nc)
            if 0 <= nr < part_h and 0 <= nc < part_w and cell in interior and cell not in cluster:
                cluster.add(cell)
                frontier.append(cell)
                grown = True
                break
        if not grown:
            frontier.remove(base)

    if forced is not None:
        cluster.add(forced)
        if len(cluster) < 2:
            nbrs = [
                (nr, nc)
                for nr, nc in _neighbors4(*forced)
                if 0 <= nr < part_h and 0 <= nc < part_w and (nr, nc) in interior
            ]
            if nbrs:
                cluster.add(random.choice(nbrs))
    return sorted(cluster)


def _build_right_region(n_layers: int, occupied: np.ndarray) -> np.ndarray:
    part_h = random.randint(3, 4)
    part_w = random.randint(3, 4)
    region = np.zeros((n_layers * part_h, n_layers * part_w), dtype=int)

    force_map = {
        (0, 0): (0, 0),
        (0, n_layers - 1): (0, part_w - 1),
        (n_layers - 1, 0): (part_h - 1, 0),
        (n_layers - 1, n_layers - 1): (part_h - 1, part_w - 1),
    }
    for qr in range(n_layers):
        for qc in range(n_layers):
            if not occupied[qr, qc]:
                continue
            forced = force_map.get((qr, qc))
            cluster = _random_cluster(part_h, part_w, forced)
            for rr, cc in cluster:
                region[qr * part_h + rr, qc * part_w + cc] = ORANGE
    return region


def _column_shortcut(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    _box, inner_to_outer = _extract_ring_info(out)
    n_layers = len(inner_to_outer)
    (r0, c0, _r1, _c1), part_h, part_w = _right_partitions(out, n_layers)
    for qr in range(n_layers):
        for qc in range(n_layers):
            rs = slice(r0 + qr * part_h, r0 + (qr + 1) * part_h)
            cs = slice(c0 + qc * part_w, c0 + (qc + 1) * part_w)
            mask = out[rs, cs] == ORANGE
            if np.any(mask):
                block = out[rs, cs]
                block[mask] = inner_to_outer[qc % n_layers]
    return out


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gap = 1

    h = top + max(left.shape[0], right.shape[0]) + bottom
    w = left_margin + left.shape[1] + gap + right.shape[1] + right_margin
    grid = np.zeros((h, w), dtype=int)

    grid[top : top + left.shape[0], left_margin : left_margin + left.shape[1]] = left
    rc0 = left_margin + left.shape[1] + gap
    grid[top : top + right.shape[0], rc0 : rc0 + right.shape[1]] = right
    return grid


def _valid_example(grid: np.ndarray) -> bool:
    expected = _paint_by_partition_order(grid)
    if np.array_equal(expected, grid):
        return False
    if np.array_equal(expected, _paint_by_partition_order(grid, reverse_colors=True)):
        return False
    if np.array_equal(expected, _paint_by_partition_order(grid, scan_mode="nonempty_col_major")):
        return False
    if np.array_equal(expected, _paint_by_partition_order(grid, scan_mode="all_row_major")):
        return False
    if np.array_equal(expected, _column_shortcut(grid)):
        return False

    right_box, _, _ = _right_partitions(grid, len(_extract_ring_info(grid)[1]))
    r0, c0, r1, c1 = right_box
    right = grid[r0 : r1 + 1, c0 : c1 + 1]
    orange_rows = np.any(right == ORANGE, axis=1)
    orange_cols = np.any(right == ORANGE, axis=0)
    if not (orange_rows[0] and orange_rows[-1] and orange_cols[0] and orange_cols[-1]):
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.choice((3, 4))
    left, _legend = _build_left_panel(n_layers)
    occupied = _occupied_mask(n_layers)
    if occupied is None:
        return None
    right = _build_right_region(n_layers, occupied)
    grid = _assemble_input(left, right)
    if max(grid.shape) > 30:
        return None
    if not _valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
