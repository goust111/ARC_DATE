"""ARC task 4: cyclic geodesic rings around red walls."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


RED = 2
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    """Find the unique corner-anchored 3-by-N grey frame and read its row."""
    h, w = grid.shape
    found: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for width in range(5, min(9, w) + 1):
        for r0 in (0, h - 3):
            for c0 in (0, w - width):
                patch = grid[r0 : r0 + 3, c0 : c0 + width]
                border = np.zeros(patch.shape, dtype=bool)
                border[0, :] = border[-1, :] = True
                border[:, 0] = border[:, -1] = True
                if not np.all(patch[border] == GREY):
                    continue
                palette = [int(x) for x in patch[1, 1:-1]]
                if not (3 <= len(palette) <= 7):
                    continue
                if len(set(palette)) != len(palette):
                    continue
                if any(x in (RED, GREY) for x in palette):
                    continue
                found.append(((r0, c0, r0 + 2, c0 + width - 1), palette))
    assert len(found) == 1, "expected one unambiguous corner legend frame"
    return found[0]


def _distance_map(shape: tuple[int, int], source: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    """Four-connected BFS distances, using the shared ARC (row, column) convention."""
    h, w = shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = source
    assert not blocked[sr, sc]
    dist[sr, sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Color cyclic geodesic rings from a lone marker while routing around red walls.

    Input structure:
    - The most common color is the background; red pixels form wall segments, including
      at least one wall-enclosed region.
    - A unique grey 3-by-N frame occupies a corner.  Its interior row is an ordered
      palette, and exactly one pixel outside the frame has the palette's first color.

    Rule:
    - Treat red walls and the complete legend rectangle as blocked.  Compute four-connected
      shortest-path distance from the lone marker through every other cell.
    - Recolor a reachable cell at distance t with palette[t modulo palette length].
      Preserve red walls and the legend; leave unreachable cells at their background color.

    Key insight:
    - The color phase comes from the visible legend, while its index is geodesic rather
      than Manhattan distance: walls delay the wave and closed wall interiors receive none.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    (r0, c0, r1, c1), palette = _find_legend(g)
    assert bg not in palette and RED != bg and GREY != bg

    protected = np.zeros(g.shape, dtype=bool)
    protected[r0 : r1 + 1, c0 : c1 + 1] = True
    marker_cells = np.argwhere((g == palette[0]) & ~protected)
    assert len(marker_cells) == 1, "the first legend color must identify one source"
    source = tuple(map(int, marker_cells[0]))

    blocked = (g == RED) | protected
    dist = _distance_map(g.shape, source, blocked)
    out = g.copy()
    reachable = (dist >= 0) & ~blocked
    for phase, color in enumerate(palette):
        out[reachable & (dist % len(palette) == phase)] = color
    out[(dist < 0) & ~blocked] = bg
    return out


def _clear_with_halo(mask: np.ndarray, cells: list[tuple[int, int]], gap: int = 1) -> bool:
    h, w = mask.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(max(0, r - gap), min(h, r + gap + 1)):
            for cc in range(max(0, c - gap), min(w, c + gap + 1)):
                if mask[rr, cc]:
                    return False
    return True


def _paint_legend(grid: np.ndarray, corner: int, palette: list[int]) -> tuple[int, int, int, int]:
    h, w = grid.shape
    fw = len(palette) + 2
    r0 = 0 if corner < 2 else h - 3
    c0 = 0 if corner % 2 == 0 else w - fw
    grid[r0 : r0 + 3, c0 : c0 + fw] = GREY
    grid[r0 + 1, c0 + 1 : c0 + fw - 1] = palette
    return r0, c0, r0 + 2, c0 + fw - 1


def _attempt(rng: random.Random) -> np.ndarray | None:
    h, w = rng.randint(18, 26), rng.randint(18, 27)
    bg = rng.choice([0, 1, 3, 4, 6, 7, 8, 9])
    available = [x for x in range(10) if x not in (bg, RED, GREY)]
    plen = rng.randint(3, min(5, len(available)))
    palette = rng.sample(available, plen)
    grid = np.full((h, w), bg, dtype=int)
    box = _paint_legend(grid, rng.randrange(4), palette)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[box[0] : box[2] + 1, box[1] : box[3] + 1] = True

    # A crosswise segment is deliberately put between the source and open cells.
    vertical = bool(rng.randrange(2))
    barrier: list[tuple[int, int]] | None = None
    source: tuple[int, int] | None = None
    for _ in range(100):
        length = rng.randint(5, 9)
        if vertical:
            r0 = rng.randint(3, h - length - 3)
            c = rng.randint(5, w - 6)
            mid = r0 + rng.randint(1, length - 2)
            side = rng.choice((-1, 1))
            s = (mid, c + side * rng.randint(2, 4))
            cells = [(r, c) for r in range(r0, r0 + length)]
        else:
            c0 = rng.randint(3, w - length - 3)
            r = rng.randint(5, h - 6)
            mid = c0 + rng.randint(1, length - 2)
            side = rng.choice((-1, 1))
            s = (r + side * rng.randint(2, 4), mid)
            cells = [(r, c) for c in range(c0, c0 + length)]
        if _clear_with_halo(occupied, cells + [s], 1):
            barrier, source = cells, s
            break
    if barrier is None or source is None:
        return None
    for cell in barrier:
        grid[cell] = RED
        occupied[cell] = True
    grid[source] = palette[0]
    occupied[source] = True

    # A red rectangular wall guarantees a visibly enclosed unreachable background patch.
    closed_rect = None
    for _ in range(150):
        bh, bw = rng.randint(4, 6), rng.randint(4, 7)
        br, bc = rng.randint(2, h - bh - 2), rng.randint(2, w - bw - 2)
        border = [
            (r, c)
            for r in range(br, br + bh)
            for c in range(bc, bc + bw)
            if r in (br, br + bh - 1) or c in (bc, bc + bw - 1)
        ]
        all_cells = [(r, c) for r in range(br, br + bh) for c in range(bc, bc + bw)]
        if _clear_with_halo(occupied, all_cells, 1):
            for cell in border:
                grid[cell] = RED
                occupied[cell] = True
            closed_rect = (br, bc, bh, bw)
            break
    if closed_rect is None:
        return None

    # Optional separated wall segments diversify count, direction, and placement.
    for _ in range(rng.randint(0, 2)):
        for _try in range(60):
            length = rng.randint(3, 7)
            if rng.randrange(2):
                rr, cc = rng.randint(2, h - 3), rng.randint(2, w - length - 2)
                cells = [(rr, cc + k) for k in range(length)]
            else:
                rr, cc = rng.randint(2, h - length - 2), rng.randint(2, w - 3)
                cells = [(rr + k, cc) for k in range(length)]
            if _clear_with_halo(occupied, cells, 1):
                for cell in cells:
                    grid[cell] = RED
                    occupied[cell] = True
                break

    # Independent semantic checks: unique parsing/source, true dead zone, and a real detour.
    try:
        parsed_box, parsed_palette = _find_legend(grid)
    except AssertionError:
        return None
    if parsed_box != box or parsed_palette != palette or _background(grid) != bg:
        return None
    frame_mask = np.zeros(grid.shape, dtype=bool)
    frame_mask[box[0] : box[2] + 1, box[1] : box[3] + 1] = True
    if int(np.count_nonzero((grid == palette[0]) & ~frame_mask)) != 1:
        return None
    blocked = (grid == RED) | frame_mask
    dist = _distance_map(grid.shape, source, blocked)
    br, bc, bh, bw = closed_rect
    interior = dist[br + 1 : br + bh - 1, bc + 1 : bc + bw - 1]
    if interior.size == 0 or not np.all(interior == -1):
        return None
    rr, cc = np.indices(grid.shape)
    manhattan = np.abs(rr - source[0]) + np.abs(cc - source[1])
    detoured = (dist >= 0) & (dist > manhattan) & ~frame_mask
    if int(detoured.sum()) < 3 or int(dist.max()) < 2 * len(palette):
        return None

    expected = grid.copy()
    for r in range(h):
        for c in range(w):
            if dist[r, c] >= 0 and not blocked[r, c]:
                expected[r, c] = palette[int(dist[r, c]) % len(palette)]
            elif dist[r, c] < 0 and not blocked[r, c]:
                expected[r, c] = bg
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(actual[frame_mask], grid[frame_mask]) or not np.all(actual[grid == RED] == RED):
        return None
    if not np.all(actual[br + 1 : br + bh - 1, bc + 1 : bc + bw - 1] == bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying all solver assumptions."""
    rng = random.Random()
    for _ in range(400):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid绕墙年轮 instance")
