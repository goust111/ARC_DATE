from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLORS = tuple(c for c in range(10) if c not in (BG, GREY))

DIRECTIONS = {
    "left": (0, 1),     # arrow is left of the container and points right
    "right": (0, -1),   # arrow is right of the container and points left
    "top": (1, 0),      # arrow is above the container and points down
    "bottom": (-1, 0),  # arrow is below the container and points up
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _containers(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = _rect_border_cells(r0, c0, r1, c1)
        actual = {(int(r), int(c)) for r, c in comp}
        if actual != expected:
            continue
        if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG):
            continue
        boxes.append((r0, c0, r1, c1))
    return boxes


def _side_tip_positions(box: tuple[int, int, int, int], side: str):
    r0, c0, r1, c1 = box
    if side == "left":
        for r in range(r0 + 1, r1):
            yield r, c0 - 1
    elif side == "right":
        for r in range(r0 + 1, r1):
            yield r, c1 + 1
    elif side == "top":
        for c in range(c0 + 1, c1):
            yield r0 - 1, c
    else:
        for c in range(c0 + 1, c1):
            yield r1 + 1, c


def _arrow_at(
    g: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, set[tuple[int, int]], int] | None:
    h, w = g.shape
    tr, tc = tip
    dr, dc = direction
    if not (0 <= tr < h and 0 <= tc < w):
        return None

    color = int(g[tr, tc])
    if color in (BG, GREY):
        return None

    base = (tr - dr, tc - dc)
    perp = (-dc, dr)
    head = {
        (tr, tc),
        (base[0] + perp[0], base[1] + perp[1]),
        (base[0] - perp[0], base[1] - perp[1]),
    }
    if any(not (0 <= r < h and 0 <= c < w) for r, c in head):
        return None
    if any(int(g[r, c]) != color for r, c in head):
        return None

    shaft_len = 0
    r, c = base
    while 0 <= r < h and 0 <= c < w and int(g[r, c]) == color:
        shaft_len += 1
        r -= dr
        c -= dc
    if shaft_len < 2:
        return None

    return color, head, shaft_len


def _container_arrow(
    g: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, set[tuple[int, int]]] | None:
    matches: list[tuple[int, set[tuple[int, int]], int]] = []
    for side, direction in DIRECTIONS.items():
        for tip in _side_tip_positions(box, side):
            found = _arrow_at(g, tip, direction)
            if found is not None:
                matches.append(found)
    if not matches:
        return None
    matches.sort(key=lambda item: item[2], reverse=True)
    color, head, _shaft_len = matches[0]
    return color, head


def _tile_container(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    head: set[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = box
    hr0 = min(r for r, _c in head)
    hc0 = min(c for _r, c in head)
    hr1 = max(r for r, _c in head)
    hc1 = max(c for _r, c in head)
    tile_h = hr1 - hr0 + 1
    tile_w = hc1 - hc0 + 1
    rel = {(r - hr0, c - hc0) for r, c in head}

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if ((r - (r0 + 1)) % tile_h, (c - (c0 + 1)) % tile_w) in rel:
                out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow grey containers with the tiled arrowhead pattern of their adjacent arrows.

    Input structure:
    - The background is black.
    - Grey pixels form one-pixel-thick hollow rectangular containers with black interiors.
    - Some containers have one adjacent colored arrow outside one side; each arrow has a straight shaft and a three-pixel V-shaped head pointing at the container.
    - Containers without an adjacent arrow may also appear and remain empty.

    Rule:
    - Keep every grey border pixel and every colored arrow pixel unchanged.
    - For each arrow-targeted container, take the arrowhead after removing the shaft.
    - Use that V-shaped head bbox as a tile, and repeat it from the container interior's top-left corner, clipping at the interior boundary.
    - Paint only the tile's V pixels in the arrow color; untouched tile cells remain black.

    Key insight: The arrow encodes both the fill color and the oriented V-shaped tile pattern, not just a solid fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for box in _containers(g):
        found = _container_arrow(g, box)
        if found is None:
            continue
        color, head = found
        _tile_container(out, box, color, head)
    return out


def _arrow_cells(
    box: tuple[int, int, int, int],
    side: str,
    shaft_len: int,
    anchor: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = box
    dr, dc = DIRECTIONS[side]
    if side == "left":
        tip = (anchor, c0 - 1)
    elif side == "right":
        tip = (anchor, c1 + 1)
    elif side == "top":
        tip = (r0 - 1, anchor)
    else:
        tip = (r1 + 1, anchor)

    base = (tip[0] - dr, tip[1] - dc)
    perp = (-dc, dr)
    head = {
        tip,
        (base[0] + perp[0], base[1] + perp[1]),
        (base[0] - perp[0], base[1] - perp[1]),
    }
    shaft = {(tip[0] - dr * k, tip[1] - dc * k) for k in range(1, shaft_len + 1)}
    return head, shaft, tip


def _paint_container(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _expanded_clear(reserved: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        rr0 = max(0, r - pad)
        rr1 = min(h, r + pad + 1)
        cc0 = max(0, c - pad)
        cc1 = min(w, c + pad + 1)
        if bool(reserved[rr0:rr1, cc0:cc1].any()):
            return False
    return True


def _reserve(reserved: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved[r, c] = True


def _frame_footprint(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _valid_frame_origins(
    rows: int, cols: int, fh: int, fw: int, side: str | None, shaft_len: int
) -> tuple[range, range]:
    r_min, r_max = 1, rows - fh - 1
    c_min, c_max = 1, cols - fw - 1
    if side == "left":
        c_min = max(c_min, shaft_len + 1)
    elif side == "right":
        c_max = min(c_max, cols - fw - shaft_len - 1)
    elif side == "top":
        r_min = max(r_min, shaft_len + 1)
    elif side == "bottom":
        r_max = min(r_max, rows - fh - shaft_len - 1)
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _place_one(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int | None,
    side: str | None,
) -> bool:
    rows, cols = grid.shape
    fh = random.randint(4, min(9, rows - 4))
    fw = random.randint(4, min(9, cols - 4))
    shaft_len = random.randint(2, 5)
    rr, cr = _valid_frame_origins(rows, cols, fh, fw, side, shaft_len)
    if not rr or not cr:
        return False

    for _ in range(120):
        r0 = random.choice(tuple(rr))
        c0 = random.choice(tuple(cr))
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        footprint = _frame_footprint(box)
        arrow = set()
        if side is not None and color is not None:
            if side in ("left", "right"):
                anchor = random.randint(box[0] + 1, box[2] - 1)
            else:
                anchor = random.randint(box[1] + 1, box[3] - 1)
            head, shaft, _tip = _arrow_cells(box, side, shaft_len, anchor)
            arrow = head | shaft
            footprint |= arrow

        if not _expanded_clear(reserved, footprint, pad=1):
            continue

        _paint_container(grid, box)
        if side is not None and color is not None:
            for r, c in arrow:
                grid[r, c] = int(color)
        _reserve(reserved, footprint)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(14, 30)
    cols = random.randint(14, 30)
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)

    n_containers = random.randint(2, 5)
    n_arrows = random.randint(1, min(n_containers, len(COLORS)))
    arrow_slots = set(random.sample(range(n_containers), n_arrows))
    color_iter = iter(random.sample(COLORS, n_arrows))

    sides = list(DIRECTIONS.keys())
    random.shuffle(sides)
    side_plan = [sides[i % len(sides)] for i in range(n_arrows)]
    random.shuffle(side_plan)
    side_i = 0

    for i in range(n_containers):
        if i in arrow_slots:
            color = next(color_iter)
            side = side_plan[side_i]
            side_i += 1
        else:
            color = None
            side = None

        placed = False
        for _ in range(60):
            if _place_one(grid, reserved, color, side):
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for arrowhead container tiling")
