from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SIDES = ("top", "right", "bottom", "left")
EDGE_COLORS = tuple(c for c in range(1, 10) if c != GREY)
GENERATION_RETRIES = 400


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(g == int(color)):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        pts: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == int(color):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _edge_colors(g: np.ndarray) -> dict[str, int]:
    h, w = g.shape
    assert h >= 5 and w >= 5
    top = int(g[0, 0])
    bottom = int(g[h - 1, 0])
    left = int(g[1, 0])
    right = int(g[1, w - 1])

    assert bool(np.all(g[0, :] == top))
    assert bool(np.all(g[h - 1, :] == bottom))
    assert bool(np.all(g[1 : h - 1, 0] == left))
    assert bool(np.all(g[1 : h - 1, w - 1] == right))

    colors = {"top": top, "right": right, "bottom": bottom, "left": left}
    assert len(set(colors.values())) == 4
    assert all(c not in (BLACK, GREY) for c in colors.values())
    return colors


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _contact_side(bbox: tuple[int, int, int, int], h: int, w: int) -> str:
    r0, c0, r1, c1 = bbox
    touches: list[str] = []
    if r0 == 1:
        touches.append("top")
    if c1 == w - 2:
        touches.append("right")
    if r1 == h - 2:
        touches.append("bottom")
    if c0 == 1:
        touches.append("left")
    assert len(touches) == 1
    return touches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor edge-touching grey rectangles and nest them concentrically by decreasing area.

    Input structure:
    - The background is black.
    - The four grid edges are one-cell-wide color bands with four distinct non-black, non-grey colors; horizontal bands own the corner cells.
    - The interior contains separated solid grey rectangles, and each rectangle is 4-adjacent to exactly one edge band.

    Rule:
    - For each grey rectangle, identify the single edge band it touches and use that band color as the rectangle's output color.
    - Sort the rectangles by area from largest to smallest.
    - On a same-size black output canvas, draw the rectangles centered on the grid center in that order, so smaller rectangles overwrite the center of larger ones.
    - When centering leaves an odd margin, place the rectangle one cell up/left.

    Key insight:
    - A grey object's color is not stored in the object itself; its touched edge band gives the color, while its area independently determines the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    colors = _edge_colors(g)

    interior = g[1 : h - 1, 1 : w - 1]
    assert bool(np.all((interior == BLACK) | (interior == GREY)))

    objects: list[dict[str, int]] = []
    for coords in _components_of_color(g, GREY):
        r0, c0, r1, c1 = _component_bbox(coords)
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        area = rect_h * rect_w
        assert int(coords.shape[0]) == area
        side = _contact_side((r0, c0, r1, c1), h, w)
        objects.append({"h": rect_h, "w": rect_w, "area": area, "color": colors[side]})

    assert objects
    areas = [obj["area"] for obj in objects]
    assert len(areas) == len(set(areas))
    objects.sort(key=lambda obj: obj["area"], reverse=True)

    out = np.zeros((h, w), dtype=int)
    for obj in objects:
        rect_h = obj["h"]
        rect_w = obj["w"]
        r0 = (h - rect_h) // 2
        c0 = (w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = obj["color"]
    return out


# --------------- generator ---------------


def _render_edge_bands(h: int, w: int, colors: dict[str, int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = colors["left"]
    grid[:, w - 1] = colors["right"]
    grid[0, :] = colors["top"]
    grid[h - 1, :] = colors["bottom"]
    return grid


def _sample_dimensions(h: int, w: int, n: int) -> list[tuple[int, int]]:
    max_h = min(10, h - 5)
    max_w = min(10, w - 5)
    if max_h < n or max_w < n:
        return []
    heights = sorted(random.sample(range(1, max_h + 1), n), reverse=True)
    widths = sorted(random.sample(range(1, max_w + 1), n), reverse=True)
    return list(zip(heights, widths))


def _sample_sides(n: int) -> list[str]:
    for _ in range(100):
        sides = list(SIDES) + [random.choice(SIDES) for _ in range(n - len(SIDES))]
        random.shuffle(sides)
        if all(sides[i] != sides[i - 1] for i in range(1, len(sides))):
            return sides
    return list(SIDES) + [SIDES[i % len(SIDES)] for i in range(n - len(SIDES))]


def _candidate_positions(h: int, w: int, rect_h: int, rect_w: int, side: str) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        if rect_h > h - 4 or rect_w > w - 4:
            return []
        row = 1 if side == "top" else h - 1 - rect_h
        return [(row, c) for c in range(2, w - rect_w - 1)]

    if rect_h > h - 4 or rect_w > w - 4:
        return []
    col = 1 if side == "left" else w - 1 - rect_w
    return [(r, col) for r in range(2, h - rect_h - 1)]


def _can_place(occupied: np.ndarray, r0: int, c0: int, rect_h: int, rect_w: int) -> bool:
    h, w = occupied.shape
    r1 = r0 + rect_h
    c1 = c0 + rect_w
    padded = occupied[max(0, r0 - 1) : min(h, r1 + 1), max(0, c0 - 1) : min(w, c1 + 1)]
    return not bool(padded.any())


def _place_rectangles(
    grid: np.ndarray,
    dims: list[tuple[int, int]],
    sides: list[str],
) -> list[dict[str, int]] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int]] = []
    for (rect_h, rect_w), side in zip(dims, sides, strict=False):
        positions = _candidate_positions(h, w, rect_h, rect_w, side)
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            if _can_place(occupied, r0, c0, rect_h, rect_w):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = GREY
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        objects.append({"h": rect_h, "w": rect_w, "area": rect_h * rect_w, "side": side})
    return objects


def _expected_output(shape: tuple[int, int], objects: list[dict[str, int]], colors: dict[str, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for obj in sorted(objects, key=lambda item: item["area"], reverse=True):
        rect_h = obj["h"]
        rect_w = obj["w"]
        r0 = (h - rect_h) // 2
        c0 = (w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = colors[obj["side"]]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n = random.randint(4, 6)
    dims = _sample_dimensions(h, w, n)
    if not dims:
        return None

    colors = dict(zip(SIDES, random.sample(EDGE_COLORS, len(SIDES)), strict=False))
    grid = _render_edge_bands(h, w, colors)
    sides = _sample_sides(n)
    objects = _place_rectangles(grid, dims, sides)
    if objects is None:
        return None

    expected = _expected_output(grid.shape, objects, colors)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-band nesting input")
