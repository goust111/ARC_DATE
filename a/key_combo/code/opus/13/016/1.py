from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _ring_gap_side(component: dict[str, object]) -> tuple[Coord, str] | None:
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[assignment]
    side = int(r1 - r0 + 1)
    if side != int(c1 - c0 + 1) or side < 3:
        return None

    coords = {tuple(map(int, p)) for p in component["coords"]}  # type: ignore[arg-type]
    if len(coords) != int(component["area"]):
        return None
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coords):
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if len(coords) != len(border) - 1:
        return None

    missing = list(border - coords)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if gr == r0:
        return gap, "up"
    if gc == c1:
        return gap, "right"
    if gr == r1:
        return gap, "down"
    if gc == c0:
        return gap, "left"
    return None


def _legend_colors(grid: np.ndarray, bg: int) -> dict[str, int]:
    h, w = grid.shape
    mapping = {
        "up": int(grid[0, 0]),
        "right": int(grid[0, w - 1]),
        "down": int(grid[h - 1, w - 1]),
        "left": int(grid[h - 1, 0]),
    }
    assert all(color != bg for color in mapping.values()), "corner legend pixels must be non-background"
    assert len(set(mapping.values())) == 4, "corner legend colors must be distinct"
    return mapping


def _paint_ring_band(out: np.ndarray, outer: BBox, inner: BBox | None, color: int) -> None:
    r0, c0, r1, c1 = outer
    if r1 - r0 + 1 <= 2 or c1 - c0 + 1 <= 2:
        return
    mask = np.zeros_like(out, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if inner is not None:
        ir0, ic0, ir1, ic1 = inner
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    out[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each concentric square ring band with the corner-legend color indexed by that ring's gap direction.

    Input structure:
    - The grid is an odd-sized square with black background and four non-background legend pixels at the corners.
    - Near the center sit 3 or 4 concentric one-pixel square ring frames; each ring has exactly one non-corner gap on one side.
    - The ring frames are nested and separated so each layer encloses an annular band or center region.

    Rule:
    - Read the corner legend as a fixed direction map: top-left means up-gap, top-right right-gap, bottom-right down-gap, bottom-left left-gap.
    - For each ring, detect which side contains its unique gap.
    - Fill the region inside that ring but outside the next inner ring with the legend color for that direction.
    - Preserve all ring pixels, keep every gap black, and leave the corner legend unchanged.

    Key insight:
    - The missing pixel is not damage to repair; its side is a directional lookup key that selects the fill color for that ring's band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1 and h <= 30, "expected an odd square ARC grid"

    bg = 0
    legend = _legend_colors(g, bg)
    components = _components_by_color(g, bg)

    rings: list[dict[str, object]] = []
    ring_cells: set[Coord] = set()
    for comp in components:
        gap_side = _ring_gap_side(comp)
        if gap_side is None:
            continue
        gap, side = gap_side
        comp = dict(comp)
        comp["gap"] = gap
        comp["side_name"] = side
        rings.append(comp)
        ring_cells.update(comp["coords"])  # type: ignore[arg-type]

    assert 3 <= len(rings) <= 4, "expected 3 or 4 concentric ring frames"

    corner_cells = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    non_bg = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    assert non_bg == ring_cells | corner_cells, "unexpected non-background pixels outside rings and corner legend"

    rings.sort(key=lambda comp: int(comp["bbox"][0]))  # type: ignore[index]
    centers = {(int(b[0] + b[2]), int(b[1] + b[3])) for b in (comp["bbox"] for comp in rings)}  # type: ignore[index]
    assert len(centers) == 1, "rings must share a common center"

    out = g.copy()
    for i, comp in enumerate(rings):
        outer = comp["bbox"]  # type: ignore[assignment]
        inner = rings[i + 1]["bbox"] if i + 1 < len(rings) else None  # type: ignore[index]
        color = legend[str(comp["side_name"])]
        _paint_ring_band(out, outer, inner, color)

    for comp in rings:
        gap_r, gap_c = comp["gap"]  # type: ignore[assignment]
        out[gap_r, gap_c] = bg
    for (r, c), color in zip(((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)), (legend["up"], legend["right"], legend["down"], legend["left"])):
        out[r, c] = color
    return out


# --------------- generator ---------------

_GRID_CHOICES = (13, 15, 17, 19, 21)
_DIRECTIONS = ("up", "right", "down", "left")
_LEGEND_COLORS = (1, 2, 3, 4)
_FRAME_COLORS = (5, 6, 7, 8, 9)
_MAX_ATTEMPTS = 300


def _max_layers(grid_size: int, margin: int) -> int:
    outer_side = grid_size - 2 * margin
    return max(0, (outer_side - 3) // 4 + 1)


def _ring_bboxes(grid_size: int, layer_count: int, margin: int) -> list[BBox]:
    boxes: list[BBox] = []
    for depth in range(layer_count):
        inset = margin + 2 * depth
        boxes.append((inset, inset, grid_size - 1 - inset, grid_size - 1 - inset))
    return boxes


def _gap_cell(box: BBox, side: str, offset: int) -> Coord:
    r0, c0, r1, c1 = box
    if side == "up":
        return r0, c0 + offset
    if side == "right":
        return r0 + offset, c1
    if side == "down":
        return r1, c0 + offset
    return r0 + offset, c0


def _draw_one_gap_ring(grid: np.ndarray, box: BBox, color: int, side: str, offset: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = _gap_cell(box, side, offset)
    grid[gr, gc] = 0


def _render_expected(inp: np.ndarray, ring_specs: list[dict[str, object]], legend: dict[str, int]) -> np.ndarray:
    out = inp.copy()
    ordered = sorted(ring_specs, key=lambda spec: int(spec["bbox"][0]))  # type: ignore[index]
    for i, spec in enumerate(ordered):
        inner = ordered[i + 1]["bbox"] if i + 1 < len(ordered) else None  # type: ignore[index]
        _paint_ring_band(out, spec["bbox"], inner, legend[str(spec["side"])])  # type: ignore[arg-type]
    for spec in ordered:
        gr, gc = spec["gap"]  # type: ignore[assignment]
        out[gr, gc] = 0
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_size = random.choice(_GRID_CHOICES)
    feasible: list[tuple[int, int]] = []
    for layer_count in (3, 4):
        for margin in range(1, grid_size // 2):
            if _max_layers(grid_size, margin) >= layer_count:
                feasible.append((layer_count, margin))
    if not feasible:
        return None

    layer_count, margin = random.choice(feasible)
    boxes = _ring_bboxes(grid_size, layer_count, margin)
    frame_color = random.choice(_FRAME_COLORS)

    perm = random.sample(_LEGEND_COLORS, 4)
    legend = {"up": perm[0], "right": perm[1], "down": perm[2], "left": perm[3]}

    directions = random.sample(_DIRECTIONS, layer_count)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[0, 0] = legend["up"]
    grid[0, -1] = legend["right"]
    grid[-1, -1] = legend["down"]
    grid[-1, 0] = legend["left"]

    ring_specs: list[dict[str, object]] = []
    for box, side in zip(boxes, directions):
        side_len = int(box[2] - box[0] + 1)
        offset = random.randint(1, side_len - 2)
        gap = _gap_cell(box, side, offset)
        _draw_one_gap_ring(grid, box, frame_color, side, offset)
        ring_specs.append({"bbox": box, "side": side, "gap": gap})

    expected = _render_expected(grid, ring_specs, legend)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-gap-ring input")
