from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
BRUSH_COLORS = (1, 2, 3, 4, 6, 7, 9)


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            oriented = np.rot90(base, k)
            key = (oriented.shape, oriented.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(oriented.copy())
    return out


RAW_TEMPLATE_LIBRARY = (
    [[1, 1]],
    [[1, 1, 1]],
    [[1, 1], [1, 0]],
    [[1, 1], [1, 1]],
    [[1, 1, 1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [0, 1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0], [1, 1], [1, 0]],
    [[1, 1], [1, 1], [1, 0]],
)
TEMPLATE_LIBRARY = tuple(np.array(shape, dtype=bool) for shape in RAW_TEMPLATE_LIBRARY)


def _room_template(sub: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    template_comps = _components(sub == GREY)
    assert len(template_comps) == 1, f"expected one grey template, found {len(template_comps)}"
    template = template_comps[0]

    brush_coords = np.argwhere((sub != BLACK) & (sub != GREY) & (sub != AZURE))
    assert brush_coords.shape == (1, 2), f"expected one brush pixel, found {brush_coords.shape[0]}"
    br, bc = map(int, brush_coords[0])
    brush_color = int(sub[br, bc])
    return template, (br, bc), brush_color


def main(grid: np.ndarray) -> np.ndarray:
    """Color each grey room-template with its brush, then stamp a 180-degree room-symmetric copy."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = _components(g != AZURE)
    assert rooms, "expected at least one room"

    for room in rooms:
        r0, c0, r1, c1 = _bbox(room)
        assert room.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "rooms must be solid rectangles"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape

        template, (br, bc), brush_color = _room_template(sub)
        out[r0 + br, c0 + bc] = BLACK

        local_template = {(int(r), int(c)) for r, c in template}
        for tr, tc in local_template:
            out[r0 + tr, c0 + tc] = brush_color
            mr = h - 1 - tr
            mc = w - 1 - tc
            rr = r0 + mr
            cc = c0 + mc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) != AZURE:
                out[rr, cc] = brush_color

    return out


ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_HEIGHT_RANGE = (4, 8)
ROOM_WIDTH_RANGE = (4, 8)
GRID_MIN = 12
GRID_MAX = 30
LAYOUT_ATTEMPTS = 200
ROOM_ATTEMPTS = 100
GEN_ATTEMPTS = 300


def _sample_layout() -> tuple[bool, list[int], list[int]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        outer_border = random.random() < 0.5
        room_rows = random.randint(*ROOM_ROWS_RANGE)
        room_cols = random.randint(*ROOM_COLS_RANGE)
        row_heights = [random.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
        col_widths = [random.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]
        h = sum(row_heights) + (room_rows - 1) + (2 if outer_border else 0)
        w = sum(col_widths) + (room_cols - 1) + (2 if outer_border else 0)
        if GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX:
            return outer_border, row_heights, col_widths
    return None


def _build_canvas(
    outer_border: bool, row_heights: list[int], col_widths: list[int]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = sum(row_heights) + (len(row_heights) - 1) + (2 if outer_border else 0)
    w = sum(col_widths) + (len(col_widths) - 1) + (2 if outer_border else 0)
    grid = np.full((h, w), BLACK, dtype=int)

    if outer_border:
        grid[0, :] = AZURE
        grid[-1, :] = AZURE
        grid[:, 0] = AZURE
        grid[:, -1] = AZURE

    row_starts: list[int] = []
    col_starts: list[int] = []

    r = 1 if outer_border else 0
    for idx, room_h in enumerate(row_heights):
        row_starts.append(r)
        r += room_h
        if idx < len(row_heights) - 1:
            grid[r, :] = AZURE
            r += 1

    c = 1 if outer_border else 0
    for idx, room_w in enumerate(col_widths):
        col_starts.append(c)
        c += room_w
        if idx < len(col_widths) - 1:
            grid[:, c] = AZURE
            c += 1

    rooms = [
        (r0, c0, room_h, room_w)
        for r0, room_h in zip(row_starts, row_heights)
        for c0, room_w in zip(col_starts, col_widths)
    ]
    return grid, rooms


def _template_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(mask):
        coords.add((top + int(r), left + int(c)))
    return coords


def _brush_candidates(h: int, w: int, template_cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    forbidden = set(template_cells)
    for tr, tc in template_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = tr + dr
                nc = tc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden.add((nr, nc))
    return [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]


def _place_room_objects(h: int, w: int) -> dict[str, object] | None:
    orientations: list[np.ndarray] = []
    for template in TEMPLATE_LIBRARY:
        for oriented in _dihedral_masks(template):
            if oriented.shape[0] <= h and oriented.shape[1] <= w:
                orientations.append(oriented)
    if not orientations:
        return None

    for _ in range(ROOM_ATTEMPTS):
        oriented = random.choice(orientations)
        th, tw = oriented.shape
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        cells = _template_cells(oriented, top, left)
        brushes = _brush_candidates(h, w, cells)
        if not brushes:
            continue
        brush = random.choice(brushes)
        mirror = {(h - 1 - r, w - 1 - c) for r, c in cells}
        return {
            "template": cells,
            "brush": brush,
            "color": random.choice(BRUSH_COLORS),
            "mirror": mirror,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    outer_border, row_heights, col_widths = layout
    grid, rooms = _build_canvas(outer_border, row_heights, col_widths)

    room_specs: list[dict[str, object]] = []
    for _r0, _c0, room_h, room_w in rooms:
        spec = _place_room_objects(room_h, room_w)
        if spec is None:
            return None
        room_specs.append(spec)

    colors = {int(spec["color"]) for spec in room_specs}
    if len(room_specs) > 1 and len(colors) < 2:
        return None
    if not any(spec["template"] != spec["mirror"] for spec in room_specs):
        return None

    for (r0, c0, _room_h, _room_w), spec in zip(rooms, room_specs):
        for tr, tc in spec["template"]:
            grid[r0 + tr, c0 + tc] = GREY
        br, bc = spec["brush"]
        grid[r0 + br, c0 + bc] = int(spec["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
