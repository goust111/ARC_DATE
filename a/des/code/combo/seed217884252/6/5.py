from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 7       # orange
CROSS = 5    # grey
PALETTE = (0, 1, 2, 3, 4, 6, 8, 9)
QUADRANTS = ("tl", "tr", "bl", "br")
ODD_SIZES = (11, 13, 15, 17, 19)
MAX_GENERATION_ATTEMPTS = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        full_rows = np.where(np.all(grid == color, axis=1))[0]
        full_cols = np.where(np.all(grid == color, axis=0))[0]
        if len(full_rows) == 1 and len(full_cols) == 1:
            candidates.append((color, int(full_rows[0]), int(full_cols[0])))
    assert len(candidates) == 1, "expected exactly one full-row/full-column cross"
    color, axis_r, axis_c = candidates[0]
    return int(color), int(axis_r), int(axis_c)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_for_bbox(bbox: tuple[int, int, int, int], axis_r: int, axis_c: int) -> str:
    r0, c0, r1, c1 = bbox
    if r1 < axis_r and c1 < axis_c:
        return "tl"
    if r1 < axis_r and c0 > axis_c:
        return "tr"
    if r0 > axis_r and c1 < axis_c:
        return "bl"
    if r0 > axis_r and c0 > axis_c:
        return "br"
    raise AssertionError("object must lie fully inside one quadrant")


def _shift_to_cross(
    bbox: tuple[int, int, int, int], quadrant: str, axis_r: int, axis_c: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if quadrant == "tl":
        return axis_r - 1 - r1, axis_c - 1 - c1
    if quadrant == "tr":
        return axis_r - 1 - r1, axis_c + 1 - c0
    if quadrant == "bl":
        return axis_r + 1 - r0, axis_c - 1 - c1
    if quadrant == "br":
        return axis_r + 1 - r0, axis_c + 1 - c0
    raise AssertionError("unknown quadrant")


def _corner_colors(grid: np.ndarray, bg: int, cross: int) -> dict[str, int]:
    h, w = grid.shape
    corners = {
        "tl": int(grid[0, 0]),
        "tr": int(grid[0, w - 1]),
        "bl": int(grid[h - 1, 0]),
        "br": int(grid[h - 1, w - 1]),
    }
    for color in corners.values():
        assert color not in (bg, cross), "corner control pixels must be visible colors"
    return corners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by its quadrant's corner control color, then pull it toward the center cross.

    Input structure:
    - The background is orange.
    - A full grey row and full grey column cross at the grid center.
    - Grey 4-connected objects lie away from the cross, and each corner has one colored control pixel.

    Rule:
    - Determine each grey object's quadrant relative to the center cross.
    - Recolor the object with that quadrant's corner pixel color.
    - Translate the whole object toward the cross until its bbox becomes edge-adjacent to the cross.
    - Remove the corner control pixels and the object's original placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    cross, axis_r, axis_c = _find_cross(g, bg)
    controls = _corner_colors(g, bg, cross)

    object_mask = g == cross
    object_mask[axis_r, :] = False
    object_mask[:, axis_c] = False

    components = _components4(object_mask)
    assert components, "expected at least one object away from the cross"

    out = np.full_like(g, bg)
    out[axis_r, :] = cross
    out[:, axis_c] = cross
    occupied = out != bg

    for coords in sorted(components, key=lambda part: tuple(part.min(axis=0))):
        box = _bbox(coords)
        quadrant = _quadrant_for_bbox(box, axis_r, axis_c)
        color = controls[quadrant]
        dr, dc = _shift_to_cross(box, quadrant, axis_r, axis_c)
        shifted = coords + np.array([dr, dc], dtype=int)
        assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < g.shape[0]))
        assert np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < g.shape[1]))
        assert not np.any(shifted[:, 0] == axis_r), "object may not overlap the cross row"
        assert not np.any(shifted[:, 1] == axis_c), "object may not overlap the cross column"
        assert not np.any(occupied[shifted[:, 0], shifted[:, 1]]), "moved objects must stay disjoint"
        occupied[shifted[:, 0], shifted[:, 1]] = True
        out[shifted[:, 0], shifted[:, 1]] = int(color)

    return out


def _random_anchor_shape(max_height: int, max_width: int) -> list[tuple[int, int]]:
    box_h = random.randint(1, max_height)
    box_w = random.randint(1, max_width)
    target_area = random.randint(1, min(5, box_h * box_w))
    cells = {(0, 0)}

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    return sorted(cells)


def _anchor_to_grid(
    cells: list[tuple[int, int]], quadrant: str, axis_r: int, axis_c: int
) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    for a, b in cells:
        if quadrant == "tl":
            placed.append((axis_r - 1 - a, axis_c - 1 - b))
        elif quadrant == "tr":
            placed.append((axis_r - 1 - a, axis_c + 1 + b))
        elif quadrant == "bl":
            placed.append((axis_r + 1 + a, axis_c - 1 - b))
        else:
            placed.append((axis_r + 1 + a, axis_c + 1 + b))
    return placed


def _shift_anchor_cells(
    cells: list[tuple[int, int]], row_shift: int, col_shift: int
) -> list[tuple[int, int]]:
    return [(a + row_shift, b + col_shift) for a, b in cells]


def _max_anchor_extent(cells: list[tuple[int, int]]) -> tuple[int, int]:
    max_a = max(a for a, _ in cells)
    max_b = max(b for _, b in cells)
    return int(max_a), int(max_b)


def _shift_limits(
    quadrant: str, cells: list[tuple[int, int]], h: int, w: int, axis_r: int, axis_c: int
) -> tuple[int, int]:
    max_a, max_b = _max_anchor_extent(cells)
    if quadrant == "tl":
        return axis_r - 2 - max_a, axis_c - 2 - max_b
    if quadrant == "tr":
        return axis_r - 2 - max_a, w - axis_c - 3 - max_b
    if quadrant == "bl":
        return h - axis_r - 3 - max_a, axis_c - 2 - max_b
    if quadrant == "br":
        return h - axis_r - 3 - max_a, w - axis_c - 3 - max_b
    raise AssertionError("unknown quadrant")


def _build_expected(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    controls: dict[str, int],
    final_by_quadrant: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    out[axis_r, :] = CROSS
    out[:, axis_c] = CROSS
    for quadrant, cells in final_by_quadrant.items():
        color = controls[quadrant]
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(ODD_SIZES)
    w = random.choice(ODD_SIZES)
    axis_r = h // 2
    axis_c = w // 2

    corner_colors = random.sample(PALETTE, 4)
    controls = dict(zip(QUADRANTS, corner_colors))

    active_count = random.randint(1, 4)
    active_quadrants = random.sample(QUADRANTS, active_count)

    final_by_quadrant: dict[str, list[tuple[int, int]]] = {}
    input_by_quadrant: dict[str, list[tuple[int, int]]] = {}

    for quadrant in active_quadrants:
        max_height = min(4, axis_r - 1, h - axis_r - 2)
        max_width = min(4, axis_c - 1, w - axis_c - 2)
        if max_height <= 0 or max_width <= 0:
            return None

        anchor_cells = _random_anchor_shape(max_height, max_width)
        max_row_shift, max_col_shift = _shift_limits(quadrant, anchor_cells, h, w, axis_r, axis_c)
        if max_row_shift < 0 or max_col_shift < 0:
            return None

        row_shift = random.randint(0, max_row_shift)
        col_shift = random.randint(0, max_col_shift)
        if row_shift == 0 and col_shift == 0:
            choices = []
            if max_row_shift > 0:
                choices.append("row")
            if max_col_shift > 0:
                choices.append("col")
            if not choices:
                return None
            picked = random.choice(choices)
            if picked == "row":
                row_shift = random.randint(1, max_row_shift)
            else:
                col_shift = random.randint(1, max_col_shift)

        final_cells = _anchor_to_grid(anchor_cells, quadrant, axis_r, axis_c)
        input_cells = _anchor_to_grid(
            _shift_anchor_cells(anchor_cells, row_shift, col_shift),
            quadrant,
            axis_r,
            axis_c,
        )
        final_by_quadrant[quadrant] = final_cells
        input_by_quadrant[quadrant] = input_cells

    inp = np.full((h, w), BG, dtype=int)
    inp[axis_r, :] = CROSS
    inp[:, axis_c] = CROSS
    inp[0, 0] = controls["tl"]
    inp[0, w - 1] = controls["tr"]
    inp[h - 1, 0] = controls["bl"]
    inp[h - 1, w - 1] = controls["br"]

    occupied = np.zeros((h, w), dtype=bool)
    occupied[axis_r, :] = True
    occupied[:, axis_c] = True
    occupied[0, 0] = True
    occupied[0, w - 1] = True
    occupied[h - 1, 0] = True
    occupied[h - 1, w - 1] = True

    for cells in input_by_quadrant.values():
        for r, c in cells:
            if occupied[r, c]:
                return None
            occupied[r, c] = True
            inp[r, c] = CROSS

    expected = _build_expected(h, w, axis_r, axis_c, controls, final_by_quadrant)
    if np.array_equal(inp, expected):
        return None
    if _background_color(inp) != BG:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
