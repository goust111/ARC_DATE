from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9
SPECIAL_COLORS = np.array([BLACK, GREY, ORANGE, MAROON], dtype=int)
SHAPE_COLORS = np.array([1, 2, 3, 4, 6, 8], dtype=int)

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300
QUADRANT_SIDE_RANGE = (6, 15)  # high-exclusive for RNG.integers
PLACEMENT_RETRIES = 60

_DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CYCLE_POSITIONS = ((0, 0), (0, 1), (1, 1), (1, 0))
_CYCLE_INDEX = {pos: idx for idx, pos in enumerate(_CYCLE_POSITIONS)}
_BBOX_OPTIONS = (
    (1, 3),
    (3, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (4, 4),
)


def _separator_indices(g: np.ndarray) -> tuple[int, int]:
    row_hits = np.flatnonzero(np.all(g == GREY, axis=1))
    col_hits = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey row and one full grey column"
    row = int(row_hits[0])
    col = int(col_hits[0])
    assert row * 2 + 1 == g.shape[0] and col * 2 + 1 == g.shape[1], "grey lines must split the grid equally"
    return row, col


def _decode_indicator(block: np.ndarray) -> str:
    maroon_cells = np.argwhere(block == MAROON)
    orange_cells = np.argwhere(block == ORANGE)
    assert maroon_cells.shape[0] == 1 and orange_cells.shape[0] == 1, "indicator must contain one maroon and one orange pixel"

    maroon_pos = tuple(map(int, maroon_cells[0]))
    orange_pos = tuple(map(int, orange_cells[0]))
    assert maroon_pos in _CYCLE_INDEX and orange_pos in _CYCLE_INDEX, "indicator pixels must lie in the 2x2 ring"

    step = (_CYCLE_INDEX[orange_pos] - _CYCLE_INDEX[maroon_pos]) % 4
    if step == 1:
        return "cw"
    if step == 3:
        return "ccw"
    raise AssertionError("indicator pixels must occupy adjacent clockwise or counterclockwise positions")


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in _DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    assert view.shape == patch.shape, "rotated object must stay inside the grid"
    colored = patch != BLACK
    assert not bool(np.any(view[colored] != BLACK)), "rotated objects must not overlap"
    view[colored] = patch[colored]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each quadrant's colored objects by the corner-indicated 90-degree direction.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column split the canvas into four equal quadrants.
    - Each grid corner contains a 2x2 direction indicator made of one maroon pixel and one orthogonally adjacent orange pixel.
    - Each quadrant interior contains one or more small colored objects that avoid the separator lines.

    Rule:
    - Read each quadrant's corner indicator: maroon->orange in clockwise order means rotate that quadrant's objects clockwise 90 degrees; the opposite order means counterclockwise 90 degrees.
    - Treat every connected colored object in a quadrant independently.
    - Rotate each object 90 degrees around the center of its own bounding box while preserving its colors.
    - Keep the black background, the grey separator lines, and all corner indicators unchanged.

    Key insight:
    - The two-pixel corner code does not move itself; it chooses the local rotation rule that must be applied separately to each object inside the corresponding quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "expected a 2D ARC grid"

    sep_r, sep_c = _separator_indices(g)
    qh, qw = sep_r, sep_c
    out = g.copy()
    movable = ~np.isin(out, SPECIAL_COLORS)
    out[movable] = BLACK

    quadrants = (
        (0, sep_r, 0, sep_c, 0, 0),
        (0, sep_r, sep_c + 1, g.shape[1], 0, qw - 2),
        (sep_r + 1, g.shape[0], 0, sep_c, qh - 2, 0),
        (sep_r + 1, g.shape[0], sep_c + 1, g.shape[1], qh - 2, qw - 2),
    )

    for r0, r1, c0, c1, ind_r, ind_c in quadrants:
        quad = g[r0:r1, c0:c1]
        direction = _decode_indicator(quad[ind_r : ind_r + 2, ind_c : ind_c + 2])
        obj_mask = ~np.isin(quad, np.array([BLACK, ORANGE, MAROON], dtype=int))

        for comp in _connected_components(obj_mask):
            rows = comp[:, 0]
            cols = comp[:, 1]
            br0 = int(rows.min())
            br1 = int(rows.max())
            bc0 = int(cols.min())
            bc1 = int(cols.max())

            patch = np.full((br1 - br0 + 1, bc1 - bc0 + 1), BLACK, dtype=int)
            patch[rows - br0, cols - bc0] = quad[rows, cols]
            rotated = np.rot90(patch, 3 if direction == "cw" else 1)

            target_r = r0 + br0 + (patch.shape[0] - rotated.shape[0]) // 2
            target_c = c0 + bc0 + (patch.shape[1] - rotated.shape[1]) // 2
            _paste_transparent(out, rotated, int(target_r), int(target_c))

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_catalog() -> dict[tuple[int, int], list[np.ndarray]]:
    catalog: dict[tuple[int, int], list[np.ndarray]] = {}
    for h, w in _BBOX_OPTIONS:
        shapes: list[np.ndarray] = []
        cell_count = h * w
        for area in range(3, min(8, cell_count) + 1):
            for combo in combinations(range(cell_count), area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in combo:
                    mask[idx // w, idx % w] = True
                if not _is_connected(mask):
                    continue
                if not bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                rotated = np.rot90(mask, 3)
                if rotated.shape == mask.shape and np.array_equal(rotated, mask):
                    continue
                shapes.append(mask)
        assert shapes, f"no shapes for bbox {(h, w)}"
        catalog[(h, w)] = shapes
    return catalog


SHAPES_BY_BBOX = _shape_catalog()


def _indicator_block(direction: str) -> np.ndarray:
    block = np.full((2, 2), BLACK, dtype=int)
    maroon_idx = int(RNG.integers(0, 4))
    orange_idx = (maroon_idx + 1) % 4 if direction == "cw" else (maroon_idx - 1) % 4
    mr, mc = _CYCLE_POSITIONS[maroon_idx]
    orr, oc = _CYCLE_POSITIONS[orange_idx]
    block[mr, mc] = MAROON
    block[orr, oc] = ORANGE
    return block


def _indicator_anchor(qh: int, qw: int, quadrant: str) -> tuple[int, int]:
    if quadrant == "tl":
        return 0, 0
    if quadrant == "tr":
        return 0, qw - 2
    if quadrant == "bl":
        return qh - 2, 0
    return qh - 2, qw - 2


def _blocked_mask(qh: int, qw: int, quadrant: str) -> np.ndarray:
    blocked = np.zeros((qh, qw), dtype=bool)
    r0, c0 = _indicator_anchor(qh, qw, quadrant)
    r1 = max(0, r0 - 1)
    r2 = min(qh, r0 + 3)
    c1 = max(0, c0 - 1)
    c2 = min(qw, c0 + 3)
    blocked[r1:r2, c1:c2] = True
    return blocked


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = mask.shape
    qh, qw = occupied.shape
    if top < 0 or left < 0 or top + h > qh or left + w > qw:
        return False
    for dr, dc in np.argwhere(mask):
        r = int(top + dr)
        c = int(left + dc)
        if bool(blocked[r, c]):
            return False
        r1 = max(0, r - 1)
        r2 = min(qh, r + 2)
        c1 = max(0, c - 1)
        c2 = min(qw, c + 2)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _paint_mask(canvas: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for dr, dc in coords:
        r = int(top + dr)
        c = int(left + dc)
        canvas[r, c] = color
        occupied[r, c] = True


def _sample_shape(qh: int, qw: int) -> np.ndarray:
    feasible = [bbox for bbox in _BBOX_OPTIONS if bbox[0] <= qh and bbox[1] <= qw]
    bbox = feasible[int(RNG.integers(0, len(feasible)))]
    shapes = SHAPES_BY_BBOX[bbox]
    return shapes[int(RNG.integers(0, len(shapes)))]


def _place_objects_in_quadrant(qh: int, qw: int, quadrant: str, direction: str, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    quad_in = np.full((qh, qw), BLACK, dtype=int)
    quad_out = np.full((qh, qw), BLACK, dtype=int)
    blocked = _blocked_mask(qh, qw, quadrant)
    occ_in = np.zeros((qh, qw), dtype=bool)
    occ_out = np.zeros((qh, qw), dtype=bool)

    for _ in range(count):
        placed = False
        for _attempt in range(PLACEMENT_RETRIES):
            mask = _sample_shape(qh, qw)
            h, w = mask.shape
            rotated = np.rot90(mask, 3 if direction == "cw" else 1)
            shift_r = (h - rotated.shape[0]) // 2
            shift_c = (w - rotated.shape[1]) // 2

            placements: list[tuple[int, int, int, int]] = []
            for top in range(qh - h + 1):
                out_top = top + shift_r
                if out_top < 0 or out_top + rotated.shape[0] > qh:
                    continue
                for left in range(qw - w + 1):
                    out_left = left + shift_c
                    if out_left < 0 or out_left + rotated.shape[1] > qw:
                        continue
                    if not _can_place(mask, top, left, occ_in, blocked):
                        continue
                    if not _can_place(rotated, out_top, out_left, occ_out, blocked):
                        continue
                    placements.append((top, left, out_top, out_left))

            if not placements:
                continue

            top, left, out_top, out_left = placements[int(RNG.integers(0, len(placements)))]
            color = int(SHAPE_COLORS[int(RNG.integers(0, len(SHAPE_COLORS)))])
            _paint_mask(quad_in, occ_in, mask, top, left, color)
            _paint_mask(quad_out, occ_out, rotated, out_top, out_left, color)
            placed = True
            break
        if not placed:
            return None

    indicator = _indicator_block(direction)
    ir, ic = _indicator_anchor(qh, qw, quadrant)
    quad_in[ir : ir + 2, ic : ic + 2] = indicator
    quad_out[ir : ir + 2, ic : ic + 2] = indicator
    return quad_in, quad_out


def _object_count(qh: int, qw: int) -> int:
    side = min(qh, qw)
    if side >= 12:
        return int(RNG.choice(np.array([1, 2, 3]), p=np.array([0.25, 0.5, 0.25])))
    if side >= 9:
        return int(RNG.choice(np.array([1, 2]), p=np.array([0.5, 0.5])))
    return 1


def _direction_assignment() -> dict[str, str]:
    quadrants = np.array(["tl", "tr", "bl", "br"], dtype=object)
    cw_quadrants = set(quadrants[RNG.choice(4, size=2, replace=False)].tolist())
    return {name: ("cw" if name in cw_quadrants else "ccw") for name in quadrants.tolist()}


def _assemble_grid(qh: int, qw: int, quads: dict[str, np.ndarray]) -> np.ndarray:
    h = 2 * qh + 1
    w = 2 * qw + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[qh, :] = GREY
    grid[:, qw] = GREY
    grid[:qh, :qw] = quads["tl"]
    grid[:qh, qw + 1 :] = quads["tr"]
    grid[qh + 1 :, :qw] = quads["bl"]
    grid[qh + 1 :, qw + 1 :] = quads["br"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    qh = int(RNG.integers(*QUADRANT_SIDE_RANGE))
    qw = int(RNG.integers(*QUADRANT_SIDE_RANGE))
    directions = _direction_assignment()

    quad_inputs: dict[str, np.ndarray] = {}
    quad_outputs: dict[str, np.ndarray] = {}
    for quadrant in ("tl", "tr", "bl", "br"):
        count = _object_count(qh, qw)
        placed = _place_objects_in_quadrant(qh, qw, quadrant, directions[quadrant], count)
        if placed is None:
            return None
        quad_inputs[quadrant], quad_outputs[quadrant] = placed

    grid = _assemble_grid(qh, qw, quad_inputs)
    expected = _assemble_grid(qh, qw, quad_outputs)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
