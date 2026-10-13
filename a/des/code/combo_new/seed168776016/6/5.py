from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
RED = 2
PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUADRANT_SIZE_RANGE = (6, 14)
OBJECT_SIZE_RANGE = (3, 6)
MAX_GEN_ATTEMPTS = 500

REPEAT_SEQUENCES = {
    0: ((0, 0, 1, 2), (0, 1, 0, 2), (0, 1, 2, 0)),
    1: ((0, 1, 1, 2), (0, 1, 2, 1)),
    2: ((0, 1, 2, 2),),
}
MISSING_POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _checkerboard(height: int, width: int, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color)).astype(int)


def _marker_origin(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 2
    if corner == "bl":
        return height - 2, 0
    if corner == "br":
        return height - 2, width - 2
    raise ValueError(f"unknown corner: {corner}")


def _quadrant_specs(mid_r: int, mid_c: int, height: int, width: int) -> list[dict[str, object]]:
    return [
        {"rows": slice(0, mid_r), "cols": slice(0, mid_c), "row0": 0, "col0": 0, "marker_corner": "br"},
        {"rows": slice(0, mid_r), "cols": slice(mid_c + 1, width), "row0": 0, "col0": mid_c + 1, "marker_corner": "bl"},
        {"rows": slice(mid_r + 1, height), "cols": slice(0, mid_c), "row0": mid_r + 1, "col0": 0, "marker_corner": "tr"},
        {"rows": slice(mid_r + 1, height), "cols": slice(mid_c + 1, width), "row0": mid_r + 1, "col0": mid_c + 1, "marker_corner": "tl"},
    ]


def _read_marker(quadrant: np.ndarray, corner: str) -> tuple[int, tuple[int, int]]:
    r0, c0 = _marker_origin(*quadrant.shape, corner)
    block = quadrant[r0 : r0 + 2, c0 : c0 + 2]
    values, counts = np.unique(block, return_counts=True)
    marker_candidates = values[counts == 3]
    assert marker_candidates.size == 1, "expected a 3-cell L marker in the inner 2x2 block"
    marker_color = int(marker_candidates[0])

    missing = None
    for dr in range(2):
        for dc in range(2):
            if int(block[dr, dc]) != marker_color:
                missing = (int(dr), int(dc))
                break
        if missing is not None:
            break
    assert missing is not None
    return marker_color, missing


def _steps_from_marker(missing: tuple[int, int], turns: int) -> int:
    clockwise = missing in {(0, 0), (1, 1)}
    return int(turns % 4) if clockwise else int((-turns) % 4)


def _rotate_coords(coords: np.ndarray, height: int, width: int, steps_cw: int) -> np.ndarray:
    steps = int(steps_cw) % 4
    if steps == 0 or coords.size == 0:
        return coords.copy()

    center_r2 = height - 1
    center_c2 = width - 1
    out = np.empty_like(coords)

    for idx, (r, c) in enumerate(coords):
        vr = 2 * int(r) - center_r2
        vc = 2 * int(c) - center_c2

        if steps == 1:
            vr2, vc2 = vc, -vr
        elif steps == 2:
            vr2, vc2 = -vr, -vc
        else:
            vr2, vc2 = -vc, vr

        nr2 = vr2 + center_r2
        nc2 = vc2 + center_c2
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on cell centers"
        nr = nr2 // 2
        nc = nc2 // 2
        assert 0 <= nr < height and 0 <= nc < width, "rotation moved a cell out of bounds"
        out[idx] = (int(nr), int(nc))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant's object by the amount and direction encoded by its inner-corner L marker.

    Input structure:
    - A black/grey checkerboard background fills the whole grid, with the top-left cell black.
    - A full red horizontal line and a full red vertical line cross at the grid center, splitting the board into four equal quadrants.
    - Each quadrant contains one small non-background object and one 3-cell L marker placed in the 2x2 inner-corner block next to the red cross.

    Rule:
    - The L marker opening gives the direction: missing top-left or bottom-right means clockwise, while missing top-right or bottom-left means counterclockwise.
    - Read quadrant marker colors in fixed order TL, TR, BL, BR; the 1st, 2nd, and 3rd newly seen colors mean 1, 2, and 3 quarter-turns, and repeated colors reuse the same count.
    - Rotate only the quadrant object about that quadrant's center by the decoded number of quarter-turns.
    - Clear every L marker back to checkerboard background, while leaving the checkerboard and red cross unchanged.

    Key insight:
    - The marker shape is only half the code: orientation gives the rotation direction, but the rotation count comes from comparing marker colors across all four quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and width % 2 == 1, "center cross requires odd dimensions"

    mid_r, mid_c = height // 2, width // 2
    cross_color = int(g[mid_r, mid_c])
    assert np.all(g[mid_r, :] == cross_color), "expected a full central horizontal line"
    assert np.all(g[:, mid_c] == cross_color), "expected a full central vertical line"

    bg_even = int(g[0, 0])
    bg_odd = int(g[0, 1] if width > 1 else g[1, 0])
    out = _checkerboard(height, width, bg_even, bg_odd)
    out[mid_r, :] = cross_color
    out[:, mid_c] = cross_color

    specs = _quadrant_specs(mid_r, mid_c, height, width)
    marker_info: list[tuple[int, tuple[int, int]]] = []
    for spec in specs:
        quadrant = g[spec["rows"], spec["cols"]]
        marker_info.append(_read_marker(quadrant, str(spec["marker_corner"])))

    turns_by_color: dict[int, int] = {}
    for color, _missing in marker_info:
        if color not in turns_by_color:
            turns_by_color[color] = len(turns_by_color) + 1

    for spec, (marker_color, missing) in zip(specs, marker_info):
        quadrant = g[spec["rows"], spec["cols"]]
        object_mask = (
            (quadrant != bg_even)
            & (quadrant != bg_odd)
            & (quadrant != cross_color)
            & (quadrant != int(marker_color))
        )
        coords = np.argwhere(object_mask)
        values = quadrant[object_mask]
        assert coords.size > 0, "each quadrant must contain an object"

        turns = int(turns_by_color[int(marker_color)])
        steps_cw = _steps_from_marker(missing, turns)
        rotated = _rotate_coords(coords, quadrant.shape[0], quadrant.shape[1], steps_cw)

        row0 = int(spec["row0"])
        col0 = int(spec["col0"])
        for (r, c), value in zip(rotated, values):
            out[row0 + int(r), col0 + int(c)] = int(value)

    return out


def _safe_mask(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    for corner in ("tl", "tr", "bl", "br"):
        r0, c0 = _marker_origin(size, size, corner)
        mask[r0 : r0 + 2, c0 : c0 + 2] = False
    return mask


def _random_object_coords(size: int) -> np.ndarray | None:
    allowed = _safe_mask(size)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if not cells:
        return None

    target_size = random.randint(*OBJECT_SIZE_RANGE)
    for _ in range(120):
        shape = {random.choice(cells)}
        while len(shape) < target_size:
            frontier = []
            for r, c in shape:
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < size and 0 <= nc < size and allowed[nr, nc] and (nr, nc) not in shape:
                        frontier.append((nr, nc))
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) == target_size:
            coords = np.array(sorted(shape), dtype=int)
            return coords
    return None


def _marker_cells(size: int, corner: str, missing: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = _marker_origin(size, size, corner)
    cells = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != tuple(missing):
                cells.append((int(r0 + dr), int(c0 + dc)))
    return cells


def _build_sequence() -> tuple[int, ...]:
    repeated = random.choice((0, 1, 2))
    return tuple(random.choice(REPEAT_SEQUENCES[int(repeated)]))


def _gen_attempt() -> np.ndarray | None:
    quadrant_size = random.randint(*QUADRANT_SIZE_RANGE)
    side = 2 * quadrant_size + 1
    grid = _checkerboard(side, side, BLACK, GREY)
    grid[quadrant_size, :] = RED
    grid[:, quadrant_size] = RED

    colors = list(PALETTE)
    random.shuffle(colors)
    marker_palette = colors[:3]
    object_palette = colors[3:7]

    sequence = _build_sequence()
    marker_colors = [int(marker_palette[label]) for label in sequence]
    turns_by_color: dict[int, int] = {}
    for color in marker_colors:
        if color not in turns_by_color:
            turns_by_color[color] = len(turns_by_color) + 1

    specs = _quadrant_specs(quadrant_size, quadrant_size, side, side)
    missing_by_quadrant = [random.choice(MISSING_POSITIONS) for _ in range(4)]
    object_coords: list[np.ndarray] = []

    for idx, spec in enumerate(specs):
        turns = int(turns_by_color[marker_colors[idx]])
        steps_cw = _steps_from_marker(missing_by_quadrant[idx], turns)

        coords = None
        for _ in range(120):
            candidate = _random_object_coords(quadrant_size)
            if candidate is None:
                return None
            rotated = _rotate_coords(candidate, quadrant_size, quadrant_size, steps_cw)
            if {tuple(map(int, rc)) for rc in candidate} == {tuple(map(int, rc)) for rc in rotated}:
                continue
            coords = candidate
            break
        if coords is None:
            return None
        object_coords.append(coords)

        row0 = int(spec["row0"])
        col0 = int(spec["col0"])
        color = int(object_palette[idx])
        for r, c in coords:
            grid[row0 + int(r), col0 + int(c)] = color

    for idx, spec in enumerate(specs):
        row0 = int(spec["row0"])
        col0 = int(spec["col0"])
        corner = str(spec["marker_corner"])
        for r, c in _marker_cells(quadrant_size, corner, missing_by_quadrant[idx]):
            grid[row0 + int(r), col0 + int(c)] = int(marker_colors[idx])

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
