from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
INDICATOR = 2

DIRECTIONS = ("up", "down", "left", "right")
MARKER_TO_DIRECTION = {
    (1, 2): "up",
    (3, 2): "down",
    (2, 1): "left",
    (2, 3): "right",
}


def _canonical_templates() -> dict[str, np.ndarray]:
    base = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    return {
        "up": base.copy(),
        "right": np.rot90(base, -1).copy(),
        "down": np.rot90(base, 2).copy(),
        "left": np.rot90(base, 1).copy(),
    }


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def _extract_arrow_templates(g: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    grey_mask = g == GREY
    templates = _canonical_templates()
    comps = _mask_components(grey_mask)

    patches: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 != 3 or c1 - c0 + 1 != 3:
            continue
        patch = grey_mask[r0 : r0 + 3, c0 : c0 + 3].copy()
        if int(patch.sum()) >= 3:
            patches.append((r0, c0, patch))

    if len(patches) >= 4:
        patches.sort(key=lambda item: (item[0], item[1]))
        row_major = patches[:4]
        # The prompt's 3x3 arrow rotations are visually identical, so the
        # library order is only needed to preserve the "choose a template" step.
        for direction, (_, _, patch) in zip(("up", "down", "left", "right"), row_major):
            templates[direction] = patch

    return templates, grey_mask


def _ring_arrow_top_left(r0: int, c0: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return r0 - 3, c0 + 1
    if direction == "down":
        return r0 + 5, c0 + 1
    if direction == "left":
        return r0 + 1, c0 - 3
    if direction == "right":
        return r0 + 1, c0 + 5
    raise ValueError(direction)


def _find_rings(g: np.ndarray) -> list[dict[str, int | str]]:
    h, w = g.shape
    rings: list[dict[str, int | str]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            patch = g[r0 : r0 + 5, c0 : c0 + 5]
            color = int(patch[0, 0])
            if color in (BG, GREY):
                continue
            border = np.zeros((5, 5), dtype=bool)
            border[0, :] = True
            border[4, :] = True
            border[:, 0] = True
            border[:, 4] = True
            if not np.all(patch[border] == color):
                continue
            inner = patch[1:4, 1:4]
            marker_cells = np.argwhere(inner != BG)
            if len(marker_cells) != 1:
                continue
            ir, ic = map(int, marker_cells[0])
            local = (ir + 1, ic + 1)
            if local not in MARKER_TO_DIRECTION:
                continue
            marker_color = int(inner[ir, ic])
            if marker_color in (color, GREY):
                continue
            direction = MARKER_TO_DIRECTION[local]
            ar, ac = _ring_arrow_top_left(r0, c0, direction)
            assert 0 <= ar and 0 <= ac and ar + 3 <= h and ac + 3 <= w, "arrow must fit outside each ring"
            rings.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "color": color,
                    "marker_r": int(r0 + local[0]),
                    "marker_c": int(c0 + local[1]),
                    "direction": direction,
                }
            )
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the arrow indicated by each hollow ring's internal marker outside that ring in the ring color.

    Input structure:
    - The background is black, with several non-grey 5x5 hollow square rings in distinct colors.
    - Each ring's 3x3 interior is black except for one shared indicator-color pixel at the upper, lower, left, or right middle interior cell.
    - A grey 2x2 library of 3x3 arrow templates sits in one corner of the grid.

    Rule:
    - Read each ring's interior marker position as a cardinal direction: upper middle, lower middle, left middle, or right middle.
    - Select the corresponding 3x3 grey arrow template, recolor its grey cells to the ring's border color, and draw it just outside that side of the ring.
    - Keep the colored rings, clear the indicator pixels, and clear the grey template library to black.

    Key insight: The marker position inside each frame is a direction code, while the corner grey shapes are reusable arrow stencils whose visible pixels inherit the frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates, grey_mask = _extract_arrow_templates(g)
    rings = _find_rings(g)
    assert rings, "expected at least one marked 5x5 ring"

    out = g.copy()
    out[grey_mask] = BG

    for ring in rings:
        out[int(ring["marker_r"]), int(ring["marker_c"])] = BG

    for ring in rings:
        direction = str(ring["direction"])
        color = int(ring["color"])
        ar, ac = _ring_arrow_top_left(int(ring["r0"]), int(ring["c0"]), direction)
        mask = templates[direction]
        for rr, cc in np.argwhere(mask):
            out[ar + int(rr), ac + int(cc)] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (4, 7)
RING_COLORS = [1, 3, 4, 6, 7, 8, 9]
TEMPLATE_GAP = 1
GEN_RETRIES = 500


def _template_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    size = 3 * 2 + TEMPLATE_GAP
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - size
    if corner == "bl":
        return h - size, 0
    if corner == "br":
        return h - size, w - size
    raise ValueError(corner)


def _draw_template_library(grid: np.ndarray, corner: str) -> set[tuple[int, int]]:
    h, w = grid.shape
    r0, c0 = _template_origin(corner, h, w)
    templates = _canonical_templates()
    order = [
        ("up", 0, 0),
        ("down", 0, 3 + TEMPLATE_GAP),
        ("left", 3 + TEMPLATE_GAP, 0),
        ("right", 3 + TEMPLATE_GAP, 3 + TEMPLATE_GAP),
    ]
    occupied: set[tuple[int, int]] = set()
    block_size = 3 * 2 + TEMPLATE_GAP
    for rr in range(r0, r0 + block_size):
        for cc in range(c0, c0 + block_size):
            occupied.add((rr, cc))
    for direction, dr, dc in order:
        mask = templates[direction]
        for pr, pc in np.argwhere(mask):
            grid[r0 + dr + int(pr), c0 + dc + int(pc)] = GREY
    return occupied


def _box_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _object_footprint(r0: int, c0: int, direction: str, h: int, w: int) -> set[tuple[int, int]] | None:
    ar, ac = _ring_arrow_top_left(r0, c0, direction)
    if ar < 0 or ac < 0 or ar + 3 > h or ac + 3 > w:
        return None
    return _box_cells(r0, c0, 5, 5) | _box_cells(ar, ac, 3, 3)


def _has_clearance(occupied: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark(occupied: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _draw_ring(grid: np.ndarray, r0: int, c0: int, color: int, direction: str) -> None:
    grid[r0, c0 : c0 + 5] = color
    grid[r0 + 4, c0 : c0 + 5] = color
    grid[r0 : r0 + 5, c0] = color
    grid[r0 : r0 + 5, c0 + 4] = color

    marker = {
        "up": (r0 + 1, c0 + 2),
        "down": (r0 + 3, c0 + 2),
        "left": (r0 + 2, c0 + 1),
        "right": (r0 + 2, c0 + 3),
    }[direction]
    grid[marker] = INDICATOR


def _place_one_ring(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    direction: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    positions = [(r, c) for r in range(h - 4) for c in range(w - 4)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        footprint = _object_footprint(r0, c0, direction, h, w)
        if footprint is None:
            continue
        if not _has_clearance(occupied, footprint):
            continue
        _draw_ring(grid, r0, c0, color, direction)
        _mark(occupied, footprint)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    template_cells = _draw_template_library(grid, corner)
    _mark(occupied, template_cells)

    max_objects = min(OBJECT_COUNT_RANGE[1], len(RING_COLORS))
    n_objects = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)
    colors = rng.sample(RING_COLORS, n_objects)
    directions = list(DIRECTIONS)
    directions.extend(rng.choice(DIRECTIONS) for _ in range(n_objects - len(DIRECTIONS)))
    rng.shuffle(directions)

    for color, direction in zip(colors, directions):
        if not _place_one_ring(grid, occupied, color, direction, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833 task 4 input")
