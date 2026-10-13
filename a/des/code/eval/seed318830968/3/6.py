import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9

CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
STEP_BY_DIR = {(-1, 0): 1, (0, 1): 2, (1, 0): 3, (0, -1): 4}

RING_COLORS = np.array([1, 2, 3, 6, 7, 8], dtype=int)
PANEL_SIZE = 9
PANEL_RADIUS = PANEL_SIZE // 2
POINTER_LEN = 2
LAYOUTS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
MARGIN_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _ring_coords_from_bbox(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _find_pointer_direction(g: np.ndarray, cr: int, cc: int) -> tuple[tuple[int, int], int]:
    runs = []
    h, w = g.shape
    for dr, dc in CARDINALS:
        length = 0
        while True:
            rr = cr + dr * (length + 1)
            cc2 = cc + dc * (length + 1)
            if not (0 <= rr < h and 0 <= cc2 < w):
                break
            if int(g[rr, cc2]) != YELLOW:
                break
            length += 1
        if length > 0:
            runs.append(((dr, dc), length))
    assert len(runs) == 1, "expected exactly one yellow pointer direction per panel"
    return runs[0]


def _detect_panel(g: np.ndarray, cr: int, cc: int) -> dict:
    assert int(g[cr, cc]) == MAROON, "panel center must be maroon"
    h, w = g.shape
    direction, pointer_len = _find_pointer_direction(g, cr, cc)

    radii = []
    for dr, dc in CARDINALS:
        k = 1 + (pointer_len if (dr, dc) == direction else 0)
        while True:
            rr = cr + dr * k
            cc2 = cc + dc * k
            assert 0 <= rr < h and 0 <= cc2 < w, "panel extends out of bounds"
            if int(g[rr, cc2]) != BLACK:
                radii.append(k)
                break
            k += 1

    radius = radii[0]
    assert all(r == radius for r in radii), "panel ring must be a centered square"

    r0, c0 = cr - radius, cc - radius
    r1, c1 = cr + radius, cc + radius
    assert 0 <= r0 and 0 <= c0 and r1 < h and c1 < w, "panel bbox out of bounds"

    ring_coords = _ring_coords_from_bbox(r0, c0, r1, c1)
    assert all(int(g[r, c]) != BLACK for r, c in ring_coords), "panel ring must be fully colored"

    pointer_cells = {(cr + direction[0] * k, cc + direction[1] * k) for k in range(1, pointer_len + 1)}
    allowed = set(ring_coords) | pointer_cells | {(cr, cc)}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in allowed:
                continue
            assert int(g[r, c]) == BLACK, "unexpected non-black cell inside panel"

    return {
        "center": (cr, cc),
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "pointer_cells": sorted(pointer_cells),
        "ring_coords": ring_coords,
        "steps": STEP_BY_DIR[direction],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance each clock panel by rotating its outer color ring clockwise according
    to the pointer direction, then erase the pointer while keeping the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    centers = sorted(map(tuple, np.argwhere(g == MAROON)))
    assert centers, "expected at least one maroon clock center"

    panels = [_detect_panel(g, int(cr), int(cc)) for cr, cc in centers]
    out = g.copy()
    for panel in panels:
        ring_coords = panel["ring_coords"]
        ring_values = np.array([int(g[r, c]) for r, c in ring_coords], dtype=int)
        rotated = np.roll(ring_values, panel["steps"])
        for (r, c), value in zip(ring_coords, rotated):
            out[r, c] = int(value)
        for r, c in panel["pointer_cells"]:
            out[r, c] = BLACK
        cr, cc = panel["center"]
        out[cr, cc] = MAROON

    return out


def _random_ring_sequence(length: int) -> np.ndarray:
    for _ in range(200):
        color_count = int(_RNG.integers(3, len(RING_COLORS) + 1))
        palette = _RNG.choice(RING_COLORS, size=color_count, replace=False)
        seq = np.empty(length, dtype=int)
        for i in range(length):
            options = palette
            if i > 0:
                options = options[options != seq[i - 1]]
            if len(options) == 0:
                break
            seq[i] = int(_RNG.choice(options))
        else:
            if seq[0] == seq[-1]:
                options = palette[(palette != seq[0]) & (palette != seq[-2])]
                if len(options) == 0:
                    continue
                seq[-1] = int(_RNG.choice(options))
            if len(set(map(int, seq))) < 3:
                continue
            if any(np.array_equal(seq, np.roll(seq, step)) for step in range(1, 5)):
                continue
            return seq
    raise RuntimeError("failed to build a non-degenerate ring sequence")


def _balanced_directions(count: int) -> list[tuple[int, int]]:
    directions: list[tuple[int, int]] = []
    while len(directions) < count:
        block = CARDINALS.copy()
        _RNG.shuffle(block)
        directions.extend(block)
    return directions[:count]


def _paint_panel(grid: np.ndarray, r0: int, c0: int, direction: tuple[int, int]) -> None:
    r1 = r0 + PANEL_SIZE - 1
    c1 = c0 + PANEL_SIZE - 1
    ring_coords = _ring_coords_from_bbox(r0, c0, r1, c1)
    ring_values = _random_ring_sequence(len(ring_coords))
    for (r, c), value in zip(ring_coords, ring_values):
        grid[r, c] = int(value)

    cr = r0 + PANEL_RADIUS
    cc = c0 + PANEL_RADIUS
    grid[cr, cc] = MAROON
    dr, dc = direction
    for k in range(1, POINTER_LEN + 1):
        grid[cr + dr * k, cc + dc * k] = YELLOW


def _gen_attempt() -> np.ndarray:
    rows, cols = LAYOUTS[int(_RNG.integers(len(LAYOUTS)))]
    top = int(_RNG.integers(*MARGIN_RANGE))
    bottom = int(_RNG.integers(*MARGIN_RANGE))
    left = int(_RNG.integers(*MARGIN_RANGE))
    right = int(_RNG.integers(*MARGIN_RANGE))

    block_h = rows * PANEL_SIZE + (rows - 1)
    block_w = cols * PANEL_SIZE + (cols - 1)
    grid = np.full((top + block_h + bottom, left + block_w + right), BLACK, dtype=int)

    if rows > 1:
        for r in range(1, rows):
            sep_r = top + r * PANEL_SIZE + (r - 1)
            grid[sep_r, left : left + block_w] = GREY
    if cols > 1:
        for c in range(1, cols):
            sep_c = left + c * PANEL_SIZE + (c - 1)
            grid[top : top + block_h, sep_c] = GREY

    directions = _balanced_directions(rows * cols)
    panel_index = 0
    for pr in range(rows):
        for pc in range(cols):
            r0 = top + pr * (PANEL_SIZE + 1)
            c0 = left + pc * (PANEL_SIZE + 1)
            _paint_panel(grid, r0, c0, directions[panel_index])
            panel_index += 1

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "generated sample must transform"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
