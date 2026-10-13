from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
CORE_COLORS = (1, 2, 3, 4, 7)
CW_CORNERS = ("NW", "NE", "SE", "SW")
GEN_RETRY_LIMIT = 500

OUTSIDE_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SE": (1, 1),
    "SW": (1, -1),
}

L_MARKER_OFFSETS = {
    "NW": ((-1, -1), (-1, 0), (0, -1)),
    "NE": ((-1, 0), (-1, 1), (0, 1)),
    "SE": ((0, 1), (1, 0), (1, 1)),
    "SW": ((0, -1), (1, -1), (1, 0)),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_coords(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SE": (r1, c1),
        "SW": (r1, c0),
    }


def _outside_pos(corner_name: str, corner_pos: tuple[int, int]) -> tuple[int, int]:
    dr, dc = OUTSIDE_OFFSETS[corner_name]
    return corner_pos[0] + dr, corner_pos[1] + dc


def _l_marker_cells(corner_name: str, corner_pos: tuple[int, int]) -> set[tuple[int, int]]:
    return {
        (corner_pos[0] + dr, corner_pos[1] + dc)
        for dr, dc in L_MARKER_OFFSETS[corner_name]
    }


def _cw_rotate(grid: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(grid, -int(steps))


def _rotation_steps(src_corner: str, dst_corner: str) -> int:
    src_idx = CW_CORNERS.index(src_corner)
    dst_idx = CW_CORNERS.index(dst_corner)
    return (dst_idx - src_idx) % 4


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, str, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    core_mask = ~np.isin(g, [BLACK, GREY, PINK])
    core_bbox = _bbox(core_mask)
    assert core_bbox is not None, "missing core grid"
    r0, c0, r1, c1 = core_bbox
    n = r1 - r0 + 1
    assert n == c1 - c0 + 1, "core grid must be square"
    core = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(core != BLACK) and np.all(core != GREY) and np.all(core != PINK), "core colors must exclude marker colors"

    corners = _corner_coords(core_bbox)

    pink_coords = np.argwhere(g == PINK)
    grey_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    assert pink_coords.shape == (1, 2), "expected one pink singleton marker"
    assert len(grey_coords) == 3, "expected one grey L marker"
    pink_coord = tuple(map(int, pink_coords[0]))

    src_matches = [
        corner_name
        for corner_name, pos in corners.items()
        if _outside_pos(corner_name, pos) == pink_coord
    ]
    tgt_matches = [
        corner_name
        for corner_name, pos in corners.items()
        if _l_marker_cells(corner_name, pos) == grey_coords
    ]
    assert len(src_matches) == 1, "pink marker must identify a unique source corner"
    assert len(tgt_matches) == 1, "grey marker must identify a unique target corner"

    src_corner = src_matches[0]
    tgt_corner = tgt_matches[0]
    steps = _rotation_steps(src_corner, tgt_corner)
    assert steps in (1, 2, 3), "rotation must be a non-zero quarter-turn"
    return core.copy(), src_corner, tgt_corner


def _scaled_output(core: np.ndarray, steps: int) -> np.ndarray:
    n = int(core.shape[0])
    rotated = _cw_rotate(core, steps)
    masks = {
        int(color): np.where(rotated == int(color), int(color), BLACK)
        for color in np.unique(core)
    }
    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = masks[int(core[r, c])]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the marker-defined quarter-turn and use the rotated core grid as the color mask for pixel expansion.

    Input structure:
    - A square colored core grid uses 3-4 colors chosen from blue, red, green, yellow, and orange.
    - Black may appear only outside the core as padding around the markers.
    - One pink singleton sits diagonally outside one core corner, and one grey 3-pixel L sits in a 2x2 corner block whose missing cell is another core corner.

    Rule:
    - Read the pink singleton as the source corner and the missing grey-corner as the target corner.
    - Compute the non-zero clockwise quarter-turn count that maps the source corner to the target corner.
    - Remove the markers conceptually, rotate the core grid by that count, and keep the original core positions unchanged.
    - Replace each core cell by an N x N block that writes its own color exactly where the rotated core also has that color, and writes black elsewhere.

    Key insight:
    - The rotation changes the global mask template used inside every enlarged block, not the spatial placement of the original core cells.
    """
    core, src_corner, tgt_corner = _parse_input(grid)
    return _scaled_output(core, _rotation_steps(src_corner, tgt_corner))


def _sample_core(n: int, color_count: int) -> np.ndarray | None:
    colors = random.sample(CORE_COLORS, color_count)
    min_cells = 2
    total = n * n
    flat = []
    for color in colors:
        flat.extend([color] * min_cells)
    flat.extend(random.choice(colors) for _ in range(total - len(flat)))
    random.shuffle(flat)
    core = np.array(flat, dtype=int).reshape(n, n)

    counts = Counter(map(int, core.flat))
    if len(counts) != color_count:
        return None
    if any(v >= total for v in counts.values()):
        return None
    if any(np.array_equal(core, _cw_rotate(core, steps)) for steps in (1, 2, 3)):
        return None
    return core


def _rotation_sensitive_colors(core: np.ndarray, steps: int) -> int:
    rotated = _cw_rotate(core, steps)
    return sum(
        not np.array_equal(core == color, rotated == color)
        for color in np.unique(core)
    )


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        core, src_corner, tgt_corner = _parse_input(inp)
    except AssertionError:
        return False

    n = int(core.shape[0])
    if n not in (4, 5):
        return False

    colors = tuple(sorted(map(int, np.unique(core))))
    if len(colors) not in (3, 4):
        return False
    if any(color not in CORE_COLORS for color in colors):
        return False
    if any(np.count_nonzero(core == color) < 2 for color in colors):
        return False

    steps = _rotation_steps(src_corner, tgt_corner)
    if _rotation_sensitive_colors(core, steps) < 2:
        return False

    outputs = [_scaled_output(core, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(outputs[i], outputs[j]):
                return False

    correct = outputs[steps]
    if np.array_equal(correct, outputs[0]):
        return False
    return True


def _build_input(core: np.ndarray, src_corner: str, tgt_corner: str) -> np.ndarray:
    n = int(core.shape[0])
    base = np.zeros((n + 2, n + 2), dtype=int)
    base[1 : n + 1, 1 : n + 1] = core

    core_bbox = (1, 1, n, n)
    corners = _corner_coords(core_bbox)

    sr, sc = _outside_pos(src_corner, corners[src_corner])
    base[sr, sc] = PINK
    for r, c in _l_marker_cells(tgt_corner, corners[tgt_corner]):
        base[r, c] = GREY

    top = random.randint(0, 1)
    bottom = random.randint(0, 1)
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    out = np.zeros((base.shape[0] + top + bottom, base.shape[1] + left + right), dtype=int)
    out[top : top + base.shape[0], left : left + base.shape[1]] = base
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((4, 5))
    color_count = random.choice((3, 4))
    src_corner = random.choice(CW_CORNERS)
    steps = random.choice((1, 2, 3))
    tgt_corner = CW_CORNERS[(CW_CORNERS.index(src_corner) + steps) % 4]

    core = _sample_core(n, color_count)
    if core is None:
        return None

    inp = _build_input(core, src_corner, tgt_corner)
    if not _semantic_checks(inp):
        return None

    expected = _scaled_output(core, steps)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid rotation-mask scaling input")
