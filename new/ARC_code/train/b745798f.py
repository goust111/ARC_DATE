from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


_LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each 3-pixel L marker into a same-color corner L on an empty background.

    Input structure
    - The grid is an odd square with one dominant background color.
    - Every foreground object is a monochromatic 4-connected 3-pixel L inside a
      2x2 box.

    Rule
    - Clear the grid to the background.
    - For each marker, map its L orientation to the corresponding outer corner
      and draw that corner L in the marker color with arm length half the grid.

    Key insight
    - The small marker is a miniature corner; its orientation, not its position,
      chooses which outer corner receives the expanded L.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    bg = get_bg(g)
    out = np.full_like(g, bg)
    arm = h // 2

    seen_corners: set[str] = set()
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components

    for comp in components:
        assert comp["area"] == 3
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0, c1 - c0) == (1, 1)

        local = frozenset(
            (int(r) - r0, int(c) - c0) for r, c in comp["coords"]
        )
        assert local in _LOCAL_TO_CORNER
        corner = _LOCAL_TO_CORNER[local]
        assert corner not in seen_corners
        seen_corners.add(corner)

        color = int(comp["color"])
        if corner == "top_left":
            out[0, :arm] = color
            out[:arm, 0] = color
        elif corner == "top_right":
            out[0, w - arm :] = color
            out[:arm, w - 1] = color
        elif corner == "bottom_left":
            out[h - arm :, 0] = color
            out[h - 1, :arm] = color
        else:
            assert corner == "bottom_right"
            out[h - arm :, w - 1] = color
            out[h - 1, w - arm :] = color

    assert seen_corners == set(_LOCAL_TO_CORNER.values())
    return out

# --------------- generator (att2) ---------------
# Canvas and palette.
_GEN_BG_COLOR = 8
_GEN_GRID_SIZES = [5, 7, 9, 11, 13, 15]
_GEN_COLOR_VALUES = range(10)
# Corner order and marker geometry.
_GEN_CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
_GEN_MARKERS = {
    "top_left": ((0, 0), (0, 1), (1, 0)),
    "top_right": ((0, 0), (0, 1), (1, 1)),
    "bottom_left": ((0, 0), (1, 0), (1, 1)),
    "bottom_right": ((0, 1), (1, 0), (1, 1)),
}
_GEN_MARKER_SHAPE = (2, 2)
# Color reuse and retry policy.
_GEN_ALL_DISTINCT_COLOR_PROB = 0.5
_GEN_DISTINCT_CORNER_COUNT = 4
_GEN_REUSED_COLOR_BASE_COUNT = 3
_GEN_MAX_ATTEMPTS = 300


def _gen_marker(color: int, bg: int, corner: str) -> np.ndarray:
    marker = np.full(_GEN_MARKER_SHAPE, int(bg), dtype=int)
    for r, c in _GEN_MARKERS[corner]:
        marker[r, c] = int(color)
    return marker


def _gen_output(n: int, bg: int, colors_by_corner: dict[str, int]) -> np.ndarray:
    out = np.full((n, n), int(bg), dtype=int)
    arm = n // 2
    out[0, :arm] = colors_by_corner["top_left"]
    out[:arm, 0] = colors_by_corner["top_left"]
    out[0, n - arm :] = colors_by_corner["top_right"]
    out[:arm, n - 1] = colors_by_corner["top_right"]
    out[n - arm :, 0] = colors_by_corner["bottom_left"]
    out[n - 1, :arm] = colors_by_corner["bottom_left"]
    out[n - arm :, n - 1] = colors_by_corner["bottom_right"]
    out[n - 1, n - arm :] = colors_by_corner["bottom_right"]
    return out


def _gen_same_color_touch(
    canvas: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, color: int
) -> bool:
    h, w = canvas.shape
    for pr, pc in np.argwhere(patch_mask):
        r = int(r0 + pr)
        c = int(c0 + pc)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) == int(color):
                return True
    return False


def _gen_valid_locations(canvas: np.ndarray, patch: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = canvas.shape
    ph, pw = patch.shape
    patch_mask = patch != int(bg)
    color = int(patch[patch_mask][0])
    locs: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            region = canvas[r0 : r0 + ph, c0 : c0 + pw]
            if bool((region != int(bg))[patch_mask].any()):
                continue
            if _gen_same_color_touch(canvas, patch_mask, r0, c0, color):
                continue
            locs.append((r0, c0))
    return locs


def _gen_corner_colors(bg: int) -> dict[str, int]:
    palette = [c for c in _GEN_COLOR_VALUES if c != int(bg)]

    if random.random() < _GEN_ALL_DISTINCT_COLOR_PROB:
        colors = random.sample(palette, _GEN_DISTINCT_CORNER_COUNT)
    else:
        base = random.sample(palette, _GEN_REUSED_COLOR_BASE_COUNT)
        repeated = random.choice(base)
        colors = base + [repeated]
        random.shuffle(colors)
    return dict(zip(_GEN_CORNERS, colors))


def _gen_place_markers(
    inp: np.ndarray, colors_by_corner: dict[str, int], bg: int
) -> bool:
    corners = list(_GEN_CORNERS)
    random.shuffle(corners)
    for corner in corners:
        marker = _gen_marker(colors_by_corner[corner], bg, corner)
        locs = _gen_valid_locations(inp, marker, bg)
        if not locs:
            return False
        r0, c0 = random.choice(locs)
        paste(inp, marker, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_GEN_GRID_SIZES)
    bg = _GEN_BG_COLOR
    colors_by_corner = _gen_corner_colors(bg)
    expected = _gen_output(n, bg, colors_by_corner)

    inp = np.full((n, n), bg, dtype=int)
    if not _gen_place_markers(inp, colors_by_corner, bg):
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid b745798f grid")
