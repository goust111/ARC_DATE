from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, tile_fill


BG = 0
GLYPH_COLOR = 1
L_COLOR = 2
FILL_COLOR = 4

CORNERS = ("tl", "tr", "bl", "br")
GLYPHS: dict[str, np.ndarray] = {
    "cross": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "h": np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "x": np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "corners": np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
}

GEN_ATTEMPTS = 300
GEN_SIZES = (7, 8, 9, 10)


def _line_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return points


def _apply_corner_transform(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask
    if corner in ("bl", "br"):
        out = np.flipud(out)
    if corner in ("tr", "br"):
        out = np.fliplr(out)
    return out


def _phase_offset(corner: str, size: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, size - 3
    if corner == "bl":
        return size - 3, 0
    if corner == "br":
        return size - 3, size - 3
    raise AssertionError("invalid corner")


def _canonical_triangle_masks(v_leg: int, h_leg: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    outline = np.zeros((size, size), dtype=bool)
    outline[0, :h_leg] = True
    outline[:v_leg, 0] = True
    for r, c in _line_points(0, h_leg - 1, v_leg - 1, 0):
        outline[r, c] = True

    inside = np.zeros((size, size), dtype=bool)
    for r in range(1, v_leg):
        for c in range(1, h_leg):
            if outline[r, c]:
                continue
            if (r + 0.5) / v_leg + (c + 0.5) / h_leg < 1.0:
                inside[r, c] = True
    return outline, inside


def _render_triangle(
    n: int,
    corner: str,
    h_len: int,
    v_len: int,
    glyph_mask: np.ndarray,
    *,
    phase_corner: str | None = None,
    use_rectangle: bool = False,
) -> np.ndarray:
    size = 3 * int(n) - 2
    h_leg = 2 * int(h_len)
    v_leg = 2 * int(v_len)

    outline, inside = _canonical_triangle_masks(v_leg, h_leg, size)
    if use_rectangle:
        inside = np.zeros((size, size), dtype=bool)
        inside[1:v_leg, 1:h_leg] = True
        inside[outline] = False

    outline = _apply_corner_transform(outline, corner)
    inside = _apply_corner_transform(inside, corner)

    tile = np.where(glyph_mask, FILL_COLOR, BG).astype(int)
    tiled = tile_fill((size, size), tile, offset=_phase_offset(phase_corner or corner, size))

    out = np.zeros((size, size), dtype=int)
    out[inside] = tiled[inside]
    out[outline] = L_COLOR
    return out


def _extract_corner_l(grid: np.ndarray) -> tuple[str, int, int, tuple[int, int, int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[L_COLOR],
    )
    assert len(comps) == 1, "expected one red L-shape"
    comp = comps[0]
    comp_bbox = tuple(map(int, comp["bbox"]))
    r0, c0, r1, c1 = comp_bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert 2 <= h <= 3 and 2 <= w <= 3, "L-shape legs must be length 2 or 3"

    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    H, W = grid.shape

    if r0 == 0 and c0 == 0:
        corner = "tl"
        h_len = int((grid[0] == L_COLOR).sum())
        v_len = int((grid[:, 0] == L_COLOR).sum())
        expected = {(0, c) for c in range(h_len)} | {(r, 0) for r in range(v_len)}
    elif r0 == 0 and c1 == W - 1:
        corner = "tr"
        h_len = int((grid[0] == L_COLOR).sum())
        v_len = int((grid[:, W - 1] == L_COLOR).sum())
        expected = {(0, c) for c in range(W - h_len, W)} | {(r, W - 1) for r in range(v_len)}
    elif r1 == H - 1 and c0 == 0:
        corner = "bl"
        h_len = int((grid[H - 1] == L_COLOR).sum())
        v_len = int((grid[:, 0] == L_COLOR).sum())
        expected = {(H - 1, c) for c in range(h_len)} | {(r, 0) for r in range(H - v_len, H)}
    elif r1 == H - 1 and c1 == W - 1:
        corner = "br"
        h_len = int((grid[H - 1] == L_COLOR).sum())
        v_len = int((grid[:, W - 1] == L_COLOR).sum())
        expected = {(H - 1, c) for c in range(W - h_len, W)} | {(r, W - 1) for r in range(H - v_len, H)}
    else:
        raise AssertionError("red component is not a valid corner L-shape")

    assert coords == expected, "red component must be a clean axis-aligned L-shape"
    assert 2 <= h_len <= 3 and 2 <= v_len <= 3, "invalid L-shape leg length"
    return corner, h_len, v_len, comp_bbox


def _extract_glyph(grid: np.ndarray, l_bbox: tuple[int, int, int, int]) -> np.ndarray:
    blue_mask = grid == GLYPH_COLOR
    blue_bbox = bbox(blue_mask)
    assert blue_bbox is not None, "expected one blue 3x3 glyph"
    r0, c0, r1, c1 = map(int, blue_bbox)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "glyph bbox must be 3x3"

    lr0, lc0, lr1, lc1 = l_bbox
    assert r1 < lr0 or lr1 < r0 or c1 < lc0 or lc1 < c0, "L-shape bbox must not overlap glyph bbox"

    patch = crop_bbox(grid, blue_bbox)
    assert set(map(int, np.unique(patch))).issubset({BG, GLYPH_COLOR}), "glyph patch must only use blue and black"
    glyph_mask = patch == GLYPH_COLOR
    assert any(np.array_equal(glyph_mask, known) for known in GLYPHS.values()), "unknown glyph pattern"

    outside = blue_mask.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not outside.any(), "blue pixels must belong to a single 3x3 glyph bbox"
    return glyph_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the canvas, draw a corner-coded right-triangle outline, and phase-tile its interior from a decoded glyph.

    Input structure:
    - The grid is square with black background.
    - One red L-shape touches exactly one corner; its horizontal and vertical legs each have length 2 or 3.
    - One blue 3x3 glyph sits away from the corner marker; the glyph is one of four fixed outline-style masks.

    Rule:
    - Build a black output whose side length is 3n - 2 for input side n.
    - Double the two L-shape leg lengths and place those doubled legs on the matching output corner edges, then connect their free endpoints to form a red right-triangle outline.
    - Decode the blue 3x3 glyph as a 3x3 yellow/black tile, and repeat that tile across the output with a phase anchored to the same corner: top-left uses a tile flush to the top-left, top-right flush to the top-right, bottom-left flush to the bottom-left, and bottom-right flush to the bottom-right.
    - Keep only the tiled pixels strictly inside the triangle; clear the outline and the exterior back to black, and keep the triangle outline red.

    Key insight:
    The same corner L-shape supplies both triangle geometry and tile phase, so solving requires reusing one visual cue for two different transformations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"
    assert max(g.shape) <= 10, "input side must stay within the puzzle's scale bounds"
    assert set(map(int, np.unique(g))).issubset({BG, GLYPH_COLOR, L_COLOR}), "unexpected input colors"

    corner, h_len, v_len, l_bbox = _extract_corner_l(g)
    glyph_mask = _extract_glyph(g, l_bbox)
    return _render_triangle(g.shape[0], corner, h_len, v_len, glyph_mask)


def _paint_l_shape(grid: np.ndarray, corner: str, h_len: int, v_len: int) -> tuple[int, int, int, int]:
    n = grid.shape[0]
    if corner == "tl":
        grid[0, :h_len] = L_COLOR
        grid[:v_len, 0] = L_COLOR
        return 0, 0, v_len - 1, h_len - 1
    if corner == "tr":
        grid[0, n - h_len :] = L_COLOR
        grid[:v_len, n - 1] = L_COLOR
        return 0, n - h_len, v_len - 1, n - 1
    if corner == "bl":
        grid[n - 1, :h_len] = L_COLOR
        grid[n - v_len :, 0] = L_COLOR
        return n - v_len, 0, n - 1, h_len - 1
    grid[n - 1, n - h_len :] = L_COLOR
    grid[n - v_len :, n - 1] = L_COLOR
    return n - v_len, n - h_len, n - 1, n - 1


def _choose_glyph_start(n: int, corner: str, l_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    lr0, lc0, lr1, lc1 = l_bbox
    center = (n - 1) / 2.0
    target_r = n - 2 if corner in ("tl", "tr") else 1
    target_c = n - 2 if corner in ("tl", "bl") else 1
    candidates: list[tuple[float, int, int, int, int]] = []

    for r0 in range(1, n - 2):
        for c0 in range(1, n - 2):
            r1, c1 = r0 + 2, c0 + 2
            if not (r1 < lr0 or lr1 < r0 or c1 < lc0 or lc1 < c0):
                continue
            center_cost = abs((r0 + 1) - center) + abs((c0 + 1) - center)
            corner_cost = abs(r0 - target_r) + abs(c0 - target_c)
            candidates.append((center_cost, corner_cost, r0, c0, r0 + c0))

    if candidates:
        _, _, r0, c0, _ = min(candidates)
        return r0, c0
    raise AssertionError("failed to place central glyph")


def _build_input(n: int, corner: str, h_len: int, v_len: int, glyph_name: str) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    l_bbox = _paint_l_shape(grid, corner, h_len, v_len)
    gr0, gc0 = _choose_glyph_start(n, corner, l_bbox)
    grid[gr0 : gr0 + 3, gc0 : gc0 + 3][GLYPHS[glyph_name]] = GLYPH_COLOR
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    n = int(grid.shape[0])
    corner, h_len, v_len, l_bbox = _extract_corner_l(grid)
    glyph_mask = _extract_glyph(grid, l_bbox)

    out = main(grid)
    expected = _render_triangle(n, corner, h_len, v_len, glyph_mask)
    assert out.shape == (3 * n - 2, 3 * n - 2)
    assert np.array_equal(out, expected), "main() must match the geometric construction"
    assert np.any(out == FILL_COLOR), "triangle interior must contain visible tiled fill"

    phase_variants = [
        _render_triangle(n, corner, h_len, v_len, glyph_mask, phase_corner=alt)
        for alt in CORNERS
        if alt != corner
    ]
    assert any(not np.array_equal(out, alt) for alt in phase_variants), "tile phase must matter"

    rect_variant = _render_triangle(n, corner, h_len, v_len, glyph_mask, use_rectangle=True)
    assert not np.array_equal(out, rect_variant), "triangle crop must matter"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = random.choice(GEN_SIZES)
        corner = random.choice(CORNERS)
        h_len = random.choice((2, 3))
        v_len = random.choice((2, 3))
        glyph_name = random.choice(tuple(GLYPHS))

        grid = _build_input(n, corner, h_len, v_len, glyph_name)
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
