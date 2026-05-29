# Combo variant of f931b4a8 (index 30)
# combo_spec: D4/1+D9/2+D11/2
# Name: 双色语义+遮罩填充+多色模板

from __future__ import annotations

from collections import deque
from itertools import combinations
from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The input is still an even square split into four equal quadrants.
#    - The upper-left and upper-right quadrants still encode output height and width by counting
#      non-black cells.
#    - The lower-left quadrant is now forced to be a two-color coarse source tile.
#    - The lower-right quadrant is no longer a black-hole stencil and no longer a literal overlay:
#      it is a fully non-black semantic template with three roles:
#        * one outside-marker color, visible on the tile border
#        * one inside-marker color, visible only inside a closed boundary
#        * one or two structure colors, forming that closed boundary and preserved in the output
#    - Small q=8 trace:
#        fill_tile uses colors {2, 7}; coarse cell (0,1) = 2, so inside uses 2 and outside uses 7.
#        stencil tile:
#          8 8 8 8 8 8 8 8
#          8 8 8 4 6 8 8 8
#          8 8 4 3 3 6 8 8
#          8 4 3 3 3 3 6 8
#          8 6 3 3 3 3 4 8
#          8 8 6 3 3 4 8 8
#          8 8 8 6 4 8 8 8
#          8 8 8 8 8 8 8 8
#      Here 8 is the outside marker, 3 is the inside marker, and {4,6} are preserved structure colors.
#      Original logic would tile all of 8/3/4/6 directly because they are all non-black.
#      D11-only mask logic would replace both the boundary and the interior side with the coarse
#      source color, erasing 4 and 6. The fused rule instead preserves 4 and 6, replaces only 3 by
#      the coarse inside color, and replaces only 8 by the other source color.
# b) Complexity calibration
#    - The source tile is limited to exactly two colors, so the inside/outside fill choice is legible.
#    - The template is limited to exactly one outside marker, one inside marker, and one or two
#      structure colors. This is enough to visibly differ from the original task without creating an
#      unlearnable many-to-many color mapping.
#    - The boundary shape is always a single clean closed loop, using a rectangle or diamond, so the
#      inside/outside split is visually obvious across a few examples.
# c) Implicit-assumption audit against the original solver
#    - `if value == bg:` used to mean "replace this cell from lower-left". That is invalid now because
#      all three semantic roles in the template are non-black.
#    - `value = stencil[r % q, c % q]` used to mean "non-black template survives". That is invalid now
#      because two template colors are markers that must be replaced rather than copied through.
#    - The old solver never had to infer different meanings among foreground colors. The new solver
#      must infer which colors are boundary structure versus outside-marker versus inside-marker.
#      It does so without hidden numeric conventions: the outside marker is the unique border color,
#      and the structure colors are exactly the inset color subset whose cells separate that border
#      region from a single enclosed marker region.
# d) Placement legality
#    - The generator only uses N=16, so Q=8. That gives enough room for an inset closed boundary with
#      both a non-empty exterior and a non-empty enclosed interior.
#    - The boundary is painted strictly away from the tile edge, so it never collides with the border
#      outside-marker region and never causes ambiguous "touching edge" cases.
#    - Output dimensions stay in [8,16], so coarse coordinates `r // Q` and `c // Q` stay within the
#      8x8 lower-left tile.
# e) Visual observability
#    - The outside marker is directly observable because it is the only template color touching the
#      tile border.
#    - The inside marker is directly observable because it fills the region enclosed by the preserved
#      multicolor boundary.
#    - The preserved structure colors repeat in the output exactly where the boundary appears in the
#      input template, making it clear that some lower-right colors survive while others are replaced.
# f) Visual cleanliness
#    - The template has only one closed boundary and two large marker regions. There is no random
#      scatter and no overlapping symbol system, so the three semantic roles remain readable.
#    - The lower-left tile uses simple binary patterns, keeping the source-color alternation visible
#      without cluttering the output.
# g) Change inferability
#    - The two source colors do not need a hidden ordering by numeric id. For each macro-block, the
#      coarse lower-left cell gives the inside fill color, and the other lower-left color becomes the
#      outside fill color. Repetition across multiple blocks provides the evidence.
#    - Marker colors are distinct from structure colors and are both removed from the output, which
#      makes their replaceable role visible to a human observer.
# h) Variant-specific non-degeneracy target
#    - Every generated sample must differ from the original overlay rule because both marker colors are
#      non-black yet never survive into the final output.
#    - Every generated sample must also differ from the plain D11 mask-fill rule because structure
#      colors remain visible in the fused output instead of being absorbed into the inside fill color.
# i) New solver assumptions and generator guarantees
#    - Assumption: the lower-left tile contains exactly two non-black colors.
#      Guarantee: `_make_lower_left` constructs a fully non-black binary tile and repairs used coarse
#      coordinates so both colors appear.
#    - Assumption: the lower-right tile has exactly one border-touching non-black color and at least
#      one enclosed region of a single marker color.
#      Guarantee: `_make_semantic_template` fills the whole exterior with one marker color and the
#      interior with another marker color.
#    - Assumption: one or two inset structure colors form the only boundary subset that separates the
#      border marker from the enclosed marker.
#      Guarantee: `_make_semantic_template` builds exactly such a boundary, and `_infer_template_roles`
#      checks the same property during oracle validation.


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Return the cells in `mask` that are reachable from the border by 4-connectivity."""
    q = mask.shape[0]
    seen = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(q):
        for c in (0, q - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(q):
        for r in (0, q - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < q and 0 <= nc < q and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _fill_colors(fill_tile: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(v) for v in np.unique(fill_tile)]
    assert bg not in colors
    assert len(colors) == 2
    return colors[0], colors[1]


def _infer_template_roles(
    stencil: np.ndarray,
    bg: int,
) -> tuple[int, int, np.ndarray]:
    """
    Infer `(outside_marker, inside_marker, structure_mask)` from a semantic template.

    The outside marker is the unique non-black border color. The structure mask is the unique
    1- or 2-color inset subset whose removal reveals a border-connected outside region of only
    the outside marker plus a non-empty enclosed inside region of one different color.
    """
    q = stencil.shape[0]
    colors = [int(v) for v in np.unique(stencil) if int(v) != bg]
    assert len(colors) >= 3

    border_values = np.concatenate(
        [stencil[0, :], stencil[q - 1, :], stencil[1 : q - 1, 0], stencil[1 : q - 1, q - 1]]
    )
    border_colors = {int(v) for v in border_values if int(v) != bg}
    assert len(border_colors) == 1
    outside_marker = next(iter(border_colors))

    candidate_colors = [c for c in colors if c != outside_marker]
    valid: list[tuple[int, np.ndarray]] = []
    max_struct_colors = min(2, len(candidate_colors) - 1)
    for size in range(1, max_struct_colors + 1):
        for struct_subset in combinations(candidate_colors, size):
            structure_mask = np.isin(stencil, struct_subset)
            if not np.any(structure_mask):
                continue
            if (
                np.any(structure_mask[0, :])
                or np.any(structure_mask[q - 1, :])
                or np.any(structure_mask[:, 0])
                or np.any(structure_mask[:, q - 1])
            ):
                continue

            non_structure = ~structure_mask
            outside_mask = _border_connected(non_structure)
            inside_mask = non_structure & ~outside_mask
            if not np.any(inside_mask) or not np.any(outside_mask):
                continue

            outside_colors = {int(v) for v in stencil[outside_mask]}
            inside_colors = {int(v) for v in stencil[inside_mask]}
            if outside_colors != {outside_marker}:
                continue
            if len(inside_colors) != 1:
                continue
            inside_marker = next(iter(inside_colors))
            if inside_marker == outside_marker or inside_marker in struct_subset:
                continue
            valid.append((inside_marker, structure_mask))

    assert len(valid) == 1
    inside_marker, structure_mask = valid[0]
    return outside_marker, inside_marker, structure_mask


def _compose_combo_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Compose the output for the fused multi-color semantic-mask rule."""
    bg = int(Color.BLACK)
    q = fill_tile.shape[0]
    outside_marker, inside_marker, _structure_mask = _infer_template_roles(stencil, bg)
    color0, color1 = _fill_colors(fill_tile, bg)

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        coarse_r = r // q
        fine_r = r % q
        assert coarse_r < q
        for c in range(out_w):
            coarse_c = c // q
            fine_c = c % q
            assert coarse_c < q
            primary = int(fill_tile[coarse_r, coarse_c])
            secondary = color1 if primary == color0 else color0
            value = int(stencil[fine_r, fine_c])
            if value == inside_marker:
                out[r, c] = primary
            elif value == outside_marker:
                out[r, c] = secondary
            else:
                out[r, c] = value
    return out


def _legacy_original_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Original f931b4a8 overlay rule, used only for degeneracy rejection."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _legacy_d11_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """D11-only mask-fill rule, used only for degeneracy rejection."""
    bg = int(Color.BLACK)
    q = fill_tile.shape[0]
    _outside_marker, _inside_marker, structure_mask = _infer_template_roles(stencil, bg)
    non_structure = ~structure_mask
    outside_mask = _border_connected(non_structure)
    inside_side = ~outside_mask

    color0, color1 = _fill_colors(fill_tile, bg)
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        coarse_r = r // q
        fine_r = r % q
        for c in range(out_w):
            coarse_c = c // q
            fine_c = c % q
            primary = int(fill_tile[coarse_r, coarse_c])
            secondary = color1 if primary == color0 else color0
            out[r, c] = primary if inside_side[fine_r, fine_c] else secondary
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control quadrants to size an output that combines a binary source palette with a multicolor semantic template.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background for the control quadrants only.
    - The upper-left and upper-right quadrants encode output height and width by foreground counts.
    - The lower-left quadrant is a two-color coarse source tile.
    - The lower-right quadrant is a fully non-black multicolor template containing one outside marker color, one inside marker color, and one or two preserved structure colors.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - Infer the template roles from the lower-right quadrant: the unique border color is the outside marker, an inset one- or two-color boundary is the preserved structure, and the single enclosed non-structure color is the inside marker.
    - Tile the lower-right template over the output. For each output macro-block, the corresponding lower-left coarse cell gives the inside fill color and the other lower-left color gives the outside fill color.
    - Replace inside-marker cells by that macro-block's inside fill color, replace outside-marker cells by the other source color, and keep structure-color cells unchanged.

    Key insight:
    - The top half still controls only the output size, but the lower-right quadrant now mixes two kinds of information at once: some colors are literal geometry that survive, while two other colors are semantic placeholders whose meaning comes from the closed-mask partition and the lower-left two-color source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    return _compose_combo_output(fill_tile, stencil, out_h, out_w)


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
STRUCTURE_TWO_COLOR_PROB = 0.72
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 8
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0 : r0 + hh, c0 : c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _make_lower_left(q: int, palette: list[int], required_coords: list[tuple[int, int]]) -> np.ndarray:
    c0, c1 = palette
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "checker", "diagonal", "quadrants"])

    if mode == "rows":
        for r in range(q):
            tile[r, :] = c0 if r % 2 == 0 else c1
    elif mode == "cols":
        for c in range(q):
            tile[:, c] = c0 if c % 2 == 0 else c1
    elif mode == "checker":
        rr, cc = np.mgrid[:q, :q]
        tile[:] = np.where((rr + cc) % 2 == 0, c0, c1)
    elif mode == "diagonal":
        rr, cc = np.mgrid[:q, :q]
        tile[:] = np.where((rr + cc) % 3 == 0, c0, c1)
    else:
        tile[: q // 2, : q // 2] = c0
        tile[: q // 2, q // 2 :] = c1
        tile[q // 2 :, : q // 2] = c1
        tile[q // 2 :, q // 2 :] = c0

    used_colors = {int(tile[r, c]) for r, c in required_coords}
    missing = [color for color in palette if color not in used_colors]
    if missing:
        fix_coords = list(required_coords)
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    assert np.all(tile != BLACK)
    assert len(np.unique(tile)) == 2
    return tile


def _rect_boundary(q: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    top = random.randint(1, 2)
    left = random.randint(1, 2)
    bottom = random.randint(q - 4, q - 2)
    right = random.randint(q - 4, q - 2)
    if bottom - top < 3:
        bottom = top + 3
    if right - left < 3:
        right = left + 3
    bottom = min(bottom, q - 2)
    right = min(right, q - 2)

    coords: list[tuple[int, int]] = []
    inside = np.zeros((q, q), dtype=bool)
    inside[top + 1 : bottom, left + 1 : right] = True
    for c in range(left, right + 1):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return coords, inside


def _diamond_boundary(q: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    center = q // 2 - 1
    radius = 2
    coords: list[tuple[int, int]] = []
    inside = np.zeros((q, q), dtype=bool)
    for r in range(q):
        for c in range(q):
            dist = abs(r - center) + abs(c - center)
            if dist == radius:
                coords.append((r, c))
            elif dist < radius:
                inside[r, c] = True
    assert all(0 < r < q - 1 and 0 < c < q - 1 for r, c in coords)
    return coords, inside


def _paint_structure(tile: np.ndarray, coords: list[tuple[int, int]], palette: list[int]) -> None:
    if len(palette) == 1:
        for r, c in coords:
            tile[r, c] = palette[0]
        return
    for i, (r, c) in enumerate(coords):
        tile[r, c] = palette[i % len(palette)]


def _make_semantic_template(
    q: int,
    outside_marker: int,
    inside_marker: int,
    structure_palette: list[int],
) -> np.ndarray:
    tile = np.full((q, q), outside_marker, dtype=int)
    mode = random.choice(["rect", "rect", "diamond"])
    coords, inside = _rect_boundary(q) if mode == "rect" else _diamond_boundary(q)
    tile[inside] = inside_marker
    _paint_structure(tile, coords, structure_palette)

    outside, inferred_inside, structure_mask = _infer_template_roles(tile, BLACK)
    assert outside == outside_marker
    assert inferred_inside == inside_marker
    assert len(np.unique(tile)) == 2 + len(structure_palette)
    assert np.all(tile != BLACK)
    assert np.any(inside)
    assert np.any(structure_mask)
    return tile


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return _compose_combo_output(fill_tile, stencil, out_h, out_w)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_palette = _palette(2)
    stencil_pool = [c for c in COLORS if c not in ll_palette]
    if len(stencil_pool) < 3:
        return None

    structure_k = 2 if random.random() < STRUCTURE_TWO_COLOR_PROB else 1
    stencil_palette = _palette(2 + structure_k, stencil_pool)
    outside_marker = stencil_palette[0]
    inside_marker = stencil_palette[1]
    structure_palette = stencil_palette[2:]

    occupied = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in occupied]
    if not control_pool:
        return None

    valid_dims = [
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for out_w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX and ceil(out_h / q) * ceil(out_w / q) >= 2
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    if len({int(fill_tile[r, c]) for r, c in used_fill}) < 2:
        return None

    stencil = _make_semantic_template(q, outside_marker, inside_marker, structure_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    legacy_original = _legacy_original_output(fill_tile, stencil, out_h, out_w)
    legacy_d11 = _legacy_d11_output(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, legacy_original):
        return None
    if np.array_equal(expected, legacy_d11):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
