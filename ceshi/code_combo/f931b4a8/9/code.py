# Combo variant of f931b4a8 (index 9)
# combo_spec: D4/3+D5/1
# Name: 对角条纹模板+旋转铺贴

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# a) Concretization and behavior trace:
#    The input frame stays identical to the base task: an even square split into four
#    quadrants; the top two quadrants encode output height/width by foreground counts.
#    The lower-left quadrant is still a non-black coarse color tile indexed by output
#    block position. The lower-right quadrant changes in two coupled ways:
#      1. Its foreground is no longer an arbitrary mask; it is a diagonal-stripe stencil.
#      2. When copied to the output, each qxq block rotates by coarse-block parity:
#         (even, even) -> original
#         (even, odd)  -> 90 deg clockwise
#         (odd, odd)   -> 180 deg
#         (odd, even)  -> 90 deg counterclockwise
#    Small example with q=4 and a main-diagonal-family stencil: under the base rule,
#    every output block would show the same "\"-leaning stripes. Under the combo rule,
#    the top-right and bottom-left blocks become "/"-leaning, so a 2x2 block patch now
#    forms an X-like crossing pattern. Black stencil holes are still filled from the
#    single lower-left coarse color at [block_r, block_c], so the rotation and the
#    coarse color lookup must be understood together rather than independently.
# b) Complexity calibration:
#    If the stencil were empty, or if it were rotationally indistinguishable, the new
#    rule would collapse back toward the original task. The generator therefore always
#    uses a non-empty diagonal stencil, forces the output to span at least 2x2 coarse
#    blocks, and prefers colorings whose four quarter-turns are all distinct. This keeps
#    the rotation cycle visible while staying within a small vocabulary: one stripe
#    family, one 2x2 rotation cycle, and the same coarse fill logic as the base task.
# c) Hidden assumptions in the original main():
#    - `stencil[r % q, c % q]` assumed a single global orientation. That fails here and
#      is replaced with blockwise selection from four rotated stencil copies.
#    - `fill_tile[r // q, c // q]` assumed the lower-left quadrant is indexed by coarse
#      block position rather than by fine within-block coordinates. That remains valid.
#    - `fill_r < q and fill_c < q` relied on output block counts never exceeding q in
#      either axis. The generator still guarantees `ceil(out_h / q) <= q` and likewise
#      for width, so the coarse lookup remains safe.
#    - The base code tolerated arbitrary stencil shapes. The combo solver itself only
#      needs a square stencil that can be rotated; the generator is what enforces the
#      stronger diagonal-stripe structure required for learnability.
# d) Placement legality:
#    There is no free-form placement; the four quadrants are fixed. Legality reduces to
#    constructing valid qxq quadrants, keeping lower-left fully non-black, keeping the
#    stencil inside its quadrant, and choosing output dimensions whose coarse block counts
#    stay within q so every black stencil hole can read a valid lower-left color.
# e) Visual observability:
#    A human can see the rule only if both changes matter simultaneously. Diagonal
#    stripes provide a clear orientation cue, and the 2x2 rotation cycle turns those
#    cues into alternating "\" and "/" blocks. The generator rejects cases where the
#    rotated output accidentally matches the old periodic output, so the fusion is not
#    merely latent.
# f) Visual cleanliness:
#    The scene remains clean because each quadrant keeps one role, and the lower-right
#    pattern family is restricted to sparse diagonal stripes rather than arbitrary noise.
#    Lower-left colors remain simple solid/stripe/diagonal fills, which keeps the hole
#    replacement readable instead of visually cluttered.
# g) Change inferability:
#    No hidden color remapping is introduced. Upper colors still only encode counts and
#    never appear in the answer. Lower-right foreground colors remain unchanged under
#    rotation; black cells are replaced by lower-left coarse colors. Because the spatial
#    structure, not the raw color ids, carries the new information, a solver can infer
#    the combo rule from a few examples without relying on non-visual conventions.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
LL_SOLID_PROB = 0.68
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_MULTI_COLOR_RANGE = (2, 4)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _rotations_for_tiling(tile: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return the four block-parity orientations used by the combo rule."""
    return {
        (0, 0): tile,
        (0, 1): np.rot90(tile, 3),
        (1, 1): np.rot90(tile, 2),
        (1, 0): np.rot90(tile, 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top-half counts, then build the output from a coarse color tile and a diagonally striped stencil whose orientation rotates by block parity.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control panels: their foreground counts set the output height and width.
    - The lower-left quadrant is a fully non-black coarse color tile.
    - The lower-right quadrant is a qxq stencil tile; in the intended combo instances it contains diagonal foreground stripes over black holes.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant.
    - Output width is the number of non-black cells in the upper-right quadrant.
    - If the stencil is all black, repeat the lower-left tile directly over the output.
    - Otherwise, partition the output into qxq blocks. The block at coarse position (block_r, block_c) uses one of four stencil orientations in a repeating 2x2 cycle:
      original, 90 degrees clockwise, 180 degrees, 90 degrees counterclockwise.
    - Within each oriented block, keep non-black stencil pixels as they are; replace black stencil pixels with the single lower-left coarse color at [block_r, block_c].

    Key insight:
    - The top half still carries only dimensions, but the bottom half now encodes a two-level pattern in which diagonal fine structure and coarse block-parity rotations must be combined to explain the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    q = qh

    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != bg):
        out = fill_tile[rr % q, cc % q].copy()
        assert np.all(out != bg)
        return out

    out = np.empty((out_h, out_w), dtype=int)
    tile_rows = ceil(out_h / q)
    tile_cols = ceil(out_w / q)
    rotations = _rotations_for_tiling(stencil)

    for block_r in range(tile_rows):
        for block_c in range(tile_cols):
            rs = block_r * q
            cs = block_c * q
            re = min(rs + q, out_h)
            ce = min(cs + q, out_w)

            block = rotations[(block_r % 2, block_c % 2)][: re - rs, : ce - cs].copy()
            bg_mask = block == bg
            if np.any(bg_mask):
                assert block_r < q and block_c < q
                fill_value = int(fill_tile[block_r, block_c])
                assert fill_value != bg
                block[bg_mask] = fill_value
            out[rs:re, cs:ce] = block

    return out


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all qxq coordinates in a layout-biased order."""
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


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (always non-black) using simple large-scale patterns."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _rotation_signature_count(tile: np.ndarray) -> int:
    """Count distinct quarter-turn array signatures."""
    return len({np.rot90(tile, k).tobytes() for k in range(4)})


def _diagonal_stripe_mask(q: int, offset: int) -> np.ndarray:
    """Return a sparse foreground mask made of stripes parallel to the main diagonal."""
    if q == 2:
        if offset % 2 == 0:
            return np.eye(q, dtype=bool)
        return np.fliplr(np.eye(q, dtype=bool))

    rr, cc = np.mgrid[:q, :q]
    return (rr - cc - offset) % 3 == 0


def _paint_diagonal_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    """Color a diagonal mask asymmetrically so its rotations stay visually distinct."""
    q = mask.shape[0]
    rr, cc = np.mgrid[:q, :q]
    tile = np.zeros((q, q), dtype=int)
    k = len(palette)
    phase = random.randrange(k)
    color_ids = (2 * rr + 3 * cc + phase) % k
    for i, color in enumerate(palette):
        tile[mask & (color_ids == i)] = color

    fg_coords = [(r, c) for r in range(q) for c in range(q) if mask[r, c]]
    used_colors = set(map(int, tile[mask]))
    missing = [color for color in palette[: min(len(palette), len(fg_coords))] if color not in used_colors]
    if missing:
        random.shuffle(fg_coords)
        for color, (r, c) in zip(missing, fg_coords):
            tile[r, c] = color
    return tile


def _make_diagonal_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a non-empty diagonal-stripe stencil whose rotations matter."""
    area = q * q
    offsets = list(range(2 if q == 2 else 3))
    for _ in range(120):
        mask = _diagonal_stripe_mask(q, random.choice(offsets))
        fg_count = int(mask.sum())
        if fg_count == 0 or fg_count == area:
            continue
        tile = _paint_diagonal_mask(mask, palette)
        if _rotation_signature_count(tile) == 4:
            return tile

    tile = np.zeros((q, q), dtype=int)
    if q == 2:
        tile[0, 0] = palette[0]
        tile[1, 1] = palette[min(1, len(palette) - 1)]
        return tile

    mask = _diagonal_stripe_mask(q, 0)
    coords = [(r, c) for r in range(q) for c in range(q) if mask[r, c]]
    for idx, (r, c) in enumerate(coords):
        tile[r, c] = palette[(idx + (r % len(palette))) % len(palette)]
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black cells."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _periodic_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Original-task oracle used to reject combo samples that visually collapse."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    if np.any(bg_mask):
        tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the combo solver's output for generator validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    out = np.empty((out_h, out_w), dtype=int)
    rotations = _rotations_for_tiling(stencil)
    tile_rows = ceil(out_h / q)
    tile_cols = ceil(out_w / q)

    for block_r in range(tile_rows):
        for block_c in range(tile_cols):
            rs = block_r * q
            cs = block_c * q
            re = min(rs + q, out_h)
            ce = min(cs + q, out_w)
            block = rotations[(block_r % 2, block_c % 2)][: re - rs, : ce - cs].copy()
            bg_mask = block == BLACK
            if np.any(bg_mask):
                block[bg_mask] = fill_tile[block_r, block_c]
            out[rs:re, cs:ce] = block
    return out


def _fusion_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int | bool]:
    """Quantify whether the generated sample actually expresses the combo behavior."""
    combo = _expected_output(fill_tile, stencil, out_h, out_w)
    periodic = _periodic_output(fill_tile, stencil, out_h, out_w)
    q = fill_tile.shape[0]
    return {
        "rotation_changes_output": bool(not np.array_equal(combo, periodic)),
        "orientation_count": int(_rotation_signature_count(stencil)),
        "tile_rows": int(ceil(out_h / q)),
        "tile_cols": int(ceil(out_w / q)),
        "stencil_fg": int(np.count_nonzero(stencil != BLACK)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid combo instance; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    st_k = random.randint(*STENCIL_MULTI_COLOR_RANGE)
    stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = []
    for out_h in range(lo, hi + 1):
        for out_w in range(lo, hi + 1):
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            tile_rows = ceil(out_h / q)
            tile_cols = ceil(out_w / q)
            if tile_rows > q or tile_cols > q:
                continue
            if tile_rows < 2 or tile_cols < 2:
                continue
            if tile_rows * tile_cols < ll_k:
                continue
            valid_dims.append((out_h, out_w))
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = _make_diagonal_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    metrics = _fusion_metrics(fill_tile, stencil, out_h, out_w)
    if not metrics["rotation_changes_output"]:
        return None
    if metrics["orientation_count"] < 4:
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
