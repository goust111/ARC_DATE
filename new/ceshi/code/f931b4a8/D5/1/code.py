# Variant of f931b4a8
# Dimension: D5 — 铺贴方式 (transform)
# Change: 模板以周期铺贴方式填充输出矩形：output[r][c] = stencil[r % H_stencil][c % W_... → 旋转铺贴（逐象限旋转90度）

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the D5 variant
# a) Concretization and behavior trace:
#    The input layout is unchanged: an even square split into four equal quadrants.
#    The top quadrants still encode only output height/width by foreground counts.
#    The lower-left quadrant still provides one coarse color per qxq output block.
#    The lower-right quadrant is still a qxq stencil, but the qxq block at coarse
#    position (block_r, block_c) no longer always uses the same orientation:
#      (even, even) -> original
#      (even, odd)  -> 90 deg clockwise
#      (odd, odd)   -> 180 deg
#      (odd, even)  -> 90 deg counterclockwise
#    Example with q=4: under the old rule, output rows 0..3 and 4..7 used the same
#    stencil orientation. Under the new rule, the top-left 4x4 block uses the raw
#    stencil, the top-right 4x4 block uses the clockwise rotation, the bottom-right
#    block uses the 180-degree rotation, and the bottom-left block uses the counter-
#    clockwise rotation. Holes in each rotated block are still filled with the single
#    coarse color from lower_left[block_r, block_c].
# b) Complexity calibration:
#    If the stencil were rotationally symmetric, the new rule would look identical to
#    the old periodic tiling. To keep the change visible, the generator enforces four
#    distinct stencil rotations and requires at least a 2x2 block span whenever the
#    stencil is non-empty. This is enough structure for a human to notice the rotation
#    cycle without introducing unrelated control signals.
# c) Hidden assumptions in the original main():
#    - `stencil[r % q, c % q]` assumed one fixed stencil orientation globally; this
#      is the main line that breaks under the variant and must be replaced.
#    - `fill_r = r // q`, `fill_c = c // q` depended only on coarse block position,
#      not on stencil orientation. That assumption still holds and is preserved.
#    - The original asserts relied on output dimensions staying within q*q so that
#      coarse block indices remain < q. The generator still guarantees that bound.
# d) Placement legality:
#    No free placement is involved; each quadrant is axis-aligned and fixed. Generator
#    legality reduces to keeping the coarse block counts <= q in both axes so every
#    hole can read a valid lower-left color, and to assembling an even square grid.
# e) Visual observability:
#    The generator rejects any non-empty stencil whose rotated output matches the old
#    periodic output on the same input. This directly prevents silent regression back
#    to the base task. Stencil-empty cases are kept rare as an edge branch only.
# f) Visual cleanliness:
#    The scene remains the same clean 2x2 quadrant composition. To avoid messy output,
#    lower-left patterns stay simple (solid/row/col/diagonal) and stencil patterns are
#    sparse masks rather than overlapping objects.
# g) Color inference:
#    No new color mapping is introduced. Colors keep their original roles: upper colors
#    are irrelevant to output identity, lower-left colors fill holes, and lower-right
#    foreground colors remain unchanged. The only new inference is spatial rotation.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.08
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _rotations_for_tiling(tile: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Precompute the four parity-based stencil orientations."""
    return {
        (0, 0): tile,
        (0, 1): np.rot90(tile, 3),
        (1, 1): np.rot90(tile, 2),
        (1, 0): np.rot90(tile, 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top controls to size an output assembled from the bottom quadrants, with a rotating stencil cycle.

    Input structure:
    - The input is an even square cut into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control panels whose foreground counts encode output height and width.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is either an empty tile or a stencil tile.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the output as before.
    - Otherwise, split the output into qxq blocks. The block orientations follow a repeating 2x2 cycle:
      original, 90 deg clockwise, 180 deg, 90 deg counterclockwise. Within each block, black stencil cells are replaced by the single coarse color from the matching lower-left cell indexed by that block's row and column.

    Key insight:
    - The top half still supplies only dimensions; the lower-left tile supplies one color per output block, while the lower-right tile supplies a fine mask whose orientation depends on block parity.
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

    if not np.any(stencil != bg):
        rr, cc = np.mgrid[:out_h, :out_w]
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

            stencil_block = rotations[(block_r % 2, block_c % 2)][: re - rs, : ce - cs].copy()
            bg_mask = stencil_block == bg
            if np.any(bg_mask):
                assert block_r < q and block_c < q
                fill_value = int(fill_tile[block_r, block_c])
                assert fill_value != bg
                stencil_block[bg_mask] = fill_value
            out[rs:re, cs:ce] = stencil_block

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
    """Return all (r,c) in a qxq grid in a layout-dependent order."""
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
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
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
    """Count how many distinct quarter-turn orientations this tile has."""
    signatures = {np.rot90(tile, k).tobytes() for k in range(4)}
    return len(signatures)


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a non-empty stencil whose four quarter-turns are all distinct."""
    area = q * q
    modes = ["random", "random", "random", "row_stripe", "col_stripe", "corner_l"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")

    for _ in range(120):
        tile = np.zeros((q, q), dtype=int)
        mode = random.choice(modes)

        if mode == "row_stripe":
            rows = [r for r in range(q) if random.random() < 0.45]
            if not rows:
                rows = [random.randrange(q)]
            for r in rows:
                tile[r, :] = _random_color_array((q,), palette)
        elif mode == "col_stripe":
            cols = [c for c in range(q) if random.random() < 0.45]
            if not cols:
                cols = [random.randrange(q)]
            for c in cols:
                tile[:, c] = _random_color_array((q,), palette)
        elif mode == "frame":
            t = random.randint(1, max(1, q // 4))
            mask = np.ones((q, q), dtype=bool)
            mask[t : q - t, t : q - t] = False
            tile[mask] = _random_color_array((int(mask.sum()),), palette)
            # Knock out one asymmetric bite so rotations remain distinguishable.
            bite_h = random.randint(1, t)
            bite_w = random.randint(1, t)
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                c0 = random.randrange(q - bite_w + 1)
                tile[:bite_h, c0 : c0 + bite_w] = BLACK
            elif side == "bottom":
                c0 = random.randrange(q - bite_w + 1)
                tile[q - bite_h :, c0 : c0 + bite_w] = BLACK
            elif side == "left":
                r0 = random.randrange(q - bite_h + 1)
                tile[r0 : r0 + bite_h, :bite_w] = BLACK
            else:
                r0 = random.randrange(q - bite_h + 1)
                tile[r0 : r0 + bite_h, q - bite_w :] = BLACK
        elif mode == "corner_l":
            arm_r = random.randint(1, q)
            arm_c = random.randint(1, q)
            color = random.choice(palette)
            tile[:arm_r, 0] = color
            tile[0, :arm_c] = _random_color_array((arm_c,), palette)
            extra = random.randint(0, max(1, area // 4))
            coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] == BLACK]
            random.shuffle(coords)
            for r, c in coords[:extra]:
                tile[r, c] = random.choice(palette)
        else:
            lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
            hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
            fg_count = random.randint(lo, hi)
            coords = [(r, c) for r in range(q) for c in range(q)]
            random.shuffle(coords)
            for r, c in coords[:fg_count]:
                tile[r, c] = random.choice(palette)

        fg_count = int(np.count_nonzero(tile != BLACK))
        if 0 < fg_count < area and _rotation_signature_count(tile) == 4:
            return tile

    # Fallback: an explicitly oriented single-corner marker always has 4 rotations.
    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = random.choice(palette)
    if q > 1:
        tile[0, 1] = random.choice(palette)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
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
    """Original-task oracle used to reject degenerate rotated samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    if np.any(bg_mask):
        tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the variant solver's tiling logic for oracle validation."""
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


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q if stencil_empty else q + 1)
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
            if not stencil_empty and (tile_rows < 2 or tile_cols < 2):
                continue
            if stencil_empty or tile_rows * tile_cols >= ll_k:
                valid_dims.append((out_h, out_w))
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if not stencil_empty and np.array_equal(expected, _periodic_output(fill_tile, stencil, out_h, out_w)):
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
