# Combo variant of f931b4a8 (index 21)
# combo_spec: D2/1+D8/2
# Name: 非对角控制区+三区域布局

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis
# ----------------
# combo_variant_spec:
#   preserved_core:
#     - The lower half still contains the two pattern sources from the original task:
#       a fully non-black coarse fill tile on the left and a stencil tile on the right.
#     - Output rendering is unchanged: tile the stencil if it is non-empty, and replace
#       stencil holes with the fill tile value chosen by coarse tile coordinates; if the
#       stencil is empty, tile the fill tile directly.
#   fused_layout_change:
#     - Apply D8/2 first: the top half becomes one shared control band of shape Q x (2Q),
#       while the bottom half remains split into fill tile and stencil tile.
#     - Then inject D2/1 semantics into that band: the two control roles are no longer
#       read from upper-left and upper-right corners. Width still uses the upper-right
#       anchor, but height is re-anchored at the lower-left corner of the control band,
#       echoing the original D2/1 move from upper-left to lower-left.
#   control_encoding:
#     - The band contains exactly two non-black control colors plus black background.
#     - Height color = color at the lower-left cell of the band.
#     - Width color = color at the upper-right cell of the band.
#     - To keep the band visibly unified rather than secretly split into two rectangles,
#       height cells may occupy only even columns and width cells may occupy only odd
#       columns. This creates an interleaved mixed region with stable capacity Q*Q per role.
#     - The top-left cell is forced to remain black. This prevents accidental collapse to
#       the D8/2 single variant, which used the top-left corner as the height anchor.
#   concrete_behavior_trace:
#     - Example with N=8 and Q=4:
#       bottom-left is a 4x4 all-non-black fill tile, bottom-right is a 4x4 stencil.
#       In the top 4x8 control band, suppose green appears 7 times on even columns and
#       includes the lower-left corner (3,0), while red appears 10 times on odd columns
#       and includes the upper-right corner (0,7). The cell (0,0) stays black.
#     - Original task would have split the grid into four quadrants and counted only the
#       upper-left / upper-right quadrants, which is wrong here because the top half is
#       one mixed band and the lower-left region is dense fill material rather than a
#       sparse control patch.
#     - D8/2 alone would have used the top-left corner as the height anchor, but that
#       cell is black in this combo, so the solver must instead locate the lower-left
#       anchor to recover the height role.
#     - The combo output is therefore 7 x 10, filled by the same stencil/fill mechanism
#       as the original task.
#   complexity_calibration:
#     - Only two control colors are used, so the role mapping is recoverable from a few
#       examples.
#     - Column-parity slotting keeps the mixed control band legible without adding a third
#       hidden code such as color ordering or shape vocabulary.
#     - Forcing each control color to appear in both left and right halves prevents the top
#       band from degenerating into a disguised pair of independent blocks.
#   implicit_assumption_review:
#     - `assert h == w and h % 2 == 0` still matters because the lower half must split into
#       two Q x Q source tiles.
#     - `height_color = control[-1, 0]` is a new assumption: the lower-left cell of the
#       top band must be non-black and uniquely mark the height role. The generator enforces
#       this.
#     - `width_color = control[0, -1]` remains an anchored non-black role marker, but now
#       it must differ from the lower-left anchor color. The generator enforces this too.
#     - `control[0, 0] == BLACK` is a deliberate anti-degeneracy assumption that breaks the
#       D8/2 top-left-anchor interpretation.
#     - `control_colors == {height_color, width_color}` assumes no extra foreground colors
#       appear in the control band, keeping counting unambiguous. The generator guarantees it.
#     - `fill_r = r // Q`, `fill_c = c // Q` still assumes output dimensions do not exceed
#       Q*Q along either axis. The generator restricts counts to the available slot capacity.
#   placement_legality:
#     - The control band, fill tile, and stencil occupy disjoint rectangular regions, so
#       no overlap is possible.
#     - The fill tile is fully non-black, guaranteeing every coarse lookup used by stencil
#       holes is legal.
#     - Height cells and width cells use disjoint column-parity slot sets, so the control
#       colors never overwrite each other inside the shared band.
#   visual_observability:
#     - Human solvers see a 1+2 layout immediately: a full-width top band over two lower
#       source tiles.
#     - The lower-left colored anchor and upper-right colored anchor make the D2/1-style
#       non-diagonal control relocation visible inside the new top band.
#     - The mixed stripe-like control band, plus both colors appearing in both halves, gives
#       repeated evidence that the top area is one shared control region rather than two
#       ordinary quadrants.
#   cleanliness_and_inference:
#     - No region mixes pattern-source pixels with control markers.
#     - Control colors are drawn outside the output palette so they read as metadata rather
#       than content colors.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a lower-left and upper-right anchor from one shared control band to size an output built from the lower source tiles.

    Input structure:
    - The grid is an even square.
    - The upper half is a single mixed control band spanning the full width.
    - The lower-left half is a fully non-black fill tile, and the lower-right half is a stencil tile or an all-black empty tile.
    - Within the control band, the lower-left corner marks the height color and the upper-right corner marks the width color.

    Rule:
    - Output height is the total number of control-band cells using the lower-left anchor color.
    - Output width is the total number of control-band cells using the upper-right anchor color.
    - If the stencil is empty, tile the lower-left fill tile directly across the output.
    - Otherwise tile the stencil, and whenever the stencil contributes black, replace it with the lower-left fill-tile value indexed by coarse tile coordinates.

    Key insight:
    - The lower pattern mechanism stays the same as the original task, but the size controls are fused into one top band and must be decoded from a non-diagonal anchor pair: height comes from the band's lower-left anchor instead of the traditional upper-left position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0

    bg = int(Color.BLACK)
    q = h // 2
    control = g[:q, :]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    assert int(control[0, 0]) == bg
    height_color = int(control[-1, 0])
    width_color = int(control[0, -1])
    assert height_color != bg and width_color != bg and height_color != width_color

    control_colors = set(map(int, np.unique(control))) - {bg}
    assert control_colors == {height_color, width_color}

    out_h = int(np.count_nonzero(control == height_color))
    out_w = int(np.count_nonzero(control == width_color))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % q, c % q])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % q, c % q])
            if value == bg:
                fill_r = r // q
                fill_c = c // q
                assert fill_r < q and fill_c < q
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x, adapted for the combo) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.16
FILL_SOLID_PROB = 0.70
FILL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300
CONTROL_MIX_RETRIES = 80


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(height: int, width: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(height) for c in range(width)]
    if mode == "rows":
        rows = list(range(height))
        cols = list(range(width))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar = random.randrange(height)
        ac = random.randrange(width)
        random.shuffle(coords)
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_fill_tile(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
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
        fix_coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t:q - t, t:q - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((q,), palette)
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _height_slots(q: int) -> list[tuple[int, int]]:
    width = 2 * q
    return [
        (r, c)
        for r in range(q)
        for c in range(width)
        if c % 2 == 0 and (r, c) != (0, 0)
    ]


def _width_slots(q: int) -> list[tuple[int, int]]:
    width = 2 * q
    return [(r, c) for r in range(q) for c in range(width) if c % 2 == 1]


def _make_control_band(q: int, out_h: int, out_w: int, pool: list[int]) -> np.ndarray | None:
    width = 2 * q
    if len(pool) < 2:
        return None

    h_slots = _height_slots(q)
    w_slots = _width_slots(q)
    if out_h < 3 or out_w < 3 or out_h > len(h_slots) or out_w > len(w_slots):
        return None

    height_color, width_color = _palette(2, pool)
    h_anchor = (q - 1, 0)
    w_anchor = (0, width - 1)
    assert h_anchor in h_slots and w_anchor in w_slots

    h_cross_candidates = [rc for rc in h_slots if rc != h_anchor and rc[1] >= q]
    w_cross_candidates = [rc for rc in w_slots if rc != w_anchor and rc[1] < q]
    if not h_cross_candidates or not w_cross_candidates:
        return None

    for _ in range(CONTROL_MIX_RETRIES):
        control = np.zeros((q, width), dtype=int)
        control[h_anchor] = height_color
        control[w_anchor] = width_color

        h_cross = random.choice(h_cross_candidates)
        w_cross = random.choice(w_cross_candidates)
        control[h_cross] = height_color
        control[w_cross] = width_color

        h_remaining = out_h - 2
        w_remaining = out_w - 2

        h_pool = [rc for rc in h_slots if rc not in {h_anchor, h_cross}]
        w_pool = [rc for rc in w_slots if rc not in {w_anchor, w_cross}]
        h_pool_set = set(h_pool)
        w_pool_set = set(w_pool)

        h_mode = random.choice(["compact", "rows", "random"])
        w_mode = random.choice(["compact", "rows", "random"])
        h_order = [rc for rc in _ordered_coords(q, width, h_mode) if rc in h_pool_set]
        w_order = [rc for rc in _ordered_coords(q, width, w_mode) if rc in w_pool_set]

        for r, c in h_order[:h_remaining]:
            control[r, c] = height_color
        for r, c in w_order[:w_remaining]:
            control[r, c] = width_color

        if np.count_nonzero(control == height_color) != out_h:
            continue
        if np.count_nonzero(control == width_color) != out_w:
            continue
        if int(control[0, 0]) != BLACK:
            continue
        if not np.any(control[:, :q] == height_color):
            continue
        if not np.any(control[:, q:] == height_color):
            continue
        if not np.any(control[:, :q] == width_color):
            continue
        if not np.any(control[:, q:] == width_color):
            continue

        left = int(np.count_nonzero(control[:, :q] != BLACK))
        right = int(np.count_nonzero(control[:, q:] != BLACK))
        if left == out_h and right == out_w:
            continue

        return control

    return None


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    fill_k = 1 if random.random() < FILL_SOLID_PROB else random.randint(*FILL_MULTI_COLOR_RANGE)
    fill_palette = _palette(fill_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(fill_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    max_h = len(_height_slots(q))
    max_w = len(_width_slots(q))
    lo_h = OUTPUT_DIM_MIN
    lo_w = max(OUTPUT_DIM_MIN, q)
    hi_h = min(OUTPUT_DIM_MAX, max_h)
    hi_w = min(OUTPUT_DIM_MAX, max_w)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo_h, hi_h + 1)
        for out_w in range(lo_w, hi_w + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        if stencil_empty or ceil(out_h / q) * ceil(out_w / q) >= fill_k
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_fill_tile(q, fill_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    control = _make_control_band(q, out_h, out_w, control_pool)
    if control is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :] = control
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
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
