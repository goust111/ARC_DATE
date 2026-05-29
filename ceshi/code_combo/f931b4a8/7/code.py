# Combo variant of f931b4a8 (index 7)
# combo_spec: D4/1+D9/2
# Name: 多色模板+双色语义分离

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The grid is still split into four equal quadrants. Upper-left and upper-right still encode
#      output height and width by counting non-black cells. Lower-left is still an all-non-black
#      coarse color source tile.
#    - The fusion change is entirely in the lower-right template:
#        * it is no longer binary black-vs-foreground
#        * it now uses at least three non-black colors
#        * exactly one template color is the dominant "fill marker"
#        * every other non-black template color is fixed skeleton that survives in the output
#    - Example with q=4:
#        fill_tile =
#          1 1 1 1
#          2 2 2 2
#          1 1 1 1
#          2 2 2 2
#        stencil =
#          7 7 7 7
#          7 3 3 7
#          7 4 4 7
#          7 7 7 7
#      Here 7 is the most frequent template color, so it is the replaceable marker; 3 and 4 are
#      preserved skeleton colors. Old logic would keep every stencil cell because there is no black.
#      The fused logic instead replaces only the 7-cells by coarse lower-left colors and keeps the
#      3/4 cells unchanged.
# b) Complexity calibration
#    - The template always has one dominant color plus only two or three skeleton colors. That is
#      enough to make the task visibly different from the original, but still regular enough that a
#      human can infer the rule from a few examples.
#    - Skeleton colors are placed as compact blobs or bands instead of noisy pixel soup, so repeated
#      evidence supports the role split. Lower-left also stays simple (solid, stripes, diagonals).
# c) Old assumptions that must change
#    - `if value == bg:` no longer identifies replaceable template cells; most replaceable cells are
#      non-black.
#    - `if not np.any(stencil != bg):` remains a harmless fallback branch, but the generator keeps
#      the fused semantic case non-empty so the new rule is actually exercised.
#    - Old code implicitly treated every non-black stencil color as fixed output. That is now false:
#      one visible stencil color disappears and is replaced by lower-left colors.
# d) Placement legality
#    - Quadrant geometry is unchanged, so no new overlap or boundary problems are introduced.
#    - Output sizes are chosen as multiples of the quadrant size. This guarantees that each used
#      coarse lower-left cell corresponds to a full qxq output block and avoids partial-block edge
#      cases that could hide lower-left colors.
#    - Lower-left remains fully non-black, so coarse replacement never reads an invalid color.
# e) Visual observability
#    - Across samples, the dominant lower-right color consistently vanishes from the output while the
#      minority template colors remain. Because lower-left colors are sampled disjointly from template
#      colors, this role split is visible without relying on numeric color ids.
#    - The dominant role is identified by frequency, a purely visual property humans can notice.
# f) Visual cleanliness
#    - The template uses only 3-4 colors total and the skeleton colors occupy disjoint compact
#      regions. That keeps the lower-right quadrant readable instead of turning into multicolor noise.
#    - Upper control colors are kept disjoint from lower-half colors, so the top half cannot be
#      mistaken for content copied into the output.
# g) Change inferability
#    - The rule is not "replace color X" by id. It is "replace the most frequent template color."
#      Repeated samples with different concrete colors still support the same structural inference.
#    - The lower-left tile is simple and repeated coarsely, so the output exposes where replacement
#      happened and what source supplied the replacement colors.
# h) Variant-specific non-degeneracy target
#    - Generated templates are fully non-black and have at least three colors, so the original solver
#      would merely tile the template. The fused solver must differ because it replaces the dominant
#      template color with disjoint lower-left colors on every sample.
# i) New solver assumptions and generator guarantees
#    - Assumption: a non-empty fused template has at least three non-black colors and one unique most
#      frequent color. Guarantee: `_make_semantic_stencil` enforces that.
#    - Assumption: every lower-left cell is non-black. Guarantee: `_make_lower_left` enforces that.
#    - Assumption: if lower-left uses multiple colors, enough coarse cells are exposed in the output.
#      Guarantee: valid output dimensions require enough used coarse cells for the lower-left palette.


def _dominant_semantic_color(stencil: np.ndarray, bg: int) -> int:
    """Return the unique most frequent non-background stencil color."""
    nz = stencil[stencil != bg]
    assert nz.size > 0
    colors, counts = np.unique(nz, return_counts=True)
    assert len(colors) >= 3
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    return int(colors[order[-1]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output built from a semantic multi-color template.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is still the only background color in the control quadrants.
    - The upper-left and upper-right quadrants encode output height and width by non-black counts.
    - The lower-left quadrant is a coarse color-source tile, while the lower-right quadrant is either
      empty or a fully non-black multi-color template with one dominant color and several accent colors.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is all black, tile the lower-left quadrant directly over the output.
    - Otherwise, tile the lower-right quadrant over the output, but treat its unique most frequent
      non-black color as a fill marker: replace only those cells by lower-left colors looked up with
      coarse coordinates `(r // q, c // q)`. All other template colors remain fixed skeleton.

    Key insight:
    - The top half still controls only the output size, but the bottom-right quadrant is no longer a
      plain stencil. One visible template color acts like replaceable background, while the other
      template colors are preserved structure, so the solver must separate foreground roles inside a
      multi-color template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % qh, c % qw])
        return out

    replace_color = _dominant_semantic_color(stencil, bg)
    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            assert value != bg
            if value == replace_color:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_SOLID_PROB = 0.42
LL_MULTI_COLOR_RANGE = (2, 3)
TWO_STRUCTURE_COLOR_PROB = 0.78
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500


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


def _make_lower_left(
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
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    assert np.all(tile != BLACK)
    return tile


def _split_total(total: int, k: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), k - 1))
    parts = []
    prev = 0
    for cut in cuts + [total]:
        parts.append(cut - prev)
        prev = cut
    return parts


def _anchor_candidates(q: int) -> list[tuple[int, int]]:
    mid = q // 2
    candidates = [
        (0, 0),
        (0, q - 1),
        (q - 1, 0),
        (q - 1, q - 1),
        (mid, mid),
        (0, mid),
        (mid, 0),
        (mid, q - 1),
        (q - 1, mid),
    ]
    uniq: list[tuple[int, int]] = []
    for rc in candidates:
        if rc not in uniq:
            uniq.append(rc)
    return uniq


def _pick_compact_cells(
    available: set[tuple[int, int]],
    anchor: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    ranked = sorted(
        available,
        key=lambda rc: (
            abs(rc[0] - anchor[0]) + abs(rc[1] - anchor[1]),
            abs(rc[0] - anchor[0]),
            abs(rc[1] - anchor[1]),
            random.random(),
        ),
    )
    chosen = ranked[:count]
    assert len(chosen) == count
    for rc in chosen:
        available.remove(rc)
    return chosen


def _make_semantic_stencil(
    q: int,
    structure_colors: list[int],
    replace_color: int,
) -> np.ndarray:
    area = q * q
    num_structure_colors = len(structure_colors)
    assert num_structure_colors >= 2
    tile = np.full((q, q), replace_color, dtype=int)

    if q == 2:
        assert num_structure_colors == 2
        counts = [1, 1]
    else:
        max_structure_total = max(num_structure_colors, area // 2 - 1)
        structure_total = random.randint(num_structure_colors, max_structure_total)
        counts = _split_total(structure_total, num_structure_colors)
        random.shuffle(counts)

    anchors = _anchor_candidates(q)
    random.shuffle(anchors)
    available = {(r, c) for r in range(q) for c in range(q)}
    for idx, color in enumerate(structure_colors):
        cells = _pick_compact_cells(available, anchors[idx], counts[idx])
        for r, c in cells:
            tile[r, c] = color

    colors, raw_counts = np.unique(tile, return_counts=True)
    assert len(colors) == num_structure_colors + 1
    assert set(map(int, colors)) == {replace_color, *structure_colors}
    replace_count = int(raw_counts[np.where(colors == replace_color)[0][0]])
    structure_counts = [
        int(raw_counts[np.where(colors == color)[0][0]])
        for color in structure_colors
    ]
    assert all(count > 0 for count in structure_counts)
    assert replace_count > max(structure_counts)
    return tile


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
        tile[r0:r0 + hh, c0:c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    replace_color = _dominant_semantic_color(stencil, BLACK)
    tiled = stencil[rr % q, cc % q].copy()
    replace_mask = tiled == replace_color
    tiled[replace_mask] = fill_tile[rr[replace_mask] // q, cc[replace_mask] // q]
    return tiled


def _old_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    return stencil[rr % q, cc % q]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        structure_colors: list[int] = []
        replace_color = None
        stencil_colors: list[int] = []
    else:
        structure_k = 2 if q == 2 or random.random() < TWO_STRUCTURE_COLOR_PROB else 3
        stencil_colors = _palette(structure_k + 1, [c for c in COLORS if c not in ll_palette])
        if len(stencil_colors) < structure_k + 1:
            return None
        replace_color = stencil_colors[0]
        structure_colors = stencil_colors[1:]

    occupied = set(ll_palette) | set(stencil_colors)
    control_pool = [c for c in COLORS if c not in occupied]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and out_h % q == 0
        and out_w % q == 0
        and ceil(out_h / q) * ceil(out_w / q) >= ll_k
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(out_h // q) for c in range(out_w // q)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    if stencil_empty:
        stencil = np.zeros((q, q), dtype=int)
    else:
        assert replace_color is not None
        stencil = _make_semantic_stencil(q, structure_colors, replace_color)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    old_like = _old_output(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, old_like):
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
