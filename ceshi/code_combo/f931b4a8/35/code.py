from __future__ import annotations

# Combo variant of f931b4a8 (index 35)
# combo_spec: D6/3+D11/0
# Name: 颜色混合+邻域决定替换色
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    In the base task, a stencil foreground cell kept its stencil color, while a stencil
#    background cell took the coarse lower-left color for the current q-by-q output block.
#    Here the lower-left coarse color participates twice:
#    - foreground seed at (r,c): boundary foreground pixels switch to the coarse LL color,
#      while interior foreground pixels keep the stencil color
#    - background cell at (r,c): inherit the color of the nearest mixed foreground seed
#      in the tiled output, not the lower-left color directly.
#    Example on one coarse block: a stencil frame with color 2 and coarse fill color 6
#    turns its exposed boundary into 6 while any protected interior foreground stays 2.
#    Holes then inherit the nearest visible boundary/interior seed. This differs from the
#    original, where foreground would stay 2 and holes would read coarse 6 directly.
# b) Complexity calibration:
#    The mixing rule is structural rather than numeric: template boundary cells visibly take
#    the source color, while interior foreground cells keep the stencil color. The propagation
#    rule uses nearest-neighbor spread. Generator palettes are kept small and stencil patterns
#    are structured so the output is not noisy, but both boundary recoloring and
#    background-propagation changes occur repeatedly in each sample.
# c) Hidden assumptions checked against the original main():
#    - Output size still comes only from upper quadrants.
#    - Lower-left remains all non-black because mixed foreground seeds need a valid source.
#    - Lower-right must be non-empty for the fusion to be exercised; generator guarantees that.
#    - Background filling no longer assumes a direct lower-left lookup, so solver first builds
#      all mixed foreground seeds and only then propagates them into holes.
#    - Nearest-color ties are resolved deterministically by (distance, row, col); generator
#      uses structured templates so the rule is still visually legible.
# d) Placement legality:
#    The grid is still an even square split into four fixed quadrants, so no free placement is
#    needed. Generator only chooses output sizes with ceil(dim/q) <= q, keeping coarse lookups
#    in-bounds.
# e) Visual observability:
#    Samples are filtered to ensure both of these happen: some foreground cells change because
#    of mixing, and some background cells differ from the original direct coarse fill. This
#    prevents accidental collapse back to the base task.
# f) Visual cleanliness:
#    Generator favors stripes, frames, plus-shapes, and light scatter over dense randomness.
#    Color counts stay low, so the propagated Voronoi-like regions remain readable.
# g) Inferability of color changes:
#    The mixed foreground uses a simple visible cue: template boundary vs template interior.
#    The same stencil pattern reappears with different coarse LL colors in different blocks,
#    so a solver can infer that exposed boundary pixels take the source color while protected
#    interior pixels keep the stencil color.

import random

import numpy as np

from arc_utils import Color


def _is_boundary_foreground(stencil: np.ndarray, r: int, c: int, bg: int) -> bool:
    """A foreground cell is boundary if a 4-neighbor is background or out of bounds."""
    qh, qw = stencil.shape
    assert int(stencil[r, c]) != bg
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < qh and 0 <= cc < qw):
            return True
        if int(stencil[rr, cc]) == bg:
            return True
    return False


def _propagate_nearest(seed_grid: np.ndarray, bg: int) -> np.ndarray:
    """
    Fill zero cells with the color of the nearest non-zero seed.

    Ties are broken deterministically by reading order of the seed coordinates.
    """
    out = np.array(seed_grid, dtype=int, copy=True)
    seed_coords = [(int(r), int(c), int(out[r, c])) for r, c in np.argwhere(out != bg)]
    assert seed_coords, "nearest-neighbor propagation requires at least one seed"

    for r, c in np.argwhere(out == bg):
        best_key: tuple[int, int, int] | None = None
        best_color = bg
        rr = int(r)
        cc = int(c)
        for sr, sc, color in seed_coords:
            key = (abs(sr - rr) + abs(sc - cc), sr, sc)
            if best_key is None or key < best_key:
                best_key = key
                best_color = color
        out[rr, cc] = best_color
    return out


def _mixed_seed_grid(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    """Create the output-sized seed grid using boundary-aware source/stencil mixing."""
    q = int(fill_tile.shape[0])
    seeds = np.full((out_h, out_w), bg, dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            sr = r % q
            sc = c % q
            stencil_value = int(stencil[sr, sc])
            if stencil_value == bg:
                continue
            src_r = r // q
            src_c = c // q
            assert src_r < q and src_c < q
            source_color = int(fill_tile[src_r, src_c])
            assert source_color != bg
            if _is_boundary_foreground(stencil, sr, sc, bg):
                seeds[r, c] = source_color
            else:
                seeds[r, c] = stencil_value
    return seeds


def _tile_fill(fill_tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Fallback used only when the stencil is empty."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = fill_tile[rr % q, cc % q]
    assert np.all(tiled != int(Color.BLACK))
    return tiled


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    """Original task output, used only for generator-side anti-degeneracy checks."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != bg):
        return fill_tile[rr % q, cc % q]
    out = stencil[rr % q, cc % q].copy()
    mask = out == bg
    out[mask] = fill_tile[rr[mask] // q, cc[mask] // q]
    return out


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> np.ndarray:
    """Oracle for the fused solver."""
    if not np.any(stencil != bg):
        return _tile_fill(fill_tile, out_h, out_w)
    seeds = _mixed_seed_grid(fill_tile, stencil, out_h, out_w, bg)
    return _propagate_nearest(seeds, bg)


def _behavior_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int, bg: int) -> dict[str, int]:
    """Quantify whether both fusion behaviors are active on this sample."""
    if not np.any(stencil != bg):
        return {"fg_source_used": 0, "fg_stencil_kept": 0, "bg_shifted": 0, "bg_total": 0}

    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    stencil_tiled = stencil[rr % q, cc % q]
    source = fill_tile[rr // q, cc // q]
    fg_mask = stencil_tiled != bg
    bg_mask = ~fg_mask

    seeds = _mixed_seed_grid(fill_tile, stencil, out_h, out_w, bg)
    fused = _propagate_nearest(seeds, bg)
    legacy = _legacy_output(fill_tile, stencil, out_h, out_w, bg)

    return {
        "fg_source_used": int(np.count_nonzero(fg_mask & (seeds == source))),
        "fg_stencil_kept": int(np.count_nonzero(fg_mask & (seeds == stencil_tiled))),
        "bg_shifted": int(np.count_nonzero(bg_mask & (fused != legacy))),
        "bg_total": int(np.count_nonzero(bg_mask)),
        "source_variety": int(len(set(map(int, np.unique(source))))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper quadrants as size controls, then expand the lower-right template with fused color logic.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - The upper-left quadrant controls output height by its non-black pixel count.
    - The upper-right quadrant controls output width by its non-black pixel count.
    - The lower-left quadrant is a non-black color-source tile read at coarse q-by-q block indices.
    - The lower-right quadrant is a template tile whose non-black cells act as foreground seeds.

    Rule:
    - Build an output of size (count_nonblack(UL), count_nonblack(UR)).
    - Tile the lower-right template across that output.
    - For each non-black template position, use a structural mix: template boundary cells take the coarse lower-left color for the current output block, while template interior cells keep their stencil color.
    - For each black template position, ignore the lower-left color directly and instead fill it with the color of the nearest fused foreground seed in the tiled output.
    - If the template is entirely black, fall back to tiling the lower-left quadrant directly.

    Key insight:
    - The lower-left tile no longer only plugs holes. It recolors the exposed template boundary first, while protected template interior stays in stencil color, and those mixed foreground colors then spread into the template background by nearest-neighbor propagation.
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
    assert out_h <= q * q and out_w <= q * q
    assert np.all(fill_tile != bg)

    if not np.any(stencil != bg):
        return _tile_fill(fill_tile, out_h, out_w)

    seeds = _mixed_seed_grid(fill_tile, stencil, out_h, out_w, bg)
    return _propagate_nearest(seeds, bg)


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
LL_COLOR_RANGE = (2, 3)
STENCIL_COLOR_RANGE = (1, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))], dtype=int).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_lower_left(q: int, palette: list[int], required_coords: list[tuple[int, int]]) -> np.ndarray:
    """Build a non-black lower-left tile while making used coarse cells informative."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "diagonal", "blocks"])

    if mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    elif mode == "diagonal":
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color
    else:
        r_mid = max(1, q // 2)
        c_mid = max(1, q // 2)
        block_colors = [palette[i % k] for i in range(4)]
        tile[:r_mid, :c_mid] = block_colors[0]
        tile[:r_mid, c_mid:] = block_colors[1]
        tile[r_mid:, :c_mid] = block_colors[2]
        tile[r_mid:, c_mid:] = block_colors[3]

    random.shuffle(required_coords)
    for i, (r, c) in enumerate(required_coords):
        tile[r, c] = palette[i % k]
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a structured stencil with both foreground and background."""
    tile = np.zeros((q, q), dtype=int)
    mode = random.choice(["checker", "row_stripe", "col_stripe", "frame", "plus", "scatter"])

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        start = random.randrange(2)
        for r in range(start, q, 2):
            tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        start = random.randrange(2)
        for c in range(start, q, 2):
            tile[:, c] = _random_color_array((q,), palette)
    elif mode == "frame":
        tile[0, :] = _random_color_array((q,), palette)
        tile[-1, :] = _random_color_array((q,), palette)
        tile[:, 0] = _random_color_array((q,), palette)
        tile[:, -1] = _random_color_array((q,), palette)
    elif mode == "plus":
        r = random.randrange(q)
        c = random.randrange(q)
        tile[r, :] = _random_color_array((q,), palette)
        tile[:, c] = _random_color_array((q,), palette)
    else:
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        fg_count = random.randint(max(q, (q * q) // 3), max(q + 1, (q * q * 3) // 5))
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)

    if not np.any(tile == BLACK):
        tile[random.randrange(q), random.randrange(q)] = BLACK
    if not np.any(tile != BLACK):
        tile[random.randrange(q), random.randrange(q)] = random.choice(palette)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0:r0 + hh, c0:c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        coords = _ordered_coords(q, random.choice(["rect", "compact", "rows"]))[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2

    out_lo = q + 1
    out_hi = min(16, 2 * q)
    valid_dims = [(hh, ww) for hh in range(out_lo, out_hi + 1) for ww in range(out_lo, out_hi + 1) if hh * ww <= OUTPUT_AREA_MAX]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range((out_h + q - 1) // q) for c in range((out_w + q - 1) // q)]
    ll_palette = _palette(random.randint(*LL_COLOR_RANGE))
    fill_tile = _make_lower_left(q, ll_palette, used_fill)

    low_colors = [c for c in COLORS if c <= max(5, min(ll_palette))]
    stencil_palette = _palette(random.randint(*STENCIL_COLOR_RANGE), low_colors if low_colors else COLORS)
    stencil = _make_stencil(q, stencil_palette)

    metrics = _behavior_metrics(fill_tile, stencil, out_h, out_w, BLACK)
    if metrics["fg_source_used"] == 0 or metrics["fg_stencil_kept"] == 0 or metrics["bg_shifted"] == 0:
        return None
    if metrics["bg_total"] == 0 or metrics["source_variety"] < 2:
        return None

    output_colors = set(map(int, np.unique(_expected_output(fill_tile, stencil, out_h, out_w, BLACK))))
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w, BLACK)
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
