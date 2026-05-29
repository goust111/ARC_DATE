# Variant of dbff022c
# Dimension: D1 — 颜色映射规则——边框颜色到填充颜色的对应方式 (match_cond)
# Change: 调色板直接给出边框颜色→填充颜色的一一对应关系，精灵的边框颜色在调色板中查找得到唯一填充颜色...
#         → 条件映射——精灵面积决定使用哪组映射

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, neighbors4


# Design analysis
# spec:
#   variant_core:
#     palette: one edge carries a 4xN or Nx4 legend made of two consecutive 2-line sub-legends.
#       Group 1 (the first pair when read from the edge inward) is for larger sprites.
#       Group 2 (the second pair) is for smaller sprites.
#       Both groups repeat the same border-key colors, but their fill colors differ.
#     sprites: keep the original idea of hollow mono-color sprites with enclosed background holes.
#       The mapped sprites are split into two visibly separated size bands:
#       small band uses 3x3 / 3x4 / 4x3 hollow rings; large band uses 4x5 / 5x4 / 5x5 rings.
#       Exactly one distractor sprite still uses a color absent from the palette keys.
#   concrete_trace:
#     Suppose the top edge shows:
#       row0 = [red, blue, green]
#       row1 = [yellow, grey, orange]   <- large mapping
#       row2 = [red, blue, green]
#       row3 = [grey, orange, yellow]   <- small mapping
#     A 5x5 red hollow sprite and a 3x4 blue hollow sprite both have border colors listed twice.
#     Old rule: every red hole would always become yellow and every blue hole would always become grey.
#     New rule: the red 5x5 hole uses the first pair and becomes yellow, while the blue 3x4 hole
#       uses the second pair and becomes orange.
#   complexity_calibration:
#     The key colors are repeated across both groups, so border color alone is no longer sufficient.
#     Only two size bands exist, with a clear geometric gap, preventing ambiguous middle cases.
#     Palette length stays small (2-3), and both mapping groups are forced to differ column-wise,
#     so the conditional behavior is visible without flooding the grid with too many colors.
#   implicit_assumptions_review:
#     _best_palette in the original code assumed a 2-line block and unique keys in one outer line.
#       That no longer holds; the variant must detect a 4-line block whose two key lines match.
#     _palette_mapping assumed one dict. The variant needs two dicts and must preserve which pair
#       is "first" versus "second" in edge-inward reading order.
#     _fill_enclosed_bg_regions originally chose a fill using only adjacent border color.
#       That is insufficient here because the same border color appears in both groups.
#       The solver now has to measure the surrounding sprite area before selecting the mapping.
#   placement_legality:
#     A 4-line palette needs more room than the original 2-line palette, so the generator raises
#     the minimum grid size. Sprites and palette still keep a 1-cell background moat, and enclosed
#     holes are kept away from the outer boundary so the solver's enclosed-region logic stays valid.
#   visual_observability:
#     Humans can see the palette is now doubled and that the same key colors appear twice.
#     The large/small split is visible from bbox size, not from hidden numeric ids.
#     Every generated sample contains at least one mapped large sprite and one mapped small sprite,
#     so both groups are exercised instead of one branch being silently unused.
#   cleanliness_and_color_inference:
#     The two fill rows use the same small pool of fill colors but permuted, keeping the number of
#     colors manageable while making the conditional remapping recoverable from a few examples.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    """Boolean mask covering the full 4xN / Nx4 palette footprint."""
    h, w = shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        pal_mask[0:4, start:end] = True
    elif side == "bottom":
        pal_mask[h - 4 : h, start:end] = True
    elif side == "left":
        pal_mask[start:end, 0:4] = True
    else:
        pal_mask[start:end, w - 4 : w] = True
    return pal_mask


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> list[np.ndarray]:
    """
    Return the palette's four lines in edge-inward order.

    For top/bottom this is rows; for left/right this is columns.
    """
    h, w = g.shape
    if side == "top":
        return [g[0, start:end], g[1, start:end], g[2, start:end], g[3, start:end]]
    if side == "bottom":
        return [g[h - 1, start:end], g[h - 2, start:end], g[h - 3, start:end], g[h - 4, start:end]]
    if side == "left":
        return [g[start:end, 0], g[start:end, 1], g[start:end, 2], g[start:end, 3]]
    return [g[start:end, w - 1], g[start:end, w - 2], g[start:end, w - 3], g[start:end, w - 4]]


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 4xN or Nx4 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal = _palette_mask(g.shape, side, start, end)
        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def valid_candidate(side: str, start: int, end: int) -> tuple[bool, tuple[int, int, int, int] | None]:
        lines = _palette_lines(g, side, start, end)
        block = np.stack(lines, axis=0)
        if np.any(block == bg):
            return False, None

        keys_a = [int(x) for x in lines[0].tolist()]
        keys_b = [int(x) for x in lines[2].tolist()]
        fills_a = [int(x) for x in lines[1].tolist()]
        fills_b = [int(x) for x in lines[3].tolist()]

        if len(keys_a) < 2:
            return False, None
        if len(keys_a) != len(set(keys_a)):
            return False, None
        if keys_a != keys_b:
            return False, None
        if len(set(fills_a)) != len(fills_a):
            return False, None
        if len(set(fills_b)) != len(fills_b):
            return False, None
        if not any(a != b for a, b in zip(fills_a, fills_b, strict=False)):
            return False, None
        if not isolated(side, start, end):
            return False, None

        distinct = len(set(keys_a) | set(fills_a) | set(fills_b))
        score = (len(keys_a), distinct, -(end - start), block.size)
        return True, score

    def consider_top() -> None:
        nonlocal best
        if h < 4:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("top", c0, c1)
                if ok:
                    cand = (score, "top", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 4:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("bottom", c0, c1)
                if ok:
                    cand = (score, "bottom", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 4:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("left", r0, r1)
                if ok:
                    cand = (score, "left", r0, r1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 4:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("right", r0, r1)
                if ok:
                    cand = (score, "right", r0, r1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mappings(g: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """
    Returns (large_mapping, small_mapping, palette_mask).
    """
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask(g.shape, side, start, end)
    lines = _palette_lines(g, side, start, end)

    large_keys = [int(x) for x in lines[0].tolist()]
    large_fills = [int(x) for x in lines[1].tolist()]
    small_keys = [int(x) for x in lines[2].tolist()]
    small_fills = [int(x) for x in lines[3].tolist()]

    assert large_keys == small_keys, "_palette_mappings: both groups must use the same keys"
    large_map = dict(zip(large_keys, large_fills, strict=False))
    small_map = dict(zip(small_keys, small_fills, strict=False))
    return large_map, small_map, pal_mask


def _component_area_from_seed(out: np.ndarray, pal_mask: np.ndarray, seed: tuple[int, int]) -> int:
    """Area of the monochrome non-palette component containing seed."""
    h, w = out.shape
    sr, sc = seed
    color = int(out[sr, sc])
    stack = [(int(sr), int(sc))]
    seen = {(int(sr), int(sc))}
    area = 0

    while stack:
        r, c = stack.pop()
        area += 1
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen:
                if pal_mask[rr, cc]:
                    continue
                if int(out[rr, cc]) != color:
                    continue
                seen.add((rr, cc))
                stack.append((rr, cc))
    return area


def _fill_enclosed_bg_regions(
    out: np.ndarray,
    bg: int,
    large_map: dict[int, int],
    small_map: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    Fill enclosed holes using one of two border->fill mappings.

    The border color still identifies the palette column, but the surrounding sprite's
    area decides whether the larger-sprite or smaller-sprite mapping row is used.
    """
    h, w = out.shape
    keyset = set(large_map) & set(small_map)
    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)

    hole_infos: list[tuple[np.ndarray, int, int]] = []
    for pts in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        adj_seed: dict[int, tuple[int, int]] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
                    adj_seed.setdefault(v, (int(rr), int(cc)))
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in keyset:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        area = _component_area_from_seed(out, pal_mask, adj_seed[border])
        hole_infos.append((pts, border, area))

    if not hole_infos:
        return

    areas = [area for _, _, area in hole_infos]
    small_area = min(areas)
    large_area = max(areas)
    pivot = (small_area + large_area) / 2.0

    for pts, border, area in hole_infos:
        if large_area == small_area:
            mapping = large_map
        else:
            mapping = large_map if area > pivot else small_map
        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites' interiors using a doubled edge legend whose active row depends on sprite size.

    Input structure:
    - bg is the dominant edge color.
    - Several mono-color hollow sprites sit away from the boundary.
    - One 4xN or Nx4 palette block lies on a grid edge and consists of two consecutive 2-line legends.
      Both legends repeat the same border-key colors, but each legend assigns a different fill row.

    Rule:
    - Locate the edge palette and read its first 2-line legend as the mapping for larger sprites
      and its second 2-line legend as the mapping for smaller sprites.
    - Find enclosed background regions that do not touch the outer grid boundary.
    - For each hole, identify the surrounding border color, measure the enclosing sprite's area,
      choose the large or small legend accordingly, and fill the hole with that mapped color.

    Key insight:
    - Border color alone no longer determines the answer because the same key appears in two palette groups.
      The solver must combine color lookup with a size comparison between the surrounding sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    large_map, small_map, pal_mask = _palette_mappings(g, bg)
    assert large_map and small_map, "main: both palette mappings must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, large_map, small_map, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 8, 14
_GRID_W_MIN, _GRID_W_MAX = 10, 16

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

# Sprite templates
_SMALL_SPECS = [(3, 3), (3, 4), (4, 3)]
_LARGE_SPECS = [(4, 5), (5, 4), (5, 5)]
_P_CORNER_NOTCH = 0.45

# Generation
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components that stay away from the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_ring(h: int, w: int) -> np.ndarray:
    """Axis-aligned hollow rectangle."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False
    return sprite


def _make_band_sprite(kind: str, rng: random.Random) -> np.ndarray:
    """Create a visibly small or large hollow sprite."""
    specs = _LARGE_SPECS if kind == "large" else _SMALL_SPECS
    h, w = rng.choice(specs)
    sprite = _make_ring(h, w)

    if rng.random() < _P_CORNER_NOTCH:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        cr, cc = rng.choice(corners)
        test = sprite.copy()
        test[cr, cc] = False
        if _get_enclosed_holes(test):
            sprite = test

    assert _get_enclosed_holes(sprite), "_make_band_sprite must preserve at least one enclosed hole"
    return sprite


def _deranged(values: list[int], rng: random.Random) -> list[int]:
    """Return a permutation with no fixed points."""
    if len(values) == 2:
        return [values[1], values[0]]
    order = list(values)
    for _ in range(40):
        rng.shuffle(order)
        if all(a != b for a, b in zip(order, values, strict=False)):
            return order
    return values[1:] + values[:1]


def _choose_colors(rng: random.Random, n: int) -> tuple[list[int], list[int], list[int], int]:
    """
    Pick repeated border keys plus two distinct fill rows.

    The two fill rows use the same small pool of colors but are permuted so that each
    key maps to a different fill in the large and small groups.
    """
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    keys = all_c[:n]
    fills_pool = all_c[n : 2 * n]
    distractor = all_c[2 * n]
    large_fills = fills_pool[:]
    rng.shuffle(large_fills)
    small_fills = _deranged(large_fills, rng)
    return keys, large_fills, small_fills, distractor


def _build_palette(side: str, border_keys: list[int], large_fills: list[int], small_fills: list[int]) -> np.ndarray:
    """Construct the 4xN or Nx4 palette block for the chosen edge."""
    edge_lines = [
        np.array(border_keys, dtype=int),
        np.array(large_fills, dtype=int),
        np.array(border_keys, dtype=int),
        np.array(small_fills, dtype=int),
    ]
    n = len(border_keys)

    if side == "top":
        pal = np.zeros((4, n), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[i] = line
        return pal
    if side == "bottom":
        pal = np.zeros((4, n), dtype=int)
        for i, line in enumerate(edge_lines[::-1]):
            pal[i] = line
        return pal
    if side == "left":
        pal = np.zeros((n, 4), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[:, i] = line
        return pal

    pal = np.zeros((n, 4), dtype=int)
    for i, line in enumerate(edge_lines[::-1]):
        pal[:, i] = line
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette on the chosen edge and reserve a 1-cell moat."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
) -> list[tuple[int, int]]:
    """All positions where the sprite fits, stays separated, and keeps holes off the outer boundary."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render one sprite; filled holes are used only when constructing the output grid."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _sprite_roles(n: int, rng: random.Random) -> list[str]:
    """Mapped sprites always include both size classes; a third mapped sprite repeats one class."""
    roles = ["large", "small"]
    if n == 3:
        roles.append(rng.choice(["large", "small"]))
    return roles


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    n = rng.randint(_PALETTE_N_MIN, _PALETTE_N_MAX)

    border_keys, large_fills, small_fills, distractor_color = _choose_colors(rng, n)
    large_map = dict(zip(border_keys, large_fills, strict=False))
    small_map = dict(zip(border_keys, small_fills, strict=False))
    pal = _build_palette(side, border_keys, large_fills, small_fills)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos: list[tuple[np.ndarray, int, int | None, str]] = []
    for key, role in zip(border_keys, _sprite_roles(n, rng), strict=False):
        mask = _make_band_sprite(role, rng)
        fill = large_map[key] if role == "large" else small_map[key]
        infos.append((mask, key, fill, role))

    distractor_role = rng.choice(["small", "large"])
    distractor_mask = _make_band_sprite(distractor_role, rng)
    infos.append((distractor_mask, distractor_color, None, distractor_role))

    infos.sort(key=lambda item: int(item[0].sum()), reverse=True)

    placed: list[tuple[int, int, np.ndarray, int | None]] = []
    for mask, color, fill, _role in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
