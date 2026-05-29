# Variant of dbff022c
# Dimension: D5 — 变换操作——内部区域的填充方式 (transform)
# Change: 使用 flood_fill 将精灵内部所有背景色像素填充为调色板查找到的单一颜色... → 精灵边框颜色与内部颜色互换

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: "palette maps border color -> hole fill color; output only changes enclosed bg cells"
#     variant: "palette maps original border color -> new border color; output fills the hole with the
#       sprite's original border color and recolors the whole sprite border to the mapped color"
#   concrete_behavior:
#     input:
#       - one edge hosts the same 2xN / Nx2 palette legend as the source task
#       - each target sprite is still hollow in the input: single border color, bg interior
#       - one distractor hollow sprite uses a color that is absent from the palette keys
#     output:
#       - for a sprite whose border color is a palette key `b` and mapped color `f`, every border pixel
#         of that sprite becomes `f`, while every enclosed bg cell becomes `b`
#       - the distractor stays unchanged
#     trace_example:
#       - original task on a 4x4 ring of color 3 with palette 3->6: border stays 3, hole becomes 6
#       - this variant on the same input: border becomes 6, hole becomes 3
#       - difference: the palette color is no longer a one-way fill; it replaces the outline itself
#   complexity_calibration:
#     - keep the palette search/orientation logic unchanged, so the puzzle still requires legend reading
#     - keep 2-4 sprites and 1 distractor, so the solver must distinguish "listed" from "unlisted"
#     - force palette values to be disjoint from palette keys for active colors; otherwise swap can look
#       identical to the original fill behavior or become visually weak
#   hidden_assumptions_review:
#     - source `_fill_enclosed_bg_regions` assumed changing only bg cells was safe; that is false now
#       because the enclosing border must also change
#     - recoloring based only on adjacent ring cells would miss outer-border pixels; therefore the new
#       solver recolors the whole monochromatic connected component reached from the hole's border seeds
#     - if a sprite had multiple disconnected same-colored pieces around one hole, a single-seed flood
#       could miss part of it; generator therefore enforces 4-connected borders
#   placement_legality:
#     - palette remains isolated from sprites by a 1-cell moat exactly as in the source task
#     - sprite holes may not touch the grid boundary, preserving the original enclosed-hole detection
#     - target border colors are unique among sprites, so recoloring one component cannot accidentally
#       recolor another sprite
#   visual_observability:
#     - every target output changes in two places at once: outline changes to the palette value and the
#       interior takes the original outline color, which is visibly different from pure filling
#     - palette keys and values are disjoint for active entries, making the swap legible in 2-3 examples
#   cleanliness:
#     - sprites remain separated by at least one bg cell, so swapped borders do not merge visually
#     - generator uses single-hole, single-component hollow sprites; this keeps the swap readable
#   color_inference:
#     - no ordering or transform uses numeric color ids; colors only participate through palette pairing
#     - active target colors are unique, so humans can match palette entry to sprite by visible equality
#   solver_assumptions_enforced_by_generator:
#     - each swappable sprite is a 4-connected monochrome border enclosing exactly one hole
#     - each swappable sprite's border color appears in the palette exactly once
#     - distractor colors do not appear in palette keys, so "unchanged" is a visible contrast case


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        h2, w2 = g.shape
        pal = np.zeros((h2, w2), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h2 - 2 : h2, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w2 - 2 : w2] = True

        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h2 and 0 <= cc < w2 and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top():
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom():
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left():
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right():
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
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


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping original_border->new_border, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask


def _collect_component_from_seeds(
    g: np.ndarray,
    pal_mask: np.ndarray,
    color: int,
    seeds: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Collect the full same-colored 4-connected component reached from the seeds."""
    h, w = g.shape
    stack = [(int(r), int(c)) for r, c in seeds]
    seen: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        if not (0 <= r < h and 0 <= c < w):
            continue
        if pal_mask[r, c] or int(g[r, c]) != int(color):
            continue
        seen.add((r, c))
        for rr, cc in neighbors4(r, c):
            stack.append((int(rr), int(cc)))
    return seen


def _swap_enclosed_bg_regions(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    For each enclosed bg hole, swap its surrounding sprite border color with the mapped palette color.
    """
    h, w = inp.shape
    bg_mask = (inp == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        seeds_by_color: dict[int, set[tuple[int, int]]] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if pal_mask[rr, cc]:
                    continue
                v = int(inp[rr, cc])
                if v == bg:
                    continue
                adj_counts[v] = adj_counts.get(v, 0) + 1
                seeds_by_color.setdefault(v, set()).add((int(rr), int(cc)))

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = int(col)
        if border is None:
            continue

        border_component = _collect_component_from_seeds(
            inp,
            pal_mask,
            border,
            sorted(seeds_by_color[border]),
        )
        new_border = int(mapping[border])
        for r, c in border_component:
            out[r, c] = new_border
        for r, c in pts:
            out[int(r), int(c)] = border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each hollow sprite's outline color with the palette-selected interior color.

    Input structure:
    - The boundary-most color is the background, with several hollow single-color sprites placed away from the edge.
    - One 2×N or N×2 legend lies on a grid edge; its outer line lists candidate sprite-border colors and its inner line lists the replacement border colors.

    Rule:
    - Read the edge legend as original-border -> replacement-border.
    - Find enclosed background holes that do not touch the grid boundary, and infer the surrounding sprite color from the dominant adjacent non-background pixels.
    - If that surrounding color is listed in the legend, recolor the whole enclosing sprite border to the mapped color and fill the hole with the sprite's original border color.

    Key insight: The legend does not merely say how to paint the cavity; it supplies the color that trades places with the sprite's own outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _swap_enclosed_bg_regions(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3
_P_DIAMOND = 0.2
_P_CORNER_CUT = 0.5

# Generation
_MAX_RETRIES = 300


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside the sprite that do not touch the sprite bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond mask with a single enclosed center cell."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _is_valid_hollow_sprite(sprite: np.ndarray) -> bool:
    """Single connected monochrome border with exactly one enclosed hole."""
    if int(sprite.sum()) > _SPRITE_CELL_MAX:
        return False
    if len(components_mask(sprite, connectivity=4)) != 1:
        return False
    return len(_get_enclosed_holes(sprite)) == 1


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a single-hole 4-connected hollow sprite suitable for border recoloring."""
    if h == 3 and w == 3 and rng.random() < _P_DIAMOND:
        return _make_diamond()

    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False
    if not _is_valid_hollow_sprite(sprite):
        return None

    if rng.random() < _P_CORNER_CUT:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        for r, c in corners[: rng.randint(1, min(2, len(corners)))]:
            test = sprite.copy()
            test[r, c] = False
            if _is_valid_hollow_sprite(test):
                sprite = test

    return sprite if _is_valid_hollow_sprite(sprite) else None


def _choose_colors(
    rng: random.Random,
    n: int,
    num_filled: int,
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick disjoint palette keys/values plus one distractor color."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    need = (2 * n) + 1
    assert need <= len(colors)
    border_keys = colors[:n]
    fill_values = colors[n : 2 * n]
    distractor_color = colors[2 * n]
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the given edge side."""
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on the chosen edge and reserve a 1-cell padded occupancy region."""
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
    """Find placements where the sprite and its enclosed hole stay clear of obstacles and edges."""
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
    border_color: int,
    hole_fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite: mask pixels use border_color; hole pixels use hole_fill when requested."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    if hole_fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = int(hole_fill)


def _valid_shape_choices(max_sh: int, max_sw: int) -> list[tuple[int, int]]:
    """All sprite bbox sizes that can support a valid hollow loop under the cell budget."""
    choices = []
    for sh in range(3, max_sh + 1):
        for sw in range(3, max_sw + 1):
            if sh + sw > _SPRITE_DIM_SUM_CAP:
                continue
            if (2 * sh) + (2 * sw) - 4 > _SPRITE_CELL_MAX:
                continue
            choices.append((sh, sw))
    return choices


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    shape_choices = _valid_shape_choices(max_sh, max_sw)
    if not shape_choices:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n))
    total_sprites = num_filled + 1

    shapes = []
    for _ in range(total_sprites):
        sh, sw = rng.choice(shape_choices)
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            key = int(filled_keys[i])
            infos.append((mask, key, int(mapping[key]), key, True))
        else:
            infos.append((mask, distractor_color, distractor_color, None, False))

    placed = []
    for mask, input_border, output_border, hole_fill, is_target in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, mask, output_border, hole_fill, holes)
        placed.append((r0, c0, mask, input_border, is_target, holes))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    inp = output.copy()
    for r0, c0, mask, input_border, is_target, holes in placed:
        if not is_target:
            continue
        sprite_view = inp[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        sprite_view[mask] = int(input_border)
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
