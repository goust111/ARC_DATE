# Combo variant of dbff022c (index 7)
# combo_spec: D4/1+D5/2
# Name: 多层嵌套精灵 + 仅填充内部边缘

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import components_mask, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
# - Keep the original edge palette: a 2xN or Nx2 legend still maps outer-border colors to fill colors.
# - Replace each relevant hollow sprite with a nested frame object:
#     outer frame color / background middle band / inner frame color / background center hole.
# - Example trace:
#   original task:
#     1. detect an enclosed background component,
#     2. infer the surrounding mapped border color,
#     3. fill the entire component.
#   D4-only behavior:
#     1. isolate the background band between outer and inner frames,
#     2. fill all of that band,
#     3. leave the center hole unchanged.
#   D5-only behavior on the same nested input:
#     1. treat the middle band and center hole as independent enclosed holes,
#     2. paint only each hole's inner edge,
#     3. if the inner-frame color is also a palette key, the center hole gets painted too.
#   combo behavior here:
#     1. first isolate only the background region that touches both the outer frame and another
#        non-background color inside the same outer-frame bbox,
#     2. then paint only that region's ring cells,
#     3. leave both the middle core and the center hole as background.
# - So the solver must combine "which enclosed region is the target layer" with
#   "inside that layer, only paint one-pixel depth".
#
# b) Complexity calibration
# - The generator forces the target middle band to have a non-empty core. Otherwise the combo
#   would collapse back to D4-only because "ring of the band" would equal the whole band.
# - It also forces at least one filled sprite whose inner-frame color is itself a palette key.
#   Otherwise the combo could collapse back to D5-only because the untouched center hole would
#   no longer contradict the naive "fill every enclosed ring" rule.
# - Palette size stays small (2-3 pairs), with exactly one filled nested sprite and one distractor
#   nested sprite, so the mapping cue is repeated but not noisy.
#
# c) Hidden assumptions audit
# - _bg_boundary / _best_palette / _palette_mapping still assume the palette is isolated on an
#   outer edge. The generator preserves that unchanged.
# - The original "_fill_enclosed_bg_regions" assumed every enclosed background component owned by
#   a mapped color should be painted. This fails for nested frames because the center hole must not
#   be treated like the outer-middle band.
# - The D4 selector assumed the entire selected middle band should be painted. This now fails
#   because the middle band's core must stay background.
# - The new solver therefore assumes:
#   1. every fillable object is a closed outer color component,
#   2. its target region is an enclosed background component inside that component's bbox,
#   3. that target region touches the outer color and also some other non-background color,
#   4. ring cells of the target region are well-defined by 4-neighbor erosion.
# - The generator guarantees all four assumptions by constructing closed rectangular outer/inner
#   frames with at least one cell of spacing between them and by keeping objects separated.
#
# d) Placement legality
# - Palette cells reserve a one-cell padded occupancy mask.
# - Sprites are placed strictly away from the global boundary and from each other with one-cell
#   padding, so no component merges or accidental edge-touching holes occur.
# - Nested sprites are generated only when both frames fit inside their bbox and the center hole
#   is non-empty.
#
# e) Visual observability
# - The output shows three distinct zones in each mapped nested sprite:
#     colored outer ring of the middle band / untouched middle-band core / untouched center hole.
# - Because the inner-frame color overlaps the palette on the filled sprite, a human can see that
#   "use any mapped border and fill its hole ring" is wrong: the center hole beside that mapped
#   inner frame still stays blank.
#
# f) Visual cleanliness
# - Only four layers exist per sprite: outer frame, middle band, inner frame, center hole.
# - No annotations or overlaps are added, and the palette remains detached from the sprites.
# - The grid is enlarged relative to the source task so the nested objects have readable spacing.
#
# g) Change inferability
# - Color change is not arbitrary: the palette still gives the only fill color source, and the
#   structure of the nested sprite identifies which background area may receive that color.
# - Unused palette entries remain possible, which prevents a trivial "replace all matching colors
#   everywhere" interpretation.
# ---------------------------------------------------------------------------


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
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

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

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "right", r0, r1)
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
    """Return (mapping border->fill, palette_mask)."""
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
        if b == bg:
            continue
        mapping[b] = int(f)

    return mapping, pal_mask


def _touches_local_boundary(pts: np.ndarray, h: int, w: int) -> bool:
    return any(int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1 for r, c in pts)


def _component_ring_and_core(pts: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ordered = [(int(r), int(c)) for r, c in pts]
    comp = set(ordered)
    ring: list[tuple[int, int]] = []
    core: list[tuple[int, int]] = []
    for r, c in ordered:
        if any((int(rr), int(cc)) not in comp for rr, cc in neighbors4(r, c)):
            ring.append((r, c))
        else:
            core.append((r, c))
    return ring, core


def _collect_target_middle_regions(
    g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    """
    Return [(outer_color, abs_pts)] for background regions that sit between two borders.

    The selected region must:
      - be enclosed inside one mapped outer-color component's bbox,
      - touch that outer-color component,
      - touch some other non-background color as well.
    """
    h, w = g.shape
    targets: list[tuple[int, np.ndarray]] = []

    for outer_color in mapping:
        color_mask = (g == int(outer_color)) & (~pal_mask)
        for comp in components_mask(color_mask, connectivity=4):
            r0 = int(comp[:, 0].min())
            c0 = int(comp[:, 1].min())
            r1 = int(comp[:, 0].max())
            c1 = int(comp[:, 1].max())
            local_h = r1 - r0 + 1
            local_w = c1 - c0 + 1
            if local_h < 3 or local_w < 3:
                continue

            local_bg = (g[r0 : r1 + 1, c0 : c1 + 1] == bg) & (~pal_mask[r0 : r1 + 1, c0 : c1 + 1])
            for hole in components_mask(local_bg, connectivity=4):
                if _touches_local_boundary(hole, local_h, local_w):
                    continue

                touches_self = False
                touches_other = False
                abs_pts = hole.copy()
                abs_pts[:, 0] += r0
                abs_pts[:, 1] += c0

                for rr, cc in abs_pts:
                    for nr, nc in neighbors4(int(rr), int(cc)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if pal_mask[nr, nc]:
                                continue
                            v = int(g[nr, nc])
                            if v == bg:
                                continue
                            if v == int(outer_color):
                                touches_self = True
                            else:
                                touches_other = True
                    if touches_self and touches_other:
                        break

                if touches_self and touches_other:
                    targets.append((int(outer_color), abs_pts))

    return targets


def _fill_nested_middle_rings(
    src: np.ndarray, out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    for outer_color, pts in _collect_target_middle_regions(src, bg, mapping, pal_mask):
        ring, _core = _component_ring_and_core(pts)
        fill = int(mapping[outer_color])
        for r, c in ring:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the ring of the middle layer in nested sprites using an edge palette.

    Input structure:
    - Background dominates the outer grid border, and one edge contains a 2xN or Nx2 legend whose
      outer line lists outer-frame colors and whose inner line lists their fill partners.
    - Sprites are nested hollow frames: outer frame, blank band, inner frame, blank center.
    - Some outer-frame colors appear in the palette; one nested distractor uses an outer color that
      is not a palette key.

    Rule:
    - Read the edge legend as an outer-border-color -> fill-color mapping.
    - For each mapped outer frame, search inside that color component's bbox for an enclosed
      background region that touches both the outer frame and another non-background color.
    - Treat that region as the target middle band and paint only its 4-neighbor boundary layer.
      The deeper band core and the center inside the inner frame both remain background.

    Key insight:
    - The solver must combine two filters: first identify which enclosed background region belongs
      to the nested two-frame structure, then restrict the write set to that region's inner edge
      instead of flood-filling the whole area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_nested_middle_rings(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator (combo) ---------------

_GRID_H_MIN, _GRID_H_MAX = 22, 26
_GRID_W_MIN, _GRID_W_MAX = 22, 26

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

_FILLED_SPRITE_H_MIN, _FILLED_SPRITE_H_MAX = 10, 12
_FILLED_SPRITE_W_MIN, _FILLED_SPRITE_W_MAX = 10, 12
_DISTRACTOR_SPRITE_H_MIN, _DISTRACTOR_SPRITE_H_MAX = 8, 10
_DISTRACTOR_SPRITE_W_MIN, _DISTRACTOR_SPRITE_W_MAX = 8, 10

_MAX_RETRIES = 500


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
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
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        r0, c0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        r0, c0 = rng.randint(0, h - pal_h), 0
    else:
        r0, c0 = rng.randint(0, h - pal_h), w - pal_w

    grid[r0 : r0 + pal_h, c0 : c0 + pal_w] = pal
    occ[max(0, r0 - 1) : min(h, r0 + pal_h + 1), max(0, c0 - 1) : min(w, c0 + pal_w + 1)] = True


def _frame_on_canvas(canvas_h: int, canvas_w: int, r0: int, c0: int, box_h: int, box_w: int) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _pts_to_mask(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


def _make_nested_sprite(h: int, w: int, rng: random.Random, require_middle_core: bool) -> dict[str, np.ndarray] | None:
    min_gap = 2 if require_middle_core else 1
    for _ in range(100):
        top_gap = rng.randint(min_gap, 3)
        bottom_gap = rng.randint(min_gap, 3)
        left_gap = rng.randint(min_gap, 3)
        right_gap = rng.randint(min_gap, 3)
        inner_h = h - 2 - top_gap - bottom_gap
        inner_w = w - 2 - left_gap - right_gap
        if inner_h < 4 or inner_w < 4:
            continue

        inner_r0 = 1 + top_gap
        inner_c0 = 1 + left_gap
        outer_mask = _frame_on_canvas(h, w, 0, 0, h, w)
        inner_mask = _frame_on_canvas(h, w, inner_r0, inner_c0, inner_h, inner_w)

        middle_mask = np.zeros((h, w), dtype=bool)
        middle_mask[1 : h - 1, 1 : w - 1] = True
        middle_mask[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = False
        middle_pts = np.argwhere(middle_mask)
        middle_ring, middle_core = _component_ring_and_core(middle_pts)
        if require_middle_core and not middle_core:
            continue

        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[inner_r0 + 1 : inner_r0 + inner_h - 1, inner_c0 + 1 : inner_c0 + inner_w - 1] = True
        if not center_mask.any():
            continue

        return {
            "outer": outer_mask,
            "inner": inner_mask,
            "middle_ring": _pts_to_mask((h, w), middle_ring),
            "middle_core": _pts_to_mask((h, w), middle_core),
            "body": outer_mask | inner_mask,
            "center": center_mask,
        }
    return None


def _choose_colors(
    rng: random.Random, n: int
) -> tuple[list[int], list[int], dict[int, int], int, int, int]:
    colors = rng.sample(list(range(1, 10)), 2 * n + 2)
    border_keys = colors[:n]
    fill_values = colors[n : 2 * n]
    distractor_outer = colors[2 * n]
    distractor_inner = colors[2 * n + 1]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_outer = int(rng.choice(border_keys))
    return border_keys, fill_values, mapping, distractor_outer, distractor_inner, filled_outer


def _pick_filled_inner_color(rng: random.Random, border_keys: list[int], outer_color: int) -> int:
    choices = [c for c in border_keys if c != outer_color]
    return int(rng.choice(choices))


def _find_valid_positions(h: int, w: int, occ: np.ndarray, body_mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = body_mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    sprite: dict[str, np.ndarray],
    outer_color: int,
    inner_color: int,
    fill_color: int | None,
) -> None:
    view = grid[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]]
    view[sprite["outer"]] = int(outer_color)
    view[sprite["inner"]] = int(inner_color)
    if fill_color is not None:
        view[sprite["middle_ring"]] = int(fill_color)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    filled_sprite = _make_nested_sprite(
        rng.randint(_FILLED_SPRITE_H_MIN, min(_FILLED_SPRITE_H_MAX, h - 2)),
        rng.randint(_FILLED_SPRITE_W_MIN, min(_FILLED_SPRITE_W_MAX, w - 2)),
        rng,
        require_middle_core=True,
    )
    distractor_sprite = _make_nested_sprite(
        rng.randint(_DISTRACTOR_SPRITE_H_MIN, min(_DISTRACTOR_SPRITE_H_MAX, h - 2)),
        rng.randint(_DISTRACTOR_SPRITE_W_MIN, min(_DISTRACTOR_SPRITE_W_MAX, w - 2)),
        rng,
        require_middle_core=False,
    )
    if filled_sprite is None or distractor_sprite is None:
        return None

    border_keys, fill_values, mapping, distractor_outer, distractor_inner, filled_outer = _choose_colors(rng, n)
    filled_inner = _pick_filled_inner_color(rng, border_keys, filled_outer)
    fill_color = int(mapping[filled_outer])
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = [
        (filled_sprite, int(filled_outer), int(filled_inner), fill_color),
        (distractor_sprite, int(distractor_outer), int(distractor_inner), None),
    ]
    infos.sort(key=lambda item: int(item[0]["body"].sum()), reverse=True)

    placed: list[tuple[int, int, dict[str, np.ndarray], int | None]] = []
    for sprite, outer_color, inner_color, maybe_fill in infos:
        positions = _find_valid_positions(h, w, occ, sprite["body"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, sprite, outer_color, inner_color, maybe_fill)
        placed.append((r0, c0, sprite, maybe_fill))
        sh, sw = sprite["body"].shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, sprite, maybe_fill in placed:
        if maybe_fill is None:
            continue
        inp[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]][sprite["middle_ring"]] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    # Guard against degenerating back to either single-axis variant.
    if not filled_sprite["middle_core"].any():
        return None
    if filled_inner not in mapping:
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
