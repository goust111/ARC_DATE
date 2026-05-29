# Combo variant of dbff022c (index 21)
# combo_spec: D9/0+D11/1
# Name: 面积排序覆盖处理 + 精灵嵌套

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Keep the original edge palette: a 2xN or Nx2 legend maps shell-border colors to
#   fill colors.
# - Replace the original "independent holes" assumption with a fused geometry:
#   1. a large mapped hollow shell,
#   2. a smaller mapped hollow shell nested inside the large shell,
#   3. a separate witness shell outside the large shell that reuses the nested shell's
#      border color.
# - The witness shell shows the normal palette mapping for the small-shell color.
#   Inside the large shell, the nested shell would also map to that small color if it
#   were processed alone, but the new D9 rule says shells are handled from smaller area
#   to larger area and later fills may overwrite earlier ones.
# - Small trace:
#   1. process the nested shell first: its inner hole gets the small shell's mapped
#      fill;
#   2. process the external witness shell: its hole gets the same mapped fill and keeps
#      it because no larger shell contains it;
#   3. process the outer shell last: its interior is defined relative only to its own
#      border, so it paints the annulus and the nested shell's inner hole, overwriting
#      that earlier small-shell fill while leaving shell borders untouched.
# - Difference from the original task:
#   the original solver fills each enclosed bg component once, based on one owning
#   border. The fused task instead lets multiple shells target overlapping interior
#   cells, so ordering by visible shell size becomes part of the rule.
#
# b) Complexity calibration
# - The layout uses exactly one nested pair plus one external witness shell. That is
#   enough to expose both dimensions without clutter.
# - Only two mapped shell colors are behaviorally active in the play area. Extra palette
#   entries may exist, but they stay unused so the color-mapping clue remains readable.
# - The nested small shell and the witness shell share the same border color. This gives
#   repeated evidence for the small shell's palette mapping while still allowing the
#   nested instance to be visually overwritten by the larger shell.
#
# c) Implicit-assumption audit of the original main()
# - Original assumption broken #1: "each hole is filled once and then done." After D9,
#   a later larger shell may repaint cells that were already filled by a smaller shell.
# - Original assumption broken #2: "other sprite borders partition the bg into final
#   ownership regions." In the fused rule, each shell defines its own inside region
#   independently; nested borders do not stop a larger shell from repainting through the
#   enclosed space.
# - Original assumption broken #3: reading evidence from the mutating output is unsafe.
#   The new solver must derive all shell masks from the immutable input snapshot and only
#   use the mutable grid for the sequential writeback.
# - Palette detection and bg-border detection remain valid because the combo only changes
#   the sprite geometry and fill order, not the legend structure.
#
# d) Placement legality
# - The generator fixes one large ring, one nested small ring, and one external witness
#   ring. All are placed fully inside the grid with at least one bg cell separating the
#   shells from the frame.
# - The palette is restricted to the left or right edge and separated from the play area
#   by a horizontal moat, so palette isolation stays true.
# - The nested shell is placed with a visible bg annulus around it. The witness shell is
#   separated from the large shell by one empty row, so it never merges with it.
#
# e) Visual observability
# - Every sample contains one same-colored witness shell outside the nest. This makes the
#   small-shell palette mapping visible in the output, so the nested overwrite is not
#   mistaken for "that color just never fills."
# - The large shell's final fill visibly occupies both the annulus and the nested hole,
#   while the nested shell border still remains visible. That makes the overwrite effect
#   directly readable from input/output pairs.
#
# f) Visual cleanliness
# - The play area uses three shells with simple rectangular outlines. No extra markers,
#   no irrelevant distractor sprites, and no deep overlap stacks are introduced.
# - The nested region is concentrated in one upper block and the witness shell sits in a
#   lower block, so the two roles are visually separated.
#
# g) Color-change inferability
# - All color changes are explained by the edge palette. There is no hidden numeric sort:
#   the only ordering signal is visible shell area.
# - Because the nested shell and the witness shell share a border color, the solver can
#   infer that this color normally maps to one fill, but that the nested instance ends in
#   the large shell's fill due to later overwrite rather than due to a different mapping.
#
# 9) New solver assumptions to check against the generator
# - The solver assumes each shell border is a single 4-connected component. The generator
#   uses rectangular rings, so this always holds.
# - The solver assumes shell areas for the semantically active components are strictly
#   ordered. The generator uses three different ring sizes, so ties do not occur.
# - The solver assumes repaintable interior cells are exactly "inside a shell while not
#   belonging to any shell border or the palette." The generator constructs only ring
#   outlines, so this model matches the intended scene exactly.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
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
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                colors_outer = set(map(int, np.unique(seg_outer))) - {bg}
                colors_inner = set(map(int, np.unique(seg_inner))) - {bg}
                if len(colors_outer) < 2 or len(colors_inner) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(colors_outer | colors_inner) < 3 or not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(colors_outer | colors_inner), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                colors_outer = set(map(int, np.unique(seg_outer))) - {bg}
                colors_inner = set(map(int, np.unique(seg_inner))) - {bg}
                if len(colors_outer) < 2 or len(colors_inner) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(colors_outer | colors_inner) < 3 or not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(colors_outer | colors_inner), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                colors_outer = set(map(int, np.unique(seg_outer))) - {bg}
                colors_inner = set(map(int, np.unique(seg_inner))) - {bg}
                if len(colors_outer) < 2 or len(colors_inner) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(colors_outer | colors_inner) < 3 or not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(colors_outer | colors_inner), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                colors_outer = set(map(int, np.unique(seg_outer))) - {bg}
                colors_inner = set(map(int, np.unique(seg_inner))) - {bg}
                if len(colors_outer) < 2 or len(colors_inner) < 2:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if len(colors_outer | colors_inner) < 3 or not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(colors_outer | colors_inner), -(r1 - r0), nonbg), "right", r0, r1)
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

    mapping: dict[int, int] = {}
    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        if b != bg:
            mapping[b] = int(f)

    return mapping, pal_mask


def _sprite_components(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Connected monochrome shell components outside the palette."""
    h, w = g.shape
    sprite_mask = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    next_id = 0

    for color in sorted(int(x) for x in np.unique(g)):
        if color == bg:
            continue
        color_mask = (g == color) & (~pal_mask)
        if not color_mask.any():
            continue
        for pts in components_mask(color_mask, connectivity=4):
            pts = np.asarray(pts, dtype=int)
            rr = pts[:, 0]
            cc = pts[:, 1]
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[rr, cc] = True
            sprite_mask |= comp_mask
            r0 = int(rr.min())
            r1 = int(rr.max())
            c0 = int(cc.min())
            c1 = int(cc.max())
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "mask": comp_mask,
                    "area": int(len(pts)),
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
            next_id += 1

    return comps, sprite_mask


def _inside_mask_for_shell(blocked: np.ndarray) -> np.ndarray:
    """
    Cells enclosed by one shell if that shell alone is treated as a wall.

    All non-shell cells, including other sprite borders, remain traversable during this
    reachability test. This is what allows larger shells to repaint nested interiors.
    """
    h, w = blocked.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if not blocked[r, c] and not reachable[r, c]:
            reachable[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))

    return (~blocked) & (~reachable)


def _fill_area_sorted_shells(base: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> np.ndarray:
    """
    Fill each mapped shell from smaller area to larger area.

    Later shells overwrite earlier fills wherever their independent interior masks
    overlap, but shell-border cells themselves are never recolored.
    """
    comps, sprite_mask = _sprite_components(base, bg, pal_mask)
    ordered = sorted(comps, key=lambda comp: (comp["area"], comp["bbox_area"], comp["id"]))
    out = base.copy()

    for comp in ordered:
        color = int(comp["color"])
        if color not in mapping:
            continue
        inside = _inside_mask_for_shell(comp["mask"])
        paint = inside & (~sprite_mask) & (~pal_mask)
        out[paint] = int(mapping[color])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow shells using an edge palette, but resolve nested cases by area-ordered overwrite.

    Input structure:
    - The border background color dominates the frame, and one 2×N or N×2 palette sits on a grid edge.
    - The play area contains hollow single-color shells, including a small shell nested inside a larger shell and a separate witness shell of the same small-shell color elsewhere.

    Rule:
    - Read the edge palette as a border-color to fill-color mapping.
    - Treat each shell independently: the cells inside a shell are those that cannot reach the grid boundary if only that shell border is considered a wall.
    - Process mapped shells from smaller visible area to larger visible area, painting their interior cells while leaving all shell borders unchanged. Later larger shells may overwrite fills written earlier by smaller nested shells.

    Key insight:
    - Nested shell borders do not partition the final answer into fixed ownership regions. The decisive clue is that shells are applied in area order, so a large enclosing shell can repaint a nested shell's earlier fill even though the nested shell would map to a different palette color on its own.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"
    return _fill_area_sorted_shells(g, bg, mapping, pal_mask)


# --------------- generator (c) ---------------

_GRID_H = 13
_GRID_W = 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 4
_PALETTE_SIDES = ("left", "right")

_OUTER_H, _OUTER_W = 7, 8
_INNER_H, _INNER_W = 3, 4
_WITNESS_H, _WITNESS_W = 4, 4

_OUTER_R0 = 1
_WITNESS_R0 = 9

_MAX_RETRIES = 200


def _ring_mask(h: int, w: int) -> np.ndarray:
    """1-cell-thick rectangular ring."""
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    return mask


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct a 2xN or Nx2 palette for the given edge side."""
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


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette on one edge."""
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    elif side == "right":
        r0, c0 = rng.randint(0, h - ph), w - pw
    elif side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    else:
        r0, c0 = h - ph, rng.randint(0, w - pw)
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    """Paint a boolean mask onto the grid."""
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _choose_colors(
    rng: random.Random, n: int
) -> tuple[list[int], list[int], dict[int, int], int, int, int, int]:
    """Choose palette entries plus distinct outer/inner mapped shell colors."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n]
    fill_values = all_colors[n : 2 * n]
    mapping = dict(zip(border_keys, fill_values))

    outer_color, inner_color = rng.sample(border_keys, 2)
    outer_fill = int(mapping[outer_color])
    inner_fill = int(mapping[inner_color])
    return border_keys, fill_values, mapping, int(outer_color), int(inner_color), outer_fill, inner_fill


def _layout_columns(side: str, rng: random.Random) -> tuple[int, int, int]:
    """Pick outer, nested-inner, and witness column positions."""
    if side == "left":
        outer_c0 = rng.choice([4, 5])
        witness_c0 = rng.choice([9, 10])
    else:
        outer_c0 = rng.choice([2, 3])
        witness_c0 = rng.choice([6, 7])
    inner_c0 = outer_c0 + rng.choice([2, 3])
    return outer_c0, inner_c0, witness_c0


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h, w = _GRID_H, _GRID_W
    side = rng.choice(_PALETTE_SIDES)
    n = rng.randint(_PALETTE_N_MIN, _PALETTE_N_MAX)

    border_keys, fill_values, _mapping, outer_color, inner_color, outer_fill, inner_fill = _choose_colors(rng, n)
    pal = _build_palette(side, border_keys, fill_values)

    outer_mask = _ring_mask(_OUTER_H, _OUTER_W)
    inner_mask = _ring_mask(_INNER_H, _INNER_W)
    witness_mask = _ring_mask(_WITNESS_H, _WITNESS_W)

    outer_c0, inner_c0, witness_c0 = _layout_columns(side, rng)

    grid = np.full((h, w), bg, dtype=int)
    _place_palette(grid, pal, side, rng)

    _paint_mask(grid, _OUTER_R0, outer_c0, outer_mask, outer_color)
    _paint_mask(grid, _OUTER_R0 + 2, inner_c0, inner_mask, inner_color)
    _paint_mask(grid, _WITNESS_R0, witness_c0, witness_mask, inner_color)

    output = grid.copy()

    outer_inside = np.zeros_like(grid, dtype=bool)
    outer_inside[_OUTER_R0 + 1 : _OUTER_R0 + _OUTER_H - 1, outer_c0 + 1 : outer_c0 + _OUTER_W - 1] = True
    nested_inner_border = np.zeros_like(grid, dtype=bool)
    nested_inner_border[_OUTER_R0 + 2 : _OUTER_R0 + 2 + _INNER_H, inner_c0 : inner_c0 + _INNER_W] = inner_mask
    output[outer_inside & (~nested_inner_border)] = outer_fill

    witness_hole = np.zeros_like(grid, dtype=bool)
    witness_hole[_WITNESS_R0 + 1 : _WITNESS_R0 + _WITNESS_H - 1, witness_c0 + 1 : witness_c0 + _WITNESS_W - 1] = True
    output[witness_hole] = inner_fill

    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
