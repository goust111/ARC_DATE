# Combo variant of dbff022c (index 6)
# combo_spec: D4/2+D5/0
# Name: 多色边框精灵 + 条纹填充

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for combo D4/2 + D5/0
# --------------------------------------
# 1. Concrete behavior and trace:
#    - Original dbff022c assumes a hollow sprite uses one border color, so one hole is
#      solved by counting adjacent border cells and then solid-filling the hole.
#    - This combo breaks both assumptions at once:
#      a) one sprite border uses two visible colors, `major` and `accent`;
#      b) output holes are striped, not solid.
#    - Example on one 5x4 ring:
#      the hole-adjacent shell cells may contain accent=4 and major=2 in a 4:2 split,
#      while the full shell contains major=2 and accent=4 in an 8:4 split.
#      Original logic would key the palette with 4 and flood-fill every hole cell.
#      Combo logic must key the palette with 2, then paint alternating stripes with
#      mapping[2] and background 0.
#    - Therefore the fusion is not a loose concatenation: the lookup step must read the
#      whole shell before the stripe writer runs, otherwise the wrong palette key is chosen.
# 2. Complexity calibration:
#    - Borders use exactly two colors. That is enough to make "local vote vs global
#      majority" visible without turning shells into noise.
#    - Stripe orientation has only two cases, derived from the hole bbox:
#      wider-or-square holes -> vertical stripes, taller holes -> horizontal stripes.
#    - Filled sprites are generated so both the majority conflict and stripe visibility
#      happen often; otherwise the task would collapse back toward the source task.
# 3. Implicit-assumption audit against the original main():
#    - `_best_palette` and `_palette_mapping` only depend on the edge legend geometry and
#      uniqueness of palette keys, so they remain valid.
#    - The original hole filler assumed one adjacent color represented the whole sprite.
#      That fails for multicolor shells, so the solver now labels full non-palette
#      connected components and counts colors over the entire touching component.
#    - The original writer could mutate every hole cell to one color. That fails for
#      striped interiors, so the solver must enumerate all bg holes before any writeback,
#      then paint each hole by row/column parity inside its own bbox.
# 4. Placement legality:
#    - The generator keeps palette and sprites separated by at least one background cell.
#    - Every filled sprite has exactly one enclosed hole with area >= 2, so stripes are
#      visible and holes never merge with the outer background.
#    - The multicolor border constructor guarantees `major` is the strict whole-shell
#      majority while `accent` is the strict local majority on cells touching the hole.
# 5. Visual observability and cleanliness:
#    - Humans can inspect the full shell and see that the repeated border color, not the
#      locally dominant accent, controls the palette lookup.
#    - They can also see that filled interiors preserve alternating background bands,
#      making the transform obviously different from a solid flood fill.
#    - No hidden ordering by numeric color id is used anywhere; every branch depends on
#      visible frequency or visible geometry.
# 6. New solver assumptions that the generator must guarantee:
#    - Each hole touches exactly one separated sprite component.
#    - A filled sprite's majority color is in the palette, while the distractor sprite's
#      majority color is not.
#    - At least one hole-adjacent accent cell uses a different palette key than the major
#      color, so the local-rule and majority-rule genuinely disagree on generated samples.


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
        """Reject edge regions that bleed into same-colored sprite cells."""
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
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
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


def _hole_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _stripe_axis_for_hole(pts: np.ndarray) -> str:
    r0, c0, r1, c1 = _hole_bbox(pts)
    return "vertical" if (c1 - c0) >= (r1 - r0) else "horizontal"


def _apply_striped_fill(out: np.ndarray, pts: np.ndarray, bg: int, fill: int) -> None:
    r0, c0, _, _ = _hole_bbox(pts)
    axis = _stripe_axis_for_hole(pts)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        if axis == "horizontal":
            out[rr, cc] = int(fill) if (rr - r0) % 2 == 0 else int(bg)
        else:
            out[rr, cc] = int(fill) if (cc - c0) % 2 == 0 else int(bg)


def _non_palette_component_info(out: np.ndarray, bg: int, pal_mask: np.ndarray) -> tuple[np.ndarray, dict[int, Counter[int]]]:
    """Group multicolor sprite cells into connected shells and count their colors."""
    comp_id = np.zeros(out.shape, dtype=int)
    comp_counts: dict[int, Counter[int]] = {}
    fg_mask = (out != bg) & (~pal_mask)
    comps = components_mask(fg_mask, connectivity=4)
    for cid, pts in enumerate(comps, start=1):
        comp_counts[cid] = Counter(int(out[int(r), int(c)]) for r, c in pts)
        for r, c in pts:
            comp_id[int(r), int(c)] = cid
    return comp_id, comp_counts


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Fill each enclosed hole by using the touching shell's majority color as the palette key,
    then write alternating stripes of the mapped fill color and background.
    """
    h, w = out.shape
    comp_id, comp_counts = _non_palette_component_info(out, bg, pal_mask)
    holes = components_mask((out == bg) & (~pal_mask), connectivity=4)

    for pts in holes:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        touching_ids = set()
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and int(comp_id[rr, cc]) != 0:
                    touching_ids.add(int(comp_id[rr, cc]))
        if not touching_ids:
            continue

        shell_counts: Counter[int] = Counter()
        for cid in touching_ids:
            shell_counts.update(comp_counts[cid])
        border_key = int(shell_counts.most_common(1)[0][0])
        if border_key not in mapping:
            continue

        _apply_striped_fill(out, pts, bg, int(mapping[border_key]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill hollow multicolor sprites by combining majority-color lookup with striped interiors.

    Input structure:
    - The outer boundary is mostly background, and one edge holds a compact 2xN or Nx2 palette legend.
    - Hollow sprites sit away from that legend; each sprite shell may mix multiple border colors instead of staying monochrome.

    Rule:
    - Detect the edge palette and read it as a border-key to fill-color mapping.
    - Find each enclosed background hole that is truly inside a sprite rather than connected to the outside.
    - For every hole, gather the whole touching shell, choose the shell's most frequent border color as the lookup key, read the mapped fill color from the palette, and paint the hole with alternating stripes of that fill color and the background. Wider-or-square holes use vertical stripes; taller holes use horizontal stripes.

    Key insight:
    - The color decision and the painting pattern are controlled by different visible cues: the shell's global majority color chooses the palette entry, while the hole's own geometry chooses whether the mapped color alternates by column or by row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_CORNER_GAP = 0.45
_MAX_RETRIES = 300


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return enclosed bg components inside a local sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """
    Create a one-hole shell with slight irregularity.

    The local hole stays single and enclosed, which keeps the majority-color signal
    legible and lets the generator reason about stripe orientation cleanly.
    """
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 4 and w >= 4 and rng.random() < _P_CORNER_GAP:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        cr, cc = corners[0]
        test = sprite.copy()
        test[cr, cc] = False
        if _get_enclosed_holes(test):
            sprite = test

    holes = _get_enclosed_holes(sprite)
    if len(holes) != 1 or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _hole_visible_for_stripes(hpts: np.ndarray) -> bool:
    """Require area >= 2 so stripes differ from solid fill."""
    return int(len(hpts)) >= 2


def _supports_axis(mask: np.ndarray, axis: str) -> bool:
    for hpts in _get_enclosed_holes(mask):
        if _hole_visible_for_stripes(hpts) and _stripe_axis_for_hole(hpts) == axis:
            return True
    return False


def _feasible_axes(max_sh: int, max_sw: int) -> list[str]:
    axes = []
    if max_sw >= 4:
        axes.append("vertical")
    if max_sh >= 4:
        axes.append("horizontal")
    return axes or ["vertical"]


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette keys, fill values, one distractor major color, and used keys."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n_pairs]

    fill_pool = list(range(1, 10))
    rng.shuffle(fill_pool)
    fill_values = fill_pool[:n_pairs]

    distractor_pool = [c for c in range(1, 10) if c not in border_keys]
    distractor_major = int(rng.choice(distractor_pool))
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_major, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the edge palette."""
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on the chosen edge and reserve one-cell padding around it."""
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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find placements that preserve non-overlap and keep holes off the grid boundary."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
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


def _hole_adj_sets(mask: np.ndarray, holes: list[np.ndarray]) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return shell cells touching holes, their union, and the full shell set."""
    border_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    per_hole = []
    inner_union: set[tuple[int, int]] = set()
    for hpts in holes:
        cells = set()
        for r, c in hpts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (rr, cc) in border_cells:
                    cells.add((rr, cc))
        if cells:
            per_hole.append(cells)
            inner_union |= cells
    return per_hole, inner_union, border_cells


def _make_multicolor_border(
    mask: np.ndarray, major_color: int, accent_color: int, rng: random.Random
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """
    Color a shell so the hole-adjacent cells prefer `accent_color` locally but the full shell
    still prefers `major_color` globally.
    """
    holes = _get_enclosed_holes(mask)
    per_hole, inner_union, border_cells = _hole_adj_sets(mask, holes)
    outer_only = border_cells - inner_union
    if len(holes) != 1 or len(per_hole) != 1 or not outer_only:
        return None

    local_cells = list(per_hole[0])
    rng.shuffle(local_cells)
    accent_need = len(local_cells) // 2 + 1
    if len(local_cells) < accent_need or len(border_cells) - accent_need <= accent_need:
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in border_cells:
        sprite[r, c] = int(major_color)
    for r, c in local_cells[:accent_need]:
        sprite[r, c] = int(accent_color)

    local_counts = Counter(int(sprite[r, c]) for r, c in per_hole[0])
    full_counts = Counter(int(sprite[r, c]) for r, c in border_cells)
    if local_counts.most_common(1)[0][0] != int(accent_color):
        return None
    if full_counts.most_common(1)[0][0] != int(major_color):
        return None
    return sprite, holes


def _paint_local_striped_hole(grid: np.ndarray, r0: int, c0: int, hpts: np.ndarray, bg: int, fill: int) -> None:
    """Paint one local hole with the same stripe rule used by the solver."""
    hr0, hc0, _, _ = _hole_bbox(hpts)
    axis = _stripe_axis_for_hole(hpts)
    for hr, hc in hpts:
        rr = r0 + int(hr)
        cc = c0 + int(hc)
        if axis == "horizontal":
            grid[rr, cc] = int(fill) if (int(hr) - hr0) % 2 == 0 else int(bg)
        else:
            grid[rr, cc] = int(fill) if (int(hc) - hc0) % 2 == 0 else int(bg)


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, sprite_colors: np.ndarray, holes: list[np.ndarray], fill: int | None, bg: int
) -> None:
    """Render a multicolor shell and optionally stripe its hole."""
    sh, sw = sprite_colors.shape
    patch = grid[r0 : r0 + sh, c0 : c0 + sw]
    mask = sprite_colors != 0
    patch[mask] = sprite_colors[mask]
    if fill is not None:
        for hpts in holes:
            _paint_local_striped_hole(grid, r0, c0, hpts, bg, int(fill))


def _random_shape_for_axis(rng: random.Random, max_sh: int, max_sw: int, axis: str) -> np.ndarray | None:
    for _ in range(80):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        cand = _make_hollow_sprite(sh, sw, rng)
        if cand is not None and _supports_axis(cand, axis):
            return cand
    return None


def _make_target_axes(rng: random.Random, num_filled: int, feasible_axes: list[str]) -> list[str]:
    if len(feasible_axes) == 1:
        return [feasible_axes[0]] * num_filled
    axes = ["horizontal", "vertical"][:]
    while len(axes) < num_filled:
        axes.append(rng.choice(feasible_axes))
    rng.shuffle(axes)
    return axes[:num_filled]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    feasible_axes = _feasible_axes(max_sh, max_sw)
    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    shapes: list[np.ndarray] = []
    for axis in _make_target_axes(rng, num_filled, feasible_axes):
        mask = _random_shape_for_axis(rng, max_sh, max_sw, axis)
        if mask is None:
            return None
        shapes.append(mask)

    while len(shapes) < total_sprites:
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_major, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos: list[tuple[np.ndarray, list[np.ndarray], int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            major = int(filled_keys[i])
            fill = int(mapping[major])
            accent_choices = [c for c in border_keys if c != major]
            accent = int(rng.choice(accent_choices))
        else:
            major = int(distractor_major)
            fill = None
            accent = int(rng.choice(border_keys))

        made = _make_multicolor_border(mask, major, accent, rng)
        if made is None:
            return None
        sprite_colors, holes = made
        if fill is not None and not all(_hole_visible_for_stripes(hpts) for hpts in holes):
            return None
        infos.append((sprite_colors, holes, fill))

    placed: list[tuple[int, int, list[np.ndarray], int | None]] = []
    for sprite_colors, holes, fill in infos:
        mask = sprite_colors != 0
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, sprite_colors, holes, fill, bg)
        placed.append((r0, c0, holes, fill))
        sh, sw = sprite_colors.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, holes, fill in placed:
        if fill is None:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if np.array_equal(inp, output):
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
