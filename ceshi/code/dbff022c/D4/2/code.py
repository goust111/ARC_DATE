# Variant of dbff022c
# Dimension: D4 — 源对象——空心精灵的形状与结构 (src)
# Change: 空心精灵具有单色边框，内部为背景色（0），形状不规则（非严格矩形），边框可能有缺口或突出部分... → 精灵边框由多种颜色组成——使用出现最多的颜色查表

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   preserved:
#     - Grid size, background, palette layout, and output-first generation all stay aligned
#       with the original task family.
#     - The palette still maps border-key colors to fill colors and sits on one outer edge.
#     - Only enclosed background holes are filled; the palette itself is untouched.
#   changed_core:
#     - A sprite border is no longer monochrome.
#     - The lookup key is the majority color over the entire border shell touching the hole,
#       not the color that happens to dominate immediately around the hole.
#   concrete_pixel_behavior:
#     - Each sprite border uses exactly two colors: `major` and `accent`.
#     - `accent` is arranged to dominate on the cells adjacent to the hole.
#     - `major` still dominates over the whole sprite because corner / outer-only cells keep
#       the major color.
#     - Therefore the old local rule and the new majority-over-shell rule disagree visibly.
#   behavior_trace:
#     - Original critical step: count colors adjacent to one hole, choose the most frequent
#       mapped color, and fill from that palette entry.
#     - Variant critical step: gather the full sprite shell surrounding that hole, count all
#       shell colors, choose the majority color, then consult the palette.
#     - Example: hole-adjacent counts may be accent=7, major=5, but full-shell counts are
#       major=8, accent=7. Old logic would choose accent; the variant must choose major.
#   complexity_calibration:
#     - Two colors per border are enough to make the change obvious without producing noisy
#       or hard-to-read samples.
#     - Accent colors for filled sprites are also palette keys, so the generator does not
#       degenerate into the original task where only one mapped border color is relevant.
#   implicit_assumption_audit:
#     - `_best_palette` and `_palette_mapping` only rely on palette uniqueness, not on sprite
#       monochromaticity, so they remain valid.
#     - The original fill routine assumed one adjacent color represented the sprite. That
#       assumption breaks immediately once a shell is multicolor, so the fill step must be
#       rewritten.
#     - The new solver assumes all border fragments touching one enclosed hole belong to the
#       same visual sprite. The generator guarantees separation between sprites and palette,
#       so this is safe.
#   placement_legality:
#     - Palette and sprites remain separated by at least one background cell.
#     - Holes are kept away from the outer grid boundary.
#     - Each sprite has at least three outer-only support cells, ensuring the generator can
#       realize "local accent majority but global major majority" without ambiguity.
#   observability_and_cleanliness:
#     - Output changes only inside holes, which preserves the original visual grammar.
#     - The border itself stays intact, so the human solver can compare border color counts
#       directly against the palette.
#     - No hidden ordering by color id is used; all decisions depend on visible frequencies.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _majority_color(values) -> int:
    """Return the visible-frequency majority color."""
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing sprite cells.
        """
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


def _non_palette_component_info(out: np.ndarray, bg: int, pal_mask: np.ndarray):
    """Label sprite cells while ignoring internal color variation."""
    h, w = out.shape
    comp_id = np.zeros((h, w), dtype=int)
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
    Fill each enclosed hole from the majority color of the full touching shell.
    """
    h, w = out.shape
    comp_id, comp_counts = _non_palette_component_info(out, bg, pal_mask)
    bg_mask = (out == bg) & (~pal_mask)
    holes = components_mask(bg_mask, connectivity=4)

    for pts in holes:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        touching_ids = set()
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    cid = int(comp_id[rr, cc])
                    if cid != 0:
                        touching_ids.add(cid)
        if not touching_ids:
            continue

        shell_counts: Counter[int] = Counter()
        for cid in touching_ids:
            shell_counts.update(comp_counts[cid])
        border_key = int(shell_counts.most_common(1)[0][0])
        if border_key not in mapping:
            continue

        fill = int(mapping[border_key])
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill hollow multicolor sprites by reading a border-to-fill legend on the edge.

    Input structure:
    - Background dominates the outer boundary, and a compact 2×N or N×2 palette sits on one grid edge.
    - Hollow sprites are drawn away from that palette, but each sprite border may now mix multiple colors.

    Rule:
    - Detect the edge palette and read it as a border-key to fill-color mapping.
    - Find enclosed background regions that are true holes rather than outside background.
    - For each hole, inspect the whole surrounding shell, take the shell's most frequent border color, and use that majority color as the palette lookup key.

    Key insight: The correct fill is controlled by the majority color of the entire multicolor shell, not by whichever border color is most common right next to the hole.
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


def _get_enclosed_holes(sprite: np.ndarray):
    """Return bg components inside the sprite that do not touch the sprite bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_hollow_sprite(h: int, w: int, rng: random.Random):
    """
    Create a one-hole shell. Most sprites are rectangular rings; some lose one corner
    to preserve the original family feeling of slight border irregularity.
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

    if int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    holes = _get_enclosed_holes(sprite)
    if len(holes) != 1:
        return None
    return sprite


def _choose_colors(rng: random.Random, n: int, num_filled: int):
    """Pick palette colors, fill colors, and one non-palette majority color for the distractor."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n]

    fill_pool = list(range(1, 10))
    rng.shuffle(fill_pool)
    fill_values = fill_pool[:n]

    distractor_pool = [c for c in range(1, 10) if c not in border_keys]
    distractor_major = int(rng.choice(distractor_pool))
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_major, mapping, filled_keys


def _build_palette(side: str, border_keys, fill_values):
    """Construct the 2xN or Nx2 palette block for the chosen edge side."""
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


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random):
    """Place the palette and reserve a 1-cell padding band around it."""
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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes):
    """Find all placements that preserve separation and keep holes away from the grid edge."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1 for hpts in holes for hr, hc in hpts):
                continue
            positions.append((r0, c0))
    return positions


def _hole_adj_sets(mask: np.ndarray, holes):
    """Return per-hole shell cells touching holes, their union, and all border cells."""
    border_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    per_hole = []
    union = set()
    for hpts in holes:
        cells = set()
        for r, c in hpts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (rr, cc) in border_cells:
                    cells.add((rr, cc))
        if cells:
            per_hole.append(cells)
            union |= cells
    return per_hole, union, border_cells


def _make_multicolor_border(mask: np.ndarray, major_color: int, accent_color: int, rng: random.Random):
    """
    Color a shell so `accent_color` dominates around the hole but `major_color`
    remains the overall shell majority.
    """
    holes = _get_enclosed_holes(mask)
    per_hole, inner, border_cells = _hole_adj_sets(mask, holes)
    outer = border_cells - inner
    if not holes or len(per_hole) != 1 or not inner or len(outer) < 3:
        return None

    local_cells = list(per_hole[0])
    rng.shuffle(local_cells)
    accent_need = len(local_cells) // 2 + 1
    if len(border_cells) - accent_need <= accent_need:
        return None

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in border_cells:
        sprite[r, c] = major_color
    for r, c in local_cells[:accent_need]:
        sprite[r, c] = accent_color

    if _majority_color(sprite[r, c] for r, c in border_cells) != major_color:
        return None
    if _majority_color(sprite[r, c] for r, c in per_hole[0]) == major_color:
        return None
    return sprite, holes


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, sprite_colors: np.ndarray, fill: int | None, holes):
    """Render a multicolor sprite; fill its hole only on the output canvas."""
    sh, sw = sprite_colors.shape
    patch = grid[r0 : r0 + sh, c0 : c0 + sw]
    mask = sprite_colors != 0
    patch[mask] = sprite_colors[mask]
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _gen_attempt():
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

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n))
    total_sprites = num_filled + 1

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_major, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            major = int(filled_keys[i])
            fill = int(mapping[major])
            accent = int(rng.choice([c for c in border_keys if c != major]))
        else:
            major = int(distractor_major)
            fill = None
            accent = int(rng.choice(border_keys))

        made = _make_multicolor_border(mask, major, accent, rng)
        if made is None:
            return None
        sprite_colors, holes = made
        infos.append((sprite_colors, holes, fill))

    placed = []
    for sprite_colors, holes, fill in infos:
        mask = sprite_colors != 0
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, sprite_colors, fill, holes)
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


def generate_input():
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
