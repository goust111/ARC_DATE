# Variant of dbff022c
# Dimension: D4 — 源对象——空心精灵的形状与结构 (src)
# Change: 空心精灵具有单色边框，内部为背景色（0），形状不规则（非严格矩形），边框可能有缺口或突出部分... → 实心精灵替代空心精灵——需要替换内部颜色

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The original task detects enclosed background holes and fills them.
#    - In this variant the same border geometry remains, but every enclosed hole is already
#      occupied by a solid non-bg interior color in the input. The output does not create a
#      region from 0; it replaces the existing interior color with the palette-selected color.
#    - Small trace:
#      original input: border 4 around a 2-cell hole of 0, palette says 4 -> 6
#      original output: those 0 cells become 6
#      variant input: border 4 around a 2-cell block of placeholder 5, palette says 4 -> 6
#      variant output: those 5 cells become 6
#      difference: the actionable region is no longer "all enclosed bg", but "an enclosed
#      monochrome island already surrounded by one border color".
# b) Complexity calibration
#    - To keep the visual rule obvious, all sprites still use irregular single-color borders.
#    - Replaceable interiors share one placeholder color within a grid, so the user can see
#      that solid interiors are being rewritten rather than discovered by color id arithmetic.
#    - Exactly one distractor sprite keeps the same placeholder interior because its border
#      color is absent from the palette mapping.
# c) Old assumptions that break
#    - The original `_fill_enclosed_bg_regions` assumes every target region is background.
#      That fails immediately once interiors are non-zero.
#    - The original adjacency tally over bg components is still useful in spirit, but now it
#      must run over enclosed non-bg components. We therefore reject candidate regions that
#      touch bg directly and only recolor components surrounded by one non-bg border color.
# d) Placement legality
#    - The generator still builds border masks with genuine enclosed holes first, because that
#      guarantees the later solid interior region is inside the border and off the canvas edge.
#    - Palette and sprites keep a one-cell gap through the existing occupancy mask, so palette
#      detection remains stable and recolorable interior components never merge.
# e) Visual observability
#    - Inputs have zero enclosed bg holes for mapped sprites; their centers are visibly solid.
#    - Outputs show the same geometry with only the interior color changed, making the D4
#      change readable from a few examples without relying on hidden numeric ordering.
# f) Visual cleanliness
#    - Borders, interiors, and palette remain spatially separated.
#    - The placeholder color is kept out of the palette and out of border colors, so interior
#      regions are easy to spot and are not visually confused with the legend.
# g) Color-change inferability
#    - The mapping clue still comes from the edge palette.
#    - The changed color is always paired with a structural cue: an interior region fully
#      enclosed by one border color. This makes the recoloring rule inferable from examples.


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
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing parts of sprites
        that touch the grid edge.
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
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
                if len(border_keys) == 0:
                    continue
                if len(border_keys) != len(set(border_keys)):
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
    """
    Returns (mapping border->fill, palette_mask).
    """
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


def _recolor_enclosed_nonbg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Find monochrome non-bg regions that:
      - are outside the palette,
      - do not touch the grid boundary,
      - do not touch background,
      - are adjacent to exactly one non-bg color.

    That adjacent color is treated as the enclosing border; if it appears in the palette,
    replace the whole interior region with the mapped fill color.
    """
    h, w = out.shape
    colors = [int(c) for c in np.unique(out[~pal_mask]) if int(c) != bg]

    for color in colors:
        comps = components_mask((out == color) & (~pal_mask), connectivity=4)
        for pts in comps:
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
                continue

            adj_counts: dict[int, int] = {}
            touches_bg = False
            for r, c in pts:
                for rr, cc in neighbors4(int(r), int(c)):
                    if not (0 <= rr < h and 0 <= cc < w):
                        touches_bg = True
                        break
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == color:
                        continue
                    if v == bg:
                        touches_bg = True
                        break
                    adj_counts[v] = adj_counts.get(v, 0) + 1
                if touches_bg:
                    break
            if touches_bg or len(adj_counts) != 1:
                continue

            border = next(iter(adj_counts))
            if border not in mapping:
                continue

            fill = int(mapping[border])
            if fill == color:
                continue
            for r, c in pts:
                out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor solid sprite interiors using a color-palette legend on the grid edge.

    Input structure:
    - The background is still the dominant edge color, and a small 2xN or Nx2 palette block sits on one grid edge.
    - Each palette entry pairs a sprite border color with the color that should appear inside that sprite.
    - Sprites are irregular single-color borders whose enclosed interiors are already filled with a solid non-bg color instead of empty background.

    Rule:
    - Locate the palette legend and read it as a border->interior-color mapping.
    - Find enclosed non-bg interior regions outside the palette that are surrounded by exactly one border color and do not touch the outer background.
    - If that surrounding border color is listed in the palette, replace the entire interior region with the mapped color; otherwise leave the sprite unchanged.

    Key insight:
    - The target is no longer an empty hole. The solver must identify an already-colored interior by its enclosure and adjacency to a border color, then rewrite that interior color according to the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _recolor_enclosed_nonbg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
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

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 300
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside a border mask that do not touch the mask bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create an irregular border mask with at least one enclosed hole."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False
    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True
    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test
    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n: int, num_active: int
) -> tuple[list[int], list[int], int, int, dict[int, int], list[int]]:
    """
    Pick palette keys, palette fills, one placeholder interior color, one distractor border,
    and the subset of palette keys that actually appear on recolorable sprites.
    """
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n]

    remaining = [c for c in all_c if c not in border_keys]
    rng.shuffle(remaining)
    interior_placeholder = remaining[0]
    distractor_border = remaining[1]

    fill_pool = [c for c in all_c if c not in {interior_placeholder, distractor_border}]
    rng.shuffle(fill_pool)
    fill_values = fill_pool[:n]

    mapping = dict(zip(border_keys, fill_values))
    active_keys = rng.sample(border_keys, num_active)
    return border_keys, fill_values, interior_placeholder, distractor_border, mapping, active_keys


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
    """Place palette on the chosen edge and mark its occupancy with one-cell padding."""
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
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements where the border mask fits and all interior cells stay off the outer border."""
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
    interior_color: int,
    holes: list[np.ndarray],
) -> None:
    """Render a border mask plus solid enclosed interior cells."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = interior_color


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

    num_active = rng.randint(1, min(_FILLED_SPRITE_MAX, n))
    total_sprites = num_active + 1

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, placeholder, distractor_border, mapping, active_keys = _choose_colors(
        rng, n, num_active
    )
    pal = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        holes = _get_enclosed_holes(mask)
        if i < num_active:
            border_color = active_keys[i]
            interior_color = mapping[border_color]
            recolorable = True
        else:
            border_color = distractor_border
            interior_color = placeholder
            recolorable = False
        infos.append((mask, holes, border_color, interior_color, recolorable))

    placed = []
    for mask, holes, border_color, interior_color, recolorable in infos:
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, mask, border_color, interior_color, holes)
        placed.append((r0, c0, holes, recolorable))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    inp = output.copy()
    recolor_cells = 0
    for r0, c0, holes, recolorable in placed:
        if not recolorable:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = placeholder
                recolor_cells += 1
    if recolor_cells == 0:
        return None

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
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
