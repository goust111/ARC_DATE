# Combo variant of dbff022c (index 39)
# combo_spec: D5/1+D8/1
# Name: 棋盘格填充 + 凸包填充区域

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The palette legend is unchanged: a 2xN or Nx2 block on one grid edge still maps sprite
#      border colors to fill colors.
#    - The original task identifies enclosed background holes and paints them solid. This combo
#      changes both the target region and the write pattern:
#      1. First isolate each sprite component whose border color appears in the palette.
#      2. Compute the convex hull of that component's border pixels.
#      3. Consider every background cell inside that convex hull, even if that cell lies outside
#         the sprite's true enclosed hole because the outer silhouette is concave.
#      4. Paint only the even-parity cells of that hull-background region; odd-parity cells stay
#         background, forming a checkerboard.
#    - Example trace: imagine a hollow 5x5 frame with one exterior corner cell removed. In the
#      original task only the central 3x3 hole would be filled solid. In this combo, the missing
#      corner sits inside the convex hull as well, so the output paints alternating cells across
#      both the true hole and that extra corner pocket.
#
# b) Complexity calibration
#    - The variant must visibly differ from both the seed task and the D5-only checkerboard task.
#      To enforce that, the generator requires at least one fillable sprite whose convex-hull
#      background region strictly exceeds its true enclosed-hole region, and at least one of those
#      extra cells must actually be painted by the checkerboard parity.
#    - Checkerboard visibility also needs both parities present, so the generator rejects outputs
#      where every fillable hull region collapses to a single parity or a single cell.
#
# c) Implicit-assumption audit of the original main()
#    - _bg_boundary and _palette_mapping only assume that the palette stays on a grid edge and the
#      outer border is mostly background. Those assumptions still hold.
#    - The original _fill_enclosed_bg_regions assumes the fill target is a background component
#      discovered from flood-fill. That is no longer valid: convex-hull fill may include cells
#      that are not part of any enclosed component in the input. The solver therefore must switch
#      from "find holes first" to "find mapped sprite components first".
#    - The original adjacency vote for border color is unnecessary once the solver processes each
#      sprite component directly: the component color itself is the border key.
#    - The new convex-hull step assumes the fillable region contains only background cells in the
#      generated tasks. The generator preserves this by reserving each sprite's full bbox plus
#      padding, keeping other objects and the palette out of the hull area.
#
# d) Placement legality
#    - Palette placement still uses a reserved padded rectangle on one edge.
#    - Sprite placement still uses bbox padding, which is sufficient here because a sprite's convex
#      hull remains inside its bbox. That prevents hull fills from colliding with other sprites or
#      with the palette.
#    - Candidate placements are filtered so the would-be hull fill cells do not touch the outer
#      grid boundary, preserving border-background dominance and keeping the fill region legible.
#
# e) Visual observability
#    - Across a few examples, a human can see two simultaneous cues: fills are checkerboarded
#      rather than solid, and some colored checker cells appear in concavity pockets that were not
#      true holes in the input. That makes the D8 contribution observable rather than hidden.
#    - The rule depends only on visible geometry and visible parity, not on color ids as numbers.
#
# f) Visual cleanliness
#    - Only one overlay is introduced: mapped fill colors on alternating background cells inside
#      each selected convex hull. Borders remain single-color, the palette remains separate, and
#      bbox padding keeps filled regions from visually merging.
#
# g) Color-change inferability
#    - The palette still supplies the border->fill color mapping, so the only new inference burden
#      is geometric: determine that the filled area is the convexified interior, then apply the
#      global checkerboard parity there.
#
# h) Audit of new main() assumptions
#    - The loop over palette keys assumes fillable sprites use colors that are unique per sprite
#      component, or at least that every component of a mapped color should be filled identically.
#      The generator uses distinct border colors for all sprites.
#    - The convex-hull rasterizer assumes each sprite component has area >= 3 and is non-degenerate.
#      Generated sprites are at least 3x3 and connected, so their hull polygon is well-defined.
#    - The write step assumes cells to be filled are background in the input. The generator checks
#      this implicitly via the oracle and structurally via padded placement.


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
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h2 and 0 <= cc < w2 and not pal[rr, cc]:
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
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
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
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
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
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
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
                if len(border_keys) == 0 or len(border_keys) != len(set(border_keys)):
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


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_in_convex_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    eps = 1e-9
    sign = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) <= eps:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def _convex_hull_fill_mask(sprite: np.ndarray) -> np.ndarray:
    """
    Rasterize the convex hull of a sprite's occupied pixel squares and return the
    background cells inside that hull.
    """
    if not sprite.any():
        return np.zeros_like(sprite, dtype=bool)

    corners: list[tuple[float, float]] = []
    for r, c in np.argwhere(sprite):
        rr = int(r)
        cc = int(c)
        corners.extend(
            [
                (float(cc), float(rr)),
                (float(cc + 1), float(rr)),
                (float(cc), float(rr + 1)),
                (float(cc + 1), float(rr + 1)),
            ]
        )

    poly = _convex_hull_polygon(corners)
    hull_mask = np.zeros_like(sprite, dtype=bool)
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            if _point_in_convex_polygon(c + 0.5, r + 0.5, poly):
                hull_mask[r, c] = True
    return hull_mask & (~sprite)


def _checkerboard_fill_region(out: np.ndarray, pts: np.ndarray, fill: int) -> tuple[int, int]:
    """Paint even-parity cells in pts with fill and leave odd-parity cells unchanged."""
    painted = 0
    skipped = 0
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        if (rr + cc) % 2 == 0:
            out[rr, cc] = fill
            painted += 1
        else:
            skipped += 1
    return painted, skipped


def _fill_convex_hull_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """Fill checkerboard patterns on background cells inside convex hulls of mapped sprite components."""
    base = out.copy()
    for border_color, fill in mapping.items():
        comps = components_mask((base == int(border_color)) & (~pal_mask), connectivity=4)
        for comp in comps:
            if comp.size == 0:
                continue
            r0 = int(comp[:, 0].min())
            c0 = int(comp[:, 1].min())
            r1 = int(comp[:, 0].max())
            c1 = int(comp[:, 1].max())
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                sprite[int(r) - r0, int(c) - c0] = True

            fill_local = _convex_hull_fill_mask(sprite)
            if not fill_local.any():
                continue

            pts = []
            for lr, lc in np.argwhere(fill_local):
                gr = r0 + int(lr)
                gc = c0 + int(lc)
                if pal_mask[gr, gc]:
                    continue
                if int(base[gr, gc]) != bg:
                    continue
                pts.append((gr, gc))
            if pts:
                _checkerboard_fill_region(out, np.asarray(pts, dtype=int), int(fill))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply palette-driven checkerboard fills inside convexified sprite interiors.

    Input structure:
    - Background dominates the outer border, with several hollow single-color sprites elsewhere.
    - A 2xN or Nx2 legend lies on one grid edge; its outer line lists sprite border colors and its
      inner line lists the corresponding fill colors.

    Rule:
    - Detect the edge legend and read it as a border-color to fill-color mapping.
    - For each non-palette sprite component whose border color appears in that mapping, compute the
      convex hull of the component's border pixels.
    - Among the background cells inside that convex hull, paint only the even (row+col) parity
      cells with the mapped fill color; leave odd-parity hull cells unchanged.

    Key insight:
    - The palette still determines which fill color belongs to each border color, but the target
      region is no longer the literal enclosed hole: it is the convexified interior of the sprite,
      and that region is rendered with a global checkerboard parity rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_convex_hull_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

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
_P_BORDER_REMOVAL = 0.6
_P_DIAMOND = 0.1

_MAX_RETRIES = 500


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _is_connected(sprite: np.ndarray) -> bool:
    return len(components_mask(sprite, connectivity=4)) == 1


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a connected hollow sprite with at least one enclosed hole."""
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
            if _is_connected(test) and _get_enclosed_holes(test):
                sprite = test
    if not _is_connected(sprite):
        return None
    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _sprite_analysis(sprite: np.ndarray) -> dict[str, object]:
    holes = _get_enclosed_holes(sprite)
    hole_mask = np.zeros_like(sprite, dtype=bool)
    for pts in holes:
        for r, c in pts:
            hole_mask[int(r), int(c)] = True
    hull_fill_mask = _convex_hull_fill_mask(sprite)
    extra_mask = hull_fill_mask & (~hole_mask)
    return {
        "holes": holes,
        "hole_mask": hole_mask,
        "hull_fill_mask": hull_fill_mask,
        "extra_mask": extra_mask,
        "hole_area": int(hole_mask.sum()),
        "hull_fill_area": int(hull_fill_mask.sum()),
        "extra_area": int(extra_mask.sum()),
    }


def _sample_sprite(rng: random.Random, max_h: int, max_w: int, require_extra: bool) -> tuple[np.ndarray, dict[str, object]] | None:
    for _ in range(80):
        sh = rng.randint(3, max_h)
        sw = rng.randint(3, max_w)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            continue
        analysis = _sprite_analysis(mask)
        if analysis["hull_fill_area"] < 2:
            continue
        if require_extra and int(analysis["extra_area"]) < 1:
            continue
        return mask, analysis
    return None


def _choose_colors(rng: random.Random, n: int, num_filled: int):
    """Pick palette border/fill colors, distractor color, and filled-sprite keys."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n]
    rng.shuffle(all_c)
    fill_values = all_c[:n]
    distractor_color = rng.choice([c for c in range(1, 10) if c not in border_keys])
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
    """Place palette on the chosen edge and mark occupancy with 1-cell padding."""
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


def _global_parity_counts(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int]:
    even = 0
    odd = 0
    for r, c in np.argwhere(mask):
        if (r0 + int(r) + c0 + int(c)) % 2 == 0:
            even += 1
        else:
            odd += 1
    return even, odd


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    require_combo_visibility: bool = False,
    extra_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Find placements where the sprite fits, stays separated, and keeps target cells off the border."""
    sh, sw = mask.shape
    positions = []
    target_pts = [(int(r), int(c)) for r, c in np.argwhere(target_mask)]
    extra_pts = [] if extra_mask is None else [(int(r), int(c)) for r, c in np.argwhere(extra_mask)]
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + tr == 0 or c0 + tc == 0 or r0 + tr == h - 1 or c0 + tc == w - 1
                for tr, tc in target_pts
            ):
                continue
            if require_combo_visibility:
                even, odd = _global_parity_counts(target_mask, r0, c0)
                if even == 0 or odd == 0:
                    continue
                if not any((r0 + er + c0 + ec) % 2 == 0 for er, ec in extra_pts):
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
    analysis: dict[str, object],
) -> dict[str, int]:
    """Render a sprite and, if mapped, checker-fill its convex-hull background cells."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    stats = {"painted": 0, "skipped": 0, "extra_painted": 0, "extra_total": int(analysis["extra_area"])}
    if fill is None:
        return stats

    fill_mask = analysis["hull_fill_mask"]
    pts = np.array([(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(fill_mask)], dtype=int)
    painted, skipped = _checkerboard_fill_region(grid, pts, fill)
    stats["painted"] = painted
    stats["skipped"] = skipped

    extra_painted = 0
    for r, c in np.argwhere(analysis["extra_mask"]):
        if (r0 + int(r) + c0 + int(c)) % 2 == 0:
            extra_painted += 1
    stats["extra_painted"] = extra_painted
    return stats


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

    special = _sample_sprite(rng, max_sh, max_sw, True)
    if special is None:
        return None

    filled_infos: list[tuple[np.ndarray, dict[str, object], bool]] = [(special[0], special[1], True)]
    for _ in range(num_filled - 1):
        sampled = _sample_sprite(rng, max_sh, max_sw, False)
        if sampled is None:
            return None
        filled_infos.append((sampled[0], sampled[1], False))

    distractor = _sample_sprite(rng, max_sh, max_sw, False)
    if distractor is None:
        return None

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    rng.shuffle(filled_infos)
    special_idx = next(i for i, (_, _, is_special) in enumerate(filled_infos) if is_special)
    if special_idx != 0:
        filled_infos[0], filled_infos[special_idx] = filled_infos[special_idx], filled_infos[0]

    for i, (mask, analysis, is_special) in enumerate(filled_infos):
        color = filled_keys[i]
        infos.append(
            {
                "mask": mask,
                "analysis": analysis,
                "color": color,
                "fill": mapping[color],
                "require_combo_visibility": is_special,
            }
        )
    infos.append(
        {
            "mask": distractor[0],
            "analysis": distractor[1],
            "color": distractor_color,
            "fill": None,
            "require_combo_visibility": False,
        }
    )

    placed = []
    total_painted = 0
    total_skipped = 0
    total_extra_painted = 0
    for info in infos:
        target_mask = info["analysis"]["hull_fill_mask"] if info["fill"] is not None else info["analysis"]["hole_mask"]
        positions = _find_valid_positions(
            h,
            w,
            occ,
            info["mask"],
            target_mask,
            require_combo_visibility=bool(info["require_combo_visibility"]),
            extra_mask=info["analysis"]["extra_mask"],
        )
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        stats = _paint_sprite(grid, r0, c0, info["mask"], info["color"], info["fill"], info["analysis"])
        placed.append((r0, c0, info))
        total_painted += stats["painted"]
        total_skipped += stats["skipped"]
        total_extra_painted += stats["extra_painted"]
        sh, sw = info["mask"].shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    if total_painted == 0 or total_skipped == 0 or total_extra_painted == 0:
        return None

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, info in placed:
        if info["fill"] is None:
            continue
        for rr, cc in np.argwhere(info["analysis"]["hull_fill_mask"]):
            inp[r0 + int(rr), c0 + int(cc)] = bg

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
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
