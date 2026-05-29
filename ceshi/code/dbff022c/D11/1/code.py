# Variant of dbff022c
# Dimension: D11 — 精灵的空间分布与间距 (meta)
# Change: 精灵散布在网格中，彼此不重叠不相邻，与调色板区域有明显间隔... → 精灵嵌套——小精灵在大精灵内部

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior and trace
# - Keep the original palette mechanic: a 2xN or Nx2 legend sits on one border and
#   maps border colors to fill colors.
# - Replace the original "scattered hollow sprites" layout with one mandatory nested
#   group: a large hollow rectangular shell contains a smaller hollow shell, and a
#   second same-colored nested sprite may sit beside it inside the same outer cavity.
# - The large shell's empty interior is split into two visually distinct enclosed bg
#   regions by the inner hollow shell:
#   1. an annulus between the outer border and the inner shell,
#   2. the inner shell's own hole.
# - The output fills the annulus using the outer shell's palette mapping. The inner
#   hole is then filled only if the inner-shell color also appears as a palette key.
#   If the inner-shell color is a distractor, its hole stays bg.
# - Small trace:
#   original logic on a nested input would inspect the annulus bg component, see both
#   the outer color and the inner color on its boundary, and might choose by raw
#   adjacency frequency. The variant instead asks: which adjacent border actually
#   encloses this bg region as its parent shell? For the annulus, that parent is the
#   outer shell; for the inner hole, it is the inner shell.
#
# b) Complexity calibration
# - The main signal is still one nested hollow shell, but the optional extra nested
#   same-colored sprite is necessary so the annulus can touch the inner color often
#   enough to defeat the original dominant-adjacency heuristic.
# - The palette still has 2-4 entries, so there can be unused mappings; this preserves
#   the original "legend contains extra keys" feel without adding unrelated complexity.
# - The inner shell has two repeated modes across samples: mapped or distractor. That
#   branch is small enough to infer and large enough to be visible.
#
# c) Implicit-assumption audit of the original main()
# - The original `_fill_enclosed_bg_regions` assumes each non-boundary bg component is
#   adjacent to one dominant border color. This was harmless when sprites were
#   separated; it becomes unsafe once an outer hole is split by an inner sprite.
# - The original code also updates the grid in place while using the current grid as
#   evidence. In the nested case, processing order should not change which shell owns a
#   region, so the new implementation reasons from an immutable snapshot and writes to a
#   separate output array.
# - Palette detection and bg-color detection do not depend on the old D11 geometry, so
#   they remain intact.
#
# d) Placement legality
# - The generator reserves a 1-cell moat between the palette and the outer shell.
# - The outer shell is kept at least 1 cell away from the global boundary so its hole
#   never touches the frame.
# - The inner shell is placed strictly inside the outer hole with at least 1 bg cell of
#   clearance from the outer border, guaranteeing a visible annulus.
# - The generator constructs the nested input layout directly, derives the intended
#   output by filling the known annulus and inner-hole regions, and finally checks
#   `main(input) == output`.
#
# e) Visual observability
# - In every generated sample there is at least one enclosed bg region adjacent to two
#   different colors, and the inner color appears multiple times inside the outer
#   cavity. That makes the nesting signature visible and also prevents regression back
#   to the old adjacency-based behavior.
# - The outer fill stops cleanly at the inner shell border, so the containment relation
#   is readable by eye.
#
# f) Visual cleanliness
# - Only four semantic structures appear in the play area: the palette, the outer
#   shell, the inner hollow shell, and one small same-colored nested sprite. No extra
#   markers are added beyond what is needed to expose the containment rule.
# - The outer shell stays rectangular for readability; the smaller shell is slightly
#   tabbed so its presence is easy to see even when the annulus wraps around it.
#
# g) Color-change inferability
# - Colors never change by numeric ordering or hidden IDs. All fills are justified only
#   by the visible edge palette and the shell color that encloses the region.
# - The inner shell's distractor mode is also visible: its color is absent from the
#   palette keys, so its hole staying bg is explainable from the examples.
#
# 9) New solver assumptions to check against the generator
# - The solver assumes each relevant shell is a single connected non-bg component. The
#   generator uses simple rectangular rings, so that always holds.
# - The solver chooses the smallest enclosing adjacent component by bbox area. The
#   generator creates proper nesting, so the immediate parent shell is always the
#   smallest enclosing shell.
# - The solver expects the palette to be isolated from sprite colors on the outside.
#   The generator keeps the original 1-cell separation from the palette and restricts
#   the palette to the left or right edge so the tall nested structure fits reliably.


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
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "top", c0, c1)
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
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "bottom", c0, c1)
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
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "left", r0, r1)
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
    """Returns (mapping border->fill, palette_mask)."""
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
        if b != bg:
            mapping[b] = f

    return mapping, pal_mask


def _sprite_components(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Connected non-bg components outside the palette, with per-cell component ids."""
    h, w = g.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    next_id = 0

    for color in sorted(int(x) for x in np.unique(g)):
        if color == bg:
            continue
        mask = (g == color) & (~pal_mask)
        if not mask.any():
            continue
        for pts in components_mask(mask, connectivity=4):
            pts = np.asarray(pts, dtype=int)
            rr = pts[:, 0]
            cc = pts[:, 1]
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[rr, cc] = True
            r0 = int(rr.min())
            c0 = int(cc.min())
            r1 = int(rr.max())
            c1 = int(cc.max())
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "mask": comp_mask,
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
            comp_ids[rr, cc] = next_id
            next_id += 1

    return comps, comp_ids


def _can_reach_boundary_without_crossing(start: tuple[int, int], blocked: np.ndarray) -> bool:
    """
    Return True if `start` can reach any grid-boundary cell while treating `blocked`
    as the only obstacle. All other colors are traversable.
    """
    h, w = blocked.shape
    sr, sc = start
    if blocked[sr, sc]:
        return False
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    seen[sr, sc] = True

    while q:
        r, c = q.popleft()
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return True
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return False


def _select_region_border(
    pts: np.ndarray,
    base: np.ndarray,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
    comps: list[dict],
    comp_ids: np.ndarray,
) -> int | None:
    """
    Choose the border color that owns this enclosed bg region.

    Prefer the smallest adjacent component that truly encloses the region when treated
    as a wall by itself. This resolves nested shells. If nothing encloses the region,
    fall back to the original dominant-adjacent-color heuristic.
    """
    adj_counts: dict[int, int] = {}
    candidate_ids: set[int] = set()

    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < base.shape[0] and 0 <= cc < base.shape[1]:
                if pal_mask[rr, cc]:
                    continue
                cid = int(comp_ids[rr, cc])
                if cid < 0:
                    continue
                candidate_ids.add(cid)
                col = int(comps[cid]["color"])
                adj_counts[col] = adj_counts.get(col, 0) + 1

    if not adj_counts:
        return None

    start = tuple(map(int, pts[0]))
    enclosing: list[tuple[int, int, int]] = []
    for cid in candidate_ids:
        color = int(comps[cid]["color"])
        if color not in mapping:
            continue
        blocked = comps[cid]["mask"]
        if not _can_reach_boundary_without_crossing(start, blocked):
            enclosing.append((int(comps[cid]["bbox_area"]), -adj_counts.get(color, 0), color))

    if enclosing:
        enclosing.sort()
        return int(enclosing[0][2])

    best_color = None
    best_count = -1
    for color, count in adj_counts.items():
        if color in mapping and count > best_count:
            best_count = count
            best_color = color
    return best_color


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Fill non-boundary bg components. In nested layouts, a region is owned by the
    nearest enclosing shell rather than by raw adjacency frequency.
    """
    base = out.copy()
    comps, comp_ids = _sprite_components(base, bg, pal_mask)
    bg_mask = (base == bg) & (~pal_mask)

    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == base.shape[0] - 1 or c == base.shape[1] - 1 for r, c in pts):
            continue
        border = _select_region_border(np.asarray(pts, dtype=int), base, mapping, pal_mask, comps, comp_ids)
        if border is None:
            continue
        out[np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]] = int(mapping[border])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested hollow sprites using a color legend placed on the grid edge.

    Input structure:
    - bg is the dominant border color, and a 2×N or N×2 palette sits on one grid edge.
    - The play area contains a large hollow shell with a smaller hollow shell inside it, so one enclosed outer cavity is split into layered interior regions.

    Rule:
    - Read the edge palette as a border-color to fill-color mapping.
    - Find each bg region that is enclosed away from the grid boundary.
    - For nested interiors, assign each enclosed region to the nearest shell that truly contains it, then fill that region with the shell's mapped color if its border color appears in the palette.

    Key insight:
    - Adjacency alone is no longer enough because an outer cavity can touch both the large shell and the inner shell; the correct fill comes from containment depth, not just from the most frequent neighboring color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 13, 13
_GRID_W_MIN, _GRID_W_MAX = 14, 15

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

# Nested-shell geometry
_OUTER_H, _OUTER_W = 11, 10
_INNER_SHELL_TEMPLATE = np.array(
    [
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    dtype=bool,
)
_INNER_BAR_LEN = 5
_P_INNER_MAPPED = 0.65

# Generation
_MAX_RETRIES = 300


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
    """Place the palette and mark a 1-cell moat around it as occupied."""
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _choose_colors(
    rng: random.Random, n: int, inner_mapped: bool
) -> tuple[list[int], list[int], dict[int, int], int, int, int | None]:
    """Pick palette keys/fills and the outer/inner shell colors."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)

    border_keys = all_colors[:n]
    fill_values = all_colors[n : 2 * n]
    mapping = dict(zip(border_keys, fill_values))
    leftovers = all_colors[2 * n :]

    outer_color = rng.choice(border_keys)
    if inner_mapped:
        inner_choices = [c for c in border_keys if c != outer_color]
        inner_color = rng.choice(inner_choices)
        inner_fill = int(mapping[inner_color])
    else:
        assert leftovers, "_choose_colors: distractor color requires a leftover color"
        inner_color = rng.choice(leftovers)
        inner_fill = None

    return border_keys, fill_values, mapping, outer_color, int(inner_color), inner_fill


def _find_outer_positions(h: int, w: int, occ: np.ndarray, oh: int, ow: int) -> list[tuple[int, int]]:
    """Candidate placements for the outer shell, leaving one bg cell to the frame."""
    positions = []
    for r0 in range(1, h - oh):
        for c0 in range(1, w - ow):
            if occ[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_ring(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    """Draw a 1-cell-thick rectangular ring."""
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    """Paint a boolean mask at (r0, c0)."""
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Flood-fill the True-valued component containing `seed`."""
    sr, sc = seed
    if not mask[sr, sc]:
        raise ValueError("_component_from_seed: seed must lie inside the target mask")
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    out[sr, sc] = True
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    inner_mapped = rng.random() < _P_INNER_MAPPED

    if n < 2:
        return None

    border_keys, fill_values, mapping, outer_color, inner_color, inner_fill = _choose_colors(rng, n, inner_mapped)
    outer_fill = int(mapping[outer_color])
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    outer_positions = _find_outer_positions(h, w, occ, _OUTER_H, _OUTER_W)
    if not outer_positions:
        return None
    outer_r0, outer_c0 = rng.choice(outer_positions)

    _paint_ring(grid, outer_r0, outer_c0, _OUTER_H, _OUTER_W, outer_color)

    inner_h, inner_w = _INNER_SHELL_TEMPLATE.shape
    inner_r0 = outer_r0 + 2
    inner_c0 = outer_c0 + rng.choice([1, 2])
    _paint_mask(grid, inner_r0, inner_c0, _INNER_SHELL_TEMPLATE, inner_color)

    bar_r0 = outer_r0 + 8
    bar_c0 = inner_c0 + 1
    grid[bar_r0, bar_c0 : bar_c0 + _INNER_BAR_LEN] = inner_color

    bg_mask = grid == bg
    annulus = _component_from_seed(bg_mask, (outer_r0 + 1, outer_c0 + 1))
    inner_hole = _component_from_seed(bg_mask, (inner_r0 + 2, inner_c0 + 3))

    output = grid.copy()
    output[annulus] = outer_fill
    if inner_fill is not None:
        output[inner_hole] = int(inner_fill)

    inp = grid.copy()

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
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
