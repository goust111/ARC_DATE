# Variant of 8f3a5a89
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出与输入尺寸完全相同（same size），在原网格上修改像素... → 裁剪到可达区域外接矩形
#
# Design analysis
# =================
# 1) Concrete behavior trace
#    - The original task solves on the full canvas: find the bottom-left marker's
#      reachable region, erase hidden blue decorations, and paint an orange inner
#      border around the reachable side of the wall/grid boundary.
#    - D9 changes only the output framing: after producing the normal solved grid,
#      keep only the smallest axis-aligned rectangle that contains every cell in
#      the marker-reachable region. Entirely unreachable rows/cols disappear.
#    - Small trace:
#      original: marker reaches rows 4..11 and cols 0..8 inside a 12x14 grid;
#      the solved output stays 12x14.
#      variant: the same orange/blue pattern is computed first, then rows 0..3
#      and cols 9..13 are cropped away, so the output becomes 8x9.
#    - This crop must be driven by reachability, not by "all non-bg pixels":
#      hidden decorations may be erased to bg before cropping, while reachable bg
#      interior must still count toward the bounding box.
#
# 2) Complexity calibration
#    - If generation allowed the reachable set to touch every row/col, the new
#      task would visually collapse back to the original same-size behavior.
#    - To make D9 observable, the generator enforces a left-bottom reachable room
#      with both a positive top trim and a positive right trim on every sample.
#      The orange border rule is unchanged, so the new information the solver
#      must infer is exactly "where does the reachable region end?".
#    - The generator still keeps small visible/hidden blue decorations and may
#      add extra structural wall components, so the task does not degenerate into
#      a plain empty rectangle crop.
#
# 3) Implicit-assumption review for main()
#    - Marker detection still assumes a unique non-bg corner pixel. The generator
#      keeps the magenta marker as a single bottom-left pixel.
#    - Wall detection still assumes the most frequent non-bg, non-marker color is
#      the structural/decor wall color. The generator uses only one wall color.
#    - Structural-wall recovery still assumes true barrier components touch the
#      border and are adjacent to the initial reach. The generator preserves that.
#    - The new crop step assumes the reachable set is non-empty and includes the
#      marker. That remains true because the flood always starts from the marker.
#    - Cropping by reachable bbox intentionally removes boundary wall rows/cols if
#      they are fully unreachable; this is required by D9 and is not a bug.
#
# 4) Placement legality
#    - The generator builds an outer L-shaped structural barrier that touches the
#      left and bottom borders and encloses a bottom-left reachable room.
#    - Optional extra structural components touch the top or right border and may
#      intrude into the kept rectangle, but they never overlap the marker.
#    - Visible decorations are placed only on reachable non-fill bg cells and are
#      kept in output; hidden decorations are placed off the border in the
#      unreachable side so the solver erases them before cropping.
#
# 5) Visual observability / cleanliness
#    - Because the marker is fixed at bottom-left, any tight crop can only trim
#      from the top and/or the right. The generator guarantees both trims are
#      present, so the variant is immediately visible in 2-3 examples.
#    - The crop boundary aligns with already-orange reachable boundary cells, so
#      the cropped outputs look intentional rather than jagged or ambiguous.
#    - Colors are unchanged from the base task: azure bg, blue walls, magenta
#      marker, orange border. No hidden color-id logic is introduced.
#
# 6) Source-task note
#    - The prompt names e87109e9 as the source task for the crop idea. Its solver
#      in this workspace family applies cropping as a final slicing step after the
#      main transform, not as a different reasoning core. This variant follows
#      the same complexity level: solve first on the full grid, crop last.
#
# 7) New solver assumptions audited against the generator
#    - main() now assumes there exists at least one fully unreachable top row and
#      one fully unreachable right column if D9 is to be observable. The solver
#      itself does not require this, but the generator enforces it to avoid
#      same-size degeneration.
#    - main() crops by reachable bbox, so any structural feature that should
#      remain visible after cropping must share kept rows/cols with reachable
#      cells. The generator optionally adds top/right components that extend into
#      the kept rectangle for that purpose.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _solve_uncropped(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out, reach


def _crop_to_reach(out: np.ndarray, reach: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(list(reach), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the reachable-side inner border, then crop to the reachable area's bbox.

    Input structure:
    - The background is the most common color; blue pixels are a mix of true
      border-touching walls and smaller decorative blue patches.
    - A single magenta pixel sits on one corner and marks the flood-fill start.
    - Grid edges together with the structural blue walls delimit the area that is
      reachable from that corner marker.

    Rule:
    - Recover the marker-reachable region by flood-filling around only the
      structural walls, while treating non-structural blue decorations as passable.
    - Erase blue decorations that lie only on the unreachable side.
    - Paint orange on reachable background cells that sit on the reachable
      region's inner boundary.
    - Finally crop the solved grid to the smallest rectangle containing every
      reachable cell.

    Key insight: The crop is determined by the full reachable region itself, so
    the solver must identify that region before it can know the output size.
    """
    out, reach = _solve_uncropped(np.asarray(grid, dtype=int))
    return _crop_to_reach(out, reach)


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

TOP_TRIM_MIN = 2
RIGHT_TRIM_MIN = 2
REACH_H_MIN = 4
REACH_W_MIN = 4

NUM_EXTRA_STRUCTS_RANGE = (0, 2)
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

REACH_RATIO_RANGE = (0.22, 0.72)
GEN_ATTEMPTS = 600


def _flood_mask(start, blocked, h, w):
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _dilate4(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask):
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _paint_output(structural, reach, marker_rc):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _reachable_bbox(reach: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(reach)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_array(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _base_barrier(h: int, w: int, rng: random.Random):
    max_top_trim = h - REACH_H_MIN
    max_right_trim = w - REACH_W_MIN
    if max_top_trim < TOP_TRIM_MIN or max_right_trim < RIGHT_TRIM_MIN:
        return None

    top_trim = rng.randint(TOP_TRIM_MIN, max_top_trim)
    right_trim = rng.randint(RIGHT_TRIM_MIN, max_right_trim)

    reach_r0 = top_trim
    reach_c1 = w - right_trim - 1
    barrier_r = reach_r0 - 1
    barrier_c = reach_c1 + 1

    structural = np.zeros((h, w), dtype=bool)
    structural[barrier_r, : barrier_c + 1] = True
    structural[barrier_r:, barrier_c] = True
    return structural, reach_r0, reach_c1, barrier_r, barrier_c


def _cells_top_rect(h, w, rng, reach_r0, reach_c1):
    if reach_c1 < 3:
        return None
    max_h = min(h, reach_r0 + rng.randint(1, 3) + 1)
    if max_h <= 1:
        return None
    rh = rng.randint(max(2, reach_r0 + 1), max_h) if max_h >= max(2, reach_r0 + 1) else None
    if rh is None:
        return None
    max_w = min(5, reach_c1)
    if max_w < 2:
        return None
    rw = rng.randint(2, max_w)
    c0 = rng.randint(1, reach_c1 - rw + 1)
    return [(r, c) for r in range(rh) for c in range(c0, c0 + rw)]


def _cells_top_triangle(h, w, rng, reach_r0, reach_c1):
    max_h = min(h, reach_r0 + rng.randint(1, 3) + 1)
    if max_h < 3 or reach_c1 < 4:
        return None
    th = rng.randint(max(3, reach_r0 + 1), max_h) if max_h >= max(3, reach_r0 + 1) else None
    if th is None:
        return None
    base_w = rng.randint(3, min(6, reach_c1))
    c0 = rng.randint(1, reach_c1 - base_w + 1)
    cells = []
    for r in range(th):
        width = max(1, base_w - r)
        for c in range(c0, c0 + width):
            cells.append((r, c))
    return cells


def _cells_right_rect(h, w, rng, reach_r0, reach_c1):
    if w - reach_c1 < 2 or h - reach_r0 < 3:
        return None
    rh = rng.randint(2, min(5, h - reach_r0 - 1))
    r0 = rng.randint(reach_r0 + 1, h - rh)
    left = rng.randint(max(1, reach_c1 - 2), reach_c1)
    return [(r, c) for r in range(r0, r0 + rh) for c in range(left, w)]


EXTRA_FACTORIES = [_cells_top_rect, _cells_top_triangle, _cells_right_rect]


def _add_extra_structs(structural, reach_r0, reach_c1, marker_rc, rng):
    h, w = structural.shape
    count = rng.randint(*NUM_EXTRA_STRUCTS_RANGE)
    for _ in range(count):
        rng.shuffle(EXTRA_FACTORIES)
        for factory in EXTRA_FACTORIES:
            cells = factory(h, w, rng, reach_r0, reach_c1)
            if not cells:
                continue
            if marker_rc in cells:
                continue
            if any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            break


def _place_decor(allowed_mask, count, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    base = _base_barrier(h, w, rng)
    if base is None:
        return None
    structural, reach_r0, reach_c1, barrier_r, barrier_c = base
    _add_extra_structs(structural, reach_r0, reach_c1, marker_rc, rng)

    if structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    if not reach[marker_rc]:
        return None

    r0, c0, r1, c1 = _reachable_bbox(reach)
    if r0 < TOP_TRIM_MIN or (w - 1 - c1) < RIGHT_TRIM_MIN:
        return None
    if c0 != 0 or r1 != h - 1:
        return None

    if (r1 - r0 + 1) < REACH_H_MIN or (c1 - c0 + 1) < REACH_W_MIN:
        return None

    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    out = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_full = out.copy()
    out_full[visible_placed] = WALL

    inp = out_full.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    expected = _crop_array(out_full, _reachable_bbox(reach))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D9 variant")
