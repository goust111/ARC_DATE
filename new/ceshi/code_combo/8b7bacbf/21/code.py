# Combo variant of 8b7bacbf (index 21)
# combo_spec: D1/0+D2/1+D12/0
# Name: 4连通追踪+分支路径+单种子多线路

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace:
#    - Canonical input now has exactly one unique seed on the border and exactly two
#      different line colors leaving that seed on two sides.
#    - Each line color grows as a 4-connected orthogonal tree, not a single chain:
#      a short entry bend leaves the seed, a vertical trunk continues inward, and
#      one side branch splits off to a second leaf.
#    - Every leaf hosts a rectangular frame ring whose cavity should fill with the
#      hosting line color. A single seed therefore activates multiple cavities of
#      multiple fill colors.
#    - A deliberate diagonal-only probe ring corner-touches one tree. Legacy
#      8-neighbor attachment would fill it, but the fused rule must leave it empty
#      because only 4-neighbor line-to-ring contact counts.
#    - Small trace:
#      seed S touches blue line B and red line R. B branches into two leaves with
#      two attached rings; R does the same. New solver separates B and R as two
#      same-color 4-connected trees, finds cavities whose ring boundary 4-touches
#      exactly one tree, and fills those cavities with B or R. The diagonal probe
#      stays background.
#
# b) Complexity calibration:
#    - The number of active lines is fixed at 2 so the 1:many mapping stays legible.
#    - Each line is guaranteed to branch once, giving two leaves and two filled
#      cavities per line. This makes the tree structure visually repeated rather
#      than a one-off accident.
#    - Frame colors are reused from a small pool, so ring color cannot explain the
#      output; only the hosting line color can.
#
# c) Implicit-assumption audit against the original main():
#    - "One seed -> one line color" is invalid. The solver must enumerate all
#      distinct non-background 4-neighbors of the single seed.
#    - "Trace whole non-background component from seed" is too coarse because it
#      merges both lines through the seed. The new solver traces same-color line
#      components separately, one per line color.
#    - "8-neighbor line attachment is enough" is invalid under D1/0. Boundary
#      contact to a line is checked in 4-neighborhood only.
#    - "Fill with seed color" is invalid under D12/0. The fill color now comes from
#      the unique touching line component.
#    - Rounded rings without corners would break 4-connectivity, so the generator
#      uses full rectangular rings.
#
# d) Placement legality:
#    - The canonical builder reserves a left and right half around the seed, so the
#      two line trees do not overlap or even 8-touch each other.
#    - Ring placement validates: empty ring+cavity, cavity not 8-adjacent to the
#      hosting line, no foreign 8-adjacency around the ring, and either required
#      4-touch (attached ring) or diagonal-only touch (probe ring).
#    - A random dihedral transform is applied only after the whole canonical scene
#      is valid, so border placement and spacing guarantees are preserved.
#
# e) Visual observability:
#    - Humans can see three simultaneous changes in 2-3 examples:
#      one seed controls two differently colored outputs, each line visibly forks,
#      and one corner-touch ring remains unfilled despite looking nearby.
#    - Because every example has four filled cavities and one empty diagonal probe,
#      the key regularities repeat enough to infer the rule.
#
# f) Visual cleanliness:
#    - Object count is moderate: one seed, two trees, four attached rings, one probe,
#      and at most one solo distractor ring.
#    - The two trees occupy opposite sides of the seed before a global rotation/flip,
#      which keeps branches and rings easy to parse.
#
# g) Inference of color change:
#    - Only two line colors participate in filling, so the color mapping is small
#      and structurally anchored. Reused frame colors prevent spurious frame-color
#      lookup strategies.
#
# h) Anti-degeneration checks:
#    - Each accepted sample must produce changed output under `main()`.
#    - Each line tree must have a degree-3 branch node.
#    - The new output must differ from a seed-fill solver, proving D12/0 is active.
#    - The new output must differ from an 8-neighbor attachment solver, proving the
#      diagonal probe makes D1/0 active on the generated input.
#
# i) New main() assumptions and generator guarantees:
#    - Assumption: there is exactly one border seed. Guaranteed by generator.
#    - Assumption: the seed has at least two differently colored 4-neighbors that
#      each start a 4-connected same-color tree. Guaranteed by canonical layout.
#    - Assumption: every fillable cavity boundary 4-touches exactly one active line
#      component. Guaranteed by per-leaf attachment and by separating the two trees.
#    - Assumption: fillable rings are single-color rectangular loops. Guaranteed by
#      explicit rectangle construction.


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    conn = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=conn)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _same_color_component_from(
    g: np.ndarray,
    start: tuple[int, int],
    color: int,
    conn8: bool,
) -> set[tuple[int, int]]:
    h, w = g.shape
    neigh_fn = neighbors8 if conn8 else neighbors4
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neigh_fn(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _collect_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    cavities = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        cavities.append((region, boundary, {int(g[p]) for p in boundary}))
    return cavities


def _seed_line_components(
    g: np.ndarray,
    seed_rc: tuple[int, int],
    bg: int,
    conn8: bool,
) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = g.shape
    starts_by_color: dict[int, tuple[int, int]] = {}
    for nr, nc in neighbors4(*seed_rc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                starts_by_color.setdefault(v, (nr, nc))
    items = []
    for line_col, start in starts_by_color.items():
        items.append((line_col, _same_color_component_from(g, start, line_col, conn8=conn8)))
    return items


def _boundary_touches_component(
    boundary: set[tuple[int, int]],
    comp: set[tuple[int, int]],
    conn8: bool,
    h: int,
    w: int,
) -> bool:
    neigh_fn = neighbors8 if conn8 else neighbors4
    for r, c in boundary:
        for nr, nc in neigh_fn(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in comp:
                return True
    return False


def _solve_combo_mode(
    grid: np.ndarray,
    *,
    path_conn8: bool,
    touch_conn8: bool,
    fill_with_seed: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    seeds = _seed_pixels(g, bg)
    if len(seeds) != 1:
        return out
    seed_rc = seeds[0]
    seed_col = int(g[seed_rc])

    line_infos = _seed_line_components(g, seed_rc, bg, conn8=path_conn8)
    if len(line_infos) < 2:
        return out

    for cav, boundary, boundary_cols in _collect_cavities(g, bg):
        if len(boundary_cols) != 1:
            continue
        touching = []
        for line_col, comp in line_infos:
            if _boundary_touches_component(boundary, comp, conn8=touch_conn8, h=h, w=w):
                touching.append(line_col)
        if len(touching) != 1:
            continue
        fill_col = seed_col if fill_with_seed else touching[0]
        for r, c in cav:
            if int(out[r, c]) == bg:
                out[r, c] = fill_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cavities attached to a single border seed's orthogonal branching lines.

    Input structure:
    - Background is the most common color.
    - Exactly one unique-color seed sits on the border.
    - Two differently colored line networks leave that seed, each as a 4-connected
      orthogonal tree with attached rectangular frame rings.

    Rule:
    - Split the seed's outgoing structures by line color.
    - For each line color, trace only its 4-connected same-color tree.
    - Any enclosed background cavity whose single-color ring boundary 4-touches
      exactly one traced tree is flood-filled with that tree's line color.

    Key insight:
    - The seed is only the shared origin. It does not supply the fill color.
      The decisive pairing is: cavity ring boundary -> uniquely touching 4-connected
      line tree -> that line's color.
    """
    return _solve_combo_mode(grid, path_conn8=False, touch_conn8=False, fill_with_seed=False)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
BG_COLORS = [0, 5]

ACTIVE_LINE_COUNT = 2
FRAME_COLORS_MAX = 2

TRUNK_COL_MARGIN = 3
TRUNK_TIP_BOTTOM_MARGIN = 5
BRANCH_TIP_BOTTOM_MARGIN = 5
RING_CAVITY_WEIGHTS = [2, 2]
SOLO_CAVITY_WEIGHTS = [2]
RING_PLACE_TRIES = 40
PROBE_PLACE_TRIES = 80
SOLO_FRAME_TRIES = 50
GEN_RETRIES = 500


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            if dr == 0 or dc == 0 or dr == fh - 1 or dc == fw - 1:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _is_orth_neighbor(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _is_diag_neighbor(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment must be axis aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i in range(len(points) - 1):
        cells.update(_cells_between(points[i], points[i + 1]))
    return cells


def _tree_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {pt: sum((nr, nc) in cells for nr, nc in neighbors4(*pt)) for pt in cells}


def _validate_line_cells(
    canvas: np.ndarray,
    cells: set[tuple[int, int]],
    *,
    color: int,
    seed_rc: tuple[int, int],
) -> bool:
    h, w = canvas.shape
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if int(canvas[r, c]) not in (-1, color):
            return False
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in cells:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if (nr, nc) == seed_rc:
                continue
            return False
    deg = _tree_degrees(cells)
    if sum(d >= 3 for d in deg.values()) == 0:
        return False
    return True


def _paint_cells(canvas: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        canvas[r, c] = color


def _validate_attached_ring(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    ring_set = set(ring)
    has_orth_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    has_orth_touch = True
                continue
            return False
    return has_orth_touch


def _validate_probe_ring(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    ring_set = set(ring)
    has_diag_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    return False
                if _is_diag_neighbor((rr, cc), (nr, nc)):
                    has_diag_touch = True
                continue
            return False
    return has_diag_touch


def _candidate_bboxes_for_leaf(
    leaf: tuple[int, int],
    side: str,
    ch: int,
    cw: int,
) -> list[tuple[int, int]]:
    r, c = leaf
    fh, fw = ch + 2, cw + 2
    if side == "down":
        r0 = r + 1
        return [(r0, c0) for c0 in range(c - fw + 1, c + 1)]
    if side == "left":
        c0 = c - fw
        return [(r0, c0) for r0 in range(r - fh + 1, r + 1)]
    if side == "right":
        c0 = c + 1
        return [(r0, c0) for r0 in range(r - fh + 1, r + 1)]
    raise ValueError(side)


def _try_attach_ring_at_leaf(
    canvas: np.ndarray,
    leaf: tuple[int, int],
    side: str,
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> bool:
    for _ in range(RING_PLACE_TRIES):
        ch = rng.choice(RING_CAVITY_WEIGHTS)
        cw = rng.choice(RING_CAVITY_WEIGHTS)
        bboxes = _candidate_bboxes_for_leaf(leaf, side, ch, cw)
        rng.shuffle(bboxes)
        for r0, c0 in bboxes:
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_attached_ring(canvas, ring, cavity, line_col):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return True
    return False


def _diag_probe_bboxes(
    anchor: tuple[int, int],
    ch: int,
    cw: int,
) -> list[tuple[int, int]]:
    r, c = anchor
    fh, fw = ch + 2, cw + 2
    return [
        (r + 1, c + 1),
        (r + 1, c - fw),
        (r - fh, c + 1),
        (r - fh, c - fw),
    ]


def _try_place_probe_ring(
    canvas: np.ndarray,
    anchors: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> bool:
    anchor_order = anchors[:]
    rng.shuffle(anchor_order)
    for anchor in anchor_order[:PROBE_PLACE_TRIES]:
        ch = rng.choice(RING_CAVITY_WEIGHTS)
        cw = rng.choice(RING_CAVITY_WEIGHTS)
        bboxes = _diag_probe_bboxes(anchor, ch, cw)
        rng.shuffle(bboxes)
        for r0, c0 in bboxes:
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_probe_ring(canvas, ring, cavity, line_col):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return True
    return False


def _try_place_solo_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(SOLO_FRAME_TRIES):
        ch = rng.choice(SOLO_CAVITY_WEIGHTS)
        cw = rng.choice(SOLO_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 2 <= 1 or w - fw - 2 <= 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _build_side_tree(
    h: int,
    w: int,
    seed_c: int,
    side: str,
    rng: random.Random,
) -> dict | None:
    if side not in {"left", "right"}:
        raise ValueError(side)

    start_c = seed_c - 1 if side == "left" else seed_c + 1
    start = (0, start_c)
    entry = (1, start_c)

    if side == "left":
        trunk_lo = 6
        trunk_hi = seed_c - TRUNK_COL_MARGIN
        if trunk_lo > trunk_hi:
            return None
        trunk_c = rng.randint(trunk_lo, trunk_hi)
        branch_c_lo = 4
        branch_c_hi = trunk_c - 2
        if branch_c_lo > branch_c_hi:
            return None
        branch_c = rng.randint(branch_c_lo, branch_c_hi)
    else:
        trunk_lo = seed_c + TRUNK_COL_MARGIN
        trunk_hi = w - 7
        if trunk_lo > trunk_hi:
            return None
        trunk_c = rng.randint(trunk_lo, trunk_hi)
        branch_c_lo = trunk_c + 2
        branch_c_hi = w - 5
        if branch_c_lo > branch_c_hi:
            return None
        branch_c = rng.randint(branch_c_lo, branch_c_hi)

    trunk_tip_r_lo = max(8, h // 2)
    trunk_tip_r_hi = h - TRUNK_TIP_BOTTOM_MARGIN
    if trunk_tip_r_lo > trunk_tip_r_hi:
        return None
    trunk_tip_r = rng.randint(trunk_tip_r_lo, trunk_tip_r_hi)

    branch_r_lo = 4
    branch_r_hi = trunk_tip_r - 3
    if branch_r_lo > branch_r_hi:
        return None
    branch_r = rng.randint(branch_r_lo, branch_r_hi)

    branch_tip_r_lo = branch_r + 3
    branch_tip_r_hi = h - BRANCH_TIP_BOTTOM_MARGIN
    if branch_tip_r_lo > branch_tip_r_hi:
        return None
    branch_tip_r = rng.randint(branch_tip_r_lo, branch_tip_r_hi)

    trunk_points = [start, entry, (1, trunk_c), (trunk_tip_r, trunk_c)]
    branch_points = [(branch_r, trunk_c), (branch_r, branch_c), (branch_tip_r, branch_c)]
    cells = _polyline_cells(trunk_points)
    cells.update(_polyline_cells(branch_points))

    leaves = [
        {"pt": (trunk_tip_r, trunk_c), "side": "down"},
        {"pt": (branch_tip_r, branch_c), "side": "left" if side == "left" else "right"},
    ]
    anchors = sorted(cells, key=lambda pt: (pt[0], pt[1]))

    return {"cells": cells, "leaves": leaves, "anchors": anchors, "start": start}


def _canonical_scene(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    if len(palette) < 5:
        return None

    seed_col = palette.pop()
    line_colors = [palette.pop() for _ in range(ACTIVE_LINE_COUNT)]
    frame_color_count = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors = [palette.pop() for _ in range(frame_color_count)]

    seed_c_lo = max(7, w // 2 - 2)
    seed_c_hi = min(w - 8, w // 2 + 2)
    if seed_c_lo > seed_c_hi:
        return None
    seed_c = rng.randint(seed_c_lo, seed_c_hi)
    seed_rc = (0, seed_c)

    canvas = -np.ones((h, w), dtype=int)
    canvas[seed_rc] = seed_col

    trees = []
    for side, line_col in zip(("left", "right"), line_colors):
        tree = _build_side_tree(h, w, seed_c, side, rng)
        if tree is None:
            return None
        if not _validate_line_cells(canvas, tree["cells"], color=line_col, seed_rc=seed_rc):
            return None
        _paint_cells(canvas, tree["cells"], line_col)
        tree["line_col"] = line_col
        trees.append(tree)

    for tree in trees:
        for leaf in tree["leaves"]:
            frame_col = rng.choice(frame_colors)
            if not _try_attach_ring_at_leaf(canvas, leaf["pt"], leaf["side"], tree["line_col"], frame_col, rng):
                return None

    probe_owner = rng.choice(trees)
    if not _try_place_probe_ring(canvas, probe_owner["anchors"], probe_owner["line_col"], rng.choice(frame_colors), rng):
        return None

    if rng.random() < 0.5:
        _try_place_solo_frame(canvas, rng.choice(frame_colors), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)
    return grid


def _apply_random_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    k = rng.randint(0, 3)
    out = np.rot90(out, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _branching_line_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    seeds = _seed_pixels(g, bg)
    if len(seeds) != 1:
        return 0
    count = 0
    for _line_col, comp in _seed_line_components(g, seeds[0], bg, conn8=False):
        deg = _tree_degrees(comp)
        if any(v >= 3 for v in deg.values()):
            count += 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base = _canonical_scene(rng)
    if base is None:
        return None
    grid = _apply_random_transform(base, rng)
    bg = get_bg(grid)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    seeds = _seed_pixels(grid, bg)
    if len(seeds) != 1:
        return None
    seed_col = int(grid[seeds[0]])
    if int((grid == seed_col).sum()) != 1:
        return None

    if len(_seed_line_components(grid, seeds[0], bg, conn8=False)) != ACTIVE_LINE_COUNT:
        return None
    if _branching_line_count(grid) != ACTIVE_LINE_COUNT:
        return None

    new_out = main(grid)
    seed_fill_out = _solve_combo_mode(grid, path_conn8=False, touch_conn8=False, fill_with_seed=True)
    legacy_touch_out = _solve_combo_mode(grid, path_conn8=True, touch_conn8=True, fill_with_seed=False)

    if np.array_equal(new_out, grid):
        return None
    if np.array_equal(new_out, seed_fill_out):
        return None
    if np.array_equal(new_out, legacy_touch_out):
        return None

    fill_colors = {int(v) for v in np.unique(new_out[(grid == bg) & (new_out != bg)])}
    if len(fill_colors) != ACTIVE_LINE_COUNT:
        return None

    for line_col, _comp in _seed_line_components(grid, seeds[0], bg, conn8=False):
        if int((new_out == line_col).sum()) <= int((grid == line_col).sum()):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
