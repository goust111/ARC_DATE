# Combo variant of 4a21e3da (index 11)
# combo_spec: D5/4+D13/0
# Name: 消失区域移至对侧角落+连通分量归属

from __future__ import annotations

"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and trace
Small canonical trace:
  - Use the original visual vocabulary: blue background (`bg=1`), red border
    controls (`ctrl=2`), orange sprite (`fg=7`).
  - Two controls are always present: one on the top/bottom edge gives a
    vertical split column; one on the left/right edge gives a horizontal split
    row.
  - Original 4a21e3da with `top + left` controls:
      * the sprite is cut into TL / TR / BL / BR rectangular regions;
      * TL, TR, BL slide to their same-side corners;
      * BR disappears.
  - D5/4 alone would keep BR as well and move it to the opposite corner, which
    in the two-control case is still BR. So D5/4 by itself is still driven by
    geometry only.
  - This fused variant changes the ownership unit first:
      * remove the fg pixels on the split row/column;
      * compute 4-connected components on the remaining fg;
      * if a component has at least one cell orthogonally adjacent to a split
        pixel, it is "same-side owned" and goes to its geometric corner;
      * otherwise it is not erased: it goes to the corner formed by the two
        missing control edges.
  - Canonical example used by the generator:
      * TL kept component touches the split row;
      * TR kept component touches the split row/column;
      * BL kept component touches the split row;
      * inside TR there is one extra 3-cell elbow that only touches the TR kept
        component diagonally, not 4-neighborly, so after removing split pixels
        it becomes a separate component not owned by the split.
      * Result with `top + left`: TL->TL, TR(main)->TR, BL->BL, TR(rogue)->BR.
    The key visible difference is that two fragments on the same geometric side
    can end up at different corners.

b) Complexity calibration
  - Two perpendicular controls are forced on every sample. Otherwise D5/4 is
    invisible, because with one control there is no unique "opposite corner"
    interpretation.
  - The generator forces exactly one rogue 4-component in a control-touching
    region, so the combo differs from both the original seed and the D5/4-only
    variant on every accepted sample.
  - Shapes stay compact and use only a few repeated motifs; the only extra
    freedom is horizontal/vertical flipping and minor grid-size variation.

c) Audit of original solver assumptions that break
  - Original assumption: one geometric region corresponds to one moved piece.
    Broken. We must first remove split-line fg and then compute 4-components.
  - Original assumption: keep/drop is decided by `touches_edges(hs, vs) & E`.
    Broken. A component can be in a control-touching quadrant and still be
    rerouted to the opposite corner if it is not split-adjacent.
  - Original assumption: the central fg object is 4-connected. Broken on
    purpose. If the whole input remained 4-connected, every component obtained
    after removing the split pixels would necessarily touch the split, so D13/0
    would degenerate. The generator therefore keeps the sprite 8-connected but
    not 4-connected by using one diagonal-only attachment.
  - Original assumption: moving each region independently is always safe.
    We now bucket all components by destination corner and translate each bucket
    by its union bbox, so multiple fragments assigned to one corner preserve
    their relative spacing without accidental overlap.

d) Placement legality
  - All fg stays at least four cells away from the border; only the control
    pixels are on the border.
  - The canonical template is placed so its bbox is 8x9, with the split row and
    split column passing roughly through the middle, matching the seed spec.
  - The rogue elbow is strictly off the split row and split column and has no
    4-neighbor contact with any split pixel, only a single diagonal contact to
    its nearby kept component.
  - Destination buckets use corner-flush translation, and the generator keeps
    exactly one opposite-corner bucket, so the moved output remains clean.

e) Visual observability
  - Every sample visibly has four semantic fragments after cutting:
    three clearly split-adjacent ones and one weakly attached rogue fragment.
  - The rogue fragment starts in a control-touching quadrant yet ends in the
    opposite corner. This makes the fused rule visibly different from both
    "pure geometry" and "erase the opposite quadrant".
  - No hidden ordering is used; the rule is inferable from shape placement and
    4-neighbor topology only.

f) Visual cleanliness
  - Still only three colors.
  - The sprite remains one compact center object rather than scattered noise.
  - The only fragile connection is a single diagonal touch, which is visible but
    does not make the grid visually messy.

g) New solver assumptions and generator guarantees
  - Assumption: every off-line 4-component lies wholly in one side of each split
    axis. The template guarantees this; no component straddles the split row or
    column.
  - Assumption: at least one component is not split-adjacent, otherwise D13/0
    would collapse back to D5/4. `_behavior_metrics()` checks this directly.
  - Assumption: the combo is not geometrically degenerate. The generator rejects
    any sample where the combo output equals the D5/4-only geometric output.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg, is_contiguous


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

CANONICAL_V_LINE = frozenset({(-3, 0), (-1, 0), (1, 0), (2, 0)})
CANONICAL_H_LINE = frozenset({(0, -4), (0, -1), (0, 1), (0, 3)})
CANONICAL_TL = frozenset(
    {
        (-3, -4),
        (-3, -3),
        (-2, -4),
        (-2, -3),
        (-2, -2),
        (-1, -3),
        (-1, -2),
        (-1, -1),
    }
)
CANONICAL_TR = frozenset(
    {
        (-3, 1),
        (-3, 2),
        (-2, 1),
        (-2, 2),
        (-1, 1),
        (-1, 2),
    }
)
CANONICAL_BL = frozenset(
    {
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (2, -4),
        (2, -3),
        (2, -2),
        (3, -2),
    }
)
CANONICAL_ROGUE = frozenset({(-4, 3), (-4, 4), (-3, 4)})


def _touches_edges(hs: str, vs: str) -> set[str]:
    te: set[str] = set()
    if hs in ("T", "A"):
        te.add("top")
    if hs in ("B", "A"):
        te.add("bottom")
    if vs in ("L", "A"):
        te.add("left")
    if vs in ("R", "A"):
        te.add("right")
    return te


def _pick_same_dest(hs: str, vs: str, r_split: int | None, c_split: int | None, edges: set[str]) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in edges) and ("bottom" not in edges):
        dh = "T"
    elif ("bottom" in edges) and ("top" not in edges):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in edges) and ("right" not in edges):
        dv = "L"
    elif ("right" in edges) and ("left" not in edges):
        dv = "R"
    else:
        dv = "L"
    return str(dh), str(dv)


def _pick_opposite_dest(hs: str, vs: str, r_split: int | None, c_split: int | None, edges: set[str]) -> tuple[str, str]:
    same_h, same_v = _pick_same_dest(hs, vs, r_split, c_split, edges)
    dh = "B" if same_h == "T" else "T"
    dv = "R" if same_v == "L" else "L"
    return dh, dv


def _infer_split_info(ctrl_pts: list[tuple[int, int]], H: int, W: int) -> tuple[set[str], int | None, int | None]:
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == H - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == W - 1:
            edges.add("right")
            r_split = r
    return edges, r_split, c_split


def _draw_rays(out: np.ndarray, g: np.ndarray, ctrl_pts: list[tuple[int, int]], fg: int, ctrl: int, bg: int) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == int(bg):
                        out[rr, c] = int(ctrl)
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == int(bg):
                        out[rr, c] = int(ctrl)
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == int(bg):
                        out[r, cc] = int(ctrl)
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == int(bg):
                        out[r, cc] = int(ctrl)

    for r, c in ctrl_pts:
        out[r, c] = int(ctrl)


def _split_fg(g: np.ndarray, fg: int, r_split: int | None, c_split: int | None) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (r_split is not None and r == r_split) or (c_split is not None and c == c_split):
            pts.add((r, c))
    return pts


def _extract_components(g: np.ndarray, fg: int, split_fg: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    mask = (g == int(fg))
    for r, c in split_fg:
        mask[r, c] = False
    comps: list[list[tuple[int, int]]] = []
    for coords in components_mask(mask, connectivity=4):
        comp = [(int(r), int(c)) for r, c in coords]
        comps.append(comp)
    return comps


def _component_region(comp: list[tuple[int, int]], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hvals = {"T" if r < r_split else "B" for r, _ in comp}
        assert len(hvals) == 1, "component straddles split row"
        hs = next(iter(hvals))
    if c_split is not None:
        vvals = {"L" if c < c_split else "R" for _, c in comp}
        assert len(vvals) == 1, "component straddles split column"
        vs = next(iter(vvals))
    return hs, vs


def _line_adjacent(comp: list[tuple[int, int]], split_fg: set[tuple[int, int]]) -> bool:
    for r, c in comp:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in split_fg:
                return True
    return False


def _translate_bucket(out: np.ndarray, pts: set[tuple[int, int]], dest: tuple[str, str], fg: int) -> None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if dest[0] == "T" else (out.shape[0] - 1 - rmax)
    dc = -cmin if dest[1] == "L" else (out.shape[1] - 1 - cmax)
    for r, c in pts:
        out[r + dr, c + dc] = int(fg)


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[tuple[int, int]], set[str], int | None, int | None, set[tuple[int, int]], list[list[tuple[int, int]]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl)) if int(r) in (0, H - 1) or int(c) in (0, W - 1)]
    edges, r_split, c_split = _infer_split_info(ctrl_pts, H, W)
    split_fg = _split_fg(g, fg, r_split, c_split)
    comps = _extract_components(g, fg, split_fg)
    return g, bg, fg, ctrl, ctrl_pts, edges, r_split, c_split, split_fg, comps


def _render_with_rule(grid: np.ndarray, *, combo_rule: bool) -> np.ndarray:
    g, bg, fg, ctrl, ctrl_pts, edges, r_split, c_split, split_fg, comps = _analyze_grid(grid)
    H, W = g.shape
    out = np.full((H, W), int(bg), dtype=int)
    _draw_rays(out, g, ctrl_pts, fg, ctrl, bg)
    for r, c in split_fg:
        out[r, c] = int(fg)

    buckets: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for comp in comps:
        hs, vs = _component_region(comp, r_split, c_split)
        if combo_rule:
            if _line_adjacent(comp, split_fg):
                dest = _pick_same_dest(hs, vs, r_split, c_split, edges)
            else:
                dest = _pick_opposite_dest(hs, vs, r_split, c_split, edges)
        else:
            dest = _pick_same_dest(hs, vs, r_split, c_split, edges)
        buckets[dest].update(comp)

    for dest, pts in buckets.items():
        _translate_bucket(out, pts, dest, fg)
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g, _bg, fg, _ctrl, _ctrl_pts, edges, r_split, c_split, split_fg, comps = _analyze_grid(grid)
    fg_mask = (g == int(fg)).astype(int)
    opposite = 0
    touching = 0
    touching_regions = 0
    rogue_in_touching = 0
    for comp in comps:
        hs, vs = _component_region(comp, r_split, c_split)
        in_touching_region = int(bool(_touches_edges(hs, vs) & edges))
        if _line_adjacent(comp, split_fg):
            touching += 1
            touching_regions += in_touching_region
        else:
            opposite += 1
            rogue_in_touching += in_touching_region
    combo_out = _render_with_rule(grid, combo_rule=True)
    d5_only_out = _render_with_rule(grid, combo_rule=False)
    return {
        "offline_components": int(len(comps)),
        "touching_components": int(touching),
        "opposite_components": int(opposite),
        "rogue_in_touching_region": int(rogue_in_touching),
        "diff_vs_d5_only": int(not np.array_equal(combo_out, d5_only_out)),
        "is_8_connected": int(is_contiguous(fg_mask, connectivity=8)),
        "is_4_connected": int(is_contiguous(fg_mask, connectivity=4)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Draw the original intermittent row/column rays from the border controls.
    - Keep fg pixels that lie on the active split row/column in place.
    - Remove those split-line fg pixels temporarily and compute 4-connected
      components on the remaining fg.
    - A component orthogonally adjacent to any split-line fg goes to its
      same-side geometric corner.
    - A component not adjacent to the split line is preserved but rerouted to
      the corner formed by the missing control edges.
    """
    return _render_with_rule(grid, combo_rule=True)


def _apply_flips(
    pts: set[tuple[int, int]],
    ctrl_pts: list[tuple[int, int]],
    H: int,
    W: int,
    flip_v: bool,
    flip_h: bool,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    def tf(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if flip_v:
            r = H - 1 - r
        if flip_h:
            c = W - 1 - c
        return int(r), int(c)

    new_pts = {tf(cell) for cell in pts}
    new_ctrl = [tf(cell) for cell in ctrl_pts]
    return new_pts, new_ctrl


def _bbox(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _canonical_fg(r_split: int, c_split: int) -> set[tuple[int, int]]:
    rels = set()
    rels |= set(CANONICAL_V_LINE)
    rels |= set(CANONICAL_H_LINE)
    rels |= set(CANONICAL_TL)
    rels |= set(CANONICAL_TR)
    rels |= set(CANONICAL_BL)
    rels |= set(CANONICAL_ROGUE)
    return {(r_split + dr, c_split + dc) for dr, dc in rels}


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    r_lo, r_hi = 8, H - 8
    c_lo, c_hi = 8, W - 9
    if r_lo > r_hi or c_lo > c_hi:
        return None

    r_split = random.randint(r_lo, r_hi)
    c_split = random.randint(c_lo, c_hi)
    fg_pts = _canonical_fg(r_split, c_split)
    ctrl_pts = [(0, c_split), (r_split, 0)]

    flip_v = bool(random.getrandbits(1))
    flip_h = bool(random.getrandbits(1))
    fg_pts, ctrl_pts = _apply_flips(fg_pts, ctrl_pts, H, W, flip_v, flip_h)

    if any((r, c) in fg_pts for r, c in ctrl_pts):
        return None
    if any(r < 4 or r > H - 5 or c < 4 or c > W - 5 for r, c in fg_pts):
        return None

    r0, c0, r1, c1 = _bbox(fg_pts)
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    density = len(fg_pts) / float(box_h * box_w)
    if not (5 <= box_h <= 8 and 8 <= box_w <= 10):
        return None
    if not (0.39 <= density <= 0.60):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in fg_pts:
        inp[r, c] = int(fg)
    for r, c in ctrl_pts:
        inp[r, c] = int(ctrl)

    metrics = _behavior_metrics(inp)
    if metrics["offline_components"] < 4:
        return None
    if metrics["touching_components"] < 3:
        return None
    if metrics["opposite_components"] < 1:
        return None
    if metrics["rogue_in_touching_region"] < 1:
        return None
    if metrics["diff_vs_d5_only"] != 1:
        return None
    if metrics["is_8_connected"] != 1:
        return None
    if metrics["is_4_connected"] != 0:
        return None

    _ = main(inp)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
