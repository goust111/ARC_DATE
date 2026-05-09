# Combo variant of 4a21e3da (index 28)
# combo_spec: D7/2+D17/2
# Name: 先变换再切割+图例区域解释规则

from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import colors_present, get_bg, is_contiguous
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, get_bg, is_contiguous


"""
Design analysis for combo D7/2 + D17/2
======================================

a) concretization_and_trace:
  original:
    - input: one connected fg sprite near the center and 1-2 ctrl pixels on the border.
    - output: border rays cut the visible sprite; split-line fg stays; edge-touching
      regions slide to corners; non-touching regions vanish.
  combo:
    - input gains a separate 3x3 legend area in the upper-left inner corner
      (`rows 1..3, cols 1..3`). The legend uses a fourth color and encodes which
      90-degree rotation must happen before the cut:
        right-turn glyph -> rotate the whole fg sprite 90 deg clockwise first
        left-turn glyph  -> rotate the whole fg sprite 90 deg counterclockwise first
    - after that hidden rotation, the task reverts to the original cut-and-slide rule.
  small trace:
    - suppose the visible input sprite is a tall 9x6 shape and the legend shows the
      right-turn glyph. The solver must first rotate the sprite into a wide 6x9 shape
      inside its tight bbox, then use the border ctrl-defined split column/row on that
      rotated intermediate state.
    - if the same visible input instead carries the left-turn glyph, the intermediate
      sprite is the opposite 6x9 orientation, so the cut line hits different fg cells
      and the corner fragments visibly differ.
    - the main difference from the source task is therefore twofold: the visible input
      sprite is not the object that gets cut, and the required pre-cut transform is
      decoded from a local legend rather than being a fixed global rule.

b) complexity_calibration:
  - The legend vocabulary is intentionally small: only two 3-cell arrow-like glyphs,
    both in the same 3x3 area and both encoding one of two 90-degree rotations.
  - Both choices keep the transformed sprite in the same size family as the D7 single
    variant, so the cut geometry remains learnable and the samples do not explode into
    many unrelated transform types.
  - The generator balances clockwise and counterclockwise legends and rejects cases
    where the two transforms would accidentally lead to the same output.

c) implicit_assumption_review_of_original_main:
  - Original `border_cols -> ctrl` logic assumed every non-bg border cell belonged to
    a single control color. This remains valid because the legend is moved off the border.
  - Original `fg = next(nonbg != ctrl)` assumed there was only one interior non-bg role.
    That breaks: the input now has both sprite fg and legend color. The new solver must
    explicitly read the legend area first and exclude its color from fg detection.
  - Every original loop using `g == fg` assumed the visible input fg was the to-be-cut
    sprite. That is false after D7/2; those loops must operate on `work_pts`, the sprite
    after legend-decoded rotation.
  - Original split-line preservation kept visible-input fg on the split row/column.
    The combo must instead preserve split-line fg from the transformed intermediate
    sprite, not from the raw input.
  - Original generator assumed the output depended only on geometry and fixed ctrl
    semantics. That is no longer enough because two identical visible ctrl layouts can
    now produce different outputs depending on the legend glyph.

d) placement_legality:
  - The legend occupies a reserved area (`1:4, 1:4`) disjoint from the sprite, which
    still keeps a >=4-cell margin from the grid border and in practice a >=1-cell gap
    from the legend block.
  - The generator first constructs the transformed wide sprite, verifies all cut regions
    and corner placements there, and only then inverse-rotates the whole sprite into the
    visible input orientation. This guarantees that `main()` sees a valid inverse image of
    a cuttable intermediate state.
  - Vanishing regions, split-line fg, and control pixels keep the original legality checks:
    no cross-split leakage, no out-of-bounds corner placement, and a connected fg sprite.

e) visual_observability:
  - The legend is a compact but explicit separate object, so a human immediately sees an
    extra symbolic cue that did not exist in the source task.
  - The input sprite is usually tall while the hidden cut-state is wide, making the
    pre-transform effect visible in the output fragments.
  - The two legend glyphs are mirror images of each other and are paired with opposing
    rotations, so repeated examples support a structural "legend orientation -> sprite
    rotation direction" inference.

f) visual_cleanliness:
  - Only four colors are used: bg, ctrl, fg, legend. The legend color appears only in
    the reserved corner area and never mixes with rays or sprite pieces.
  - Rays stay on their split row/column, corner pieces stay in corners, and the legend
    disappears from the output, so the three semantic layers remain easy to read.

g) change_inference:
  - No hidden numeric color ordering is used. The only semantic color novelty is that a
    dedicated legend color marks a small glyph in a fixed corner area, and the glyph's
    visible left/right turning shape determines the first-step rotation.

h) fusion-metric plan:
  - During validation, measure how many generated samples use the CCW glyph, and compare
    the combo solver against a fixed-CW solver on those inputs. Every CCW-legend sample
    should produce a different output; otherwise the legend would be semantically inert.

i) new solver assumptions and generator guarantees:
  - Assumption: the legend is the only non-bg content in `rows 1..3, cols 1..3`.
    Guarantee: generator reserves that block exclusively for the legend.
  - Assumption: after excluding ctrl and legend colors, exactly one fg sprite color
    remains. Guarantee: generator uses fixed roles `bg=1, ctrl=2, fg=7, legend=8`.
  - Assumption: only one vertical split and at most one horizontal split exist.
    Guarantee: generator mirrors the source task's 1-or-2 perpendicular ctrl scheme.
  - Assumption: the hidden pre-cut rotation is visible. Guarantee: transformed sprites
    are wide and non-rotation-invariant, and CCW samples are rejected if they match the
    fixed-CW solver or the legacy no-rotation solver too closely.
"""


BG = 1
CTRL = 2
FG = 7
LEGEND = 8

LEGEND_R0 = 1
LEGEND_C0 = 1
LEGEND_H = 3
LEGEND_W = 3

LEGEND_CW = frozenset({(0, 1), (1, 1), (1, 2)})
LEGEND_CCW = frozenset({(0, 1), (1, 1), (1, 0)})


def _bbox_of_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    return int(min(rs)), int(max(rs)), int(min(cs)), int(max(cs))


def _normalize_piece(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0 = int(min(rs))
    c0 = int(min(cs))
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _piece_hw(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return int(max(rs) + 1), int(max(cs) + 1)


def _rotate_piece_cw(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]], turns: int = 1
) -> frozenset[tuple[int, int]]:
    piece = _normalize_piece(cells)
    for _ in range(int(turns) % 4):
        height, _width = _piece_hw(piece)
        piece = frozenset((int(c), int(height - 1 - r)) for r, c in piece)
        piece = _normalize_piece(piece)
    return piece


def _rotation_is_visible(cells: set[tuple[int, int]] | frozenset[tuple[int, int]], turns: int) -> bool:
    piece = _normalize_piece(cells)
    return _rotate_piece_cw(piece, turns) != piece


def _rotate_points_tl(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]], turns: int
) -> set[tuple[int, int]]:
    if not pts:
        return set()
    out = set((int(r), int(c)) for r, c in pts)
    for _ in range(int(turns) % 4):
        r0, r1, c0, c1 = _bbox_of_points(out)
        height = int(r1 - r0 + 1)
        out = {(int(r0 + (c - c0)), int(c0 + (height - 1 - (r - r0)))) for r, c in out}
    return out


def _legend_cells(g: np.ndarray, bg: int) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for lr in range(LEGEND_H):
        for lc in range(LEGEND_W):
            rr = int(LEGEND_R0 + lr)
            cc = int(LEGEND_C0 + lc)
            if int(g[rr, cc]) != int(bg):
                cells.add((int(lr), int(lc)))
    return frozenset(cells)


def _decode_legend_turns(g: np.ndarray, bg: int) -> int:
    cells = _legend_cells(g, bg)
    if cells == LEGEND_CW:
        return 1
    if cells == LEGEND_CCW:
        return 3
    raise AssertionError(f"unexpected legend pattern: {sorted(cells)}")


def _infer_roles_and_controls(
    g: np.ndarray,
) -> tuple[int, int, int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert len(border_cols) == 1
    ctrl = int(next(iter(border_cols)))

    legend_cols = set()
    for lr in range(LEGEND_H):
        for lc in range(LEGEND_W):
            color = int(g[int(LEGEND_R0 + lr), int(LEGEND_C0 + lc)])
            if color != int(bg):
                legend_cols.add(int(color))
    assert len(legend_cols) == 1
    legend = int(next(iter(legend_cols)))

    fg_candidates = [int(c) for c in nonbg if int(c) not in {int(ctrl), int(legend)}]
    assert len(fg_candidates) == 1
    fg = int(fg_candidates[0])

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = int(c)
        elif r == H - 1:
            E.add("bottom")
            c_split = int(c)
        elif c == 0:
            E.add("left")
            r_split = int(r)
        elif c == W - 1:
            E.add("right")
            r_split = int(r)
    return int(bg), int(ctrl), int(fg), int(legend), ctrl_pts, E, c_split, r_split


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


def _pick_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in E) and ("bottom" not in E):
        dh = "T"
    elif ("bottom" in E) and ("top" not in E):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in E) and ("right" not in E):
        dv = "L"
    elif ("right" in E) and ("left" not in E):
        dv = "R"
    else:
        dv = "L"
    return str(dh), str(dv)


def _render_cut_and_slide(
    sprite_pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl: int,
    ctrl_pts: list[tuple[int, int]],
    E: set[str],
    c_split: int | None,
    r_split: int | None,
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    sprite_set = set((int(r), int(c)) for r, c in sprite_pts)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr, cc in sprite_set if int(cc) == int(c)]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if (int(rr), int(c)) not in sprite_set:
                        out[int(rr), int(c)] = int(ctrl)
        elif r == H - 1:
            rows = [int(rr) for rr, cc in sprite_set if int(cc) == int(c)]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if (int(rr), int(c)) not in sprite_set:
                        out[int(rr), int(c)] = int(ctrl)
        elif c == 0:
            cols = [int(cc) for rr, cc in sprite_set if int(rr) == int(r)]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if (int(r), int(cc)) not in sprite_set:
                        out[int(r), int(cc)] = int(ctrl)
        elif c == W - 1:
            cols = [int(cc) for rr, cc in sprite_set if int(rr) == int(r)]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if (int(r), int(cc)) not in sprite_set:
                        out[int(r), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    split_line_pts = {
        (int(r), int(c))
        for r, c in sprite_set
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split))
    }
    for r, c in split_line_pts:
        out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in sprite_set:
        if (int(r), int(c)) in split_line_pts:
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        rs = [int(r) for r, _ in pts]
        cs = [int(c) for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _solve_with_turns(grid: np.ndarray, turns: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, ctrl, fg, _legend, ctrl_pts, E, c_split, r_split = _infer_roles_and_controls(g)
    sprite_pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    work_pts = _rotate_points_tl(sprite_pts, turns)
    return _render_cut_and_slide(work_pts, H, W, bg, fg, ctrl, ctrl_pts, E, c_split, r_split)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_turns(grid, 0)


def _fixed_cw_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_turns(grid, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Read the 3x3 legend in the upper-left inner corner.
    2. Rotate the whole fg sprite inside its tight bbox according to that legend.
    3. Apply the original border-ray cut + corner-slide rule to the rotated sprite.
    """
    g = np.array(grid, dtype=int)
    turns = _decode_legend_turns(g, int(get_bg(g)))
    return _solve_with_turns(g, turns)


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (int(nr), int(nc))
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(int(nr), int(nc))}) - min(r for r, _ in cells | {(int(nr), int(nc))}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(int(nr), int(nc))}) - min(c for _, c in cells | {(int(nr), int(nc))}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    return _normalize_piece(cells)


def _mask_is_contiguous(H: int, W: int, pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    mask = np.zeros((H, W), dtype=int)
    for r, c in pts:
        mask[int(r), int(c)] = 1
    return bool(is_contiguous(mask))


def _legend_grid(turns: int) -> np.ndarray:
    area = np.full((LEGEND_H, LEGEND_W), int(BG), dtype=int)
    pattern = LEGEND_CW if int(turns) % 4 == 1 else LEGEND_CCW
    for r, c in pattern:
        area[int(r), int(c)] = int(LEGEND)
    return area


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S

    turns = random.choice([1, 3])
    legend_patch = _legend_grid(turns)

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1))
            has_h = True

    def touches_E(hs: str, vs: str) -> bool:
        return bool(_touches_edges(hs, vs) & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [r for r in all_regions if touches_E(*r)]
    vanish = [r for r in all_regions if r not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)
    safe_side = int(max(sprite_h, sprite_w))
    max_r0 = int(H - 5 - safe_side + 1)
    max_c0 = int(W - 5 - safe_side + 1)
    if max_r0 < 4 or max_c0 < 4:
        return None

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(sprite_c0, max_c0))
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        sprite_c0 = random.randint(4, max_c0)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(sprite_r0, max_r0))
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        sprite_r0 = random.randint(4, max_r0)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [int(r) for r in range(sprite_r0, sprite_r1 + 1) if int(r) != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [int(c) for c in range(sprite_c0, sprite_c1 + 1) if int(c) != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_total = int(random.uniform(0.39, 0.60) * area)

    n_kept = len(kept)
    budget_pieces = int(target_total - n_split)
    if budget_pieces < n_kept * 3:
        return None

    if vanish:
        max_vanish = min(7, max(3, budget_pieces // 3))
        vanish_budget = random.randint(3, max_vanish)
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (len(kept) - i - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    transformed_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        piece = _grow_piece(sub_h, sub_w, int(piece_budgets[idx]))
        if len(piece) < 3:
            return None
        if not _rotation_is_visible(piece, 1):
            return None

        ph, pw = _piece_hw(piece)

        if has_h:
            inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        else:
            split_rows = sorted(int(r) for r, c in split_pixels if int(c) == int(c_split))
            if split_rows:
                mid = int((split_rows[0] + split_rows[-1]) // 2)
                inp_r0 = max(int(r0), min(int(mid - ph // 2), int(r1 - ph + 1)))
            else:
                inp_r0 = int(r0)

        if has_v:
            inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        else:
            split_cols = sorted(int(c) for r, c in split_pixels if int(r) == int(r_split))
            if split_cols:
                mid = int((split_cols[0] + split_cols[-1]) // 2)
                inp_c0 = max(int(c0), min(int(mid - pw // 2), int(c1 - pw + 1)))
            else:
                inp_c0 = int(c0)

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        transformed_pieces[reg] = inp_pts

    all_transformed_fg = set()
    for pts in transformed_pieces.values():
        all_transformed_fg |= set(pts)
    all_transformed_fg |= split_pixels
    if not all_transformed_fg:
        return None

    fg_arr = np.array(list(all_transformed_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None
    if aw <= ah:
        return None
    if not _mask_is_contiguous(H, W, all_transformed_fg):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vr0, vr1, vc0, vc1 = sub_bbox(*vreg)
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(int(vanish_budget), int(vh * vw))
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph, ppw = _piece_hw(piece)
            vhs, vvs = vreg
            vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
            vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (int(r), int(c)) in all_transformed_fg:
                    return None
            vanish_fg |= vpts

    combined_rot = set(all_transformed_fg) | set(vanish_fg)
    if not _mask_is_contiguous(H, W, combined_rot):
        return None

    inverse_turns = int((4 - turns) % 4)
    combined_in = _rotate_points_tl(combined_rot, inverse_turns)
    if combined_in == combined_rot:
        return None
    if not _mask_is_contiguous(H, W, combined_in):
        return None

    in_arr = np.array(list(combined_in))
    in_h = int(in_arr[:, 0].max() - in_arr[:, 0].min() + 1)
    in_w = int(in_arr[:, 1].max() - in_arr[:, 1].min() + 1)
    if in_h <= in_w:
        return None

    for r, c in combined_in:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if LEGEND_R0 <= r < LEGEND_R0 + LEGEND_H and LEGEND_C0 <= c < LEGEND_C0 + LEGEND_W:
            return None
    if any((int(r), int(c)) in combined_in for r, c in ctrl_pts):
        return None

    density = len(combined_in) / float(in_h * in_w)
    if density < 0.38 or density > 0.62:
        return None

    inp = np.full((H, W), int(BG), dtype=int)
    for r, c in combined_in:
        inp[int(r), int(c)] = int(FG)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(CTRL)
    for lr in range(LEGEND_H):
        for lc in range(LEGEND_W):
            inp[int(LEGEND_R0 + lr), int(LEGEND_C0 + lc)] = int(legend_patch[int(lr), int(lc)])

    out = _render_cut_and_slide(all_transformed_fg, H, W, BG, FG, CTRL, ctrl_pts, E, c_split, r_split)
    variant_out = main(inp)
    if not np.array_equal(variant_out, out):
        return None

    legacy_out = _legacy_main(inp)
    if np.array_equal(legacy_out, out):
        return None

    fixed_cw_out = _fixed_cw_main(inp)
    if turns == 3:
        diff_cw = int(np.count_nonzero(fixed_cw_out != out))
        if diff_cw < 6:
            return None
    diff_legacy = int(np.count_nonzero(legacy_out != out))
    if diff_legacy < 6:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
