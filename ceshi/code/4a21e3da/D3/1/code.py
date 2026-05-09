# Variant of 4a21e3da
# Dimension: D3 — 源对象类型 (src)
# Change: control_pixels — 网格边缘的1-2个红色控制像素作为射线源... → 精灵内部标记像素

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
Design analysis for the D3 variant
==================================

source_task:
  reference_task: none

a) concretization_and_trace:
  original:
    input:
      - a single connected orange sprite near the center
      - one or two red control pixels on the grid border
    output:
      - red rays are drawn from the border inward along a split row/column
      - orange pixels on the split line stay
      - off-line sprite fragments slide to border corners if their region touches
        the chosen border edge; otherwise they vanish
  variant:
    input:
      - the orange sprite still sits near the center
      - the 1-2 red controls are embedded in the sprite instead of sitting on the grid border
      - each red marker lies on its split line and is the extreme occupied cell in the ray direction:
        top-marker = topmost occupied cell on its split column, bottom-marker = bottommost, etc.
    output:
      - the red marker stays inside the sprite
      - a red ray is drawn from that internal marker outward to its chosen grid edge
      - the split row/column still partitions the sprite exactly as before
      - orange fragments still slide to the same corners as the original edge choice implies
  small_trace:
    - original top-edge control at column c: draw downward from row 0 to the farthest orange on c.
    - variant top-marker at (r,c): keep the same split column c and same destination logic "top",
      but draw upward from row r to row 0.
    - difference: the solver must detect the rare red pixel inside the sprite, infer whether it means
      top/bottom/left/right from its placement on the sprite outline, then reuse the old partition logic.

b) complexity_calibration:
  - The main rule change is visually obvious: there are no border control pixels in the input anymore.
  - I keep the old "one vertical split and optionally one horizontal split" structure so the puzzle
    remains learnable from 2-4 examples.
  - To avoid ambiguous internal markers, each marker is placed on the relevant outer edge of the sprite
    bbox and is the only control on that line. This preserves a small number of directional cases:
    {top, bottom, left, right}.

c) implicit_assumption_review_of_original_main:
  - original border_cols detection assumed the control color appears on the grid border. This fails now.
    New solver identifies ctrl as the rarer non-bg color (1-2 pixels) and fg as the other color.
  - original loop over ctrl_pts assumed every control point sits exactly on top/bottom/left/right border,
    and used that to infer both split orientation and destination edge. This also fails now.
    New solver infers direction from the marker's relation to the non-bg bbox plus the continuation
    of orange pixels on the same row/column.
  - original ray drawing assumed the ray starts on the border and moves inward to the farthest fg.
    In the variant, the ray starts at the internal marker and moves outward to the border.
  - original region movement and corner selection remain valid once E, c_split, and r_split are recovered.

d) placement_legality:
  - The generator still keeps the sprite bbox at least 4 cells away from the grid border.
  - Split lines still pass through the sprite interior around 40-60% of bbox width/height.
  - Control markers are placed on the sprite's own outline, never on the grid border.
  - Kept/vanishing pieces are generated strictly inside their subregions, never on the split line,
    and output placements are checked not to cross their split side.
  - The combined input object (orange pixels plus red markers) is required to be 4-connected.

e) visual_observability:
  - A human sees a central sprite containing one or two rare red pixels, and in the outputs those
    red pixels emit rays outward to the border. That is a strong, visible inversion of the original task.
  - The marker-to-edge mapping is supported by repeated cases: top/bottom markers sit on the top/bottom
    of the sprite component, left/right markers on the left/right.
  - The sliding behavior is intentionally unchanged, so the new difficulty is localizing the source.

f) visual_cleanliness:
  - Only three colors are used, same as the original.
  - Red is reserved for markers and rays; orange remains the movable sprite material.
  - The rays occupy empty background space outside the centered sprite, so there is little overlap noise.

g) color_change_inference:
  - No new color remapping is introduced.
  - The only semantic change is where the red source pixels live and which way the red lines grow.

h) new_solver_assumptions_and_generator_guarantees:
  - Assumption: exactly two non-bg colors exist, with the control color appearing 1-2 times.
    Guarantee: generator always uses fg=7 for the bulk sprite and ctrl=2 for 1-2 markers.
  - Assumption: each control marker yields exactly one directional interpretation.
    Guarantee: the marker is placed on one side of the non-bg bbox and the same row/column contains
    orange continuation only on the opposite side, eliminating corner ambiguity.
  - Assumption: at most one vertical and one horizontal split exist.
    Guarantee: generator mirrors the original task and never creates two vertical or two horizontal markers.
"""


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    """Grow a connected piece up to max_h x max_w targeting target_px pixels."""
    cells = {(0, 0)}
    for _ in range(target_px - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= max_h
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= max_w
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _nonbg_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(g != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ctrl_and_fg(g: np.ndarray, bg: int) -> tuple[int, int]:
    nonbg = colors_present(g, exclude={bg})
    assert len(nonbg) == 2
    counts = {int(col): int(np.count_nonzero(g == int(col))) for col in nonbg}
    ctrl = min(counts, key=counts.get)
    fg = next(col for col in nonbg if int(col) != int(ctrl))
    assert 1 <= counts[int(ctrl)] <= 2
    return int(ctrl), int(fg)


def _infer_control_info(
    g: np.ndarray, bg: int, ctrl: int, fg: int
) -> list[tuple[int, int, str]]:
    """
    Infer each marker as one of {top, bottom, left, right}.

    The generator guarantees:
    - top/bottom markers sit on the top/bottom of the non-bg bbox and have fg continuation
      only on the opposite side of the same column;
    - left/right markers are analogous on the same row.
    """
    r0, c0, r1, c1 = _nonbg_bbox(g, bg)
    info: list[tuple[int, int, str]] = []

    for r, c in np.argwhere(g == int(ctrl)):
        r = int(r)
        c = int(c)
        col_rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
        row_cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]

        candidates: list[str] = []
        if col_rows and r == r0 and min(col_rows) > r:
            candidates.append("top")
        if col_rows and r == r1 and max(col_rows) < r:
            candidates.append("bottom")
        if row_cols and c == c0 and min(row_cols) > c:
            candidates.append("left")
        if row_cols and c == c1 and max(row_cols) < c:
            candidates.append("right")

        assert len(candidates) == 1
        info.append((r, c, candidates[0]))

    return info


def _pick_dest(E: set[str], r_split: int | None, c_split: int | None, hs: str, vs: str) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    else:
        if ("top" in E) and ("bottom" not in E):
            dh = "T"
        elif ("bottom" in E) and ("top" not in E):
            dh = "B"
        else:
            dh = "T"
    if c_split is not None:
        dv = vs
    else:
        if ("left" in E) and ("right" not in E):
            dv = "L"
        elif ("right" in E) and ("left" not in E):
            dv = "R"
        else:
            dv = "L"
    return str(dh), str(dv)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - The 1-2 red control pixels are embedded in the sprite instead of on the grid border.
    - Each marker sits on a split row/column and indicates one border direction.
    - Rays are drawn from the marker outward to that border.
    - The split row/column still partitions the orange sprite, and the kept regions slide to corners
      using the same corner logic as the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ctrl, fg = _ctrl_and_fg(g, bg)
    ctrl_info = _infer_control_info(g, bg, ctrl, fg)

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c, edge in ctrl_info:
        if edge in ("top", "bottom"):
            assert c_split is None or c_split == c
            c_split = int(c)
        else:
            assert r_split is None or r_split == r
            r_split = int(r)
        E.add(edge)

    out = np.full((H, W), int(bg), dtype=int)

    # Stage 1: draw rays from internal markers outward to the corresponding border.
    for r, c, edge in ctrl_info:
        if edge == "top":
            for rr in range(0, r):
                if int(out[rr, c]) == int(bg):
                    out[rr, c] = int(ctrl)
        elif edge == "bottom":
            for rr in range(r + 1, H):
                if int(out[rr, c]) == int(bg):
                    out[rr, c] = int(ctrl)
        elif edge == "left":
            for cc in range(0, c):
                if int(out[r, cc]) == int(bg):
                    out[r, cc] = int(ctrl)
        elif edge == "right":
            for cc in range(c + 1, W):
                if int(out[r, cc]) == int(bg):
                    out[r, cc] = int(ctrl)

    for r, c, _edge in ctrl_info:
        out[r, c] = int(ctrl)

    # Orange pixels on split line(s) remain in place. Control markers themselves stay red.
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = int(fg)

    # Stage 2: move off-line pieces from each region to its target corner.
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if c < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue

        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(fg)

    return out


def _slot_start(slots: list[int], target: int, span: int) -> int | None:
    if span > len(slots):
        return None
    best_i = 0
    best_cost = None
    for i in range(len(slots) - span + 1):
        center = slots[i + span // 2]
        cost = abs(center - int(target))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_i = i
    return int(best_i)


def _sub_bbox(
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
    hs: str,
    vs: str,
) -> tuple[int, int, int, int]:
    r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
    r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if r_split is not None else sprite_r1
    c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
    c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if c_split is not None else sprite_c1
    return int(r0), int(r1), int(c0), int(c1)


def _available_axes(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    forbidden_rows: set[int],
    forbidden_cols: set[int],
) -> tuple[list[int], list[int]]:
    rows = [r for r in range(r0, r1 + 1) if int(r) not in forbidden_rows]
    cols = [c for c in range(c0, c1 + 1) if int(c) not in forbidden_cols]
    return rows, cols


def _place_piece(
    rows: list[int],
    cols: list[int],
    piece: frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    has_h: bool,
    has_v: bool,
    split_rows: list[int],
    split_cols: list[int],
) -> set[tuple[int, int]] | None:
    ph = max(r for r, _ in piece) + 1
    pw = max(c for _, c in piece) + 1
    if ph > len(rows) or pw > len(cols):
        return None

    if has_h:
        r_start = len(rows) - ph if hs == "T" else 0
    else:
        target_r = split_rows[len(split_rows) // 2] if split_rows else rows[len(rows) // 2]
        r_start = _slot_start(rows, target_r, ph)
        if r_start is None:
            return None

    if has_v:
        c_start = len(cols) - pw if vs == "L" else 0
    else:
        target_c = split_cols[len(split_cols) // 2] if split_cols else cols[len(cols) // 2]
        c_start = _slot_start(cols, target_c, pw)
        if c_start is None:
            return None

    return {(rows[r + int(r_start)], cols[c + int(c_start)]) for r, c in piece}


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[2, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    E = set(ctrl_edges)
    has_v = any(edge in ("top", "bottom") for edge in ctrl_edges)
    has_h = any(edge in ("left", "right") for edge in ctrl_edges)

    c_split = random.randint(int(W * 0.4), int(W * 0.6)) if has_v else None
    r_split = random.randint(int(H * 0.4), int(H * 0.6)) if has_h else None

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split) - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if int(c_split) <= sprite_c0 or int(c_split) >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split) - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if int(r_split) <= sprite_r0 or int(r_split) >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    ctrl_info: list[tuple[int, int, str]] = []
    forbidden_rows: set[int] = set()
    forbidden_cols: set[int] = set()

    if has_v:
        edge = next(edge for edge in ctrl_edges if edge in ("top", "bottom"))
        v_ctrl_r = sprite_r0 if edge == "top" else sprite_r1
        ctrl_info.append((int(v_ctrl_r), int(c_split), edge))
        forbidden_rows.add(int(v_ctrl_r))

    if has_h:
        edge = next(edge for edge in ctrl_edges if edge in ("left", "right"))
        h_ctrl_c = sprite_c0 if edge == "left" else sprite_c1
        ctrl_info.append((int(r_split), int(h_ctrl_c), edge))
        forbidden_cols.add(int(h_ctrl_c))

    ctrl_pts = [(r, c) for r, c, _edge in ctrl_info]
    if len(set(ctrl_pts)) != len(ctrl_pts):
        return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        edge = next(edge for edge in ctrl_edges if edge in ("top", "bottom"))
        if edge == "top":
            rows_pool = [r for r in range(sprite_r0 + 1, sprite_r1 + 1) if r != r_split]
        else:
            rows_pool = [r for r in range(sprite_r0, sprite_r1) if r != r_split]
        if len(rows_pool) < 2:
            return None
        n = random.randint(2, min(4, len(rows_pool)))
        for r in random.sample(rows_pool, n):
            split_pixels.add((int(r), int(c_split)))

    if has_h:
        edge = next(edge for edge in ctrl_edges if edge in ("left", "right"))
        if edge == "left":
            cols_pool = [c for c in range(sprite_c0 + 1, sprite_c1 + 1) if c != c_split]
        else:
            cols_pool = [c for c in range(sprite_c0, sprite_c1) if c != c_split]
        if len(cols_pool) < 2:
            return None
        n = random.randint(2, min(4, len(cols_pool)))
        for c in random.sample(cols_pool, n):
            split_pixels.add((int(r_split), int(c)))

    total_area = sprite_h * sprite_w
    target_density = random.uniform(0.40, 0.58)
    target_nonbg = int(round(target_density * total_area))

    n_fixed = len(split_pixels) + len(ctrl_pts)
    budget_pieces = target_nonbg - n_fixed
    if budget_pieces < len(kept) * 3:
        return None

    if vanish:
        max_vanish = min(7, max(3, budget_pieces // 3))
        vanish_budget = random.randint(3, max_vanish)
        budget_kept = budget_pieces - vanish_budget
    else:
        vanish_budget = 0
        budget_kept = budget_pieces

    if budget_kept < len(kept) * 3:
        return None

    capacities: dict[tuple[str, str], int] = {}
    axes_cache: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    for reg in kept + vanish:
        r0, r1, c0, c1 = _sub_bbox(sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split, *reg)
        if r1 < r0 or c1 < c0:
            return None
        rows, cols = _available_axes(r0, r1, c0, c1, forbidden_rows, forbidden_cols)
        if not rows or not cols:
            return None
        axes_cache[reg] = (rows, cols)
        capacities[reg] = len(rows) * len(cols)

    piece_budgets: list[int] = []
    remaining = budget_kept
    for i, reg in enumerate(kept):
        cap = capacities[reg]
        if i == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - i)
            lo_px = max(3, avg - 2)
            hi_px = min(cap, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, cap, remaining - (len(kept) - i - 1) * 3))
        if px > cap:
            return None
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0:
        return None

    split_rows = sorted([r for r, c in split_pixels if c_split is not None and c == int(c_split)])
    split_cols = sorted([c for r, c in split_pixels if r_split is not None and r == int(r_split)])

    input_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        rows, cols = axes_cache[reg]
        piece = _grow_piece(len(rows), len(cols), piece_budgets[idx])
        if len(piece) < 3:
            return None
        pts = _place_piece(rows, cols, piece, hs, vs, has_h, has_v, split_rows, split_cols)
        if pts is None:
            return None

        for r, c in pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if c_split is not None and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                return None
            if r_split is not None and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                return None
            if r in forbidden_rows or c in forbidden_cols:
                return None

        input_pieces[reg] = pts

        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
        dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in pts)

        for r, c in out_pts:
            if c_split is not None and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                return None
            if r_split is not None and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                return None
        output_pieces[reg] = out_pts

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        vreg = random.choice(vanish)
        rows, cols = axes_cache[vreg]
        max_px = min(vanish_budget, capacities[vreg])
        if max_px < 3:
            return None
        piece = _grow_piece(len(rows), len(cols), max_px)
        if len(piece) < 3:
            return None
        vpts = _place_piece(rows, cols, piece, vreg[0], vreg[1], has_h, has_v, split_rows, split_cols)
        if vpts is None:
            return None
        for r, c in vpts:
            if r in forbidden_rows or c in forbidden_cols:
                return None
        vanish_fg = set(vpts)

    all_fg = set().union(*input_pieces.values()) if input_pieces else set()
    all_fg |= split_pixels
    all_fg |= vanish_fg
    all_nonbg = set(all_fg) | set(ctrl_pts)

    if not all_nonbg:
        return None

    arr = np.array(list(all_nonbg))
    ah = int(arr[:, 0].max() - arr[:, 0].min() + 1)
    aw = int(arr[:, 1].max() - arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_nonbg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    actual_density = len(all_nonbg) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_fg:
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg
    for r, c, edge in ctrl_info:
        if edge == "top":
            for rr in range(0, r):
                if out[rr, c] == bg:
                    out[rr, c] = ctrl_color
        elif edge == "bottom":
            for rr in range(r + 1, H):
                if out[rr, c] == bg:
                    out[rr, c] = ctrl_color
        elif edge == "left":
            for cc in range(0, c):
                if out[r, cc] == bg:
                    out[r, cc] = ctrl_color
        elif edge == "right":
            for cc in range(c + 1, W):
                if out[r, cc] == bg:
                    out[r, cc] = ctrl_color
        out[r, c] = ctrl_color

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
