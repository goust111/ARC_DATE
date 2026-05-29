# Combo variant of 4a21e3da (index 31)
# combo_spec: D9/1+D16/0
# Name: 分割线像素消失+消失区域作为干扰项

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


"""
design_analysis:
  concrete_behavior:
    input:
      - Keep the original centered connected fg sprite and border ctrl pixels.
      - Use exactly 2 ctrl pixels in generation so both halves of the combo are
        always visible: one vertical split and one horizontal split.
      - Split-line fg still exists in the input because it connects the sprite and
        defines where the intermittent rays stop.
    output:
      - Regions touching a control edge still slide to their corresponding corners.
      - The unique non-touching quadrant no longer disappears; it stays at its
        original coordinates as a static distractor.
      - Split-line fg no longer survives in the output at all.
      - Therefore the only fg left in the output is:
        1. corner-moved pieces
        2. the static distractor quadrant
      - Rays remain ctrl-colored and are painted only on background cells.
  behavior_trace:
    - Example: controls on top and left create split column c=9 and split row r=8.
    - Original task:
      - TL/TR/BL regions move to corners.
      - BR region vanishes.
      - Any fg on row 8 or column 9 stays in place.
    - Combo task:
      - TL/TR/BL still move to corners.
      - BR now stays in the center as a distractor.
      - Any fg on row 8 or column 9 is removed from the output.
    - Net visual delta:
      - The center loses its cut-line markers but keeps one intact off-line chunk.
  complexity_calibration:
    - Fix generation to the 2-control regime; otherwise D16 would be absent and the
      combo could collapse into plain D9.
    - Keep the original 3-color palette, 18-20 square grid, 5-8 by 8-10 sprite
      bbox, and 2-4 split pixels per split so the task still reads as the same
      family while the combo effect stays obvious.
    - Reserve 3-7 pixels for the static distractor so it is visible but does not
      dominate the central sprite.
  implicit_assumption_audit:
    - Base main wrote split-line fg back into the output; that assumption is invalid
      and the write-back loop must be removed.
    - Base main skipped non-touching regions entirely; that assumption is invalid and
      such regions must now be copied in place.
    - Base/D16 generators built rays by scanning fg already drawn into the output.
      That fails under D9 because split-line fg is absent there. The combo generator
      must paint rays from input-side split_pixels instead.
    - Region classification still depends only on split coordinates, so removing
      output-side split pixels does not affect quadrant assignment.
  placement_legality:
    - Kept pieces stay strictly off the split row/column in both input and output.
    - The static distractor is placed strictly inside the unique non-touching
      quadrant, so it cannot overlap rays or moved corner pieces.
    - Input fg remains one connected component after adding the static quadrant.
    - Output may have multiple disconnected fg components, which is intended.
  visual_observability:
    - Every sample shows both combo effects simultaneously:
      - missing split-line pixels
      - one unmoved central distractor quadrant
    - Because the distractor is off the split line, solvers can visually separate
      "removed cut markers" from "retained central piece".
  neatness:
    - The original >=4-cell margin and centered placement are preserved, keeping the
      distractor away from the border rays and the corner-moved pieces.
    - No extra colors are introduced, so the scene remains readable.
  inference_safety:
    - There is no color remapping. The inference is entirely positional:
      touching regions move, the non-touching region stays, split-line fg disappears.
  new_main_assumptions_and_generator_guarantees:
    - main assumes at most one horizontal split and one vertical split; generator
      always constructs exactly that.
    - main assumes any fg on the split row/column may be deleted from the output;
      generator always places split witnesses in the input.
    - main assumes the retained distractor is off both split lines; generator
      enforces that by placing it strictly inside the static quadrant.
    - ray painting assumes the stopping witnesses come from split_pixels rather than
      output-side fg; generator supplies 2-4 witnesses on each split.
"""


def _detect_structure(
    g: np.ndarray,
) -> tuple[int, int, int, int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = (
        set(map(int, g[0, :]))
        | set(map(int, g[-1, :]))
        | set(map(int, g[:, 0]))
        | set(map(int, g[:, -1]))
    )
    border_cols.discard(bg)
    assert border_cols

    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = c
        elif r == H - 1:
            E.add("bottom")
            c_split = c
        elif c == 0:
            E.add("left")
            r_split = r
        elif c == W - 1:
            E.add("right")
            r_split = r

    return H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split


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


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
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


def _paint_rays_from_input_split_pixels(
    out: np.ndarray,
    ctrl_pts: list[tuple[int, int]],
    split_pixels: set[tuple[int, int]],
    ctrl_color: int,
    bg: int,
) -> None:
    H, W = out.shape
    for cr, cc in ctrl_pts:
        if cr == 0:
            rows = sorted(r for r, c in split_pixels if c == cc)
            if rows:
                for r in range(0, max(rows) + 1):
                    if (r, cc) not in split_pixels and int(out[r, cc]) == bg:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            rows = sorted(r for r, c in split_pixels if c == cc)
            if rows:
                for r in range(H - 1, min(rows) - 1, -1):
                    if (r, cc) not in split_pixels and int(out[r, cc]) == bg:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            cols = sorted(c for r, c in split_pixels if r == cr)
            if cols:
                for c in range(0, max(cols) + 1):
                    if (cr, c) not in split_pixels and int(out[cr, c]) == bg:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            cols = sorted(c for r, c in split_pixels if r == cr)
            if cols:
                for c in range(W - 1, min(cols) - 1, -1):
                    if (cr, c) not in split_pixels and int(out[cr, c]) == bg:
                        out[cr, c] = ctrl_color

    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo variant:
    - split-line fg disappears in the output
    - the non-touching quadrant stays in place as a distractor
    """
    g = np.array(grid, dtype=int)
    H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _detect_structure(g)

    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            continue

        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            for r, c in pts:
                out[r, c] = fg
            continue

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = -rmin if dh == "T" else (H - 1) - rmax
        dc = -cmin if dv == "L" else (W - 1) - cmax

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg

    return out


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    # Fix to 2 controls so the static distractor quadrant is always present.
    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = None
    r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1))
            has_h = True

    assert has_v and has_h and r_split is not None and c_split is not None

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    static_regs = [reg for reg in all_regions if reg not in kept]
    if len(static_regs) != 1:
        return None

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac = random.uniform(0.4, 0.6)
    sprite_c0 = c_split - int(frac * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
    if sprite_c0 > W - 5 - sprite_w + 1:
        return None
    if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
        return None

    frac = random.uniform(0.4, 0.6)
    sprite_r0 = r_split - int(frac * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
    if sprite_r0 > H - 5 - sprite_h + 1:
        return None
    if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else int(r_split) + 1
        r1 = int(r_split) - 1 if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else int(c_split) + 1
        c1 = int(c_split) - 1 if vs == "L" else sprite_c1
        return r0, r1, c0, c1

    for reg in kept + static_regs:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    v_rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
    h_cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
    n_v = random.randint(2, 4)
    n_h = random.randint(2, 4)
    if len(v_rows) < n_v or len(h_cols) < n_h:
        return None
    for r in random.sample(v_rows, n_v):
        split_pixels.add((int(r), int(c_split)))
    for c in random.sample(h_cols, n_h):
        split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    if n_split < 4:
        return None

    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    budget_pieces = target_total - n_split
    static_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
    budget_pieces -= static_budget
    if budget_pieces < len(kept) * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
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
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)

        for r, c in out_pts:
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
        output_pieces[reg] = out_pts

    all_inp_fg = set().union(*input_pieces.values()) if input_pieces else set()
    all_inp_fg |= split_pixels
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    static_fg: set[tuple[int, int]] = set()
    reg = static_regs[0]
    vhs, vvs = reg
    vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
    vh = vr1 - vr0 + 1
    vw = vc1 - vc0 + 1
    if vh < 1 or vw < 1:
        return None
    max_px = min(static_budget, vh * vw)
    if max_px < 3:
        return None
    piece = _grow_piece(vh, vw, max_px)
    if len(piece) < 3:
        return None

    pph = max(r for r, _ in piece) + 1
    ppw = max(c for _, c in piece) + 1
    vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
    vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
    vpts = {(r + vr_off, c + vc_off) for r, c in piece}
    for r, c in vpts:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (r, c) in all_inp_fg:
            return None
        if r == r_split or c == c_split:
            return None
    static_fg = vpts
    if len(static_fg) < 3:
        return None

    combined = all_inp_fg | static_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in static_fg:
        out[r, c] = fg
    _paint_rays_from_input_split_pixels(out, ctrl_pts, split_pixels, ctrl_color, bg)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
