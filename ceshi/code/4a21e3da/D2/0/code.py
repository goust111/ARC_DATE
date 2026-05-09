# Variant of 4a21e3da
# Dimension: D2 — 射线轨迹模式 (trajectory)
# Change: intermittent_skip — 射线跳过前景像素，仅在背景像素上涂色... → 连续轨迹（覆盖前景）
"""
Design analysis for the D2 variant.

variant_spec:
  core_change:
    original: the ctrl ray only colored background cells and skipped fg cells on the split line
    variant: the ctrl ray is continuous from the border marker to the farthest split-line fg cell,
      recoloring every traversed cell to ctrl, including former fg cells
  concrete_visual_effect:
    input:
      - same three-color setup as the source task
      - a connected central fg sprite still crosses the active split row/col
      - each active split line still contains 2-4 fg pixels so the future overwrite is visible
    output:
      - ctrl rays are solid, not intermittent
      - fg pixels that lie on the split line disappear as fg and become ctrl instead
      - off-line pieces keep the same slide/vanish logic as the source task
  behavior_trace:
    example:
      - top ctrl at column c, split-line fg at rows 6 and 9
      - original output on column c: rows 0..9 become ctrl only where input had bg, while rows 6 and 9 stay fg
      - variant output on column c: every row 0..9 becomes ctrl, so rows 6 and 9 no longer remain fg
      - off-line pieces are classified exactly as before because the partition is still induced by the same row/col
  complexity_calibration:
    - keep 1-ctrl and 2-ctrl cases so the rule still needs quadrant/half reasoning
    - keep 2-4 overwritten split pixels per active ray so the continuous-vs-skip distinction is visible in every sample
    - keep the same three colors and same motion rule to isolate D2 as the only conceptual change
  hidden_assumptions_audit:
    - source Stage 1 assumed "painting only bg" and therefore guarded each write with g[...] == bg; this must be removed
    - source Stage 1 + preservation loop assumed split-line fg survive as fg; that loop must be deleted
    - source generator drew rays by scanning fg already present in the output; this no longer works because the
      overwritten split-line fg vanish, so ray endpoints must instead be derived from input split pixels
    - region extraction can still ignore split-line fg entirely because in the variant they are consumed by the ray
  placement_legality:
    - input pieces stay strictly on one side of each split line; only the designated split pixels lie on the line
    - output corner pieces are still kept away from the split lines, so continuous ctrl rays do not overwrite moved pieces
    - the input sprite remains contiguous because the split-line fg still bridge the pieces before being overwritten
  observability:
    - every active ray has at least two split-line fg cells, guaranteeing that the variant changes at least two visible
      fg cells into ctrl cells
    - with two ctrls, both a vertical and horizontal solid ctrl segment appear, making the "continuous overwrite"
      pattern easy to infer from a few examples
  cleanliness:
    - still only three colors and a single central sprite
    - no extra markers are introduced; the visual difference is solely that ctrl lines pass through the sprite continuously
  new_main_assumptions_and_generator_support:
    - main assumes each active split line has at least one input fg cell; generator enforces 2-4 such cells per line
    - main assumes ctrl-colored border pixels are the only non-bg border color; generator preserves that
    - main assumes moved pieces do not need to survive on split lines; generator keeps all movable pieces strictly off the lines
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _parse_controls(
    g: np.ndarray, bg: int
) -> tuple[int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    nonbg = colors_present(g, exclude={bg})
    border_cols = (
        set(map(int, g[0, :]))
        | set(map(int, g[-1, :]))
        | set(map(int, g[:, 0]))
        | set(map(int, g[:, -1]))
    )
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    H, W = g.shape
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
    return ctrl, fg, ctrl_pts, edges, c_split, r_split


def _on_split_line(r: int, c: int, c_split: int | None, r_split: int | None) -> bool:
    return (c_split is not None and c == c_split) or (r_split is not None and r == r_split)


def _ray_stop(g: np.ndarray, fg: int, r: int, c: int) -> int | None:
    H, W = g.shape
    if r == 0:
        rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
        return int(max(rows)) if rows else None
    if r == H - 1:
        rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
        return int(min(rows)) if rows else None
    if c == 0:
        cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
        return int(max(cols)) if cols else None
    if c == W - 1:
        cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
        return int(min(cols)) if cols else None
    return None


def _paint_continuous_ray(out: np.ndarray, ctrl: int, r: int, c: int, stop: int | None) -> None:
    if stop is None:
        return
    H, W = out.shape
    if r == 0:
        for rr in range(0, int(stop) + 1):
            out[rr, c] = int(ctrl)
    elif r == H - 1:
        for rr in range(H - 1, int(stop) - 1, -1):
            out[rr, c] = int(ctrl)
    elif c == 0:
        for cc in range(0, int(stop) + 1):
            out[r, cc] = int(ctrl)
    elif c == W - 1:
        for cc in range(W - 1, int(stop) - 1, -1):
            out[r, cc] = int(ctrl)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: ctrl rays are solid and overwrite fg on the split line.

    The partition logic is unchanged from the source task: off-line fg cells are grouped
    by split-defined regions, pieces touching a control edge slide to their corner, and the
    remaining regions vanish. The difference is that any input fg cell on the active split
    row/col is consumed by the ctrl ray and does not survive as fg in the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ctrl, fg, ctrl_pts, edges, c_split, r_split = _parse_controls(g, bg)

    out = np.full((H, W), int(bg), dtype=int)

    # Stage 1: paint each ray continuously from the border ctrl to the farthest input fg on that line.
    for r, c in ctrl_pts:
        _paint_continuous_ray(out, ctrl, r, c, _ray_stop(g, fg, r, c))

    def touches_edges(hs: str, vs: str) -> set[str]:
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

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
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

    # Stage 2: move off-line fg pieces from each split-defined region to their destination corners.
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if _on_split_line(r, c, c_split, r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & edges):
            continue

        dh, dv = pick_dest(hs, vs)
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

    for r, c in ctrl_pts:
        out[r, c] = int(ctrl)
    return out


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


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    edges = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, c_split))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else W - 1))
            has_h = True

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        elif "top" in edges and "bottom" not in edges:
            dh = "T"
        elif "bottom" in edges and "top" not in edges:
            dh = "B"
        else:
            dh = "T"

        if c_split is not None:
            dv = vs
        elif "left" in edges and "right" not in edges:
            dv = "L"
        elif "right" in edges and "left" not in edges:
            dv = "R"
        else:
            dv = "L"
        return dh, dv

    def touches_E(hs: str, vs: str) -> bool:
        te = set()
        if hs in ("T", "A"):
            te.add("top")
        if hs in ("B", "A"):
            te.add("bottom")
        if vs in ("L", "A"):
            te.add("left")
        if vs in ("R", "A"):
            te.add("right")
        return bool(te & edges)

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

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = c_split - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = r_split - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
        r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
        c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    # Keep split-line fg in the input so the overwrite is visible in the output.
    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((r, int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

    n_split = len(split_pixels)
    if n_split == 0:
        return None

    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = target_total - n_split
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= vanish_budget
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets = []
    remaining = budget_pieces
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3))
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        if has_h:
            inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        else:
            split_rows = sorted(r for r, c in split_pixels if c == c_split)
            if split_rows:
                mid = (split_rows[0] + split_rows[-1]) // 2
                inp_r0 = max(r0, min(mid - ph // 2, r1 - ph + 1))
            else:
                inp_r0 = r0

        if has_v:
            inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        else:
            split_cols = sorted(c for r, c in split_pixels if r == r_split)
            if split_cols:
                mid = (split_cols[0] + split_cols[-1]) // 2
                inp_c0 = max(c0, min(mid - pw // 2, c1 - pw + 1))
            else:
                inp_c0 = c0

        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)

        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        output_pieces[reg] = out_pts

    all_inp_fg = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
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

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
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
            vanish_fg = vpts

    combined = all_inp_fg | vanish_fg
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

    # Draw solid rays using the input split pixels as the stopping evidence.
    for cr, cc in ctrl_pts:
        stop = _ray_stop(inp, fg, cr, cc)
        if stop is None:
            return None
        _paint_continuous_ray(out, ctrl_color, cr, cc, stop)
    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color

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
