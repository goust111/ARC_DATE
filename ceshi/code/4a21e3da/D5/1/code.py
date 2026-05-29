# Variant of 4a21e3da
# Dimension: D5 — 切割后片段的变换规则 (transform)
# Change: 与控制边接壤的区域滑动到对应角落；不接壤的区域消失（erase）... → 片段镜像翻转后滑入角落
#
# Design analysis for the D5 variant
# a) Concretization / behavior trace
#    - The input remains the same family as 4a21e3da: one connected orange sprite,
#      one or two red control pixels on the border, and intermittent red rays drawn
#      only in the output.
#    - The change is only in Stage 2. Original behavior: each kept cut-piece is
#      translated so its bounding box becomes flush with the destination corner.
#      Variant behavior: before that corner placement, normalize the piece inside
#      its own bounding box and mirror it along the axes implied by the movement.
#      Since a corner move has both a vertical destination side and a horizontal
#      destination side, the piece is flipped vertically and horizontally, then
#      anchored to the corner. In shape space this is a 180-degree reorientation,
#      but it is implemented as composition of the two axis reflections.
#    - Small example: suppose a kept region contains an L-shape occupying
#      {(0,0),(1,0),(1,1)} in its local bbox. Original rule sends the same L to the
#      corner. Variant rule flips rows and cols inside the 2x2 bbox first, giving
#      {(0,0),(0,1),(1,1)}, then places that mirrored shape in the corner. The split
#      line fg still stays fixed, and any region not touching a control edge still
#      disappears.
# b) Complexity calibration
#    - The task still has only 1 or 2 controls and at most 4 cut regions. The new
#      freedom is restricted to one deterministic transform family: per-axis mirror
#      before corner placement. There is no color-dependent or piece-dependent branch.
#    - To keep the reflection visible, the generator rejects symmetric kept pieces
#      whose mirrored corner placement would match the legacy slide-only output.
# c) Hidden assumptions audit
#    - Original Stage 2 assumes "local pixel offsets are preserved" and directly
#      adds (dr, dc) to each pixel. That is invalid now because mirrored pieces must
#      reverse local row/col order before placement.
#    - The code that preserves split-line fg is still valid because those pixels are
#      explicitly excluded from regions and remain fixed in both old and new tasks.
#    - Ray drawing is unchanged: it only depends on control positions and the final
#      fg extent along the split line, not on the interior orientation of moved pieces.
# d) Placement legality
#    - The generator keeps the original centered sprite and subregion logic so every
#      input piece lies strictly inside its cut rectangle and the whole sprite stays
#      connected.
#    - Output pieces are produced by the same solver-side transform helper used in
#      main(), then checked to stay on the correct side of split lines.
# e) Visual observability
#    - With a single-color sprite, reflection is only visible through silhouette. The
#      generator therefore requires a measurable difference between mirrored output
#      and legacy slide-only output, preventing accidental degeneration to the base
#      task on symmetric pieces.
# f) Visual cleanliness
#    - Layers remain clean: border controls and rays are red, the sprite is orange,
#      and mirrored corner pieces occupy disjoint corners or corner-side regions.
# g) Color inference
#    - No new color logic is introduced; colors keep the original roles.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _normalize_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - rmin), int(c - cmin)) for r, c in pts)
    h = int(max(r for r, _ in shape) + 1)
    w = int(max(c for _, c in shape) + 1)
    return shape, h, w


def _flip_shape(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[tuple[int, int]]:
    out = set()
    for r, c in shape:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.add((int(rr), int(cc)))
    return frozenset(out)


def _place_shape_in_corner(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _mirror_then_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    shape, h, w = _normalize_points(pts)
    mirrored = _flip_shape(shape, h, w, flip_v=(dh in ("T", "B")), flip_h=(dv in ("L", "R")))
    return _place_shape_in_corner(mirrored, h, w, dh, dv, H, W)


def _slide_only_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    shape, h, w = _normalize_points(pts)
    return _place_shape_in_corner(shape, h, w, dh, dv, H, W)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and move surviving pieces to corners.

    Variant rule:
    - Draw rays inward from each ctrl pixel along its row/col, painting only bg
      cells up to the farthest fg pixel (intermittent).
    - Rays partition the grid; fg on the cut line stays in place.
    - Pieces whose region touches a control edge survive. Before sliding into the
      destination corner, each surviving piece is mirrored inside its own bbox
      along the axes implied by the corner move; non-touching pieces vanish.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) == 0:
            E.add("top")
            c_split = int(c)
        elif int(r) == int(H - 1):
            E.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            E.add("left")
            r_split = int(r)
        elif int(c) == int(W - 1):
            E.add("right")
            r_split = int(r)

    out = np.full((H, W), int(bg), dtype=int)

    for r, c in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(r) == int(H - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)
        elif int(c) == int(W - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

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

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue
        dh, dv = pick_dest(hs, vs)
        for rr, cc in _mirror_then_corner(frozenset(pts), dh, dv, H, W):
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if (int(nr), int(nc)) not in cells:
                    neighbors.add((int(nr), int(nc)))
        valid = [
            (int(nr), int(nc))
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(int(nr), int(nc))}) - min(r for r, _ in cells | {(int(nr), int(nc))}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(int(nr), int(nc))}) - min(c for _, c in cells | {(int(nr), int(nc))}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 12])[0]
    if int(num_ctrl) == 1:
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
            ctrl_pts.append((0 if edge == "top" else int(H - 1), int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else int(W - 1)))
            has_h = True

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        elif "top" in E and "bottom" not in E:
            dh = "T"
        elif "bottom" in E and "top" not in E:
            dh = "B"
        else:
            dh = "T"
        if c_split is not None:
            dv = vs
        elif "left" in E and "right" not in E:
            dv = "L"
        elif "right" in E and "left" not in E:
            dv = "R"
        else:
            dv = "L"
        return str(dh), str(dv)

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
        return bool(te & E)

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
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(int(sprite_c0), int(W - 5 - sprite_w + 1)))
        if sprite_c0 > int(W - 5 - sprite_w + 1):
            return None
        if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
            return None
    else:
        hi = int(W - 5 - sprite_w + 1)
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(int(sprite_r0), int(H - 5 - sprite_h + 1)))
        if sprite_r0 > int(H - 5 - sprite_h + 1):
            return None
        if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
            return None
    else:
        hi = int(H - 5 - sprite_h + 1)
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else r_split + 1)
        r1 = int((r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else c_split + 1)
        c1 = int((c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [int(r) for r in range(int(sprite_r0), int(sprite_r1) + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [int(c) for c in range(int(sprite_c0), int(sprite_c1) + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = int(target_total - n_split)
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < int(n_kept * 3):
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == int(n_kept - 1):
            px = int(remaining)
        else:
            avg = int(remaining // (n_kept - i))
            lo_px = int(max(3, avg - 2))
            hi_px = int(min(max_area, avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = int(max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3)))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    legacy_output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = int(max(r for r, _ in piece) + 1)
        pw = int(max(c for _, c in piece) + 1)
        if ph < 2 or pw < 2:
            return None

        if has_h:
            inp_r0 = int((r1 - ph + 1) if hs == "T" else r0)
        else:
            split_rows = sorted(int(r) for r, c in split_pixels if int(c) == int(c_split))
            if split_rows:
                mid = int((split_rows[0] + split_rows[-1]) // 2)
                inp_r0 = int(max(r0, min(mid - ph // 2, r1 - ph + 1)))
            else:
                inp_r0 = int(r0)

        if has_v:
            inp_c0 = int((c1 - pw + 1) if vs == "L" else c0)
        else:
            split_cols = sorted(int(c) for r, c in split_pixels if int(r) == int(r_split))
            if split_cols:
                mid = int((split_cols[0] + split_cols[-1]) // 2)
                inp_c0 = int(max(c0, min(mid - pw // 2, c1 - pw + 1)))
            else:
                inp_c0 = int(c0)

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                return None
            if has_v and ((vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split))):
                return None
            if has_h and ((hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split))):
                return None

        input_pieces[reg] = inp_pts

        out_pts = _mirror_then_corner(inp_pts, dh, dv, H, W)
        legacy_pts = _slide_only_corner(inp_pts, dh, dv, H, W)
        if out_pts == legacy_pts:
            return None

        for r, c in out_pts:
            if has_v and ((vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split))):
                return None
            if has_h and ((hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split))):
                return None
        output_pieces[reg] = out_pts
        legacy_output_pieces[reg] = legacy_pts

    variant_delta = sum(len(output_pieces[reg] ^ legacy_output_pieces[reg]) for reg in kept)
    if variant_delta < 4:
        return None

    all_inp_fg: set[tuple[int, int]] = set()
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
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = int(vr1 - vr0 + 1), int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = int(min(vanish_budget, vh * vw))
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = int(max(r for r, _ in piece) + 1)
            ppw = int(max(c for _, c in piece) + 1)
            vr_off = int((vr1 - pph + 1) if vhs == "T" else vr0)
            vc_off = int((vc1 - ppw + 1) if vvs == "L" else vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                    return None
                if (int(r), int(c)) in all_inp_fg:
                    return None
            vanish_fg = vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc in ctrl_pts:
        if int(cr) == 0:
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(max(fg_rows)) + 1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cr) == int(H - 1):
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(H - 1), int(min(fg_rows)) - 1, -1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cc) == 0:
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(max(fg_cols)) + 1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif int(cc) == int(W - 1):
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(W - 1), int(min(fg_cols)) - 1, -1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
