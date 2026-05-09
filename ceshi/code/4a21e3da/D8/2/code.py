# Variant of 4a21e3da
# Dimension: D8 — 控制像素数量与分布 (meta)
# Change: 1或2个控制像素，2个时在垂直边缘（上/下之一和左/右之一）... → 同一边缘两个控制像素

from __future__ import annotations

"""
Design analysis for the D8 variant
==================================

a) Concrete behavior and execution trace
YAML-style summary:
  input:
    - square grid, bg/fg/ctrl colors stay the same as the seed task
    - one ctrl pixel is still allowed
    - when there are two ctrl pixels, they lie on the same edge:
      top/bottom -> two distinct columns -> two parallel vertical rays
      left/right -> two distinct rows -> two parallel horizontal rays
    - the sprite remains a single connected fg component in the input
    - fg pixels placed on split lines remain in place in the output, exactly as before
  output:
    - the rays are drawn inward on bg cells only
    - with one split line: the two outer pieces survive and slide to the two corners on the ctrl edge
    - with two parallel split lines: the sprite is divided into three stripes
      outer stripes survive and slide to the two corners on the ctrl edge
      middle stripe disappears completely

Small vertical example:
  original single-top-control logic:
    top ctrl at c=8 makes one vertical cut
    fg left of c=8 slides to top-left, fg right of c=8 slides to top-right
  new double-top-control logic:
    top ctrls at c=7 and c=11 make two vertical cuts
    fg left of c=7 slides to top-left
    fg between c=7 and c=11 vanishes
    fg right of c=11 slides to top-right
    fg sitting on c=7 or c=11 stays put
  key difference:
    the old code only had one c_split, so it could only classify L/R.
    The new variant needs L/M/R (or T/M/B) classification and a special fate for M.

b) Complexity calibration
  - The middle stripe is the only new branch, which keeps the rule compact.
  - The generator fixes all produced samples to the two-control case so every example visibly exhibits
    the D8 change instead of occasionally falling back to the source-task look.
  - Only one axis is active per sample. This avoids a 3x3 region system, which would add too much free
    variation for 2-4 ARC examples.

c) Implicit-assumption audit of the original main()
  - `c_split` / `r_split` were scalar slots. That silently overwrote earlier ctrl positions and breaks as
    soon as two same-edge controls appear. The new solver uses sorted split lists.
  - Region labels were binary (`L/R` or `T/B`). The variant needs ternary labels on the active axis.
  - `touches_edges()` was safe before because regions spanned the inactive axis. In the variant, using
    that logic would wrongly keep the middle stripe, since it still spans the control edge. The new rule
    therefore decides survival directly from stripe position: only outer stripes survive.
  - The ray drawing loop itself is still valid because it already iterates per control pixel.

d) Placement legality guaranteed by the generator
  - All fg stays at least 4 cells away from every grid edge in the input.
  - Split lines pass through the sprite interior, never on the bbox boundary.
  - For two-control cases, the middle stripe has positive width/height and contains its own fg piece.
  - Mandatory bridge cells force the input sprite to be a single 4-connected component.
  - The solver assumptions are matched exactly: one active axis, one edge family, 1 or 2 splits.

e) Visual observability
  - The output clearly shows two parallel rays on the same edge.
  - The outer pieces move to two same-edge corners, while the center stripe disappears.
  - Because the middle stripe is present in the input and absent in the output, the variant is visible
    even from a small number of examples.

f) Visual cleanliness
  - Only three colors are used, as in the source task.
  - The sprite remains centered with generous margins, so rays and corner-slid pieces do not visually
    collide with unrelated structure.
  - The middle stripe is a compact connected piece instead of noisy scattered pixels.

g) Color inferability
  - No color remapping is introduced.
  - Color identity is not used for hidden ordering; only visible geometry matters.

Post-edit solver assumptions to audit against the generator
  - Generated samples always contain two controls on the same active edge with distinct split
    coordinates. The solver still accepts the one-control subset.
  - Every split line used by a ctrl contains fg somewhere inside the sprite, so the ray has a stopping
    point. The generator places 2-4 fg pixels on every split line.
  - Each surviving stripe is treated as one translated fg set. The generator builds exactly one connected
    piece per stripe, so this grouping is correct.
"""

import random
import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _edge_of_ctrl(H: int, W: int, r: int, c: int) -> str:
    if r == 0:
        return "top"
    if r == H - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == W - 1:
        return "right"
    raise ValueError("control pixel must lie on the border")


def _draw_ray(out: np.ndarray, g: np.ndarray, bg: int, fg: int, ctrl: int, r: int, c: int) -> None:
    H, W = g.shape
    if r == 0:
        rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
        if rows:
            for rr in range(0, max(rows) + 1):
                if int(g[rr, c]) == bg:
                    out[rr, c] = ctrl
    elif r == H - 1:
        rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
        if rows:
            for rr in range(H - 1, min(rows) - 1, -1):
                if int(g[rr, c]) == bg:
                    out[rr, c] = ctrl
    elif c == 0:
        cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
        if cols:
            for cc in range(0, max(cols) + 1):
                if int(g[r, cc]) == bg:
                    out[r, cc] = ctrl
    elif c == W - 1:
        cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
        if cols:
            for cc in range(W - 1, min(cols) - 1, -1):
                if int(g[r, cc]) == bg:
                    out[r, cc] = ctrl


def _slide_points_to_corner(
    out: np.ndarray,
    pts: list[tuple[int, int]],
    H: int,
    W: int,
    fg: int,
    dh: str,
    dv: str,
) -> None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if dh == "T" else (H - 1 - rmax)
    dc = -cmin if dv == "L" else (W - 1 - cmax)
    for r, c in pts:
        out[r + dr, c + dc] = fg


def _legacy_main(g: np.ndarray, fg: int, ctrl: int, ctrl_pts: list[tuple[int, int]], out: np.ndarray) -> np.ndarray:
    H, W = g.shape
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

    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = fg

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
        return dh, dv

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue
        dh, dv = pick_dest(hs, vs)
        _slide_points_to_corner(out, pts, H, W, fg, dh, dv)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant:
    - one ctrl pixel: same as the source task for a single active axis
    - two ctrl pixels on the same edge: two parallel cuts; the middle stripe vanishes
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    edge_splits: dict[str, list[int]] = {"top": [], "bottom": [], "left": [], "right": []}
    for r, c in ctrl_pts:
        edge = _edge_of_ctrl(H, W, r, c)
        edge_splits[edge].append(c if edge in ("top", "bottom") else r)

    c_splits = sorted(edge_splits["top"] + edge_splits["bottom"])
    r_splits = sorted(edge_splits["left"] + edge_splits["right"])

    out = np.full((H, W), bg, dtype=int)
    for r, c in ctrl_pts:
        _draw_ray(out, g, bg, fg, ctrl, r, c)
    for r, c in ctrl_pts:
        out[r, c] = ctrl

    if c_splits and r_splits:
        return _legacy_main(g, fg, ctrl, ctrl_pts, out)

    split_cols = set(c_splits)
    split_rows = set(r_splits)
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if c in split_cols or r in split_rows:
            out[r, c] = fg

    if c_splits:
        active_edge = "top" if edge_splits["top"] else "bottom"
        assert len(split_cols) in (1, 2)
        regions: dict[str, list[tuple[int, int]]] = {}
        if len(c_splits) == 1:
            s = c_splits[0]
            for r, c in np.argwhere(g == fg):
                r = int(r)
                c = int(c)
                if c == s:
                    continue
                band = "L" if c < s else "R"
                regions.setdefault(band, []).append((r, c))
            keep_to_dest = {
                "L": ("T" if active_edge == "top" else "B", "L"),
                "R": ("T" if active_edge == "top" else "B", "R"),
            }
        else:
            s1, s2 = c_splits
            for r, c in np.argwhere(g == fg):
                r = int(r)
                c = int(c)
                if c == s1 or c == s2:
                    continue
                if c < s1:
                    band = "L"
                elif c > s2:
                    band = "R"
                else:
                    band = "M"
                regions.setdefault(band, []).append((r, c))
            keep_to_dest = {
                "L": ("T" if active_edge == "top" else "B", "L"),
                "R": ("T" if active_edge == "top" else "B", "R"),
            }

        for band, pts in regions.items():
            if band not in keep_to_dest:
                continue
            dh, dv = keep_to_dest[band]
            _slide_points_to_corner(out, pts, H, W, fg, dh, dv)
        return out

    active_edge = "left" if edge_splits["left"] else "right"
    assert len(split_rows) in (1, 2)
    regions_h: dict[str, list[tuple[int, int]]] = {}
    if len(r_splits) == 1:
        s = r_splits[0]
        for r, c in np.argwhere(g == fg):
            r = int(r)
            c = int(c)
            if r == s:
                continue
            band = "T" if r < s else "B"
            regions_h.setdefault(band, []).append((r, c))
        keep_to_dest_h = {
            "T": ("T", "L" if active_edge == "left" else "R"),
            "B": ("B", "L" if active_edge == "left" else "R"),
        }
    else:
        s1, s2 = r_splits
        for r, c in np.argwhere(g == fg):
            r = int(r)
            c = int(c)
            if r == s1 or r == s2:
                continue
            if r < s1:
                band = "T"
            elif r > s2:
                band = "B"
            else:
                band = "M"
            regions_h.setdefault(band, []).append((r, c))
        keep_to_dest_h = {
            "T": ("T", "L" if active_edge == "left" else "R"),
            "B": ("B", "L" if active_edge == "left" else "R"),
        }

    for band, pts in regions_h.items():
        if band not in keep_to_dest_h:
            continue
        dh, dv = keep_to_dest_h[band]
        _slide_points_to_corner(out, pts, H, W, fg, dh, dv)
    return out


def _grow_piece_in_box(sub_h: int, sub_w: int, target_px: int, must: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    cells = set(must)
    if not cells:
        cells.add((0, 0))
    while len(cells) < target_px:
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < sub_h and 0 <= nc < sub_w and (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        if not neighbors:
            break
        cells.add(random.choice(sorted(neighbors)))
    return frozenset(cells)


def _bbox_of_points(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _build_vertical_case(num_ctrl: int, edge: str) -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    sprite_h = random.randint(6, 8)
    if num_ctrl == 1:
        sprite_w = random.randint(8, 10)
        local_split = random.randint(3, sprite_w - 4)
        widths = [local_split, sprite_w - local_split - 1]
        split_cols_local = [local_split]
    else:
        sprite_w = random.randint(9, 11)
        possible = []
        for s1 in range(2, sprite_w - 4):
            for s2 in range(s1 + 3, sprite_w - 2):
                left_w = s1
                mid_w = s2 - s1 - 1
                right_w = sprite_w - s2 - 1
                if min(left_w, mid_w, right_w) >= 2:
                    possible.append((s1, s2))
        if not possible:
            return None
        s1, s2 = random.choice(possible)
        widths = [s1, s2 - s1 - 1, sprite_w - s2 - 1]
        split_cols_local = [s1, s2]

    sprite_r0 = random.randint(4, H - sprite_h - 4)
    bridge_len = random.randint(2, min(4, sprite_h))
    bridge_r0 = random.randint(sprite_r0 + 1, sprite_r0 + sprite_h - bridge_len - 1)

    valid_c0 = []
    for sprite_c0 in range(4, W - sprite_w - 3):
        global_splits = [sprite_c0 + s for s in split_cols_local]
        if num_ctrl == 1:
            s = global_splits[0]
            if int(W * 0.35) <= s <= int(W * 0.65):
                valid_c0.append(sprite_c0)
        else:
            mid = sum(global_splits) / 2
            if int(W * 0.3) <= global_splits[0] < global_splits[1] <= int(W * 0.7) and int(W * 0.4) <= mid <= int(W * 0.6):
                valid_c0.append(sprite_c0)
    if not valid_c0:
        return None
    sprite_c0 = random.choice(valid_c0)

    bridge_rows = list(range(bridge_r0, bridge_r0 + bridge_len))
    split_pixels = {(r, sprite_c0 + s) for s in split_cols_local for r in bridge_rows}

    pieces: list[set[tuple[int, int]]] = []

    left_w = widths[0]
    left_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * left_w, bridge_len + 6))
    left_must = {(r - sprite_r0, left_w - 1) for r in bridge_rows}
    left_local = _grow_piece_in_box(sprite_h, left_w, left_target, left_must)
    pieces.append({(sprite_r0 + r, sprite_c0 + c) for r, c in left_local})

    if num_ctrl == 2:
        mid_w = widths[1]
        mid_c0 = sprite_c0 + split_cols_local[0] + 1
        mid_row = random.choice(bridge_rows) - sprite_r0
        mid_must = {(mid_row, c) for c in range(mid_w)}
        mid_target = random.randint(max(mid_w + 1, 4), min(sprite_h * mid_w, mid_w + 5))
        mid_local = _grow_piece_in_box(sprite_h, mid_w, mid_target, mid_must)
        pieces.append({(sprite_r0 + r, mid_c0 + c) for r, c in mid_local})

    right_w = widths[-1]
    right_c0 = sprite_c0 + sprite_w - right_w
    right_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * right_w, bridge_len + 6))
    right_must = {(r - sprite_r0, 0) for r in bridge_rows}
    right_local = _grow_piece_in_box(sprite_h, right_w, right_target, right_must)
    pieces.append({(sprite_r0 + r, right_c0 + c) for r, c in right_local})

    combined: set[tuple[int, int]] = set(split_pixels)
    for pts in pieces:
        if combined & pts:
            return None
        combined |= pts

    if not combined:
        return None

    b0, b1, b2, b3 = _bbox_of_points(combined)
    bbox_h = b2 - b0 + 1
    bbox_w = b3 - b1 + 1
    density = len(combined) / (bbox_h * bbox_w)
    if not (5 <= bbox_h <= 8 and 8 <= bbox_w <= 11 and 0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg

    ctrl_row = 0 if edge == "top" else H - 1
    ctrl_pts = [(ctrl_row, sprite_c0 + s) for s in split_cols_local]
    for r, c in ctrl_pts:
        inp[r, c] = ctrl

    out = main(inp)

    out_fg = set(map(tuple, np.argwhere(out == fg)))
    inp_fg = set(map(tuple, np.argwhere(inp == fg)))
    split_fg = split_pixels
    if not split_fg.issubset(out_fg):
        return None

    if num_ctrl == 2:
        middle_cols = range(sprite_c0 + split_cols_local[0] + 1, sprite_c0 + split_cols_local[1])
        middle_input = {(r, c) for r, c in inp_fg if c in middle_cols}
        middle_output = {(r, c) for r, c in out_fg if c in middle_cols}
        if len(middle_input) < 4 or middle_output:
            return None

    return inp


def _gen_attempt() -> np.ndarray | None:
    num_ctrl = 2
    final_edge = random.choice(["top", "bottom", "left", "right"])

    if final_edge in ("top", "bottom"):
        inp = _build_vertical_case(num_ctrl, final_edge)
        if inp is None:
            return None
        return inp

    mapped_edge = "top" if final_edge == "left" else "bottom"
    inp = _build_vertical_case(num_ctrl, mapped_edge)
    if inp is None:
        return None
    return np.array(inp.T, dtype=int)


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
