# Combo variant of 4a21e3da (index 15)
# combo_spec: D8/2+D9/1
# Name: 同侧双控制像素+分割线像素消失

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


"""
Design analysis for the combo variant
-------------------------------------

a) Concrete behavior and execution trace
  - Keep the source-task vocabulary: one connected fg sprite, bg background, ctrl
    pixels on the border, and intermittent rays that travel inward on bg cells only.
  - Combo input always uses two ctrl pixels on the same top/bottom edge, at two
    distinct columns c1 < c2. These induce two parallel vertical cuts.
  - Small trace:
      input:
        top ctrls at c=7 and c=11
        sprite spans across both split columns
        fg exists in three vertical bands: left of c=7, between 7 and 11, right of 11
        fg also exists on c=7 and c=11 themselves, so the input stays connected and
        both rays have stopping witnesses
      source D8/2-only behavior:
        left band slides to top-left
        middle band vanishes
        right band slides to top-right
        fg on c=7 and c=11 stays in place
      fused combo behavior:
        left band still slides to the left corner on the control edge
        middle band still vanishes
        right band still slides to the right corner on the control edge
        but fg on c=7 and c=11 now also disappears
    Difference:
      the old same-edge-double-control variant left two stationary fg divider
      columns behind; the combo removes those divider pixels completely, so the
      middle region is detached from both sides by blank cut columns.

b) Complexity calibration
  - To keep the rule legible from 2-4 ARC examples, the generator always exhibits
    the combo branch: exactly two same-edge controls, one active axis, and a non-empty
    middle stripe.
  - Orientation is restricted to top/bottom controls, so every training example shows
    the same left/middle/right partition instead of mixing in a transposed top/middle/
    bottom version.
  - Each split column carries 2-4 fg anchor pixels, and the middle stripe has at least
    4 fg cells, so both "parallel cuts" and "cut-line disappearance" are visible.

c) Implicit-assumption audit of the original solver
  - The seed solver stored only one `c_split` and one `r_split`; that silently breaks
    when two controls sit on the same edge. The combo solver therefore tracks sorted
    split lists and detects the parallel-cut case explicitly.
  - The seed solver assumed split-line fg survives and wrote it back into the output.
    D9/1 invalidates that assumption, so the write-back block must be removed.
  - The original `touches_edges()` logic is correct for one split or perpendicular
    two-split cases, but it would wrongly keep the middle stripe in the same-edge case
    because the middle stripe still spans the top/bottom edge family. The combo solver
    bypasses `touches_edges()` there and decides survival by stripe class: only outer
    bands survive.
  - Ray drawing itself remains valid because it already scans the input fg on each
    controlled line; the combo only changes what survives afterward.

d) Placement legality guaranteed by the generator
  - The sprite stays at least 4 cells away from every border except for the ctrl pixels.
  - Both split columns pass through the sprite interior, never through its bbox boundary.
  - Left, middle, and right stripe widths are all at least 2, so every band can host a
    visible connected piece without overlap.
  - Mandatory bridge cells on both split columns and both outer stripes guarantee the
    input fg is a single 4-connected component.
  - The output corner pieces fit strictly on their own sides of the split columns, so
    they cannot overwrite the central cut columns.

e) Visual observability
  - Every sample visibly contains two ctrl pixels on the same edge and two parallel rays.
  - The output differs from the D8-only same-edge variant in a human-visible way: the
    stationary fg on the two cut columns is gone.
  - Because the middle stripe is also present in the input but absent in the output,
    the solver must infer both the three-band partition and the disappearing cut columns.

f) Visual cleanliness
  - Only the original three colors are used.
  - The sprite stays compact and centered, while the surviving pieces land in opposite
    corners on the same edge, so the moved pieces and the empty cut columns do not
    visually collide.
  - The generator uses contiguous stripe pieces rather than scattered noise, making the
    left/middle/right decomposition readable.

g) Change inferability
  - No recoloring is introduced, so the entire combo is inferred from visible geometry:
    same-edge double controls, two parallel cuts, middle stripe disappears, and cut-line
    fg also disappears.
  - The rule does not depend on hidden color ordering or numeric color ids.

Post-edit solver assumptions audited against the generator
  - The combo branch assumes exactly two split columns on one horizontal edge family and
    no horizontal split row. The generator enforces that.
  - The combo branch assumes each split column contains input fg, so each ray has a stop.
    The generator places 2-4 split-line fg per column.
  - The combo branch treats each outer stripe as a single translated set. The generator
    creates exactly one connected piece per stripe.
  - The D9 part assumes split-line fg may vanish entirely from the output. The generator
    verifies that output fg on both split columns is exactly zero.
"""


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


def _legacy_d9_main(g: np.ndarray, fg: int, ctrl_pts: list[tuple[int, int]], out: np.ndarray) -> np.ndarray:
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
        return dh, dv

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue
        dh, dv = pick_dest(hs, vs)
        _slide_points_to_corner(out, pts, H, W, fg, dh, dv)
    return out


def _solve_parallel_same_edge(
    g: np.ndarray,
    fg: int,
    edge_splits: dict[str, list[int]],
    out: np.ndarray,
) -> bool:
    H, W = g.shape
    c_splits = sorted(edge_splits["top"] + edge_splits["bottom"])
    r_splits = sorted(edge_splits["left"] + edge_splits["right"])

    # Two vertical split columns from the same top/bottom edge.
    if (
        len(c_splits) == 2
        and not r_splits
        and bool(edge_splits["top"]) != bool(edge_splits["bottom"])
    ):
        active_edge = "top" if edge_splits["top"] else "bottom"
        s1, s2 = c_splits
        regions: dict[str, list[tuple[int, int]]] = {}
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
        return True

    # Optional symmetric support for left/right same-edge controls.
    if (
        len(r_splits) == 2
        and not c_splits
        and bool(edge_splits["left"]) != bool(edge_splits["right"])
    ):
        active_edge = "left" if edge_splits["left"] else "right"
        s1, s2 = r_splits
        regions: dict[str, list[tuple[int, int]]] = {}
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
            regions.setdefault(band, []).append((r, c))

        keep_to_dest = {
            "T": ("T", "L" if active_edge == "left" else "R"),
            "B": ("B", "L" if active_edge == "left" else "R"),
        }
        for band, pts in regions.items():
            if band not in keep_to_dest:
                continue
            dh, dv = keep_to_dest[band]
            _slide_points_to_corner(out, pts, H, W, fg, dh, dv)
        return True

    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - two controls may define parallel same-edge cuts
    - the middle stripe vanishes
    - fg lying on the split line(s) also vanishes instead of staying in place
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

    out = np.full((H, W), bg, dtype=int)
    for r, c in ctrl_pts:
        _draw_ray(out, g, bg, fg, ctrl, r, c)
    for r, c in ctrl_pts:
        out[r, c] = ctrl

    if _solve_parallel_same_edge(g, fg, edge_splits, out):
        return out
    return _legacy_d9_main(g, fg, ctrl_pts, out)


def _grow_piece_in_box(
    sub_h: int,
    sub_w: int,
    target_px: int,
    must: set[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
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


def _paint_rays_from_split_pixels(
    out: np.ndarray,
    ctrl_pts: list[tuple[int, int]],
    split_pixels: set[tuple[int, int]],
    ctrl_color: int,
) -> None:
    H, W = out.shape
    for cr, cc in ctrl_pts:
        if cr == 0:
            rows = sorted(r for r, c in split_pixels if c == cc)
            if rows:
                for r in range(0, max(rows) + 1):
                    if (r, cc) not in split_pixels:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            rows = sorted(r for r, c in split_pixels if c == cc)
            if rows:
                for r in range(H - 1, min(rows) - 1, -1):
                    if (r, cc) not in split_pixels:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            cols = sorted(c for r, c in split_pixels if r == cr)
            if cols:
                for c in range(0, max(cols) + 1):
                    if (cr, c) not in split_pixels:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            cols = sorted(c for r, c in split_pixels if r == cr)
            if cols:
                for c in range(W - 1, min(cols) - 1, -1):
                    if (cr, c) not in split_pixels:
                        out[cr, c] = ctrl_color
    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color


def _build_vertical_case(edge: str) -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(9, 11)

    possible_splits: list[tuple[int, int]] = []
    for s1 in range(2, sprite_w - 4):
        for s2 in range(s1 + 3, sprite_w - 2):
            left_w = s1
            mid_w = s2 - s1 - 1
            right_w = sprite_w - s2 - 1
            if min(left_w, mid_w, right_w) >= 2:
                possible_splits.append((s1, s2))
    if not possible_splits:
        return None
    s1, s2 = random.choice(possible_splits)

    sprite_r0 = random.randint(4, H - sprite_h - 4)
    bridge_len = random.randint(2, min(4, sprite_h - 2))
    bridge_r0 = random.randint(sprite_r0 + 1, sprite_r0 + sprite_h - bridge_len - 1)

    valid_c0: list[int] = []
    for sprite_c0 in range(4, W - sprite_w - 3):
        g1 = sprite_c0 + s1
        g2 = sprite_c0 + s2
        mid = (g1 + g2) / 2
        if int(W * 0.3) <= g1 < g2 <= int(W * 0.7) and int(W * 0.4) <= mid <= int(W * 0.6):
            valid_c0.append(sprite_c0)
    if not valid_c0:
        return None
    sprite_c0 = random.choice(valid_c0)

    bridge_rows = list(range(bridge_r0, bridge_r0 + bridge_len))
    split_pixels = {(r, sprite_c0 + s) for s in (s1, s2) for r in bridge_rows}

    left_w = s1
    mid_w = s2 - s1 - 1
    right_w = sprite_w - s2 - 1

    left_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * left_w, bridge_len + 6))
    left_must = {(r - sprite_r0, left_w - 1) for r in bridge_rows}
    left_local = _grow_piece_in_box(sprite_h, left_w, left_target, left_must)

    mid_target = random.randint(max(mid_w + 2, 4), min(sprite_h * mid_w, mid_w + 6))
    mid_row = random.choice(bridge_rows) - sprite_r0
    mid_must = {(mid_row, c) for c in range(mid_w)}
    mid_local = _grow_piece_in_box(sprite_h, mid_w, mid_target, mid_must)

    right_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * right_w, bridge_len + 6))
    right_must = {(r - sprite_r0, 0) for r in bridge_rows}
    right_local = _grow_piece_in_box(sprite_h, right_w, right_target, right_must)

    left_pts = {(sprite_r0 + r, sprite_c0 + c) for r, c in left_local}
    mid_pts = {(sprite_r0 + r, sprite_c0 + s1 + 1 + c) for r, c in mid_local}
    right_pts = {(sprite_r0 + r, sprite_c0 + s2 + 1 + c) for r, c in right_local}

    combined = set(split_pixels)
    for pts in [left_pts, mid_pts, right_pts]:
        if combined & pts:
            return None
        combined |= pts

    if not combined:
        return None

    r0, c0, r1, c1 = _bbox_of_points(combined)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1
    density = len(combined) / (bbox_h * bbox_w)
    if not (5 <= bbox_h <= 8 and 9 <= bbox_w <= 11 and 0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    ctrl_row = 0 if edge == "top" else H - 1
    ctrl_pts = [(ctrl_row, sprite_c0 + s1), (ctrl_row, sprite_c0 + s2)]

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl

    out_expected = np.full((H, W), bg, dtype=int)
    keep_pieces = [list(left_pts), list(right_pts)]
    dests = [
        ("T" if edge == "top" else "B", "L"),
        ("T" if edge == "top" else "B", "R"),
    ]
    for pts, (dh, dv) in zip(keep_pieces, dests):
        _slide_points_to_corner(out_expected, pts, H, W, fg, dh, dv)
    _paint_rays_from_split_pixels(out_expected, ctrl_pts, split_pixels, ctrl)

    out = main(inp)
    if not np.array_equal(out, out_expected):
        return None

    out_fg = set(map(tuple, np.argwhere(out == fg)))
    g1 = sprite_c0 + s1
    g2 = sprite_c0 + s2
    split_input = {(r, c) for r, c in combined if c in (g1, g2)}
    split_output = {(r, c) for r, c in out_fg if c in (g1, g2)}
    middle_input = {(r, c) for r, c in combined if g1 < c < g2}
    middle_output = {(r, c) for r, c in out_fg if g1 < c < g2}

    if len(split_input) < 4 or split_output:
        return None
    if len(middle_input) < 4 or middle_output:
        return None

    edge_row = 0 if edge == "top" else H - 1
    if not any(r == edge_row and c < g1 for r, c in out_fg):
        return None
    if not any(r == edge_row and c > g2 for r, c in out_fg):
        return None

    return inp


def _gen_attempt() -> np.ndarray | None:
    return _build_vertical_case(random.choice(["top", "bottom"]))


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
