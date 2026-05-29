# Combo variant of 4a21e3da (index 19)
# combo_spec: D13/1+D15/0
# Name: 按面积筛选保留+不对称切割

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
  - Original task:
    - 1 or 2 border control pixels define one vertical split and/or one
      horizontal split.
    - The sprite is cut by those split lines.
    - Split-line fg stays in place.
    - Pieces in regions touching a control edge survive and slide to the
      corresponding corner; the rest vanish.
  - Combo task:
    - The cut geometry is no longer roughly central. Each active split is
      pushed near one side: 20-30% or 70-80% of the sprite span.
    - This creates one thin region and one much larger region on each active
      axis.
    - After partitioning, keep the largest N rectangular regions, where
      N = number of control pixels. Selection is by region area, not by whether
      the region touches a marked border.
    - Surviving pieces still slide using the original corner rule that depends
      on their side of the split(s).
  - Small trace:
    - Suppose there are two controls, top + left.
    - The vertical split is near the left side, so widths are 2 and 7.
    - The horizontal split is near the top, so heights are 1 and 5.
    - Region areas become TL=2, TR=7, BL=10, BR=35.
    - Original task would keep TL/TR/BL because those touch top or left edges.
    - Combo task keeps only the largest N=2 regions: BR and BL.
    - The solver must therefore infer asymmetric cutting first, then compare
      the resulting rectangle areas.

b) Complexity calibration
  - The asymmetry is discrete and repeated, not free-form: width-small-side is
    always 2 columns; height-small-side is 1 row for sprite heights 5-6 and
    2 rows for height 8. That keeps the pattern learnable from a few examples.
  - Each region contains a sparse connected piece that spans its whole region
    bbox, so humans can see region extent rather than just raw fg count.
  - Every geometrical region is populated, preventing silent degeneration where
    the "dropped" region had no fg anyway.
  - Tied area rankings at the keep boundary are rejected in the generator.

c) Implicit-assumption audit against the original solver
  - Original assumption: `touches_edges(hs, vs) & E` determines survival.
    This is invalid now; survival depends on area ranking instead.
  - Original assumption: splits near the center make region areas comparable.
    This is invalid now; the generator must enforce strongly asymmetric splits.
  - Original assumption: arbitrary occupancy inside a region is acceptable.
    For this combo we need each region's bbox to be visually recoverable, so the
    generator uses spanning pieces in every region.
  - New solver assumption: if a region is present, its side relative to the
    split is unambiguous from pixel coordinates. The generator guarantees at
    most one vertical and one horizontal split and keeps all region fg strictly
    off the split line(s).
  - New solver assumption: selecting the top-N areas should be deterministic.
    The generator rejects cases with a tie at the keep/drop boundary.

d) Placement legality
  - Grid stays square, 18-20 on a side, with margin >= 4 around the sprite.
  - Control pixels remain on distinct border edges, with at most one vertical
    and one horizontal split.
  - Each region piece is placed entirely inside its own sub-rectangle and
    touches the side(s) adjacent to the split so that the whole input sprite is
    connected through split-line fg.
  - After corner translation, kept pieces still remain on their own side of the
    split and cannot overflow because translation only moves them outward.

e) Visual observability
  - The thin-vs-wide split is obvious: a 2-column strip versus a much wider
    remainder, and similarly 1/2 rows versus a much taller remainder.
  - Because each region piece spans its sub-bbox, the intended comparison is
    "rectangle area", not color id or hidden metadata.
  - With two controls, the new outputs visibly differ from the original rule:
    the original would keep 3 regions, while this combo keeps only 2.
  - With one control, the original would keep both sides, while this combo
    keeps only the single larger side.

f) Visual cleanliness
  - Only the original three colors are used.
  - Split-line fg is capped at 2-4 per active split, keeping the center legible.
  - Region pieces are single connected sparse components, so the sprite looks
    structured instead of noisy even when all regions are populated.

g) Color inference
  - No color remapping occurs. Colors keep their original semantic roles:
    bg, fg sprite, ctrl rays.
"""


REGION_ORDER = {
    ("T", "L"): 0,
    ("T", "R"): 1,
    ("B", "L"): 2,
    ("B", "R"): 3,
    ("A", "L"): 4,
    ("A", "R"): 5,
    ("T", "A"): 6,
    ("B", "A"): 7,
}


def _all_regions(has_h: bool, has_v: bool) -> list[tuple[str, str]]:
    if has_h and has_v:
        return [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    if has_v:
        return [("A", "L"), ("A", "R")]
    return [("T", "A"), ("B", "A")]


def _pick_dest(
    hs: str,
    vs: str,
    *,
    r_split: int | None,
    c_split: int | None,
    edges: set[str],
) -> tuple[str, str]:
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
    return dh, dv


def _region_area(
    hs: str,
    vs: str,
    *,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> int:
    if r_split is None:
        h = sprite_r1 - sprite_r0 + 1
    elif hs == "T":
        h = r_split - sprite_r0
    else:
        h = sprite_r1 - r_split

    if c_split is None:
        w = sprite_c1 - sprite_c0 + 1
    elif vs == "L":
        w = c_split - sprite_c0
    else:
        w = sprite_c1 - c_split
    return int(h * w)


def _select_largest_regions(
    present_regions: list[tuple[str, str]],
    *,
    num_keep: int,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> set[tuple[str, str]]:
    scored: list[tuple[int, int, tuple[str, str]]] = []
    for hs, vs in present_regions:
        area = _region_area(
            hs,
            vs,
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c1,
            r_split=r_split,
            c_split=c_split,
        )
        scored.append((-area, REGION_ORDER[(hs, vs)], (hs, vs)))
    scored.sort()
    return {reg for _, _, reg in scored[: min(num_keep, len(scored))]}


def _draw_rays(
    out: np.ndarray,
    g: np.ndarray,
    ctrl_pts: list[tuple[int, int]],
    *,
    fg: int,
    ctrl: int,
    bg: int,
) -> None:
    H, W = out.shape
    for r, c in ctrl_pts:
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
    for r, c in ctrl_pts:
        out[r, c] = ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Rays and split-line preservation follow the original task.
    - The active split(s) partition the sprite bbox into 2 or 4 rectangles.
    - Let N be the number of control pixels.
    - Keep only the N largest regions by rectangle area.
    - Each surviving piece slides to the same destination corner it would have
      used in the original task; the rest vanish.
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
    num_keep = len(ctrl_pts)

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

    out = np.full((H, W), bg, dtype=int)
    _draw_rays(out, g, ctrl_pts, fg=fg, ctrl=ctrl, bg=bg)

    fg_coords = np.argwhere(g == fg)
    sprite_r0 = int(fg_coords[:, 0].min())
    sprite_r1 = int(fg_coords[:, 0].max())
    sprite_c0 = int(fg_coords[:, 1].min())
    sprite_c1 = int(fg_coords[:, 1].max())

    for r, c in fg_coords:
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in fg_coords:
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

    selected = _select_largest_regions(
        list(regions.keys()),
        num_keep=num_keep,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        r_split=r_split,
        c_split=c_split,
    )

    for (hs, vs), pts in regions.items():
        if (hs, vs) not in selected:
            continue
        dh, dv = _pick_dest(hs, vs, r_split=r_split, c_split=c_split, edges=edges)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
        dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
        for r, c in pts:
            out[r + dr, c + dc] = fg

    return out


def _make_spanning_piece(
    hs: str,
    vs: str,
    *,
    sub_h: int,
    sub_w: int,
    has_h: bool,
    has_v: bool,
    edges: set[str],
) -> frozenset[tuple[int, int]]:
    if has_h:
        row_toward_split = sub_h - 1 if hs == "T" else 0
    else:
        row_toward_split = 0 if ("top" in edges and "bottom" not in edges) else sub_h - 1

    if has_v:
        col_toward_split = sub_w - 1 if vs == "L" else 0
    else:
        col_toward_split = 0 if ("left" in edges and "right" not in edges) else sub_w - 1

    cells = {(row_toward_split, c) for c in range(sub_w)} | {(r, col_toward_split) for r in range(sub_h)}

    extra_candidates = []
    for r in range(sub_h):
        for c in range(sub_w):
            if (r, c) in cells:
                continue
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells):
                extra_candidates.append((r, c))
    random.shuffle(extra_candidates)
    extra_budget = min(len(extra_candidates), (sub_h * sub_w) // 8)
    for r, c in extra_candidates[:extra_budget]:
        cells.add((r, c))

    return frozenset(cells)


def _layout_regions(
    *,
    sprite_r0: int,
    sprite_c0: int,
    sprite_h: int,
    sprite_w: int,
    r_split: int | None,
    c_split: int | None,
    has_h: bool,
    has_v: bool,
) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    return {reg: sub_bbox(*reg) for reg in _all_regions(has_h, has_v)}


def _make_split_pixels(
    *,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    has_h: bool,
    has_v: bool,
    r_split: int | None,
    c_split: int | None,
) -> set[tuple[int, int]]:
    split_pixels: set[tuple[int, int]] = set()

    if has_v:
        assert c_split is not None
        if has_h:
            assert r_split is not None
            top_rows = list(range(sprite_r0, r_split))
            bottom_rows = list(range(r_split + 1, sprite_r1 + 1))
            chosen = {random.choice(top_rows), random.choice(bottom_rows)}
            remaining = [r for r in top_rows + bottom_rows if r not in chosen]
            extra_n = random.randint(0, min(2, len(remaining)))
            if extra_n:
                chosen |= set(random.sample(remaining, extra_n))
            for r in chosen:
                split_pixels.add((r, c_split))
        else:
            rows = list(range(sprite_r0, sprite_r1 + 1))
            n = random.randint(2, min(4, len(rows)))
            for r in random.sample(rows, n):
                split_pixels.add((r, c_split))

    if has_h:
        assert r_split is not None
        if has_v:
            assert c_split is not None
            left_cols = list(range(sprite_c0, c_split))
            right_cols = list(range(c_split + 1, sprite_c1 + 1))
            chosen = {random.choice(left_cols), random.choice(right_cols)}
            remaining = [c for c in left_cols + right_cols if c not in chosen]
            extra_n = random.randint(0, min(2, len(remaining)))
            if extra_n:
                chosen |= set(random.sample(remaining, extra_n))
            for c in chosen:
                split_pixels.add((r_split, c))
        else:
            cols = list(range(sprite_c0, sprite_c1 + 1))
            n = random.randint(2, min(4, len(cols)))
            for c in random.sample(cols, n):
                split_pixels.add((r_split, c))

    return split_pixels


def _expected_output(
    *,
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl_color: int,
    ctrl_pts: list[tuple[int, int]],
    edges: set[str],
    r_split: int | None,
    c_split: int | None,
    selected_regions: set[tuple[str, str]],
    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]],
    split_pixels: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((H, W), bg, dtype=int)
    for reg in selected_regions:
        pts = input_pieces[reg]
        dh, dv = _pick_dest(reg[0], reg[1], r_split=r_split, c_split=c_split, edges=edges)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
        dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
        for r, c in pts:
            out[r + dr, c + dc] = fg
    for r, c in split_pixels:
        out[r, c] = fg
    ray_source = out.copy()
    _draw_rays(out, ray_source, ctrl_pts, fg=fg, ctrl=ctrl_color, bg=bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 5])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    edges = set(ctrl_edges)
    has_v = any(edge in {"top", "bottom"} for edge in ctrl_edges)
    has_h = any(edge in {"left", "right"} for edge in ctrl_edges)

    if has_h:
        sprite_h = random.choice([5, 6, 8])
        small_h = 1 if sprite_h in {5, 6} else 2
        large_h = sprite_h - small_h - 1
        if random.choice([False, True]):
            top_h, bottom_h = small_h, large_h
        else:
            top_h, bottom_h = large_h, small_h
    else:
        sprite_h = random.randint(5, 8)
        top_h = bottom_h = None

    if has_v:
        sprite_w = random.randint(8, 10)
        small_w = 2
        large_w = sprite_w - small_w - 1
        if large_w < 2:
            return None
        if random.choice([False, True]):
            left_w, right_w = small_w, large_w
        else:
            left_w, right_w = large_w, small_w
    else:
        sprite_w = random.randint(8, 10)
        left_w = right_w = None

    max_r0 = H - sprite_h - 4
    max_c0 = W - sprite_w - 4
    if max_r0 < 4 or max_c0 < 4:
        return None
    sprite_r0 = random.randint(4, max_r0)
    sprite_c0 = random.randint(4, max_c0)

    r_split = sprite_r0 + top_h if has_h else None
    c_split = sprite_c0 + left_w if has_v else None

    ctrl_pts = []
    for edge in ctrl_edges:
        if edge == "top":
            ctrl_pts.append((0, int(c_split)))
        elif edge == "bottom":
            ctrl_pts.append((H - 1, int(c_split)))
        elif edge == "left":
            ctrl_pts.append((int(r_split), 0))
        elif edge == "right":
            ctrl_pts.append((int(r_split), W - 1))

    regions = _all_regions(has_h, has_v)
    region_boxes = _layout_regions(
        sprite_r0=sprite_r0,
        sprite_c0=sprite_c0,
        sprite_h=sprite_h,
        sprite_w=sprite_w,
        r_split=r_split,
        c_split=c_split,
        has_h=has_h,
        has_v=has_v,
    )

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1
    areas = {
        reg: _region_area(
            reg[0],
            reg[1],
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c1,
            r_split=r_split,
            c_split=c_split,
        )
        for reg in regions
    }
    sorted_areas = sorted(areas.values(), reverse=True)
    if len(sorted_areas) > num_ctrl and sorted_areas[num_ctrl - 1] <= sorted_areas[num_ctrl]:
        return None

    selected_regions = _select_largest_regions(
        regions,
        num_keep=num_ctrl,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        r_split=r_split,
        c_split=c_split,
    )

    split_pixels = _make_split_pixels(
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        has_h=has_h,
        has_v=has_v,
        r_split=r_split,
        c_split=c_split,
    )

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    all_fg = set(split_pixels)
    for reg in regions:
        r0, r1, c0, c1 = region_boxes[reg]
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        if sub_h <= 0 or sub_w <= 0:
            return None
        local_piece = _make_spanning_piece(
            reg[0],
            reg[1],
            sub_h=sub_h,
            sub_w=sub_w,
            has_h=has_h,
            has_v=has_v,
            edges=edges,
        )
        pts = frozenset((r0 + r, c0 + c) for r, c in local_piece)
        for r, c in pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((reg[1] == "L" and c >= int(c_split)) or (reg[1] == "R" and c <= int(c_split))):
                return None
            if has_h and ((reg[0] == "T" and r >= int(r_split)) or (reg[0] == "B" and r <= int(r_split))):
                return None
        input_pieces[reg] = pts
        all_fg |= set(pts)

    fg_arr = np.array(list(all_fg))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h != sprite_h or bbox_w != sprite_w:
        return None

    density = len(all_fg) / float(sprite_h * sprite_w)
    if not (0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_fg:
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    out = _expected_output(
        H=H,
        W=W,
        bg=bg,
        fg=fg,
        ctrl_color=ctrl_color,
        ctrl_pts=ctrl_pts,
        edges=edges,
        r_split=r_split,
        c_split=c_split,
        selected_regions=selected_regions,
        input_pieces=input_pieces,
        split_pixels=split_pixels,
    )
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
