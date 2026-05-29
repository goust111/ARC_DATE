# Variant of 4a21e3da
# Dimension: D13 — 射线切割与区域归属判定 (meta)
# Change: ray_splitting — 射线将精灵分为矩形区域，区域与控制边接壤则保留... → 按面积大小筛选

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
Design analysis for the D13 variant
===================================

a) Concrete behavior and trace
YAML-like summary:
  input:
    - square grid, size 18-20
    - bg=1, fg=7, ctrl=2
    - 1 or 2 control pixels on border edges
    - ctrl rays define one vertical split and/or one horizontal split
    - the centered fg sprite is still a single connected component
    - unlike the original seed, every cut region is populated with fg so the
      choice of kept regions is actually visible
    - each region piece is a sparse connected "inward-facing L" whose bbox
      matches the whole rectangular partition, so the region's width/height are
      visually recoverable from the fg itself
  output:
    - rays are drawn the same way as before
    - split-line fg stays in place
    - the sprite is partitioned into 2 or 4 rectangles
    - compute each rectangle's area from its height x width
    - let N = number of control pixels
    - keep only the N largest regions, slide each kept piece to its geometric
      destination corner using the original translation rule, delete the rest

Small trace:
  - With two controls, suppose the four partition rectangles have sizes
    TL=2x3, TR=2x5, BL=4x3, BR=4x5, so their areas are 6, 10, 12, 20.
  - Original rule: keep regions touching the chosen edges, e.g. for
    {top,left} controls that would keep TL/TR/BL and delete BR.
  - New rule: keep only the largest N=2 regions, i.e. BR and BL, regardless of
    which edges the controls lie on.
  - The rest of the pipeline stays unchanged: rays are still intermittent and
    each surviving piece still slides to the corner determined by its side of
    the split.

b) Complexity calibration
  - The new decision is not trivialized into "touches a marked edge".
  - Region areas are forced to have a strict ranking at the keep/drop boundary.
  - The sparse L-shapes make fg cell count depend on perimeter-like size
    (h+w-1) instead of area (h*w), so counting fg pixels is not a reliable
    shortcut for the intended rule.
  - Two-control cases are preserved because they are the main source of
    non-local comparison across multiple candidate regions.

c) Implicit assumptions in the original solver that must change
  - Original `touches_edges(hs, vs) & E` assumed ownership is determined only by
    adjacency to control edges. This is the core obsolete assumption.
  - Original region iteration only needed region labels; now we also need the
    sprite bbox and split positions to derive each partition rectangle's area.
  - Original tie handling was irrelevant because control-edge ownership was
    boolean. Now generator must avoid ambiguous ties or near-ties.
  - The original solver was safe with arbitrary region occupancy. Here the
    generator guarantees every region is populated, otherwise selecting an empty
    "largest region" would make the new rule less observable.

d) Placement legality
  - Control pixels remain on distinct edges and define at most one horizontal
    and one vertical split.
  - Split positions stay inside the sprite interior.
  - Every region piece is kept strictly off the split line, inside its own
    partition rectangle, and the whole fg remains connected via split-line fg.
  - After sliding, pieces still fit on their side of the split lines, so they
    do not collide with the centered split-line fg or overflow the grid.

e) Visual observability
  - Humans can estimate region area because each piece spans the full bbox of
    its region even though the interior is sparse.
  - The generator rejects two-control layouts where the new selected set would
    equal the original edge-touch set, preventing silent degeneration.
  - The area ranking is strict, so from 2-4 examples the "pick the biggest
    partitions" pattern is visible.

f) Visual cleanliness
  - Only three colors are used and the layering is unchanged: bg, fg, ctrl rays.
  - Each region contains a single connected sparse piece, avoiding clutter while
    still making partition extents legible.
  - Split-line fg is limited to 2-4 per split, keeping the center readable.

g) Color inference
  - No color remapping is introduced. Colors retain their original roles.

Reference-task note:
  - `source_task` is None, so there is no extra solver to inspect.
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
    scored = []
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
    Variant rule:
    - Rays and split-line preservation are unchanged.
    - The split creates 2 or 4 rectangular regions.
    - Let N be the number of control pixels.
    - Keep only the N regions with the largest rectangle areas; slide those
      pieces to their usual destination corners. Other regions vanish.
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
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        dr = -rmin if dh == "T" else (H - 1 - rmax)
        dc = -cmin if dv == "L" else (W - 1 - cmax)
        for r, c in pts:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg

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

    # Add a few interior cells so density stays inside the original visual band
    # without changing the piece bbox or the intended selection signal.
    extra_candidates = []
    for r in range(sub_h):
        for c in range(sub_w):
            if (r, c) in cells:
                continue
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells):
                extra_candidates.append((r, c))
    random.shuffle(extra_candidates)
    for r, c in extra_candidates[: random.randint(0, min(3, len(extra_candidates)))]:
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
    _draw_rays(out, out.copy(), ctrl_pts, fg=fg, ctrl=ctrl_color, bg=bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    edges = set(ctrl_edges)
    has_v = any(edge in {"top", "bottom"} for edge in ctrl_edges)
    has_h = any(edge in {"left", "right"} for edge in ctrl_edges)

    if has_h:
        top_h = random.randint(2, 5)
        bottom_h = random.randint(2, 5)
        sprite_h = top_h + bottom_h + 1
        if not (5 <= sprite_h <= 8):
            return None
    else:
        top_h = bottom_h = None
        sprite_h = random.randint(5, 8)

    if has_v:
        left_w = random.randint(3, 6)
        right_w = random.randint(3, 6)
        sprite_w = left_w + right_w + 1
        if not (8 <= sprite_w <= 10):
            return None
    else:
        left_w = right_w = None
        sprite_w = random.randint(8, 10)

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

    areas = {
        reg: _region_area(
            reg[0],
            reg[1],
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r0 + sprite_h - 1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c0 + sprite_w - 1,
            r_split=r_split,
            c_split=c_split,
        )
        for reg in regions
    }
    sorted_areas = sorted(areas.values(), reverse=True)
    if len(sorted_areas) > num_ctrl and sorted_areas[num_ctrl - 1] <= sorted_areas[num_ctrl]:
        return None
    if num_ctrl == 2 and len(sorted_areas) > 2 and sorted_areas[1] - sorted_areas[2] < 2:
        return None

    selected_regions = _select_largest_regions(
        regions,
        num_keep=num_ctrl,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r0 + sprite_h - 1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c0 + sprite_w - 1,
        r_split=r_split,
        c_split=c_split,
    )
    original_regions = {reg for reg in regions if _touches_edges(reg[0], reg[1]) & edges}
    if selected_regions == original_regions:
        return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        rows = list(range(sprite_r0, sprite_r0 + sprite_h))
        if has_h:
            rows.remove(int(r_split))
        n = random.randint(2, min(4, len(rows)))
        for r in random.sample(rows, n):
            split_pixels.add((r, int(c_split)))
    if has_h:
        cols = list(range(sprite_c0, sprite_c0 + sprite_w))
        if has_v:
            cols.remove(int(c_split))
        n = random.randint(2, min(4, len(cols)))
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

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

    if num_ctrl == 2:
        by_count = {
            reg
            for reg in sorted(regions, key=lambda reg: (-len(input_pieces[reg]), REGION_ORDER[reg]))[:2]
        }
        if by_count == selected_regions:
            return None

    if not all_fg:
        return None

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
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
