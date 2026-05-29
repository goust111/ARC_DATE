# Combo variant of 4a21e3da (index 38)
# combo_spec: D14/0+D15/0
# Name: 小网格大精灵+不对称切割

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

# Design analysis for the fused variant.
# - Concrete behavior:
#   Original samples use a centered 18-20 grid and a roughly middle split. Here the
#   transformation rule stays the same, but the generator compresses the whole scene
#   into a 10-12 grid and pushes each split to a discrete asymmetric slot:
#   20/25% or 75/80% of the sprite span. Example: in an 11x11 grid, a 5x5 sprite
#   with a top control at column c0+1 produces a 1-column strip on the left and a
#   3-column body on the right. The left strip contains only 2-3 fg pixels, while
#   the right body is larger. After solving, the strip slides to the left corner and
#   the large body slides to the right corner, so the output is visibly different
#   from the original centered-cut version.
# - Complexity calibration:
#   The change must stay readable from a few examples, so the asymmetric split is
#   quantized to a tiny set of ratios: {0.20, 0.25, 0.75, 0.80}. This keeps the
#   variation visible without creating too many mutually unrelated layouts.
# - Implicit assumptions checked against main():
#   1) Border ctrl color must still be the only non-bg color on the border.
#   2) There is still at most one vertical split and at most one horizontal split.
#   3) Rays still stop at the farthest fg on the controlled line; small grids do not
#      change that logic.
#   4) Regions are still rectangular partitions defined only by split row/col. The
#      generator therefore keeps every off-line fg pixel strictly inside its region.
#   5) Translation in stage 2 assumes a region piece fits on its destination side.
#      The generator guarantees this by creating each piece inside the same side's
#      available rectangle before translation.
# - Placement legality:
#   The generator places the sprite with margin >= 2, keeps ctrl pixels on the border,
#   ensures regions are non-overlapping by construction, and validates contiguity.
# - Visual observability and neatness:
#   Small strips are deliberately only 2-3 pixels, the large region is extended to
#   the relevant outer bbox sides, and split-line pixels form a short contiguous run
#   or cross. This makes the "tiny fragment vs large body" contrast legible even in
#   the crowded 10-12 grids.
# - New assumptions introduced by the variant:
#   Because the split is biased, dimensions that are cut must be at least 5 cells,
#   otherwise 20-30% / 70-80% is not representable on an integer lattice. The
#   generator enforces width/height >= 5 exactly on the split dimensions.


def _edge_info(ctrl_pts: Iterable[tuple[int, int]], H: int, W: int) -> tuple[set[str], int | None, int | None]:
    edges: set[str] = set()
    r_split: int | None = None
    c_split: int | None = None
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


def _region_list(has_h: bool, has_v: bool) -> list[tuple[str, str]]:
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


def _pick_dest(hs: str, vs: str, edges: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the sprite along the control rays and slide each surviving fragment to a corner.

    The fused variant only changes the input distribution: smaller grids, larger
    relative sprites, and asymmetric split positions. The solver logic is unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])

    nonbg = colors_present(g, exclude={bg})
    fg_candidates = [int(c) for c in nonbg if int(c) != ctrl]
    assert fg_candidates
    fg = int(fg_candidates[0])

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    edges, r_split, c_split = _edge_info(ctrl_pts, H, W)

    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = max(rows)
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = min(rows)
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = max(cols)
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = min(cols)
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

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
        hs = "A" if r_split is None else ("T" if r < r_split else "B")
        vs = "A" if c_split is None else ("L" if c < c_split else "R")
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & edges):
            continue
        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
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


def _sub_bbox(
    hs: str,
    vs: str,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int, int, int]:
    r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
    r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if r_split is not None else sprite_r1
    c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
    c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if c_split is not None else sprite_c1
    return r0, r1, c0, c1


def _grow_anchored_piece(sub_h: int, sub_w: int, target_px: int, anchors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    target_px = max(1, min(target_px, sub_h * sub_w))
    start = random.choice(anchors)
    cells = {start}
    while len(cells) < target_px:
        candidates = []
        seen = set()
        for r, c in list(cells):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < sub_h and 0 <= nc < sub_w):
                    continue
                if (nr, nc) in cells or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                shared = 0
                for ar, ac in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (nr + ar, nc + ac) in cells:
                        shared += 1
                anchor_dist = min(abs(nr - ar) + abs(nc - ac) for ar, ac in anchors)
                anchor_bonus = 1 if (nr, nc) in anchors else 0
                candidates.append((shared, anchor_bonus, -anchor_dist, random.random(), (nr, nc)))
        if not candidates:
            break
        candidates.sort(reverse=True)
        best = candidates[: min(5, len(candidates))]
        cells.add(random.choice(best)[4])
    return cells


def _extend_piece_to_side(piece: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> None:
    r0, r1, c0, c1 = bbox
    assert piece
    if side == "top":
        r, c = min(piece, key=lambda rc: rc[0])
        for rr in range(r - 1, r0 - 1, -1):
            piece.add((rr, c))
    elif side == "bottom":
        r, c = max(piece, key=lambda rc: rc[0])
        for rr in range(r + 1, r1 + 1):
            piece.add((rr, c))
    elif side == "left":
        r, c = min(piece, key=lambda rc: rc[1])
        for cc in range(c - 1, c0 - 1, -1):
            piece.add((r, cc))
    elif side == "right":
        r, c = max(piece, key=lambda rc: rc[1])
        for cc in range(c + 1, c1 + 1):
            piece.add((r, cc))


def _measure_variant_features(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in colors_present(g, exclude={bg}) if int(c) != ctrl))
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    edges, r_split, c_split = _edge_info(ctrl_pts, H, W)

    fg_pts = [(int(r), int(c)) for r, c in np.argwhere(g == fg)]
    rs = [r for r, _ in fg_pts]
    cs = [c for _, c in fg_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    regions: dict[tuple[str, str], int] = {}
    for r, c in fg_pts:
        if (r_split is not None and r == r_split) or (c_split is not None and c == c_split):
            continue
        hs = "A" if r_split is None else ("T" if r < r_split else "B")
        vs = "A" if c_split is None else ("L" if c < c_split else "R")
        regions[(hs, vs)] = regions.get((hs, vs), 0) + 1

    split_fracs: dict[str, float] = {}
    if c_split is not None and cmax > cmin:
        split_fracs["vertical"] = (c_split - cmin) / (cmax - cmin)
    if r_split is not None and rmax > rmin:
        split_fracs["horizontal"] = (r_split - rmin) / (rmax - rmin)

    kept_counts = {k: v for k, v in regions.items() if _touches_edges(*k) & edges}
    return {
        "grid_size": (H, W),
        "bbox": (rmin, cmin, rmax, cmax),
        "split_fracs": split_fracs,
        "region_counts": regions,
        "kept_region_counts": kept_counts,
    }


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(10, 12)
    H = W = S
    bg, fg, ctrl = 1, 7, 2

    ctrl_edges = [random.choice(["top", "bottom", "left", "right"])] if random.random() < 0.65 else [
        random.choice(["top", "bottom"]),
        random.choice(["left", "right"]),
    ]
    edges = set(ctrl_edges)
    has_v = any(e in ("top", "bottom") for e in ctrl_edges)
    has_h = any(e in ("left", "right") for e in ctrl_edges)

    sprite_h = random.randint(5, 6) if has_h else random.randint(4, 6)
    sprite_w = random.randint(5, 6) if has_v else random.randint(4, 6)

    sprite_r0 = random.randint(2, H - sprite_h - 2)
    sprite_c0 = random.randint(2, W - sprite_w - 2)
    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    if has_h and has_v:
        vanish_h = "B" if "top" in edges else "T"
        vanish_v = "R" if "left" in edges else "L"
        small_h = vanish_h
        small_v = vanish_v
    else:
        small_h = random.choice(["T", "B"]) if has_h else None
        small_v = random.choice(["L", "R"]) if has_v else None

    r_split = None
    c_split = None
    if has_h:
        r_split = sprite_r0 + (1 if small_h == "T" else sprite_h - 2)
    if has_v:
        c_split = sprite_c0 + (1 if small_v == "L" else sprite_w - 2)

    ctrl_pts: list[tuple[int, int]] = []
    for edge in ctrl_edges:
        if edge == "top":
            ctrl_pts.append((0, int(c_split)))
        elif edge == "bottom":
            ctrl_pts.append((H - 1, int(c_split)))
        elif edge == "left":
            ctrl_pts.append((int(r_split), 0))
        else:
            ctrl_pts.append((int(r_split), W - 1))

    split_pixels: set[tuple[int, int]] = set()
    if has_h and has_v:
        for rr in [int(r_split) - 1, int(r_split), int(r_split) + 1]:
            split_pixels.add((rr, int(c_split)))
        for cc in [int(c_split) - 1, int(c_split), int(c_split) + 1]:
            split_pixels.add((int(r_split), cc))
    elif has_v:
        run_len = random.randint(2, min(3, sprite_h))
        start = random.randint(sprite_r0, sprite_r1 - run_len + 1)
        for rr in range(start, start + run_len):
            split_pixels.add((rr, int(c_split)))
    else:
        run_len = random.randint(2, min(3, sprite_w))
        start = random.randint(sprite_c0, sprite_c1 - run_len + 1)
        for cc in range(start, start + run_len):
            split_pixels.add((int(r_split), cc))

    all_regions = _region_list(has_h, has_v)
    kept = [reg for reg in all_regions if _touches_edges(*reg) & edges]
    vanish = [reg for reg in all_regions if reg not in kept]

    region_bounds = {
        reg: _sub_bbox(reg[0], reg[1], sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split)
        for reg in all_regions
    }
    if any(r1 < r0 or c1 < c0 for r0, r1, c0, c1 in region_bounds.values()):
        return None

    pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for reg in kept + vanish:
        hs, vs = reg
        r0, r1, c0, c1 = region_bounds[reg]
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        area = sub_h * sub_w

        anchors: list[tuple[int, int]] = []
        if c_split is not None and vs != "A":
            edge_c = c1 if vs == "L" else c0
            boundary_rows = sorted(rr for rr, cc in split_pixels if cc == c_split and r0 <= rr <= r1)
            anchors.extend((rr - r0, edge_c - c0) for rr in boundary_rows)
        if r_split is not None and hs != "A":
            edge_r = r1 if hs == "T" else r0
            boundary_cols = sorted(cc for rr, cc in split_pixels if rr == r_split and c0 <= cc <= c1)
            anchors.extend((edge_r - r0, cc - c0) for cc in boundary_cols)
        if not anchors:
            anchors = [((sub_h - 1) // 2, (sub_w - 1) // 2)]
        anchors = list(dict.fromkeys(anchors))

        if reg in kept:
            if sub_h == 1 or sub_w == 1:
                target_px = random.randint(2, min(3, area))
            else:
                hi = min(area, 7 if len(kept) <= 2 else 6)
                lo = 4 if hi >= 4 else hi
                target_px = random.randint(lo, hi)
        else:
            if area == 1:
                target_px = 1
            else:
                target_px = random.randint(1, min(2, area))

        local_piece = _grow_anchored_piece(sub_h, sub_w, target_px, anchors)
        if len(local_piece) != target_px:
            return None

        abs_piece = {(r0 + r, c0 + c) for r, c in local_piece}

        if reg in kept and sub_h > 1 and sub_w > 1:
            wanted: list[str] = []
            if hs in ("T", "A"):
                wanted.append("top")
            if hs in ("B", "A"):
                wanted.append("bottom")
            if vs in ("L", "A"):
                wanted.append("left")
            if vs in ("R", "A"):
                wanted.append("right")
            for side in wanted:
                _extend_piece_to_side(abs_piece, (r0, r1, c0, c1), side)

        pieces[reg] = abs_piece

    combined = set(split_pixels)
    for pts in pieces.values():
        if combined & pts:
            return None
        combined |= pts

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl

    fg_pts = np.argwhere(inp == fg)
    if fg_pts.size == 0:
        return None
    rmin = int(fg_pts[:, 0].min())
    rmax = int(fg_pts[:, 0].max())
    cmin = int(fg_pts[:, 1].min())
    cmax = int(fg_pts[:, 1].max())
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    if not (4 <= bbox_h <= 6 and 4 <= bbox_w <= 6):
        return None
    if min(rmin, cmin, H - 1 - rmax, W - 1 - cmax) < 2:
        return None

    if c_split is not None:
        frac = (int(c_split) - cmin) / (cmax - cmin)
        if not (0.19 <= frac <= 0.31 or 0.69 <= frac <= 0.81):
            return None
    if r_split is not None:
        frac = (int(r_split) - rmin) / (rmax - rmin)
        if not (0.19 <= frac <= 0.31 or 0.69 <= frac <= 0.81):
            return None

    density = len(fg_pts) / (bbox_h * bbox_w)
    if not (0.38 <= density <= 0.72):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    metrics = _measure_variant_features(inp)
    kept_sizes = sorted(int(v) for v in metrics["kept_region_counts"].values())
    if not any(2 <= v <= 3 for v in kept_sizes):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 500 attempts")
