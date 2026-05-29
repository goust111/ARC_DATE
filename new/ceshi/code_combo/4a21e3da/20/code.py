# Combo variant of 4a21e3da (index 20)
# combo_spec: D3/1+D5/1
# Name: 精灵内部标记+镜像翻转

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
Design analysis for combo D3/1 + D5/1
=====================================

a) concretization_and_trace:
  original:
    - input: one connected orange sprite near the center; 1-2 red controls live on
      grid edges and define one vertical split and optionally one horizontal split.
    - output: red rays are drawn inward from the border controls; orange pixels on
      split lines stay fixed; each surviving cut-piece slides to a corner without
      changing orientation; non-touching regions vanish.
  combo:
    - input: the red controls are no longer on the grid border. They are rare red
      marker pixels embedded in the sprite and placed on the sprite outline at the
      side they represent: top / bottom / left / right.
    - output stage 1: each internal red marker emits a ray outward from the sprite
      to the matching grid edge, so the visible ray direction is reversed relative
      to the source task.
    - output stage 2: each surviving off-line piece is normalized in its own bbox,
      mirrored on both movement axes, then anchored to the destination corner.
      With corner moves this is equivalent to a 180-degree reorientation inside the
      local bbox before corner placement.
  small_trace:
    - suppose a top-marker sits at (r,c) on the sprite's top outline and a left-marker
      sits at (r2,c2) on the sprite's left outline.
    - original would have used border controls at row 0 / col 0 and drawn rays inward.
    - combo instead keeps red at (r,c) and (r2,c2), draws red upward / leftward to
      the border, partitions the orange sprite by column c and row r2, then sends
      the kept fragments farther outward into corners after mirroring them.
    - difference summary: source localization, ray direction, and piece orientation
      all change together, so the solver must infer direction from the embedded markers
      before it can apply the mirrored corner motion.

b) complexity_calibration:
  - I keep the original task family: at most one vertical split and one horizontal
    split, 1-2 controls, and the same corner-destination logic. That keeps the rule
    learnable from a few examples.
  - To avoid ambiguity, each internal marker is the extreme occupied cell in its ray
    direction on its split row/column. This limits the control vocabulary to the same
    four edge labels {top, bottom, left, right}.
  - Reflection is restricted to a single deterministic rule: mirror the kept piece
    inside its own bbox along the vertical and horizontal axes implied by the target
    corner. There is no piece-dependent branch.
  - The generator rejects shapes where mirrored-corner placement would equal legacy
    slide-only placement, so the D5 contribution stays visually observable.

c) implicit_assumption_review_of_original_main:
  - Original ctrl detection assumed a non-bg border color. That breaks because combo
    controls are internal. New solver identifies ctrl as the rarer non-bg color.
  - Original edge inference assumed ctrl coordinates lie on row 0 / row H-1 / col 0 /
    col W-1. That also breaks. New solver infers each marker's semantic edge by checking
    whether it sits on the non-bg bbox boundary and whether fg continuation exists only
    on the opposite side of the same row/column.
  - Original ray drawing used the input grid to stop at the farthest fg pixel from the
    border inward. Combo rays run outward from the internal marker to the border, and
    the generator guarantees there is no fg beyond the marker in that ray direction.
  - Original region motion preserved pixel offsets. Combo motion cannot do that; it
    must normalize the region, mirror it, then anchor it in the destination corner.
  - Original split-line preservation still holds because control markers are not fg,
    and kept/vanishing pieces are generated strictly off the split lines except for
    the dedicated fixed split pixels.

d) placement_legality:
  - Grid stays square with size 18-20. Sprite bbox stays 5-8 by 8-10 and remains at
    least 4 cells away from every grid border, which guarantees visible outward rays.
  - Internal markers occupy forbidden sprite-outline row/col slots so other orange
    pieces cannot extend beyond them and invalidate edge inference.
  - Each piece is generated inside its own cut subregion, never crossing the split
    row/column or the reserved marker row/column. Output pieces are checked to remain
    on the correct side of the split lines after mirrored corner placement.
  - The combined non-bg input object is required to stay 4-connected.

e) visual_observability:
  - A human sees a central orange sprite with 1-2 rare red pixels inside it rather
    than on the frame; in the output those same internal red pixels emit rays outward
    to the border. That makes the D3 inversion explicit.
  - The D5 contribution is visible because corner fragments change silhouette; the
    generator rejects 180-degree-symmetric kept pieces and requires a measurable delta
    versus legacy slide-only outputs.
  - Across 2-4 examples, repeated top/bottom/left/right marker placements support the
    inference that marker position on the sprite outline determines ray direction.

f) visual_cleanliness:
  - Only three colors are used, unchanged from the source task.
  - Red is reserved for markers and rays. Orange is reserved for the sprite material.
  - Rays occupy empty background outside the centered sprite, while mirrored corner
    pieces move to border corners, so overlap is limited and the output remains legible.

g) color_change_inference:
  - No new color remapping is introduced. The combo changes geometry only:
    internal-source detection, outward ray growth, and mirrored corner placement.

h) new_solver_assumptions_and_generator_guarantees:
  - Assumption: exactly two non-bg colors exist and ctrl appears 1-2 times.
    Guarantee: generator always uses fg=7 for bulk sprite material and ctrl=2 for
    exactly 1 or 2 internal markers.
  - Assumption: each ctrl marker has a unique edge interpretation.
    Guarantee: a vertical marker is placed on the sprite's top or bottom outline at
    the split column, with orange continuation only on the opposite side; horizontal
    markers are analogous on the split row.
  - Assumption: at most one vertical split and one horizontal split exist.
    Guarantee: generator mirrors the source task structure and never creates two
    vertical markers or two horizontal markers.
  - Assumption: reflection should be visible.
    Guarantee: kept pieces must change under mirrored-corner placement, and the total
    mirrored-vs-slide-only pixel delta is bounded away from zero.
"""


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
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
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


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


def _infer_control_info(g: np.ndarray, bg: int, ctrl: int, fg: int) -> list[tuple[int, int, str]]:
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
        info.append((int(r), int(c), candidates[0]))

    return info


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
    Combo rule:
    - 1-2 red control pixels are embedded in the sprite instead of on the border.
    - Each marker sits on a split row/column and indicates a border direction.
    - Rays grow outward from the internal marker to the corresponding border.
    - Split-line orange pixels remain in place.
    - Surviving off-line pieces are mirrored inside their own bbox before being
      anchored to their destination corner; non-touching pieces vanish.
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
            assert c_split is None or c_split == int(c)
            c_split = int(c)
        else:
            assert r_split is None or r_split == int(r)
            r_split = int(r)
        E.add(edge)

    out = np.full((H, W), int(bg), dtype=int)

    for r, c, edge in ctrl_info:
        if edge == "top":
            for rr in range(0, int(r)):
                if int(out[int(rr), int(c)]) == int(bg):
                    out[int(rr), int(c)] = int(ctrl)
        elif edge == "bottom":
            for rr in range(int(r) + 1, H):
                if int(out[int(rr), int(c)]) == int(bg):
                    out[int(rr), int(c)] = int(ctrl)
        elif edge == "left":
            for cc in range(0, int(c)):
                if int(out[int(r), int(cc)]) == int(bg):
                    out[int(r), int(cc)] = int(ctrl)
        elif edge == "right":
            for cc in range(int(c) + 1, W):
                if int(out[int(r), int(cc)]) == int(bg):
                    out[int(r), int(cc)] = int(ctrl)

    for r, c, _edge in ctrl_info:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if c < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        for rr, cc in _mirror_then_corner(frozenset(pts), dh, dv, H, W):
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _slot_start(slots: list[int], target: int, span: int) -> int | None:
    if span > len(slots):
        return None
    best_i = 0
    best_cost = None
    for i in range(len(slots) - span + 1):
        center = slots[i + span // 2]
        cost = abs(int(center) - int(target))
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
    rows = [int(r) for r in range(int(r0), int(r1) + 1) if int(r) not in forbidden_rows]
    cols = [int(c) for c in range(int(c0), int(c1) + 1) if int(c) not in forbidden_cols]
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
    ph = int(max(r for r, _ in piece) + 1)
    pw = int(max(c for _, c in piece) + 1)
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

    return {(int(rows[r + int(r_start)]), int(cols[c + int(c_start)])) for r, c in piece}


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    ctrl, fg = _ctrl_and_fg(inp, bg)
    ctrl_info = _infer_control_info(inp, bg, ctrl, fg)
    H, W = inp.shape
    border_ctrl = 0
    ray_len_sum = 0
    for r, c, edge in ctrl_info:
        if r in (0, H - 1) or c in (0, W - 1):
            border_ctrl += 1
        if edge == "top":
            ray_len_sum += int(r)
        elif edge == "bottom":
            ray_len_sum += int(H - 1 - r)
        elif edge == "left":
            ray_len_sum += int(c)
        else:
            ray_len_sum += int(W - 1 - c)

    E = {edge for _r, _c, edge in ctrl_info}
    c_split = next((c for _r, c, edge in ctrl_info if edge in ("top", "bottom")), None)
    r_split = next((r for r, _c, edge in ctrl_info if edge in ("left", "right")), None)
    legacy_delta = 0
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(inp == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if c < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))
    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        legacy = _slide_only_corner(frozenset(pts), dh, dv, H, W)
        combo = _mirror_then_corner(frozenset(pts), dh, dv, H, W)
        legacy_delta += len(legacy ^ combo)
    return {"border_ctrl": int(border_ctrl), "ray_len_sum": int(ray_len_sum), "legacy_delta": int(legacy_delta)}


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[2, 4])[0]
    if int(num_ctrl) == 1:
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
        if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
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
        if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    ctrl_info: list[tuple[int, int, str]] = []
    forbidden_rows: set[int] = set()
    forbidden_cols: set[int] = set()

    if has_v:
        edge = next(edge for edge in ctrl_edges if edge in ("top", "bottom"))
        v_ctrl_r = int(sprite_r0) if edge == "top" else int(sprite_r1)
        ctrl_info.append((int(v_ctrl_r), int(c_split), edge))
        forbidden_rows.add(int(v_ctrl_r))

    if has_h:
        edge = next(edge for edge in ctrl_edges if edge in ("left", "right"))
        h_ctrl_c = int(sprite_c0) if edge == "left" else int(sprite_c1)
        ctrl_info.append((int(r_split), int(h_ctrl_c), edge))
        forbidden_cols.add(int(h_ctrl_c))

    ctrl_pts = [(int(r), int(c)) for r, c, _edge in ctrl_info]
    if len(set(ctrl_pts)) != len(ctrl_pts):
        return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        edge = next(edge for edge in ctrl_edges if edge in ("top", "bottom"))
        if edge == "top":
            rows_pool = [int(r) for r in range(int(sprite_r0) + 1, int(sprite_r1) + 1) if r != r_split]
        else:
            rows_pool = [int(r) for r in range(int(sprite_r0), int(sprite_r1)) if r != r_split]
        if len(rows_pool) < 2:
            return None
        n = random.randint(2, min(4, len(rows_pool)))
        for r in random.sample(rows_pool, n):
            split_pixels.add((int(r), int(c_split)))

    if has_h:
        edge = next(edge for edge in ctrl_edges if edge in ("left", "right"))
        if edge == "left":
            cols_pool = [int(c) for c in range(int(sprite_c0) + 1, int(sprite_c1) + 1) if c != c_split]
        else:
            cols_pool = [int(c) for c in range(int(sprite_c0), int(sprite_c1)) if c != c_split]
        if len(cols_pool) < 2:
            return None
        n = random.randint(2, min(4, len(cols_pool)))
        for c in random.sample(cols_pool, n):
            split_pixels.add((int(r_split), int(c)))

    total_area = int(sprite_h * sprite_w)
    target_density = random.uniform(0.40, 0.58)
    target_nonbg = int(round(target_density * total_area))
    n_fixed = int(len(split_pixels) + len(ctrl_pts))
    budget_pieces = int(target_nonbg - n_fixed)
    if budget_pieces < int(len(kept) * 3):
        return None

    if vanish:
        max_vanish = int(min(7, max(3, budget_pieces // 3)))
        vanish_budget = random.randint(3, max_vanish)
        budget_kept = int(budget_pieces - vanish_budget)
    else:
        vanish_budget = 0
        budget_kept = int(budget_pieces)

    if budget_kept < int(len(kept) * 3):
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
        capacities[reg] = int(len(rows) * len(cols))

    piece_budgets: list[int] = []
    remaining = int(budget_kept)
    for i, reg in enumerate(kept):
        cap = int(capacities[reg])
        if i == len(kept) - 1:
            px = int(remaining)
        else:
            avg = int(remaining // (len(kept) - i))
            lo_px = int(max(3, avg - 2))
            hi_px = int(min(cap, avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = int(max(3, min(px, cap, remaining - (len(kept) - i - 1) * 3)))
        if px > cap:
            return None
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0:
        return None

    split_rows = sorted([int(r) for r, c in split_pixels if c_split is not None and int(c) == int(c_split)])
    split_cols = sorted([int(c) for r, c in split_pixels if r_split is not None and int(r) == int(r_split)])

    input_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    variant_delta = 0

    for idx, reg in enumerate(kept):
        hs, vs = reg
        rows, cols = axes_cache[reg]
        piece = _grow_piece(len(rows), len(cols), piece_budgets[idx])
        if len(piece) < 3:
            return None

        ph = int(max(r for r, _ in piece) + 1)
        pw = int(max(c for _, c in piece) + 1)
        if ph < 2 or pw < 2:
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

        input_pieces[reg] = set(pts)

        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        out_pts = _mirror_then_corner(pts, dh, dv, H, W)
        legacy_pts = _slide_only_corner(pts, dh, dv, H, W)
        if out_pts == legacy_pts:
            return None

        for r, c in out_pts:
            if c_split is not None and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                return None
            if r_split is not None and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                return None
        output_pieces[reg] = out_pts
        variant_delta += int(len(out_pts ^ legacy_pts))

    if variant_delta < 4:
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        vreg = random.choice(vanish)
        rows, cols = axes_cache[vreg]
        max_px = int(min(vanish_budget, capacities[vreg]))
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
    all_fg |= set(split_pixels)
    all_fg |= set(vanish_fg)
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
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    actual_density = len(all_nonbg) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in all_fg:
        inp[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = int(ctrl_color)

    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)
    for r, c, edge in ctrl_info:
        if edge == "top":
            for rr in range(0, int(r)):
                if out[int(rr), int(c)] == int(bg):
                    out[int(rr), int(c)] = int(ctrl_color)
        elif edge == "bottom":
            for rr in range(int(r) + 1, H):
                if out[int(rr), int(c)] == int(bg):
                    out[int(rr), int(c)] = int(ctrl_color)
        elif edge == "left":
            for cc in range(0, int(c)):
                if out[int(r), int(cc)] == int(bg):
                    out[int(r), int(cc)] = int(ctrl_color)
        elif edge == "right":
            for cc in range(int(c) + 1, W):
                if out[int(r), int(cc)] == int(bg):
                    out[int(r), int(cc)] = int(ctrl_color)
        out[int(r), int(c)] = int(ctrl_color)

    if not np.array_equal(main(inp), out):
        return None

    metrics = _combo_metrics(inp)
    if metrics["border_ctrl"] != 0:
        return None
    if metrics["ray_len_sum"] <= 0 or metrics["legacy_delta"] < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
