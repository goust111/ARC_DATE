# Combo variant of 4a21e3da (index 36)
# combo_spec: D5/2+D17/0
# Name: 片段旋转+颜色编码指令

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

RED_CTRL = 2
BLUE_CTRL = 8

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) concretization_and_trace:
#   - The input still looks like the base task: one orange connected sprite near the
#     center, one or two border control pixels, and no foreground on the outer frame.
#   - Geometry from the base task is preserved:
#       border position -> ray direction and split line
#       touching a control edge -> region is kept
#       touching none of them -> region vanishes
#   - The fused change is in how kept regions transform after cutting.
#       red control (2): the touched region quarter-turns, then slides to its corner
#       blue control (8): the touched region half-turns, then slides to its corner
#   - Small trace:
#       controls = top red + left blue
#       kept regions = TL, TR, BL ; vanished = BR
#       TL touches top+left, so blue dominates -> rotate 180 -> move to TL corner
#       TR touches only top, so red applies -> rotate 90 toward TR -> move to TR
#       BL touches only left, so blue applies -> rotate 180 -> move to BL
#     In the base task these three pieces only translated; here the silhouettes change.
#
# b) complexity_calibration:
#   - Only two instruction colors are used, so the legend is small.
#   - Only two angle classes exist: 90 and 180 degrees.
#   - The 90-degree case still needs a direction; that direction is not a third symbol.
#     It is inherited from the D5 corner relation: rotate "toward the destination
#     corner", which yields either CW or CCW depending on the active edge and corner.
#   - Two-control samples are biased toward red+blue or blue+blue, so examples usually
#     contain repeated support for both angle classes without introducing a red+red
#     ambiguity for a single region.
#
# c) hidden_assumption_audit_on_main:
#   - The original solver assumed a single border control color. That is no longer true,
#     so border scanning must collect every non-background border pixel as a control.
#   - The original solver inferred fg as "the other non-bg color". That breaks with two
#     control colors, so fg must be inferred as the unique non-border non-bg color.
#   - Stage 1 can no longer paint all rays with one shared color; each control point must
#     paint its own color.
#   - Stage 2 can no longer use pure translation. Each kept region must be cropped to its
#     local bbox, rotated, then placed flush to the chosen corner.
#   - The combo adds one generator-side assumption: no kept region in a two-control sample
#     is touched by two red controls at once. Otherwise "red means 90 degrees" would not
#     specify which quarter-turn direction to use. The generator enforces that by avoiding
#     red+red two-control color assignments.
#
# d) placement_legality:
#   - Input pieces are generated inside their own split-defined sub-boxes.
#   - Rotated output pieces are checked against the split-side capacity of their
#     destination corner so they do not cross the split lines.
#   - Vanishing pieces live only in untouching regions and are checked not to overlap kept
#     pieces or split-line pixels.
#
# e) visual_observability:
#   - Any kept piece assigned 90 degrees is forced to be non-invariant under quarter-turn.
#   - Any kept piece assigned 180 degrees is forced to be non-invariant under half-turn.
#   - Mixed red/blue two-control boards are common, so one example can show both angle
#     classes at once.
#
# f) visual_cleanliness:
#   - The board uses bg + fg + at most two control colors.
#   - Rays remain on their own split lines, split-line fg stays near the center, and moved
#     pieces occupy corners, so the three visual roles remain separated.
#
# g) color_rule_inference:
#   - Color is the only symbolic change. Red and blue appear both as border markers and as
#     rays, which gives repeated support for the instruction mapping in each sample.
#   - The numeric ids 2 and 8 are irrelevant; only visible color difference matters.
#
# h) metric_plan:
#   - After implementation, measure how many kept regions in generated samples are 90-turn
#     vs 180-turn, and compare this combo solver with the original base solver on the same
#     inputs. If samples with blue-controlled kept regions frequently match the base output,
#     the generator has collapsed back to the old task and must be fixed.
#
# i) new_main_assumptions_review:
#   - `_rotation_k_for_region` assumes that if no blue control touches a kept region, then
#     exactly one red control touches it in generated samples. The generator guarantees that.
#   - Corner placement assumes one connected piece per region. The generator still builds
#     one piece container per region.
#   - Output overlap is safe only because rotated pieces stay inside their destination-side
#     capacities. The generator checks those capacities before accepting a sample.


def _normalize_piece(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0 = int(min(rs))
    c0 = int(min(cs))
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _piece_hw(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return int(max(rs) + 1), int(max(cs) + 1)


def _rotate_piece(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    krot: int,
) -> frozenset[tuple[int, int]]:
    piece = _normalize_piece(cells)
    for _ in range(int(krot) % 4):
        height, _width = _piece_hw(piece)
        piece = frozenset((int(c), int(height - 1 - r)) for r, c in piece)
        piece = _normalize_piece(piece)
    return piece


def _rotation_visible(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    krot: int,
) -> bool:
    piece = _normalize_piece(cells)
    return _rotate_piece(piece, int(krot)) != piece


def _edge_of(r: int, c: int, H: int, W: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(H - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    if int(c) == int(W - 1):
        return "right"
    raise ValueError("control point is not on a border")


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


def _quarter_turn_k(edge: str, dh: str, dv: str) -> int:
    if edge == "top":
        assert dh == "T"
        return 1 if dv == "L" else 3
    if edge == "bottom":
        assert dh == "B"
        return 3 if dv == "L" else 1
    if edge == "left":
        assert dv == "L"
        return 3 if dh == "T" else 1
    assert edge == "right"
    assert dv == "R"
    return 1 if dh == "T" else 3


def _rotation_k_for_region(
    touched_edges: set[str],
    edge_to_color: dict[str, int],
    dh: str,
    dv: str,
) -> int:
    if any(int(edge_to_color[edge]) == int(BLUE_CTRL) for edge in touched_edges):
        return 2
    red_edges = [edge for edge in touched_edges if int(edge_to_color[edge]) == int(RED_CTRL)]
    assert len(red_edges) == 1
    return int(_quarter_turn_k(red_edges[0], dh, dv))


def _place_piece_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
    krot: int,
) -> frozenset[tuple[int, int]]:
    local = _normalize_piece(set((int(r), int(c)) for r, c in pts))
    local = _rotate_piece(local, int(krot))
    ph, pw = _piece_hw(local)
    base_r = 0 if dh == "T" else int(H - ph)
    base_c = 0 if dv == "L" else int(W - pw)
    return frozenset((int(base_r + r), int(base_c + c)) for r, c in local)


def _region_tags_for_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


def _draw_ray(out: np.ndarray, g: np.ndarray, fg: int, r: int, c: int, color: int) -> None:
    H, W = map(int, g.shape)
    if int(r) == 0:
        rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
        if rows:
            stop = int(max(rows))
            for rr in range(0, int(stop) + 1):
                if int(g[int(rr), int(c)]) == int(get_bg(g)):
                    out[int(rr), int(c)] = int(color)
    elif int(r) == int(H - 1):
        rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
        if rows:
            stop = int(min(rows))
            for rr in range(int(H - 1), int(stop) - 1, -1):
                if int(g[int(rr), int(c)]) == int(get_bg(g)):
                    out[int(rr), int(c)] = int(color)
    elif int(c) == 0:
        cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
        if cols:
            stop = int(max(cols))
            for cc in range(0, int(stop) + 1):
                if int(g[int(r), int(cc)]) == int(get_bg(g)):
                    out[int(r), int(cc)] = int(color)
    elif int(c) == int(W - 1):
        cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
        if cols:
            stop = int(min(cols))
            for cc in range(int(W - 1), int(stop) - 1, -1):
                if int(g[int(r), int(cc)]) == int(get_bg(g)):
                    out[int(r), int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Border position still determines the split line and ray direction.
    - Control color determines the kept-region rotation amount.
      red(2)  -> quarter-turn toward the destination corner
      blue(8) -> half-turn before moving to the destination corner
    - Regions touching no active control edge still vanish.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})
    ctrl_pts: list[tuple[int, int, int]] = []
    border_nonbg: set[int] = set()
    for r in range(H):
        for c in range(W):
            val = int(g[int(r), int(c)])
            if val == int(bg):
                continue
            if int(r) in (0, H - 1) or int(c) in (0, W - 1):
                ctrl_pts.append((int(r), int(c), int(val)))
                border_nonbg.add(int(val))

    assert ctrl_pts
    fg_candidates = [int(c) for c in nonbg if int(c) not in border_nonbg]
    assert len(fg_candidates) == 1
    fg = int(fg_candidates[0])

    E: set[str] = set()
    edge_to_color: dict[str, int] = {}
    c_split: int | None = None
    r_split: int | None = None
    for r, c, color in ctrl_pts:
        edge = _edge_of(r, c, H, W)
        E.add(edge)
        edge_to_color[edge] = int(color)
        if edge in ("top", "bottom"):
            c_split = int(c)
        else:
            r_split = int(r)

    out = np.full((H, W), int(bg), dtype=int)

    for r, c, color in ctrl_pts:
        _draw_ray(out, g, int(fg), int(r), int(c), int(color))
    for r, c, color in ctrl_pts:
        out[int(r), int(c)] = int(color)

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
        hs, vs = _region_tags_for_pixel(r, c, r_split, c_split)
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        touched = _touches_edges(hs, vs) & E
        if not touched:
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        krot = _rotation_k_for_region(touched, edge_to_color, dh, dv)
        for rr, cc in _place_piece_to_corner(pts, H, W, dh, dv, int(krot)):
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    return _normalize_piece(cells)


def _output_caps(
    dh: str,
    dv: str,
    H: int,
    W: int,
    has_h: bool,
    has_v: bool,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int]:
    if has_h:
        assert r_split is not None
        cap_h = int(r_split) if dh == "T" else int(H - 1 - r_split)
    else:
        cap_h = int(H)

    if has_v:
        assert c_split is not None
        cap_w = int(c_split) if dv == "L" else int(W - 1 - c_split)
    else:
        cap_w = int(W)
    return int(cap_h), int(cap_w)


def _region_rotation_k(
    hs: str,
    vs: str,
    E: set[str],
    edge_to_color: dict[str, int],
    r_split: int | None,
    c_split: int | None,
) -> int:
    touched = _touches_edges(hs, vs) & E
    dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
    return int(_rotation_k_for_region(touched, edge_to_color, dh, dv))


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg = 1, 7

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
        ctrl_colors = [random.choice([RED_CTRL, BLUE_CTRL])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
        ctrl_colors = random.choices(
            [
                [RED_CTRL, BLUE_CTRL],
                [BLUE_CTRL, RED_CTRL],
                [BLUE_CTRL, BLUE_CTRL],
            ],
            weights=[3, 3, 2],
        )[0]

    c_split = None
    r_split = None
    ctrl_pts: list[tuple[int, int, int]] = []
    edge_to_color: dict[str, int] = {}
    E = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge, color in zip(ctrl_edges, ctrl_colors):
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split), int(color)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1, int(color)))
            has_h = True
        edge_to_color[edge] = int(color)

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
        assert c_split is not None
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
        assert r_split is not None
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
        r0 = int(sprite_r0 if hs in ("T", "A") else int(r_split) + 1)
        r1 = int((int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else int(c_split) + 1)
        c1 = int((int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        assert c_split is not None
        rows = [
            r
            for r in range(int(sprite_r0), int(sprite_r1) + 1)
            if r_split is None or r != int(r_split)
        ]
        n = random.randint(2, 4)
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        assert r_split is not None
        cols = [
            c
            for c in range(int(sprite_c0), int(sprite_c1) + 1)
            if c_split is None or c != int(c_split)
        ]
        n = random.randint(2, 4)
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_total = int(random.uniform(0.39, 0.60) * area)

    n_kept = len(kept)
    budget_pieces = int(target_total - n_split)
    if budget_pieces < n_kept * 3:
        return None

    if vanish:
        max_vanish = max(3, min(7, budget_pieces // 3))
        if max_vanish < 3:
            return None
        vanish_budget = random.randint(3, max_vanish)
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        cap_h, cap_w = _output_caps(dh, dv, H, W, has_h, has_v, r_split, c_split)
        max_area = int(min(sub_h * sub_w, cap_h * cap_w))
        if idx == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - idx - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        krot = _region_rotation_k(hs, vs, E, edge_to_color, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        cap_h, cap_w = _output_caps(dh, dv, H, W, has_h, has_v, r_split, c_split)

        placed = False
        for _ in range(80):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) != target_px:
                continue
            if not _rotation_visible(piece, int(krot)):
                continue

            ph, pw = _piece_hw(piece)
            out_shape = _rotate_piece(piece, int(krot))
            oh, ow = _piece_hw(out_shape)
            if ph > sub_h or pw > sub_w or oh > cap_h or ow > cap_w:
                continue

            if has_h:
                inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
            else:
                split_rows = sorted(r for r, c in split_pixels if c == int(c_split))
                if split_rows:
                    mid = int((split_rows[0] + split_rows[-1]) // 2)
                    inp_r0 = max(int(r0), min(int(mid - ph // 2), int(r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
            else:
                split_cols = sorted(c for r, c in split_pixels if r == int(r_split))
                if split_cols:
                    mid = int((split_cols[0] + split_cols[-1]) // 2)
                    inp_c0 = max(int(c0), min(int(mid - pw // 2), int(c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
            ok = True
            for r, c in inp_pts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    ok = False
                    break
                if has_v and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _place_piece_to_corner(inp_pts, H, W, dh, dv, int(krot))
            for r, c in out_pts:
                if has_v and ((dv == "L" and c >= int(c_split)) or (dv == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((dh == "T" and r >= int(r_split)) or (dh == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            input_pieces[reg] = inp_pts
            output_pieces[reg] = out_pts
            placed = True
            break

        if not placed:
            return None

    all_inp_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in input_pieces.values():
        if all_inp_fg & set(pts):
            return None
        all_inp_fg |= set(pts)
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
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(int(vanish_budget), int(vh * vw))
            if max_px < 3:
                return None

            accepted = False
            for _ in range(60):
                piece = _grow_piece(vh, vw, max_px)
                if len(piece) < 3:
                    continue
                pph, ppw = _piece_hw(piece)
                vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
                vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
                vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
                ok = True
                for r, c in vpts:
                    if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                        ok = False
                        break
                    if (r, c) in all_inp_fg:
                        ok = False
                        break
                if not ok:
                    continue
                vanish_fg = set(vpts)
                accepted = True
                break
            if not accepted:
                return None

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc, color in ctrl_pts:
        inp[int(cr), int(cc)] = int(color)

    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    ray_src = out.copy()
    for cr, cc, color in ctrl_pts:
        _draw_ray(out, ray_src, int(fg), int(cr), int(cc), int(color))
    for cr, cc, color in ctrl_pts:
        out[int(cr), int(cc)] = int(color)

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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
