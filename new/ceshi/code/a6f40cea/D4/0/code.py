# Variant of a6f40cea
# Dimension: D4 — 终点/停止条件 (dst)
# Change: 路径连接同一开放矩形框的两个触碰点（即补全开放框的缺失部分）... → 跨组件连接

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the D4 variant
# ----------------------------------
# yaml:
#   variant_core:
#     original: Each outside open rectangle was solved locally: its own two frame-touch
#       points were projected inward and then connected inside the frame.
#     new: Two separate outside components with the same visible palette are paired.
#       Their touchpoints are connected across the frame, so the destination of a path
#       lies on another component, not on the source component itself.
#   concrete_shape:
#     frame:
#       Same as the original task: one 1px complete rectangle, border color 3, uniform fill.
#     ring_components:
#       Exactly two 4-connected open rectangular traces live outside the frame.
#       They sit in opposite corners: top-left and bottom-right.
#       Each trace touches the frame ring at two cells on adjacent sides.
#       The two traces share either:
#         - one solid color, or
#         - the same 2-color alternating palette.
#     output:
#       Crop the frame interior.
#       Use the top touchpoint from the top-left component and the right touchpoint from
#       the bottom-right component to draw one long Manhattan L-path.
#       Use the left touchpoint from the top-left component and the bottom touchpoint from
#       the bottom-right component to draw the complementary L-path.
#       So each outside component contributes endpoints, but no component is completed by itself.
#   behavior_trace:
#     example:
#       interior_size: 7x9
#       top_left_touches: top@(0,2), left@(3,0)
#       bottom_right_touches: bottom@(6,5), right@(4,8)
#     original_solver:
#       It would connect (0,2) to (3,0) and separately connect (6,5) to (4,8),
#       yielding the two local missing corners.
#     new_solver:
#       It instead connects (0,2) to (4,8) and (3,0) to (6,5),
#       swapping the destinations across components.
#     visible_difference:
#       The output now uses the opposite component's endpoint as the destination,
#       which is visually distinct from "repair each corner locally".
#   complexity_calibration:
#     chosen_constraint:
#       Use exactly one palette-paired component pair per grid.
#     reason:
#       This keeps the pairing rule unambiguous and easy to infer from 2-4 samples,
#       while still making the D4 change obvious because the produced paths are global.
#     retained_variation:
#       The pair can be monochrome or 2-color alternating, and the crossing row/column vary.
#   implicit_assumption_audit:
#     old_main_dependency_1:
#       The original loop grouped by connected component and assumed each relevant path was
#       recoverable entirely from one component's own touchpoints.
#       This no longer holds, so per-component painting must be replaced by palette pairing.
#     old_main_dependency_2:
#       The original deferred monochrome logic merged same-color points from many components.
#       Under the new rule that would accidentally reconstruct original-like stripes, so it is removed.
#     old_main_dependency_3:
#       The original 2-color phase logic only constrained one local path.
#       The new variant has two cross-component paths whose destinations are swapped, so generator
#       and solver must agree on endpoint colors and parity to avoid contradictory overlap cells.
#   placement_legality:
#     generator_guarantees:
#       - the two ring-touching components are opposite-corner open rectangles and do not overlap
#       - the top-left component uses an earlier interior row/column than the bottom-right component,
#         so the two cross-component L-paths are well-formed and visibly swapped
#       - the chosen crossing row/column stay away from the frame corners, so every touchpoint
#         belongs to exactly one side and side detection is unambiguous
#       - for 2-color cases, component endpoint parity is searched so the vertical and horizontal
#         stripes agree at their intersection
#   visual_observability:
#     pairing_signal:
#       The only repeated non-frame palette on the ring is the palette of the two true components.
#       That repeated visible palette is the cue for pairing; no hidden numeric ordering is used.
#     cleanliness:
#       Only one swapped pair is produced inside the frame, so the interior stays legible and does not
#       accumulate multiple incompatible crossings.


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the largest 1px complete rectangle with uniform fill."""
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _other_color(palette: tuple[int, int], col: int) -> int:
    a, b = map(int, palette)
    return int(b) if int(col) == int(a) else int(a)


def _toggle_steps(palette: tuple[int, int], start: int, steps: int) -> int:
    cur = int(start)
    for _ in range(int(steps)):
        cur = _other_color(palette, cur)
    return int(cur)


def _paint_path_with_start(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...], start: int | None = None) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    palette = (int(colors[0]), int(colors[1]))
    cur = int(palette[0] if start is None else start)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = _other_color(palette, cur)
    grid[rs, cs] = vals


def _component_outer(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    inner_r: int,
    inner_c: int,
    off_r: int,
    off_c: int,
) -> tuple[list[tuple[int, int]], dict[str, tuple[int, int]]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + inner_c)
    ring_b = (r0 + 1 + inner_r, c0 - 1 if is_left else c1 + 1)
    side_a = "top" if is_top else "bottom"
    side_b = "left" if is_left else "right"

    R = (r0 - int(off_r)) if is_top else (r1 + int(off_r))
    C = (c0 - int(off_c)) if is_left else (c1 + int(off_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return outer, {side_a: ring_a, side_b: ring_b}


def _candidate_components(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    inner_r: int,
    inner_c: int,
    offsets: tuple[int, int],
) -> list[dict]:
    out = []
    lo, hi = offsets
    for off_r in range(lo, hi + 1):
        for off_c in range(lo, hi + 1):
            path, touches = _component_outer(kind, H, W, r0, c0, r1, c1, inner_r, inner_c, off_r, off_c)
            if len(set(path)) != len(path):
                continue
            out.append(
                {
                    "path": path,
                    "touches": touches,
                    "same_endpoints": (len(path) % 2 == 1),
                    "off_r": int(off_r),
                    "off_c": int(off_c),
                }
            )
    return out


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    *,
    start: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path_with_start(grid, path, colors, start=start)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
    *,
    tries: int,
    size_range: tuple[int, int],
    clearance: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    sz_lo, sz_hi = size_range
    for _ in range(int(tries)):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - clearance <= rr <= r1 + clearance and c0 - clearance <= cc <= c1 + clearance for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _side_of_point(r: int, c: int, outH: int, outW: int) -> str:
    sides = []
    if int(r) == 0:
        sides.append("top")
    if int(r) == int(outH - 1):
        sides.append("bottom")
    if int(c) == 0:
        sides.append("left")
    if int(c) == int(outW - 1):
        sides.append("right")
    assert len(sides) == 1
    return sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover cross-component Manhattan paths inside a framed rectangle by pairing outside corner traces.

    Input structure:
    - A single closed 1-cell rectangular frame with uniform interior fill.
    - Two separate open rectangular traces outside the frame, each touching two adjacent sides.
    - The two traces share either one solid color or the same 2-color alternating palette.

    Rule:
    - Output only the frame interior, starting from the fill color.
    - Project each ring-touch cell to the corresponding interior-edge coordinate.
    - Pair the two components by shared palette instead of closing either component locally.
    - Draw one connection from the top-left component's top touchpoint to the bottom-right component's right touchpoint.
    - Draw the complementary connection from the top-left component's left touchpoint to the bottom-right component's bottom touchpoint.
      Monochrome pairs stay solid; 2-color pairs alternate and keep the endpoint colors seen on the ring.

    Key insight:
    - The repeated palette identifies which two separate components belong together,
      and the visible touchpoint colors determine the phase after the two destinations are swapped.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def paint_checked(path: list[tuple[int, int]], colors: list[int]) -> None:
        for (rr, cc), col in zip(path, colors):
            prev = int(out[rr, cc])
            if prev != int(fill):
                assert prev == int(col)
            out[rr, cc] = int(col)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, int(g[rr, cc])))

    groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for cid, pts in comp_pts.items():
        palette = tuple(sorted(set(map(int, g[clab == int(cid)].flatten()))))
        assert len(palette) in (1, 2)

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        touches = {}
        for (r, c), ctr in hist.items():
            side = _side_of_point(int(r), int(c), outH, outW)
            touches[side] = {"coord": (int(r), int(c)), "color": int(ctr.most_common(1)[0][0])}

        if len(touches) != 2:
            continue
        groups[palette].append({"cid": int(cid), "palette": palette, "touches": touches})

    used_group = False
    for palette, comps in groups.items():
        if len(comps) != 2:
            continue
        used_group = True
        comp_a, comp_b = comps
        sides_a = set(comp_a["touches"])
        sides_b = set(comp_b["touches"])
        assert sides_a == {"top", "left"} and sides_b == {"bottom", "right"}

        top = comp_a["touches"]["top"]
        left = comp_a["touches"]["left"]
        bottom = comp_b["touches"]["bottom"]
        right = comp_b["touches"]["right"]

        top_to_right = _join(
            _vpath(int(top["coord"][0]), int(right["coord"][0]), int(top["coord"][1])),
            _hpath(int(right["coord"][0]), int(top["coord"][1]), int(right["coord"][1])),
        )
        left_to_bottom = _join(
            _hpath(int(left["coord"][0]), int(left["coord"][1]), int(bottom["coord"][1])),
            _vpath(int(left["coord"][0]), int(bottom["coord"][0]), int(bottom["coord"][1])),
        )

        if len(palette) == 1:
            col = int(palette[0])
            paint_checked(top_to_right, [col] * len(top_to_right))
            paint_checked(left_to_bottom, [col] * len(left_to_bottom))
            continue

        pal2 = (int(palette[0]), int(palette[1]))

        def alt_colors(path: list[tuple[int, int]], start_col: int, end_col: int) -> list[int]:
            chosen = None
            for start in pal2:
                cur_end = _toggle_steps(pal2, int(start), len(path) - 1)
                if int(start) == int(start_col) and int(cur_end) == int(end_col):
                    chosen = int(start)
                    break
            assert chosen is not None
            vals = []
            cur = int(chosen)
            for _ in path:
                vals.append(int(cur))
                cur = _other_color(pal2, cur)
            return vals

        top_to_right_colors = alt_colors(top_to_right, int(top["color"]), int(right["color"]))
        left_to_bottom_colors = alt_colors(left_to_bottom, int(left["color"]), int(bottom["color"]))
        paint_checked(top_to_right, top_to_right_colors)
        paint_checked(left_to_bottom, left_to_bottom_colors)

    assert used_group
    return out


# --------------- generator (a6 D4) ---------------
_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)

    palette_pool = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette_pool)
    if len(palette_pool) < 2:
        return None

    inner_r_tl = random.randint(1, H - 3)
    inner_c_tl = random.randint(1, W - 3)
    inner_r_br = random.randint(inner_r_tl + 1, H - 2)
    inner_c_br = random.randint(inner_c_tl + 1, W - 2)

    cand_tl = _candidate_components("tl", H, W, r0, c0, r1, c1, inner_r_tl, inner_c_tl, _OUTER_OFFSET_RANGE)
    cand_br = _candidate_components("br", H, W, r0, c0, r1, c1, inner_r_br, inner_c_br, _OUTER_OFFSET_RANGE)
    if not cand_tl or not cand_br:
        return None

    if random.random() < _DUAL_COLOR_PROB and len(palette_pool) >= 2:
        palette = (int(palette_pool[0]), int(palette_pool[1]))
        need_same_tl = (inner_r_tl % 2) == (inner_c_tl % 2)
        tl_options = [cand for cand in cand_tl if bool(cand["same_endpoints"]) == bool(need_same_tl)]
        if not tl_options:
            return None

        top_color = random.choice(palette)
        left_color = int(top_color) if need_same_tl else _other_color(palette, int(top_color))
        right_steps = int(inner_r_br + (W - 1 - inner_c_tl))
        bottom_steps = int(inner_c_br + (H - 1 - inner_r_tl))
        right_color = _toggle_steps(palette, int(top_color), right_steps)
        bottom_color = _toggle_steps(palette, int(left_color), bottom_steps)
        need_same_br = int(bottom_color) == int(right_color)
        br_options = [cand for cand in cand_br if bool(cand["same_endpoints"]) == bool(need_same_br)]
        if not br_options:
            return None

        chosen_tl = random.choice(tl_options)
        chosen_br = random.choice(br_options)
        if not _place_component(grid, comp_mask, chosen_tl["path"], palette, bg, start=int(top_color)):
            return None
        if not _place_component(grid, comp_mask, chosen_br["path"], palette, bg, start=int(bottom_color)):
            return None

        _paint_path_with_start(
            expected,
            _join(_vpath(0, inner_r_br, inner_c_tl), _hpath(inner_r_br, inner_c_tl, W - 1)),
            palette,
            start=int(top_color),
        )
        _paint_path_with_start(
            expected,
            _join(_hpath(inner_r_tl, 0, inner_c_br), _vpath(inner_r_tl, H - 1, inner_c_br)),
            palette,
            start=int(left_color),
        )
    else:
        color = int(palette_pool[0])
        chosen_tl = random.choice(cand_tl)
        chosen_br = random.choice(cand_br)
        if not _place_component(grid, comp_mask, chosen_tl["path"], (color,), bg):
            return None
        if not _place_component(grid, comp_mask, chosen_br["path"], (color,), bg):
            return None
        _paint_path_with_start(
            expected,
            _join(_vpath(0, inner_r_br, inner_c_tl), _hpath(inner_r_br, inner_c_tl, W - 1)),
            (color,),
        )
        _paint_path_with_start(
            expected,
            _join(_hpath(inner_r_tl, 0, inner_c_br), _vpath(inner_r_tl, H - 1, inner_c_br)),
            (color,),
        )

    next_idx = 2 if np.unique(expected).size > 2 else 1
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if next_idx >= len(palette_pool):
            break
        _add_distractor(
            grid,
            comp_mask,
            bg,
            (r0, c0, r1, c1),
            int(palette_pool[next_idx]),
            tries=_DISTRACTOR_TRIES,
            size_range=_DISTRACTOR_SIZE_RANGE,
            clearance=_DISTRACTOR_CLEARANCE,
        )
        next_idx += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea D4 input")
