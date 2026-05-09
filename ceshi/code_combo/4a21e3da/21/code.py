# Combo variant of 4a21e3da (index 21)
# combo_spec: D3/0+D8/2
# Name: 边缘色条源+同侧双控制
#
# Design analysis
# a) Concrete behavior and trace:
#    - The seed task used 1-2 border pixels. Here the input always has exactly two ctrl-colored
#      short bars, each length 2 or 3, both anchored to the same edge.
#    - Example: two top bars at columns 7 and 11 cut the sprite into left / middle / right
#      vertical stripes. Rays still travel inward on those two columns and only paint bg cells.
#    - Outer stripes behave like the D8 same-edge-double-control variant:
#      left stripe -> top-left, right stripe -> top-right.
#    - The D3 signal is fused into the middle stripe. Each bar length encodes a side vote:
#      vertical bars: len 2 votes left, len 3 votes right.
#      horizontal bars: len 2 votes up, len 3 votes down.
#      If both bars vote the same side, the middle stripe survives and slides to the opposite
#      edge corner on that side. If the votes disagree, the middle stripe disappears.
#    - This differs from plain D8 because the middle stripe is not always erased, and it differs
#      from plain D3 because the encoded lengths only matter once the two parallel cuts create
#      a third stripe to reason about.
# b) Complexity calibration:
#    - Only three middle outcomes exist: opposite-left/up, vanish, opposite-right/down.
#    - The only code values are bar lengths 2 and 3, reused on both bars, so the visual cue is
#      repeated and human-visible.
#    - The generator balances the three middle states so the task does not collapse to the
#      always-vanish D8 behavior.
# c) Implicit assumption audit against the seed main():
#    - Original code assumed every ctrl cell was an independent pixel. That breaks here; ctrl
#      cells must be grouped into two edge-anchored straight bar components.
#    - Original code stored one split per axis. That breaks here because two splits on one axis
#      are active simultaneously; the solver now uses sorted split pairs and ternary band labels.
#    - Original survival logic depended on touching the control edge. That would wrongly keep the
#      middle band, so survival is now decided directly from outer/middle band role plus bar votes.
#    - The ray-painting loop remains valid once bars are parsed into (edge, split) sources.
# d) Placement legality guaranteed by the generator:
#    - Bars are length 2-3 and the sprite keeps a >=4-cell edge margin, so bars never overlap fg.
#    - The two splits are distinct and separated enough to give positive left/middle/right widths.
#    - Input fg is built as one connected component using bridge rows on both split lines.
#    - Output destinations use three different corners at most, so moved pieces do not overlap.
# e) Visual observability:
#    - Two parallel same-edge bars are easy to spot.
#    - The middle stripe is visibly present in the input in every sample.
#    - Across samples, the middle stripe sometimes vanishes and sometimes reappears in the
#      opposite-side corner, with the deciding cue being the repeated 2-vs-3 bar length.
# f) Visual cleanliness:
#    - Only the original three colors are used.
#    - The sprite stays near the center; bars and rays stay in the empty margins.
#    - The middle behavior is expressed with one extra moved piece or absence, not noisy pixel edits.
# g) Color inferability:
#    - No color remapping is introduced; geometry and bar length carry the rule.
# Post-edit solver assumptions matched by the generator:
#    - Exactly two ctrl components exist, both straight 2-3-cell bars on the same edge.
#    - Every split line contains fg bridge pixels, so every ray has a stopping point.
#    - There is one connected fg stripe piece in each of the left/middle/right bands.

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import colors_present, get_bg, is_contiguous


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bar_cells(edge: str, split: int, length: int, H: int, W: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(r, split) for r in range(length)]
    if edge == "bottom":
        return [(H - 1 - r, split) for r in range(length)]
    if edge == "left":
        return [(split, c) for c in range(length)]
    if edge == "right":
        return [(split, W - 1 - c) for c in range(length)]
    raise ValueError(edge)


def _edge_corner(edge: str, side: str) -> tuple[str, str]:
    if edge == "top":
        return "T", side
    if edge == "bottom":
        return "B", side
    if edge == "left":
        return side, "L"
    if edge == "right":
        return side, "R"
    raise ValueError(edge)


def _opposite_edge(edge: str) -> str:
    if edge == "top":
        return "bottom"
    if edge == "bottom":
        return "top"
    if edge == "left":
        return "right"
    if edge == "right":
        return "left"
    raise ValueError(edge)


def _draw_ray(out: np.ndarray, g: np.ndarray, bg: int, fg: int, ctrl: int, edge: str, split: int) -> None:
    H, W = g.shape
    if edge == "top":
        rows = [int(rr) for rr in np.argwhere(g[:, split] == fg).flatten()]
        if rows:
            for rr in range(0, max(rows) + 1):
                if int(g[rr, split]) == bg:
                    out[rr, split] = ctrl
    elif edge == "bottom":
        rows = [int(rr) for rr in np.argwhere(g[:, split] == fg).flatten()]
        if rows:
            for rr in range(H - 1, min(rows) - 1, -1):
                if int(g[rr, split]) == bg:
                    out[rr, split] = ctrl
    elif edge == "left":
        cols = [int(cc) for cc in np.argwhere(g[split, :] == fg).flatten()]
        if cols:
            for cc in range(0, max(cols) + 1):
                if int(g[split, cc]) == bg:
                    out[split, cc] = ctrl
    else:
        cols = [int(cc) for cc in np.argwhere(g[split, :] == fg).flatten()]
        if cols:
            for cc in range(W - 1, min(cols) - 1, -1):
                if int(g[split, cc]) == bg:
                    out[split, cc] = ctrl


def _slide_points_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if dh == "T" else (H - 1 - rmax)
    dc = -cmin if dv == "L" else (W - 1 - cmax)
    return {(r + dr, c + dc) for r, c in pts}


def _parse_bars(g: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    bars: list[dict[str, object]] = []
    for comp in _components(g == ctrl):
        rs = sorted({int(r) for r, _ in comp})
        cs = sorted({int(c) for _, c in comp})
        comp_set = set(comp)
        if len(cs) == 1 and rs[0] == 0:
            edge = "top"
            split = int(cs[0])
            expect = set((r, split) for r in range(len(rs)))
        elif len(cs) == 1 and rs[-1] == H - 1:
            edge = "bottom"
            split = int(cs[0])
            expect = set((H - 1 - r, split) for r in range(len(rs)))
        elif len(rs) == 1 and cs[0] == 0:
            edge = "left"
            split = int(rs[0])
            expect = set((split, c) for c in range(len(cs)))
        elif len(rs) == 1 and cs[-1] == W - 1:
            edge = "right"
            split = int(rs[0])
            expect = set((split, W - 1 - c) for c in range(len(cs)))
        else:
            raise AssertionError("Control component is not an edge-anchored straight bar")

        assert comp_set == expect
        length = len(comp)
        assert length in (2, 3)
        bars.append(
            {
                "edge": edge,
                "split": int(split),
                "length": int(length),
                "cells": list(expect),
            }
        )

    assert len(bars) == 2
    edges = {str(bar["edge"]) for bar in bars}
    assert len(edges) == 1
    edge = next(iter(edges))
    if edge in ("top", "bottom"):
        bars.sort(key=lambda bar: int(bar["split"]))
    else:
        bars.sort(key=lambda bar: int(bar["split"]))
    assert int(bars[0]["split"]) < int(bars[1]["split"])
    return ctrl, fg, bars


def _middle_vote(edge: str, lengths: list[int]) -> str | None:
    if edge in ("top", "bottom"):
        prefs = ["L" if length == 2 else "R" for length in lengths]
    else:
        prefs = ["T" if length == 2 else "B" for length in lengths]
    return prefs[0] if prefs[0] == prefs[1] else None


def _classify_band(edge: str, splits: list[int], r: int, c: int) -> str | None:
    s1, s2 = splits
    if edge in ("top", "bottom"):
        if c == s1 or c == s2:
            return None
        if c < s1:
            return "L"
        if c > s2:
            return "R"
        return "M"
    if r == s1 or r == s2:
        return None
    if r < s1:
        return "T"
    if r > s2:
        return "B"
    return "M"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ctrl, fg, bars = _parse_bars(g)

    edge = str(bars[0]["edge"])
    splits = [int(bar["split"]) for bar in bars]
    lengths = [int(bar["length"]) for bar in bars]
    middle_pref = _middle_vote(edge, lengths)

    out = np.full((H, W), bg, dtype=int)
    for bar in bars:
        _draw_ray(out, g, bg, fg, ctrl, str(bar["edge"]), int(bar["split"]))
    for bar in bars:
        for r, c in bar["cells"]:
            out[int(r), int(c)] = ctrl

    regions: dict[str, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        band = _classify_band(edge, splits, rr, cc)
        if band is None:
            out[rr, cc] = fg
            continue
        regions.setdefault(band, []).append((rr, cc))

    if edge in ("top", "bottom"):
        keep_to_dest: dict[str, tuple[str, str]] = {
            "L": _edge_corner(edge, "L"),
            "R": _edge_corner(edge, "R"),
        }
        if middle_pref is not None:
            keep_to_dest["M"] = _edge_corner(_opposite_edge(edge), middle_pref)
    else:
        keep_to_dest = {
            "T": _edge_corner(edge, "T"),
            "B": _edge_corner(edge, "B"),
        }
        if middle_pref is not None:
            keep_to_dest["M"] = _edge_corner(_opposite_edge(edge), middle_pref)

    for band, pts in regions.items():
        if band not in keep_to_dest:
            continue
        dh, dv = keep_to_dest[band]
        for rr, cc in _slide_points_to_corner(pts, H, W, dh, dv):
            out[rr, cc] = fg

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


def _state_to_lengths(state: str) -> list[int]:
    if state == "left":
        return [2, 2]
    if state == "right":
        return [3, 3]
    return list(random.choice([(2, 3), (3, 2)]))


def _draw_expected_ray(
    out: np.ndarray,
    bg: int,
    ctrl: int,
    edge: str,
    split: int,
    split_pixels: set[tuple[int, int]],
) -> None:
    H, W = out.shape
    if edge == "top":
        rows = [r for r, c in split_pixels if c == split]
        for r in range(0, max(rows) + 1):
            if out[r, split] == bg:
                out[r, split] = ctrl
    elif edge == "bottom":
        rows = [r for r, c in split_pixels if c == split]
        for r in range(H - 1, min(rows) - 1, -1):
            if out[r, split] == bg:
                out[r, split] = ctrl
    elif edge == "left":
        cols = [c for r, c in split_pixels if r == split]
        for c in range(0, max(cols) + 1):
            if out[split, c] == bg:
                out[split, c] = ctrl
    else:
        cols = [c for r, c in split_pixels if r == split]
        for c in range(W - 1, min(cols) - 1, -1):
            if out[split, c] == bg:
                out[split, c] = ctrl


def _build_vertical_case(state: str, edge: str) -> tuple[np.ndarray, np.ndarray] | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(9, 11)
    possible_widths = []
    for left_w in range(2, sprite_w - 3):
        for mid_w in range(2, sprite_w - left_w - 1):
            right_w = sprite_w - 2 - left_w - mid_w
            if right_w >= 2:
                possible_widths.append((left_w, mid_w, right_w))
    if not possible_widths:
        return None
    left_w, mid_w, right_w = random.choice(possible_widths)
    s1 = left_w
    s2 = left_w + 1 + mid_w

    sprite_r0 = random.randint(4, H - sprite_h - 4)
    bridge_len = random.randint(2, min(4, sprite_h - 1))
    bridge_r0 = random.randint(sprite_r0 + 1, sprite_r0 + sprite_h - bridge_len - 1)
    bridge_rows = list(range(bridge_r0, bridge_r0 + bridge_len))

    valid_c0 = []
    for sprite_c0 in range(4, W - sprite_w - 3):
        global_s1 = sprite_c0 + s1
        global_s2 = sprite_c0 + s2
        if int(W * 0.3) <= global_s1 < global_s2 <= int(W * 0.7) and int(W * 0.4) <= (global_s1 + global_s2) / 2 <= int(W * 0.6):
            valid_c0.append(sprite_c0)
    if not valid_c0:
        return None
    sprite_c0 = random.choice(valid_c0)

    split_pixels = {(r, sprite_c0 + s1) for r in bridge_rows} | {(r, sprite_c0 + s2) for r in bridge_rows}

    left_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * left_w, bridge_len + 7))
    left_must = {(r - sprite_r0, left_w - 1) for r in bridge_rows}
    left_local = _grow_piece_in_box(sprite_h, left_w, left_target, left_must)
    left_pts = {(sprite_r0 + r, sprite_c0 + c) for r, c in left_local}

    mid_target = random.randint(max(mid_w + 1, 4), min(sprite_h * mid_w, mid_w + bridge_len + 4))
    mid_row = random.choice(bridge_rows) - sprite_r0
    mid_must = {(mid_row, c) for c in range(mid_w)}
    mid_local = _grow_piece_in_box(sprite_h, mid_w, mid_target, mid_must)
    mid_pts = {(sprite_r0 + r, sprite_c0 + s1 + 1 + c) for r, c in mid_local}

    right_target = random.randint(max(bridge_len + 2, 4), min(sprite_h * right_w, bridge_len + 7))
    right_must = {(r - sprite_r0, 0) for r in bridge_rows}
    right_local = _grow_piece_in_box(sprite_h, right_w, right_target, right_must)
    right_pts = {(sprite_r0 + r, sprite_c0 + s2 + 1 + c) for r, c in right_local}

    combined = set(split_pixels)
    for pts in [left_pts, mid_pts, right_pts]:
        if combined & pts:
            return None
        combined |= pts

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

    lengths = _state_to_lengths(state)
    bar_specs = [
        {"edge": edge, "split": sprite_c0 + s1, "length": lengths[0]},
        {"edge": edge, "split": sprite_c0 + s2, "length": lengths[1]},
    ]

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for bar in bar_specs:
        for r, c in _bar_cells(str(bar["edge"]), int(bar["split"]), int(bar["length"]), H, W):
            if inp[r, c] != bg:
                return None
            inp[r, c] = ctrl

    expected = np.full((H, W), bg, dtype=int)
    left_out = _slide_points_to_corner(left_pts, H, W, *_edge_corner(edge, "L"))
    right_out = _slide_points_to_corner(right_pts, H, W, *_edge_corner(edge, "R"))
    for pts in [left_out, right_out]:
        for r, c in pts:
            expected[r, c] = fg

    if state == "left":
        mid_out = _slide_points_to_corner(mid_pts, H, W, *_edge_corner(_opposite_edge(edge), "L"))
        for r, c in mid_out:
            expected[r, c] = fg
    elif state == "right":
        mid_out = _slide_points_to_corner(mid_pts, H, W, *_edge_corner(_opposite_edge(edge), "R"))
        for r, c in mid_out:
            expected[r, c] = fg

    for r, c in split_pixels:
        expected[r, c] = fg
    for bar in bar_specs:
        _draw_expected_ray(expected, bg, ctrl, str(bar["edge"]), int(bar["split"]), split_pixels)
    for bar in bar_specs:
        for r, c in _bar_cells(str(bar["edge"]), int(bar["split"]), int(bar["length"]), H, W):
            expected[r, c] = ctrl

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    state = random.choice(["left", "vanish", "right"])
    final_edge = random.choice(["top", "bottom", "left", "right"])

    if final_edge in ("top", "bottom"):
        built = _build_vertical_case(state, final_edge)
        if built is None:
            return None
        inp, expected = built
        if not np.array_equal(main(inp), expected):
            return None
        return inp

    mapped_edge = "top" if final_edge == "left" else "bottom"
    built = _build_vertical_case(state, mapped_edge)
    if built is None:
        return None
    inp_v, expected_v = built
    inp = np.array(inp_v.T, dtype=int)
    expected = np.array(expected_v.T, dtype=int)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
