# Variant of 35ab12c3
# Dimension: D8 — 终止条件/端点规则 (dst)
# Change: 无独立终点——路径在同色像素对之间直接连接，终点就是配对的另一个像素... → 遇到障碍物停止

# Design analysis for the D8 variant
# a) Concrete behavior / trace:
#    The original task connects each same-color vertex pair across the full H/V/45deg span,
#    then copies one parallel "shadow" path from the chosen start vertex's foreign-color
#    neighbor. In this variant, a segment is no longer all-or-nothing with respect to the
#    other endpoint: it advances cell by cell and halts immediately when it reaches a cell
#    already occupied by a different group. The distant endpoint still remains because the
#    solver starts from a copy of the input grid. Small example:
#      - Original: a late horizontal pair at row 8 would fill all cells from c=2..12 even
#        if an earlier vertical group already occupies (8,7); it would simply skip that
#        occupied cell and continue to the right.
#      - Variant: the same horizontal pair fills only c=2..6, stops before (8,7), and the
#        right endpoint at c=12 stays isolated.
#    The change therefore affects the segment-drawing stage, not vertex detection.
# b) Complexity calibration:
#    If all generated groups remain far apart, the new rule degenerates to the original.
#    To keep D8 visible, the generator always creates one early vertical blocker and one or
#    more later horizontal victims that would cross it under the old rule. Repeating that
#    interaction within a sample provides direct evidence that "later paths stop at earlier
#    regions" is the governing rule.
# c) Hidden assumptions in the original main():
#    - Iterating colors in sorted color-id order was harmless before because groups never
#      interacted. Under D8, that would create an invisible order dependency. The variant
#      replaces it with a visible order: groups are processed by the chosen start vertex
#      position (top-left first).
#    - `_clear_in_original(...)` assumes a segment is drawable iff the whole interior is
#      obstacle-free in the input. That is wrong for D8 because a partially drawable segment
#      should still produce a prefix. The variant removes this binary precheck.
#    - `_draw_line_bg_only(...)` assumes "skip occupied cell and continue" is safe. Under
#      D8 it is incorrect, because the first foreign occupied cell is a terminal event.
#    The reference task 195c6913 also uses stepwise motion with stop/turn logic instead of a
#    global clearance precheck, which is the right complexity baseline for this change.
# d) Placement legality:
#    The generator keeps all input pixels disjoint and in-bounds. It reserves a tall central
#    vertical line as the blocker, then places later horizontal pairs on distinct rows that
#    cross that column. Optional shadow anchors sit one row above/below the left endpoint, so
#    they identify the start vertex without colliding with other rows.
# e) Visual observability:
#    Order is encoded by geometry, not color ids. The blocker's start vertex is visibly
#    higher than the victims' starts, so a human can infer that upper groups are drawn first.
#    The output shows truncated horizontals plus intact far endpoints, which is a strong cue
#    that stopping happens mid-segment rather than at the peer vertex.
# f) Visual cleanliness:
#    Only one crossing family is allowed per sample: a single vertical blocker with 1-3
#    horizontal victims. This keeps overlap localized and readable. Optional distractors stay
#    away from every segment corridor so they do not create accidental clutter.
# g) Color inference:
#    Colors remain identity-preserving. The only color change is the original shadow-copy
#    behavior, now also subject to stopping. No color remapping or color-based sorting is
#    introduced.
# 9) New solver assumptions and generator guarantees:
#    - `main()` now assumes the intended draw order is readable from start-vertex positions.
#      The generator guarantees strictly increasing start rows across interacting groups.
#    - `main()` assumes same-group colors should not block each other; the generator uses
#      unique colors per group/shadow so cross-group blocking is unambiguous.
#    - `main()` assumes at least one victim truly crosses the blocker for the variant to be
#      observable; the generator verifies this by comparing the new solver with an unblocked
#      baseline using the same visible group order.

from __future__ import annotations

import math
import numpy as np

from arc_utils import colors_present, get_bg as _bg_color


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _collect_groups(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    groups: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs: list[int] = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        groups.append(
            {
                "color": col,
                "cycle": cycle,
                "start": cycle[0],
                "edges": edges,
                "carried": carried,
                "group_colors": {int(col), *(int(v) for _, _, v in carried)},
            }
        )

    groups.sort(key=lambda info: info["start"])
    return groups


def _draw_line_until_block(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    allow: set[int],
) -> tuple[bool, int]:
    first = True
    painted = 0
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        v = int(out[rr, cc])
        if v != bg and v not in allow:
            return True, painted
        if v == bg:
            out[rr, cc] = col
            painted += 1
    return False, painted


def _draw_line_skip_occupied(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _run_solver(grid: np.ndarray, *, stop_on_block: bool, collect_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()
    groups = _collect_groups(g, bg)

    meta = {
        "blocked_main": 0,
        "blocked_shadow": 0,
        "groups": len(groups),
    }

    for info in groups:
        col = int(info["color"])
        carried = info["carried"]
        group_colors = set(int(v) for v in info["group_colors"])
        edges = [edge for edge in info["edges"] if _aligned_8(edge[0][0], edge[0][1], edge[1][0], edge[1][1])]

        for (r0, c0), (r1, c1) in edges:
            if stop_on_block:
                blocked, _ = _draw_line_until_block(out, r0, c0, r1, c1, col, bg, group_colors)
                if blocked:
                    meta["blocked_main"] += 1
            else:
                _draw_line_skip_occupied(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if stop_on_block:
                    blocked, _ = _draw_line_until_block(out, rr0, cc0, rr1, cc1, int(ncol), bg, group_colors)
                    if blocked:
                        meta["blocked_shadow"] += 1
                else:
                    _draw_line_skip_occupied(out, rr0, cc0, rr1, cc1, int(ncol), bg)

    if collect_meta:
        return out, meta
    return out


def _main_unblocked_visible_order(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, stop_on_block=False, collect_meta=False)


def main(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, stop_on_block=True, collect_meta=False)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.35
GENERATE_RETRIES = 400

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count - 1, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count : shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _shape_output_cells(shape: dict) -> set[tuple[int, int]]:
    cells = set(_shape_cells(shape["vertices"]))
    shadow = shape.get("shadow")
    if shadow is not None:
        dr, dc = shadow["offset"]
        cells.update((r + dr, c + dc) for r, c in _shape_cells(shape["vertices"]))
    return cells


def _shape_input_cells(shape: dict) -> set[tuple[int, int]]:
    cells = set(shape["vertices"])
    shadow = shape.get("shadow")
    if shadow is not None:
        ar, ac = shape["vertices"][int(shadow["anchor_idx"])]
        dr, dc = shadow["offset"]
        cells.add((ar + dr, ac + dc))
    return cells


def _sample_shadow_offset(
    row: int,
    col: int,
    h: int,
    w: int,
    used_input: set[tuple[int, int]],
    preferred: list[tuple[int, int]],
) -> tuple[int, int] | None:
    for dr, dc in preferred:
        rr = row + dr
        cc = col + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if (rr, cc) in used_input:
            continue
        return dr, dc
    return None


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned_8(coords[i][0], coords[i][1], coords[j][0], coords[j][1]):
                return None
    return coords


def _place_distractor(
    inp: np.ndarray,
    color: int,
    reserved: set[tuple[int, int]],
) -> bool:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        for _ in range(DISTRACTOR_RETRIES):
            r = int(np.random.randint(inp.shape[0]))
            c = int(np.random.randint(inp.shape[1]))
            if inp[r, c] != BG or (r, c) in reserved:
                continue
            inp[r, c] = color
            return True
        return False

    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        r0 = int(np.random.randint(0, inp.shape[0] - h + 1))
        c0 = int(np.random.randint(0, inp.shape[1] - w + 1))
        placed = [(r0 + r, c0 + c) for r, c in coords]
        if any(inp[r, c] != BG or (r, c) in reserved for r, c in placed):
            continue
        for r, c in placed:
            inp[r, c] = color
        return True
    return False


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    blocked_out, meta = _run_solver(inp, stop_on_block=True, collect_meta=True)
    unblocked_out = _main_unblocked_visible_order(inp)
    diff_cells = int(np.sum(blocked_out != unblocked_out))
    return {
        "blocked_main": int(meta["blocked_main"]),
        "blocked_shadow": int(meta["blocked_shadow"]),
        "diff_cells": diff_cells,
    }


def _build_interacting_shapes(
    h: int,
    w: int,
    shape_colors: list[int],
    shadow_colors: list[int],
) -> list[dict] | None:
    shape_count = len(shape_colors)
    victim_count = shape_count - 1
    if victim_count < 1:
        return None

    c_bar_min = 4
    c_bar_max = w - 5
    if c_bar_min > c_bar_max:
        return None
    c_bar = int(np.random.randint(c_bar_min, c_bar_max + 1))

    min_span = 6 if victim_count == 1 else (8 if victim_count == 2 else 10)
    r_top_max = max(1, h - min_span - 3)
    if r_top_max < 1:
        return None
    r_top = int(np.random.randint(1, r_top_max + 1))
    r_bot_min = r_top + min_span
    if r_bot_min > h - 2:
        return None
    r_bot = int(np.random.randint(r_bot_min, h - 1))

    rows = np.linspace(r_top + 2, r_bot - 2, victim_count)
    victim_rows = [int(round(v)) for v in rows]
    if len(set(victim_rows)) != victim_count:
        return None
    if any(abs(victim_rows[i] - victim_rows[i - 1]) < 2 for i in range(1, len(victim_rows))):
        return None

    shapes: list[dict] = [
        {
            "color": int(shape_colors[0]),
            "vertices": [(r_top, c_bar), (r_bot, c_bar)],
            "shadow": None,
        }
    ]

    used_input = set(_shape_input_cells(shapes[0]))
    shadow_iter = iter(shadow_colors)
    for i, row in enumerate(victim_rows, start=1):
        c_left = int(np.random.randint(1, c_bar - 1))
        c_right = int(np.random.randint(c_bar + 3, w - 1))
        if c_left >= c_bar or c_right <= c_bar:
            return None
        shape = {
            "color": int(shape_colors[i]),
            "vertices": [(row, c_left), (row, c_right)],
            "shadow": None,
        }

        shadow_color = next(shadow_iter, None)
        if shadow_color is not None:
            anchor_r, anchor_c = shape["vertices"][0]
            preferred = [(-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
            offset = _sample_shadow_offset(anchor_r, anchor_c, h, w, used_input, preferred)
            if offset is None:
                return None
            shape["shadow"] = {
                "anchor_idx": 0,
                "offset": offset,
                "color": int(shadow_color),
            }

        input_cells = _shape_input_cells(shape)
        if input_cells & used_input:
            return None
        used_input |= input_cells
        shapes.append(shape)

    return shapes


def _paint_shapes(inp: np.ndarray, shapes: list[dict]) -> None:
    for shape in shapes:
        for r, c in shape["vertices"]:
            inp[r, c] = int(shape["color"])
        shadow = shape.get("shadow")
        if shadow is not None:
            ar, ac = shape["vertices"][int(shadow["anchor_idx"])]
            dr, dc = shadow["offset"]
            inp[ar + dr, ac + dc] = int(shadow["color"])


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette

    shapes = _build_interacting_shapes(h, w, shape_colors, shadow_colors)
    if shapes is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    _paint_shapes(inp, shapes)

    reserved: set[tuple[int, int]] = set()
    for shape in shapes:
        reserved.update(_shape_output_cells(shape))

    if has_distractor and distractor_color is not None:
        if not _place_distractor(inp, int(distractor_color), reserved):
            return None

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if _bg_color(inp) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None

    blocked_out = main(inp)
    if blocked_out.shape != inp.shape:
        return None

    metrics = _variant_metrics(inp)
    if metrics["blocked_main"] < 1:
        return None
    if metrics["diff_cells"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
