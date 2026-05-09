# Combo variant of 35ab12c3 (index 24)
# combo_spec: D8/1+D13/3
# Name: 遇障碍物停止+按颜色值排序处理

# Design analysis for the combo variant
# a) Concrete behavior / trace:
#    The original task connects each same-color vertex set into a line/polygon and then
#    paints a parallel shadow copied from the chosen start vertex's foreign-color neighbor.
#    In the fused variant, groups are processed by ascending base color, and every segment
#    advances cell by cell, stopping as soon as it reaches a non-background cell belonging
#    to another group that is already present in the current canvas.
#    Small trace used by the generator:
#      - color 1: a horizontal bar A is drawn first and fills across the center.
#      - color 2: a vertical bar B has a shadow anchor next to its bottom endpoint, so B
#        starts from the bottom and grows upward. It first passes through the lower rows,
#        then halts when it reaches A. Its shadow grows in parallel and also halts at A.
#      - color 4: a lower horizontal bar C is processed after B, so C stops when it hits
#        the already-drawn lower part of B. Under the original non-blocking behavior, C
#        would continue through the crossing and reach its far endpoint.
#    The visible difference is therefore a color-order-dependent chain:
#      color 1 blocks color 2, and color 2 blocks color 4.
# b) Complexity calibration:
#    If shapes are far apart, both D8 and D13 disappear. To prevent that degeneration, the
#    generator always creates at least three interacting shapes in one corridor, with at
#    least two independent block events and one blocked shadow. Some samples add a fourth
#    higher-color horizontal victim, repeating the same ordering evidence.
# c) Hidden assumptions review:
#    - The original color iteration order existed already, but it was latent because shapes
#      were spatially separated. With collision-stop enabled, that order becomes semantic,
#      so the solver must make it explicit and the generator must create crossings where the
#      order changes the output.
#    - The original `_clear_in_original(...)` binary clearance test is invalid here because
#      a partially drawable prefix must still be painted. The combo solver removes that
#      all-or-nothing precheck and instead walks each segment incrementally.
#    - The original "draw all mains, then all shadows" remains necessary inside one group,
#      otherwise a group's own shadow could prematurely interfere with its main edge. The
#      new global order is therefore: process groups by color; within each group draw mains
#      first, then shadows; only foreign colors block.
#    - Start-vertex choice matters more now because it decides which side of a segment meets
#      the obstacle first. The generator guarantees the middle vertical blocker has exactly
#      one foreign neighbor, at its bottom endpoint, so its upward direction is unambiguous.
# d) Placement legality:
#    All constructed vertices, shadow anchors, and optional extra victims are kept in-bounds
#    with fixed row/column gaps. Intersections occur only at background cells in the input,
#    so the task remains a genuine "connect sparse vertices" puzzle. No distractors are
#    placed adjacent to shape vertices, so solver seed detection stays stable.
# e) Visual observability:
#    Sorting by hidden numeric color id is intrinsically weaker than geometry-based order, so
#    the generator compensates by reusing the same small precedence ladder of visible colors:
#    shape colors are drawn from the fixed ascending chain 1 < 2 < 4 < 7. Their positions
#    vary, but the same colors repeatedly win or lose crossings in a consistent hierarchy.
#    The middle color also blocks a higher color even when its start vertex is spatially
#    later, making it clear that top-left order is not the governing rule.
# f) Visual cleanliness:
#    The interaction is confined to one central corridor: one early blocker, one middle
#    blocker/victim, and one or two late victims. This yields truncated straight segments and
#    one truncated shadow without dense overpainting or ambiguous nested overlaps.
# g) Change inferability:
#    There is no color remapping; colors keep their identity. The only new dependency is
#    precedence between a small set of recurring colors, and that precedence is supported by
#    repeated pairwise collisions inside every generated sample.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes every meaningful blocking interaction is between different groups; the
#      generator uses unique base colors and at most one shadow color per group.
#    - `main()` assumes the intended start vertex can be inferred from foreign neighbors; the
#      generator gives the key vertical line exactly one foreign neighbor at its bottom end.
#    - `main()` assumes color-order competition is observable; the generator enforces at least
#      one output difference versus an alternative start-position order and at least one
#      output difference versus the non-blocking baseline.

from __future__ import annotations

import math
import numpy as np

from arc_utils import colors_present, get_bg as _bg_color


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
SHAPE_COLOR_CHAIN = [1, 2, 4, 7]
BLOCKER_SHADOW_COLOR = 6
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)
GENERATE_RETRIES = 400
EXTRA_VICTIM_PROB = 0.45


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
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
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
                "color": int(col),
                "start": cycle[0],
                "edges": edges,
                "carried": carried,
                "group_colors": {int(col), *(int(v) for _, _, v in carried)},
            }
        )
    return groups


def _order_groups(groups: list[dict], order: str) -> list[dict]:
    if order == "color_asc":
        return sorted(groups, key=lambda info: int(info["color"]))
    if order == "start":
        return sorted(groups, key=lambda info: (info["start"][0], info["start"][1], int(info["color"])))
    raise ValueError(f"unknown order: {order}")


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
) -> int:
    painted = 0
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col
            painted += 1
    return painted


def _run_solver(
    grid: np.ndarray,
    *,
    order: str,
    stop_on_block: bool,
    collect_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()
    groups = _order_groups(_collect_groups(g, bg), order)

    meta = {
        "blocked_main": 0,
        "blocked_shadow": 0,
        "groups": len(groups),
        "order": [int(info["color"]) for info in groups],
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


# Overview:
# Connect each shape's same-color vertices as in the original task, but process shapes in
# ascending base-color order and let every segment stop at the first foreign occupied cell.
# Input structure:
# Sparse same-color vertices define straight 8-direction segments; a different-colored
# 8-neighbor next to one chosen vertex marks a parallel shadow offset for that shape.
# Rule:
# Sort drawable shape groups by their own color value (ascending). For each group, draw all
# main edges first and then the parallel shadow edges. Each edge advances stepwise from its
# chosen start vertex and halts immediately when it reaches any already-occupied foreign
# color; own group colors do not count as blockers.
# Key insight:
# The original latent color iteration becomes visible only because collision stopping turns
# earlier groups into obstacles. Lower-color shapes therefore claim space first and can cut
# off higher-color shapes and their shadows mid-segment.
def main(grid: np.ndarray) -> np.ndarray:
    return _run_solver(grid, order="color_asc", stop_on_block=True, collect_meta=False)


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _paint_shape_inputs(inp: np.ndarray, shapes: list[dict]) -> None:
    for shape in shapes:
        for r, c in shape["vertices"]:
            inp[r, c] = int(shape["color"])
        shadow = shape.get("shadow")
        if shadow is not None:
            ar, ac = shape["vertices"][int(shadow["anchor_idx"])]
            dr, dc = shadow["offset"]
            inp[ar + dr, ac + dc] = int(shadow["color"])


def _build_chain_shapes(h: int, w: int, shape_count: int) -> list[dict] | None:
    if shape_count not in (3, 4):
        return None
    if h < 15 or w < 15:
        return None

    for _ in range(120):
        c_b = int(np.random.randint(5, w - 5))
        left_min = 1
        left_max = c_b - 3
        right_min = c_b + 3
        right_max = w - 2
        if left_min > left_max or right_min > right_max:
            continue

        base_top = int(np.random.randint(1, 3))
        gap1 = int(np.random.randint(2, 5))
        gap2 = int(np.random.randint(3, 6))
        if shape_count == 3:
            gap3 = int(np.random.randint(3, 6))
            r_top = base_top
            r_a = r_top + gap1
            r_c = r_a + gap2
            r_bot = r_c + gap3
            if r_bot >= h - 1:
                continue
            rows_for_victims = [r_c]
        else:
            gap3a = int(np.random.randint(2, 4))
            gap3b = int(np.random.randint(3, 6))
            r_top = base_top
            r_a = r_top + gap1
            r_c = r_a + gap2
            r_d = r_c + gap3a
            r_bot = r_d + gap3b
            if r_bot >= h - 1:
                continue
            rows_for_victims = [r_c, r_d]

        a_left = int(np.random.randint(left_min, left_max + 1))
        a_right = int(np.random.randint(right_min, right_max + 1))
        if a_right - a_left < 6:
            continue

        shapes: list[dict] = [
            {
                "color": SHAPE_COLOR_CHAIN[0],
                "vertices": [(r_a, a_left), (r_a, a_right)],
            },
            {
                "color": SHAPE_COLOR_CHAIN[1],
                "vertices": [(r_top, c_b), (r_bot, c_b)],
                "shadow": {"anchor_idx": 1, "offset": (0, 1), "color": BLOCKER_SHADOW_COLOR},
            },
        ]

        ok = True
        for idx, row in enumerate(rows_for_victims):
            left = int(np.random.randint(left_min, left_max + 1))
            right = int(np.random.randint(right_min, right_max + 1))
            if right - left < 6:
                ok = False
                break
            shapes.append(
                {
                    "color": SHAPE_COLOR_CHAIN[2 + idx],
                    "vertices": [(row, left), (row, right)],
                }
            )
        if not ok:
            continue

        input_cells: set[tuple[int, int]] = set()
        for shape in shapes:
            for cell in shape["vertices"]:
                if cell in input_cells:
                    ok = False
                    break
                input_cells.add(cell)
            if not ok:
                break
            shadow = shape.get("shadow")
            if shadow is not None:
                ar, ac = shape["vertices"][int(shadow["anchor_idx"])]
                dr, dc = shadow["offset"]
                pos = (ar + dr, ac + dc)
                if pos in input_cells:
                    ok = False
                    break
                if not (0 <= pos[0] < h and 0 <= pos[1] < w):
                    ok = False
                    break
                input_cells.add(pos)
        if not ok:
            continue

        return shapes
    return None


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    combo_out, meta = _run_solver(inp, order="color_asc", stop_on_block=True, collect_meta=True)
    no_stop = _run_solver(inp, order="color_asc", stop_on_block=False, collect_meta=False)
    start_stop = _run_solver(inp, order="start", stop_on_block=True, collect_meta=False)
    return {
        "blocked_main": int(meta["blocked_main"]),
        "blocked_shadow": int(meta["blocked_shadow"]),
        "diff_vs_no_stop": int(np.sum(combo_out != no_stop)),
        "diff_vs_start_order": int(np.sum(combo_out != start_stop)),
        "shape_groups": int(meta["groups"]),
    }


def _gen_attempt(shape_count: int | None = None) -> np.ndarray | None:
    h, w = _sample_canvas_size()
    if shape_count is None:
        shape_count = 4 if (h >= 18 and np.random.rand() < EXTRA_VICTIM_PROB) else 3
    if shape_count == 4 and h < 18:
        return None
    shapes = _build_chain_shapes(h, w, shape_count)
    if shapes is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    _paint_shape_inputs(inp, shapes)

    if _bg_color(inp) != BG:
        return None

    expected_colors = {shape["color"] for shape in shapes}
    expected_colors.add(BLOCKER_SHADOW_COLOR)
    if set(colors_present(inp, exclude=[BG])) != expected_colors:
        return None

    combo_out = main(inp)
    if combo_out.shape != inp.shape:
        return None

    metrics = _behavior_metrics(inp)
    if metrics["shape_groups"] != shape_count:
        return None
    if metrics["blocked_main"] < 2:
        return None
    if metrics["blocked_shadow"] < 1:
        return None
    if metrics["diff_vs_no_stop"] < 4:
        return None
    if metrics["diff_vs_start_order"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    preferred = 4 if np.random.rand() < 0.5 else 3
    fallback = 3 if preferred == 4 else 4
    primary_budget = GENERATE_RETRIES // 2

    for _ in range(primary_budget):
        result = _gen_attempt(preferred)
        if result is not None:
            return result

    for _ in range(GENERATE_RETRIES - primary_budget):
        result = _gen_attempt(fallback)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
