# Combo variant of 195c6913 (index 7)
# combo_spec: D3/0+D5/1+D18/0
# Name: 间歇涂色+固定步数停止+固定2色循环
#
# Design analysis
# a) Concrete behavior and trace:
#    The scene grammar stays close to the source task: a thick anti-diagonal
#    path band, 1-2 markers on col 0, a top-left row of 2x2 template blocks,
#    and one isolated 2x2 end block. The fused change is entirely in how the
#    particle interprets the template and how long/when it paints.
#    Example with template blocks [A][B][A], marker color A, and step budget 12:
#    - Source task:
#      move1->paint B, move2->paint A, move3->paint B, ... until a blocked turn.
#    - Fused task:
#      move1->paint B, move2->skip, move3->paint A, move4->skip, ...
#      The cycle is always only A/B, even though the template may show 3 or 4
#      blocks. The particle still turns on collisions using CCW/CW alternation,
#      but it halts exactly after 12 successful path moves, whether or not more
#      path remains. So the visible output is a sparse two-color dashed prefix.
# b) Complexity calibration:
#    The dash period is fixed at 2, the cycle size is fixed at 2 colors, and the
#    distance budget comes from a small visible set {8,12,16}. Those are enough
#    to be learnable from 2-4 ARC examples while still creating a visibly
#    different output from the base task and from the single-change variants.
# c) Hidden assumptions audit:
#    The original code assumed every move paints and that the full list of block
#    colors is the cycle. Both fail here: movement and painting must be decoupled,
#    and repeated template blocks such as [A,B,A,B] must collapse to the semantic
#    cycle [A,B]. The old stop condition "blocked turn means semantic end" also
#    no longer holds; it is only a malformed-input safeguard before the fixed
#    budget is exhausted. State memoization now needs the dash phase because the
#    same position/direction can lead to different future output depending on
#    whether the next successful step is a paint or skip step.
# d) Placement legality:
#    The generator keeps the original spatial separation: template in top-left bg,
#    end block in bottom-right bg, markers embedded in a col-0 path lobe, and the
#    path a single 4-connected component. New legality checks ensure the template
#    alternates between exactly two colors and that every marker can complete the
#    full encoded budget without an early blocked turn.
# e) Visual observability:
#    A human can see three independent cues in the paired examples:
#    the template itself alternates between only two colors, the output trail is
#    dashed because every other traversed path cell stays path-colored, and the
#    recolored prefix stops early while the path corridor continues. No rule uses
#    hidden properties like numeric color ids.
# f) Visual cleanliness:
#    Skipped cells remain path_color instead of receiving extra symbols, so the
#    dash pattern is legible inside the existing corridor. Collision marks stay
#    sparse single pixels in end_color, and trajectories remain disjoint when two
#    markers exist, preventing ambiguous overlaps.
# g) Change inferability:
#    The generator requires both dash gaps and two-color alternation to be visible
#    in the output, and it also requires a longer unbounded continuation so fixed
#    stopping is observable rather than accidental. This prevents degeneration
#    back to the base task or to a simpler single-variant instance.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


_MARKER_BRANCHES = (1, 2)
_BLOCK_BRANCHES = (2, 3, 4)
_marker_branch_cursor = 0
_block_branch_cursor = 0


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker on col 0 sits inside a vertical path run."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _ordered_unique(vals: list[int]) -> list[int]:
    out: list[int] = []
    for v in vals:
        if int(v) not in out:
            out.append(int(v))
    return out


def _detect_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    pattern_blocks: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern_blocks.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern_blocks, pat_cells


def _decode_two_color_cycle(pattern_blocks: list[int]) -> list[int]:
    cycle = _ordered_unique([int(v) for v in pattern_blocks])
    assert len(cycle) == 2, f"template must expose exactly two cycle colors, got {cycle}"
    assert len(pattern_blocks) >= 2, "template must contain at least two 2x2 blocks"
    for i, v in enumerate(pattern_blocks):
        assert int(v) == int(cycle[i % 2]), (
            f"template blocks must alternate between the two cycle colors; "
            f"block {i} has {v}, expected {cycle[i % 2]}"
        )
    return [int(cycle[0]), int(cycle[1])]


def _detect_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                return int(v), (int(r), int(c))
    return None, None


def _detect_markers(
    g: np.ndarray, path_color: int, pat_set: set[int]
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))
    return markers


def _scan_scene(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    list[int],
    list[int],
    list[tuple[int, int]],
    int,
    int | None,
    tuple[int, int] | None,
    list[tuple[int, int, int]],
]:
    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern_blocks, pat_cells = _detect_pattern(g, template_bg, path_color)
    cycle = _decode_two_color_cycle(pattern_blocks)
    step_budget = int(len(pat_cells))
    pat_set = set(map(int, pattern_blocks))
    end_color, end_pos = _detect_end_block(g, template_bg, path_color, pat_set)
    markers = _detect_markers(g, path_color, pat_set)
    _assert_marker_embedded_in_path(g, markers, int(path_color))
    return (
        template_bg,
        path_color,
        pattern_blocks,
        cycle,
        pat_cells,
        step_budget,
        end_color,
        end_pos,
        markers,
    )


def _trace_particle(
    g: np.ndarray,
    cycle: list[int],
    path_color: int,
    end_color: int | None,
    start: tuple[int, int, int],
    step_budget: int | None,
    paint_period: int,
    out: np.ndarray | None = None,
) -> dict[str, object]:
    H, W = g.shape
    sr, sc, sv = start
    template_bg = int(g[0, 0])
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break
    if init is None or not cycle:
        return {
            "cells": set(),
            "steps": 0,
            "painted_steps": 0,
            "skipped_steps": 0,
            "turns": 0,
            "collision_marks": 0,
            "blocked": True,
            "looped": False,
            "finished_budget": False,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(cycle.index(int(sv))) if int(sv) in cycle else 0
    hits = 0
    steps = 0
    turns = 0
    painted_steps = 0
    skipped_steps = 0
    collision_marks = 0
    my_cells: set[tuple[int, int]] = set()
    visited: set[tuple[int, int, int, int, int, int, int]] = set()

    while True:
        if step_budget is not None and int(steps) >= int(step_budget):
            return {
                "cells": my_cells,
                "steps": int(steps),
                "painted_steps": int(painted_steps),
                "skipped_steps": int(skipped_steps),
                "turns": int(turns),
                "collision_marks": int(collision_marks),
                "blocked": False,
                "looped": False,
                "finished_budget": True,
            }

        state = (
            int(r),
            int(c),
            int(dr),
            int(dc),
            int(idx),
            int(hits & 1),
            int(steps % paint_period),
        )
        if state in visited:
            return {
                "cells": my_cells,
                "steps": int(steps),
                "painted_steps": int(painted_steps),
                "skipped_steps": int(skipped_steps),
                "turns": int(turns),
                "collision_marks": int(collision_marks),
                "blocked": False,
                "looped": True,
                "finished_budget": False,
            }
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            return {
                "cells": my_cells,
                "steps": int(steps),
                "painted_steps": int(painted_steps),
                "skipped_steps": int(skipped_steps),
                "turns": int(turns),
                "collision_marks": int(collision_marks),
                "blocked": True,
                "looped": False,
                "finished_budget": False,
            }

        if int(g[nr, nc]) == int(path_color):
            steps += 1
            my_cells.add((int(nr), int(nc)))
            if ((int(steps) - 1) % int(paint_period)) == 0:
                idx = int((idx + 1) % len(cycle))
                if out is not None:
                    out[nr, nc] = int(cycle[idx])
                painted_steps += 1
            else:
                skipped_steps += 1
            r, c = nr, nc
            continue

        if out is not None and int(g[nr, nc]) == int(template_bg) and end_color is not None:
            out[nr, nc] = int(end_color)
            collision_marks += 1

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            turns += 1
            continue

        return {
            "cells": my_cells,
            "steps": int(steps),
            "painted_steps": int(painted_steps),
            "skipped_steps": int(skipped_steps),
            "turns": int(turns),
            "collision_marks": int(collision_marks),
            "blocked": True,
            "looped": False,
            "finished_budget": False,
        }


def _simulate(
    grid: np.ndarray,
    *,
    paint_period: int,
    use_fixed_budget: bool,
) -> tuple[np.ndarray, list[set[tuple[int, int]]], list[dict[str, object]], dict[str, object]]:
    assert paint_period >= 1
    g = np.array(grid, dtype=int)
    (
        template_bg,
        path_color,
        pattern_blocks,
        cycle,
        pat_cells,
        step_budget,
        end_color,
        end_pos,
        markers,
    ) = _scan_scene(g)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    traces: list[dict[str, object]] = []
    budget = int(step_budget) if use_fixed_budget else None
    for marker in markers:
        trace = _trace_particle(
            g,
            cycle,
            int(path_color),
            end_color,
            marker,
            budget,
            paint_period,
            out=out,
        )
        per_marker_cells.append(set(trace["cells"]))
        traces.append(trace)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    scene = {
        "template_bg": int(template_bg),
        "path_color": int(path_color),
        "pattern_blocks": [int(v) for v in pattern_blocks],
        "cycle": [int(v) for v in cycle],
        "pat_cells": list(pat_cells),
        "step_budget": int(step_budget),
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": list(markers),
    }
    return out, per_marker_cells, traces, scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate dashed two-color particles that stop after a template-encoded move budget.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks; the blocks may repeat, but they alternate between exactly two colors and simultaneously expose both the 2-color cycle and the step budget.
    - Left edge: 1-2 isolated marker pixels embedded inside the path band; each marker uses the first cycle color.
    - Elsewhere: one separate 2x2 monochrome block supplies the collision-mark color, while the thick meandering band is the traversable path.

    Rule:
    - Each marker launches a particle that follows the path with the same initial direction preference and strict CCW/CW alternating turn rule as the source task.
    - The particle counts every successful move onto a path cell toward a fixed budget equal to the number of colored cells in the template blocks.
    - Only every other successful move is painted; painted moves alternate between the two cycle colors, skipped moves stay path-colored.
    - Collisions may still leave single end-color corner marks and trigger turns, but they do not determine termination: the particle stops exactly when the budget is exhausted.

    Key insight:
    - The template now carries three coupled clues at once: two visible colors define the only paint cycle, repeated block count defines travel distance, and the output's untouched path cells reveal that movement and painting happen on different cadences.
    """
    out, _, traces, scene = _simulate(grid, paint_period=2, use_fixed_budget=True)
    step_budget = int(scene["step_budget"])
    for marker, trace in zip(scene["markers"], traces):
        assert int(trace["steps"]) == int(step_budget), (
            f"marker at {marker[:2]} could not complete encoded step budget "
            f"{step_budget} (got {trace['steps']})"
        )
        assert not bool(trace["blocked"]), f"marker at {marker[:2]} stopped before budget"
        assert not bool(trace["looped"]), f"marker at {marker[:2]} looped before budget"
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [
            0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
            for r in range(H)
        ]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg fjord into the left side of the path band."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _behavior_metrics(g: np.ndarray) -> dict[str, object]:
    combo_out, _, combo_traces, scene = _simulate(g, paint_period=2, use_fixed_budget=True)
    solid_out, _, solid_traces, _ = _simulate(g, paint_period=1, use_fixed_budget=True)
    unbounded_out, _, unbounded_traces, _ = _simulate(g, paint_period=2, use_fixed_budget=False)

    path_color = int(scene["path_color"])
    cycle = [int(v) for v in scene["cycle"]]
    step_budget = int(scene["step_budget"])

    dash_gap_cells = int(np.count_nonzero((combo_out == path_color) & (solid_out != path_color)))
    extra_unbounded_cells = 0
    for fixed, free in zip(combo_traces, unbounded_traces):
        extra_unbounded_cells += max(0, int(free["steps"]) - int(fixed["steps"]))

    trail_color_counts = {
        int(c): int(np.count_nonzero(combo_out == int(c))) for c in cycle
    }
    active_stats = [t for t in combo_traces if int(t["steps"]) > 0]
    solid_active = [t for t in solid_traces if int(t["steps"]) > 0]
    free_active = [t for t in unbounded_traces if int(t["steps"]) > 0]

    return {
        "step_budget": int(step_budget),
        "cycle": cycle,
        "dash_gap_cells": int(dash_gap_cells),
        "extra_unbounded_cells": int(extra_unbounded_cells),
        "trail_color_counts": trail_color_counts,
        "combo_traces": active_stats,
        "solid_traces": solid_active,
        "unbounded_traces": free_active,
        "combo_out": combo_out,
        "solid_out": solid_out,
        "unbounded_out": unbounded_out,
    }


def _gen_attempt(n_markers: int, n_blocks: int) -> np.ndarray | None:
    H = W = random.randint(20, 30)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    cycle_base = random.sample(palette[3:], k=2)
    cycle_colors = [int(cycle_base[i % 2]) for i in range(n_blocks)]
    step_budget = 4 * n_blocks

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        marker_rows = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        marker_rows = [r1, r2]

    mask = _build_band(H, W, lobe_rows=marker_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in marker_rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        if random.random() < 0.25:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in marker_rows:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in marker_rows:
        g[r, 0] = int(cycle_base[0])

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        out = main(g)
        metrics = _behavior_metrics(g)
    except Exception:
        return None

    combo_traces = metrics["combo_traces"]
    solid_traces = metrics["solid_traces"]
    free_traces = metrics["unbounded_traces"]
    if len(combo_traces) != n_markers or len(solid_traces) != n_markers or len(free_traces) != n_markers:
        return None

    if metrics["step_budget"] != step_budget:
        return None
    if metrics["cycle"] != [int(cycle_base[0]), int(cycle_base[1])]:
        return None
    if metrics["dash_gap_cells"] < max(4, 2 * n_markers):
        return None
    if metrics["extra_unbounded_cells"] < max(3, n_markers + 1):
        return None

    total_turns = 0
    for fixed, free in zip(combo_traces, free_traces):
        if int(fixed["steps"]) != step_budget:
            return None
        if bool(fixed["blocked"]) or bool(fixed["looped"]):
            return None
        if int(fixed["painted_steps"]) < 4 or int(fixed["skipped_steps"]) < 4:
            return None
        if int(free["steps"]) < step_budget + 3:
            return None
        total_turns += int(fixed["turns"])

    if total_turns < n_markers:
        return None

    combo_palette_hits = metrics["trail_color_counts"]
    if min(combo_palette_hits.values()) < 2:
        return None

    if np.array_equal(metrics["combo_out"], metrics["solid_out"]):
        return None
    if np.array_equal(metrics["combo_out"], metrics["unbounded_out"]):
        return None
    if np.array_equal(out, g):
        return None

    return g


def generate_input() -> np.ndarray:
    global _marker_branch_cursor, _block_branch_cursor
    target_markers = int(_MARKER_BRANCHES[_marker_branch_cursor % len(_MARKER_BRANCHES)])
    target_blocks = int(_BLOCK_BRANCHES[_block_branch_cursor % len(_BLOCK_BRANCHES)])
    _marker_branch_cursor += 1
    _block_branch_cursor += 1
    for _ in range(500):
        try:
            g = _gen_attempt(target_markers, target_blocks)
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed after 500 attempts "
        f"(markers={target_markers}, blocks={target_blocks})"
    )
