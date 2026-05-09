# Combo variant of 195c6913 (index 1)
# combo_spec: D2/0+D5/0
# Name: 始终同向转弯+到达目标停止

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import label_mask


# Design analysis for the combo variant (D2/0 + D5/0).
# a) Concrete behavior:
#    The scene keeps the original visual grammar: a thick anti-diagonal path band,
#    2x2 cycle blocks in the top-left, 1-2 isolated markers on col 0, and one 2x2
#    legend block in the bottom-right background pocket. The fusion adds isolated
#    in-path finish pixels colored like the legend block, and changes turning to a
#    single global CCW rule. Small trace sketch:
#      original: marker goes Right, hits wall, turns Left; at next wall it turns
#      Right; if that scheduled turn is blocked it stops.
#      combo: marker goes Right, hits wall, turns Left; at the next wall it turns
#      Left again, so the route curls instead of zig-zagging. Later it reaches a
#      finish-colored pixel already embedded on that curled route and stops there
#      immediately, without waiting for a blocked-turn event.
# b) Complexity calibration:
#    If a sample has only 0-1 successful turns, fixed-left and alternating look
#    identical. If a target is placed at a cell where the no-target solver would
#    already stop, the destination rule becomes visually inert. The generator
#    therefore rejects such cases and keeps only inputs where:
#      - at least one marker makes >=2 turns under the fixed-left policy;
#      - the combo output differs clearly from alternating-turn-with-targets; and
#      - the combo output also differs clearly from fixed-left-without-target-stop.
# c) Implicit assumptions that break:
#    The original solver assumed every traversable cell had the unique path color
#    and that future motion depended on collision parity. Both fail here:
#      - target cells use end_color but must still count as traversable;
#      - fixed-left motion has no parity state, so visited-state tracking must not
#        include odd/even hit parity or the simulator can spin through fake states.
#    End-color detection also changes: a 2x2 block still identifies the legend
#    color, but isolated cells of that same color now encode the actual stopping
#    sites and must not be mistaken for noise.
# d) Placement legality:
#    Template blocks remain on background-only cells in rows 1-3 and cols
#    {1,4,7,10}. Markers still overwrite col-0 path cells while preserving the
#    band's 4-connectivity. Targets are chosen only from the fixed-left trajectory
#    of each marker, are kept isolated from other end-color cells, and preserve a
#    single passable component when recolored from path_color to end_color.
# e) Visual observability:
#    The same end_color now appears twice with distinct but related structure:
#    one removable 2x2 legend block and one isolated in-path finish pixel per
#    marker. In output examples, trails stop exactly at those isolated pixels, so
#    the destination mechanism is visible. Fixed-left turning is made visible by
#    forcing multi-turn samples whose outputs differ from the alternating baseline.
# f) Visual cleanliness:
#    Targets are selected from the later half of each trajectory and preferably
#    after at least two turns, keeping them away from the crowded marker/template
#    corner. The original separator logic for 2-marker cases is preserved so the
#    two channels stay visually separated.
# g) Color inferability:
#    No color-id arithmetic is used. The cycle is still read from left to right
#    from the 2x2 template blocks. end_color keeps a single semantic family:
#    collision/finish annotation. Human solvers can infer the target role from the
#    isolated end-colored path pixels that persist in the outputs while the 2x2
#    legend block is erased.
# h) New solver assumptions audited:
#    main() assumes exactly one 2x2 legend block of end_color, one isolated target
#    per marker, and globally fixed CCW turning for every marker. generate_input()
#    guarantees those assumptions by construction and by oracle filtering with both
#    baseline comparisons.

_TURN_POLICY_FIXED_LEFT = "fixed_left"
_TURN_POLICY_ALTERNATING = "alternating"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each col-0 marker must sit inside a vertical path segment."""
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


def _assert_targets_embedded_in_passable(
    g: np.ndarray, targets: list[tuple[int, int]], path_color: int, end_color: int
) -> None:
    """Targets are isolated end-color cells touching the path neighborhood."""
    H, W = g.shape
    for r, c in targets:
        assert int(g[r, c]) == int(end_color), f"target at ({r},{c}) must use end_color"
        same = 0
        touches_path = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                if int(g[rr, cc]) == int(end_color):
                    same += 1
                if int(g[rr, cc]) == int(path_color):
                    touches_path += 1
        assert same == 0, f"target at ({r},{c}) must be isolated"
        assert touches_path >= 1, f"target at ({r},{c}) must touch the path"


def _assert_passable_single_component(passable_mask: np.ndarray) -> None:
    """Path cells plus targets must remain one 4-connected component."""
    _, n_path = label_mask(passable_mask, connectivity=4)
    assert n_path == 1, f"passable region must be a single 4-connected component, got {n_path}"


def _find_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    pattern: list[int] = []
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
                pattern.append(v)
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    end_color = None
    end_pos = None
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
                end_color = int(v)
                end_pos = (int(r), int(c))
                return end_color, end_pos
    return end_color, end_pos


def _find_markers(
    g: np.ndarray, pat_set: set[int], path_color: int
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
                markers.append((int(r), int(c), v))
    return markers


def _find_targets(
    g: np.ndarray, end_color: int | None, path_color: int
) -> list[tuple[int, int]]:
    if end_color is None:
        return []
    H, W = g.shape
    targets: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != int(end_color):
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == int(end_color):
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                targets.append((int(r), int(c)))
    return targets


def _turn_left(dr: int, dc: int) -> tuple[int, int]:
    return -int(dc), int(dr)


def _turn_right(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), -int(dr)


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )

    pattern, pat_cells = _find_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, template_bg, path_color, pat_set)
    markers = _find_markers(g, pat_set, path_color)
    targets = _find_targets(g, end_color, path_color)

    _assert_marker_embedded_in_path(g, markers, path_color)
    assert pattern, "template pattern must be present"
    assert end_color is not None and end_pos is not None, "end legend block must be present"
    assert targets, "at least one in-path target must be present"
    assert len(targets) == len(markers), "generator contract: one target per marker"
    _assert_targets_embedded_in_passable(g, targets, path_color, int(end_color))

    passable = g == int(path_color)
    for r, c in targets:
        passable[int(r), int(c)] = True
    _assert_passable_single_component(passable)

    return {
        "grid": g,
        "H": H,
        "W": W,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": int(end_color),
        "end_pos": end_pos,
        "markers": markers,
        "targets": targets,
    }


def _simulate_from_parsed(
    parsed: dict[str, object],
    turn_policy: str,
    use_targets: bool,
) -> dict[str, object]:
    g = np.array(parsed["grid"], dtype=int)
    H = int(parsed["H"])
    W = int(parsed["W"])
    template_bg = int(parsed["template_bg"])
    path_color = int(parsed["path_color"])
    pattern = list(parsed["pattern"])
    pat_cells = list(parsed["pat_cells"])
    end_color = int(parsed["end_color"])
    end_pos = parsed["end_pos"]
    markers = list(parsed["markers"])
    targets = [tuple(map(int, t)) for t in parsed.get("targets", [])]
    target_cells = set(targets) if use_targets else set()

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    def is_passable(rr: int, cc: int) -> bool:
        return 0 <= rr < H and 0 <= cc < W and (
            int(g[rr, cc]) == int(path_color) or (int(rr), int(cc)) in target_cells
        )

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    per_marker_cells: list[set[tuple[int, int]]] = []
    traces: list[dict[str, object]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if is_passable(rr, cc):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            traces.append(
                {
                    "cells": [],
                    "hit_counts": [],
                    "turns": 0,
                    "reached_target": False,
                    "target": None,
                    "corner_marks": 0,
                }
            )
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        corner_marks = 0
        visited: set[tuple[int, ...]] = set()
        cells: list[tuple[int, int]] = []
        hit_counts: list[int] = []
        reached_target = False
        target_hit = None

        for _ in range(H * W * 6):
            if turn_policy == _TURN_POLICY_ALTERNATING:
                state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            else:
                state = (int(r), int(c), int(dr), int(dc), int(idx))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (int(nr), int(nc)) in target_cells:
                out[nr, nc] = int(end_color)
                cells.append((int(nr), int(nc)))
                hit_counts.append(int(hits))
                reached_target = True
                target_hit = (int(nr), int(nc))
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                cells.append((int(nr), int(nc)))
                hit_counts.append(int(hits))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg):
                out[nr, nc] = int(end_color)
                corner_marks += 1

            if turn_policy == _TURN_POLICY_ALTERNATING:
                tdr, tdc = _turn_left(dr, dc) if (int(hits) % 2 == 0) else _turn_right(dr, dc)
            else:
                tdr, tdc = _turn_left(dr, dc)

            rr = int(r + tdr)
            cc = int(c + tdc)
            if is_passable(rr, cc):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        my_cells = set((int(rr), int(cc)) for rr, cc in cells)
        per_marker_cells.append(my_cells)
        traces.append(
            {
                "cells": cells,
                "hit_counts": hit_counts,
                "turns": int(hits),
                "reached_target": bool(reached_target),
                "target": target_hit,
                "corner_marks": int(corner_marks),
            }
        )

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return {
        "output": out,
        "traces": traces,
        "total_turns": int(sum(int(t["turns"]) for t in traces)),
        "max_turns": int(max((int(t["turns"]) for t in traces), default=0)),
        "total_steps": int(sum(len(t["cells"]) for t in traces)),
    }


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    parsed = _parse_scene(grid)
    fused = _simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT, use_targets=True)
    alt_target = _simulate_from_parsed(parsed, _TURN_POLICY_ALTERNATING, use_targets=True)
    fixed_no_target = _simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT, use_targets=False)
    return {
        "n_markers": len(parsed["markers"]),
        "targets": len(parsed["targets"]),
        "fused_total_turns": int(fused["total_turns"]),
        "fused_max_turns": int(fused["max_turns"]),
        "fused_total_steps": int(fused["total_steps"]),
        "reached_targets": int(sum(1 for t in fused["traces"] if bool(t["reached_target"]))),
        "diff_vs_alternating_target": int(
            np.count_nonzero(np.array(fused["output"]) != np.array(alt_target["output"]))
        ),
        "diff_vs_fixed_no_target": int(
            np.count_nonzero(np.array(fused["output"]) != np.array(fixed_no_target["output"]))
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace particles through a path band with one fixed turn direction and explicit stop targets.

    Input structure:
    - The top-left contains a row of 2x2 solid-color blocks whose left-to-right order defines a repeating color cycle.
    - One or two isolated single pixels on the left edge reuse the first cycle color and act as launch markers inside the path.
    - A separate 2x2 legend block gives the finish color, and isolated pixels of that same color appear inside the path as destinations.

    Rule:
    - Each marker launches a particle that advances straight and repaints passable path cells with successive cycle colors.
    - When the next cell is background, that collision cell is marked with the finish color.
    - Every successful turn is the same 90-degree counterclockwise turn; there is no left/right alternation.
    - A finish-colored isolated path pixel is traversable and terminal: stepping onto it stops that particle immediately.

    Key insight:
    - The top-left blocks still only encode the paint cycle, while the real motion rule is the combination of fixed CCW turning and explicit in-path finish pixels; termination is read from where the path contains the finish color, not from a blocked next turn.
    """
    parsed = _parse_scene(grid)
    result = _simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT, use_targets=True)
    assert int(result["total_steps"]) > 0, "at least one marker must traverse the path"
    assert int(sum(1 for t in result["traces"] if bool(t["reached_target"]))) == len(parsed["targets"])
    return np.array(result["output"], dtype=int)


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal path band with a left-edge lobe for markers."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
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
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - width / 2))
            c_right[r] = c_left[r] + width - 1

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
    """Carve a narrow bg separator from the left edge without disconnecting path."""
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


def _path_neighbor_count(g: np.ndarray, r: int, c: int, path_color: int) -> int:
    H, W = g.shape
    total = 0
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            total += 1
    return total


def _target_candidates(
    g: np.ndarray, trace: dict[str, object], path_color: int
) -> list[tuple[int, int]]:
    cells = list(trace["cells"])
    hit_counts = list(trace["hit_counts"])
    if len(cells) < 8:
        return []

    later_start = max(5, len(cells) // 2)
    strong: list[tuple[int, int]] = []
    medium: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for idx in range(len(cells) - 1, later_start - 1, -1):
        r, c = cells[idx]
        if _path_neighbor_count(g, int(r), int(c), int(path_color)) < 2:
            continue
        cell = (int(r), int(c))
        if int(hit_counts[idx]) >= 2:
            strong.append(cell)
        elif int(hit_counts[idx]) >= 1:
            medium.append(cell)
        else:
            fallback.append(cell)
    return strong + medium + fallback


def _targets_are_legal(
    g: np.ndarray, path_color: int, end_color: int, targets: list[tuple[int, int]]
) -> bool:
    if len(set(targets)) != len(targets):
        return False
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            if abs(targets[i][0] - targets[j][0]) + abs(targets[i][1] - targets[j][1]) <= 1:
                return False

    trial = g.copy()
    for r, c in targets:
        if int(trial[r, c]) not in (int(path_color), int(end_color)):
            return False
        trial[int(r), int(c)] = int(end_color)

    for r, c in targets:
        same = 0
        has_path = False
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < trial.shape[0] and 0 <= cc < trial.shape[1]:
                if int(trial[rr, cc]) == int(end_color):
                    same += 1
                if int(trial[rr, cc]) == int(path_color):
                    has_path = True
        if same != 0 or not has_path:
            return False

    passable = trial == int(path_color)
    for r, c in targets:
        passable[int(r), int(c)] = True
    _, n_pass = label_mask(passable, connectivity=4)
    return n_pass == 1


def _choose_targets(
    g: np.ndarray, traces: list[dict[str, object]], path_color: int, end_color: int
) -> list[tuple[int, int]] | None:
    candidate_lists = [_target_candidates(g, trace, path_color) for trace in traces]
    if any(not candidates for candidates in candidate_lists):
        return None

    if len(candidate_lists) == 1:
        for target in candidate_lists[0][:18]:
            if _targets_are_legal(g, path_color, end_color, [target]):
                return [target]
        return None

    left = candidate_lists[0][:16]
    right = candidate_lists[1][:16]
    for t1 in left:
        for t2 in right:
            targets = [t1, t2]
            if _targets_are_legal(g, path_color, end_color, targets):
                return targets
    return None


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
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
        for r in markers:
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
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

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

    for r in markers:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    parsed_seed = {
        "grid": g.copy(),
        "H": H,
        "W": W,
        "template_bg": int(bg),
        "path_color": int(path_color),
        "pattern": list(cycle_colors),
        "pat_cells": [
            (template_row + dr, bc + dc)
            for bc in block_cols
            for dr in range(2)
            for dc in range(2)
        ],
        "end_color": int(end_color),
        "end_pos": (int(er), int(ec)),
        "markers": [(int(r), 0, int(cycle_colors[0])) for r in markers],
        "targets": [],
    }

    try:
        fixed_seed = _simulate_from_parsed(parsed_seed, _TURN_POLICY_FIXED_LEFT, use_targets=False)
        alt_seed = _simulate_from_parsed(parsed_seed, _TURN_POLICY_ALTERNATING, use_targets=False)
    except Exception:
        return None

    traces = list(fixed_seed["traces"])
    if any(len(trace["cells"]) < 8 for trace in traces):
        return None
    if int(fixed_seed["max_turns"]) < 2:
        return None
    if int(fixed_seed["total_turns"]) < 2:
        return None
    if int(np.count_nonzero(np.array(fixed_seed["output"]) != np.array(alt_seed["output"]))) < 4:
        return None

    targets = _choose_targets(g, traces, int(path_color), int(end_color))
    if targets is None or len(targets) != len(markers):
        return None
    for tr, tc in targets:
        g[int(tr), int(tc)] = int(end_color)

    if not _targets_are_legal(g, int(path_color), int(end_color), targets):
        return None

    try:
        metrics = _fusion_metrics(g)
        out = main(g)
    except Exception:
        return None

    if metrics["reached_targets"] != len(markers):
        return None
    if metrics["fused_max_turns"] < 2:
        return None
    if metrics["diff_vs_alternating_target"] < 4:
        return None
    if metrics["diff_vs_fixed_no_target"] < 3:
        return None
    if int(np.count_nonzero(np.array(out) != np.array(g))) < 10:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 800 attempts")
