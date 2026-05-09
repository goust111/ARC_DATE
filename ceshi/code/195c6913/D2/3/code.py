# Variant of 195c6913
# Dimension: D2 — 碰撞转弯规则：严格交替左右转 (path)
# Change: 碰撞时第1/3/5次左转(CCW)，第2/4/6次右转(CW)，严格交替；若计划转向方向不是path则停止（无后备方案）... → 碰撞时尝试双向并选较长路径

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    The visible objects stay the same as the base task: a top-left row of 2x2
#    template blocks defines a color cycle, col-0 single-pixel markers launch
#    particles into a thick diagonal path band, and a separate 2x2 end block
#    provides the collision color. The changed behavior happens only at turns.
#    Old rule: on collision, the kth turn direction was forced by parity
#    (left, right, left, right, ...); if that side was blocked, the walk ended.
#    New rule: on collision, inspect both 90-degree turn candidates. For each
#    open candidate, count how many consecutive path cells can be traversed
#    immediately after that turn before the next obstruction. Choose the longer
#    straight continuation; ties are broken to the left for determinism, but the
#    generator rejects samples whose main evidence relies only on ties.
#    Example trace at a rightward collision:
#      left branch run = 2 cells, right branch run = 6 cells.
#      Old parity solver may still choose left on odd collisions and stop early
#      or enter a shorter corridor. The new solver must compare both branches
#      and choose the 6-cell corridor, producing a visibly different painted
#      trail deeper inside the band.
# b) Complexity calibration:
#    The path generator remains a thick meandering band plus optional separator,
#    because that already creates local fork-like collision sites without
#    introducing new object types. To keep the variant non-trivial, accepted
#    samples must contain at least one collision where both turn directions are
#    open and their straight-run lengths differ, and the greedy output must
#    differ from the legacy alternating output on the same input.
# c) Hidden assumptions from the original solver:
#    - `hits` parity was part of the visited-state key. That no longer reflects
#      behavior, so the new state key must not depend on turn count.
#    - The old collision block always inspected a single scheduled side. Under
#      the variant, both side-adjacent cells must be checked before deciding.
#    - The original generator only needed disjoint trajectories under the old
#      turn parity. With greedy turns, trajectories can shift, so the generator
#      must oracle-check the new solver and reject overlaps or degenerate cases.
#    - A deterministic tie-break is now required because equal left/right run
#      lengths are possible. The solver uses left preference, while the
#      generator ensures that at least one accepted decision is a strict win.
# d) Placement legality:
#    Template blocks still live in rows 1-3 and cols {1,4,7,10}; end block
#    still sits in the bottom-right background region; markers still overwrite
#    col-0 path cells whose vertical neighbors remain path. The generator keeps
#    the path 4-connected and checks that template/end placements stay off path.
# e) Visual observability:
#    The variant is observable because the changed rule alters which corridor is
#    chosen at a collision, changing multiple recolored path cells and often the
#    placement of end-color collision dots. The generator explicitly rejects
#    inputs whose greedy output matches the old alternating output.
# f) Visual cleanliness:
#    No new symbols are introduced. The only difference is which existing path
#    corridor gets painted, so the I/O remains readable: template in the top
#    left, markers on col 0, end block in bottom-right background, recoloring
#    along a single band with separator keeping multi-marker traces apart.
# g) Color inferability:
#    Colors keep the original meaning. No rule depends on numeric color ids.
#    The only color logic is the existing template-defined cycle and the fixed
#    collision/end color, both already recoverable from spatial structure.
# New solver-side assumptions and generator guarantees:
#    - Both-turn comparison assumes local 90-degree neighbors faithfully encode
#      the candidate corridors. The band generator keeps turns orthogonal and
#      grid-aligned, so this comparison is well-defined.
#    - The tie-break should not be the main evidence. The generator requires at
#      least one strict greedy choice and a nonzero output difference versus the
#      old rule before accepting a sample.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker stays embedded in a path segment."""
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


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
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
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
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
                break
        if end_pos is not None:
            break

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

    _assert_marker_embedded_in_path(g, markers, int(path_color))
    return {
        "grid": g,
        "H": H,
        "W": W,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _straight_run_length(
    g: np.ndarray, r: int, c: int, dr: int, dc: int, path_color: int
) -> int:
    length = 0
    nr = int(r + dr)
    nc = int(c + dc)
    H, W = g.shape
    while 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color):
        length += 1
        nr += int(dr)
        nc += int(dc)
    return int(length)


def _pick_turn(
    g: np.ndarray, r: int, c: int, dr: int, dc: int, path_color: int
) -> tuple[tuple[int, int] | None, dict]:
    left = (-int(dc), int(dr))
    right = (int(dc), -int(dr))
    H, W = g.shape
    options: list[tuple[int, int, int, int]] = []
    for side_rank, (tdr, tdc) in enumerate([left, right]):
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            run = _straight_run_length(g, r, c, tdr, tdc, path_color)
            options.append((int(run), int(-side_rank), int(tdr), int(tdc)))

    info = {
        "both_open": len(options) == 2,
        "strict_preference": False,
        "chosen_run": 0,
        "other_run": 0,
    }
    if not options:
        return None, info

    options.sort(reverse=True)
    chosen_run, _prefer_left, tdr, tdc = options[0]
    if len(options) == 2:
        other_run = int(options[1][0])
        info["other_run"] = other_run
        info["strict_preference"] = int(chosen_run) != int(other_run)
    info["chosen_run"] = int(chosen_run)
    return (int(tdr), int(tdc)), info


def _simulate(grid: np.ndarray, mode: str) -> tuple[np.ndarray, list[set[tuple[int, int]]], dict]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    pat_cells = list(scene["pat_cells"])
    end_color = scene["end_color"]
    end_pos = scene["end_pos"]
    markers = list(scene["markers"])

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    per_marker_cells: list[set[tuple[int, int]]] = []
    stats = {
        "total_turns": 0,
        "both_open_collisions": 0,
        "strict_greedy_decisions": 0,
        "tie_collisions": 0,
    }

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, ...]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            if mode == "alternate":
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

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            if mode == "alternate":
                left = (-int(dc), int(dr))
                right = (int(dc), -int(dr))
                tdr, tdc = left if (int(hits) % 2 == 0) else right
                rr = int(r + tdr)
                cc = int(c + tdc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                    dr, dc = int(tdr), int(tdc)
                    hits += 1
                    stats["total_turns"] += 1
                    continue
                break

            turn, info = _pick_turn(g, r, c, dr, dc, path_color)
            if info["both_open"]:
                stats["both_open_collisions"] += 1
                if info["strict_preference"]:
                    stats["strict_greedy_decisions"] += 1
                else:
                    stats["tie_collisions"] += 1
            if turn is None:
                break
            dr, dc = turn
            stats["total_turns"] += 1

        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out, per_marker_cells, stats


def _behavior_metrics(grid: np.ndarray) -> dict:
    greedy_out, greedy_cells, greedy_stats = _simulate(grid, mode="greedy")
    alt_out, _, _ = _simulate(grid, mode="alternate")
    diff_cells = int(np.count_nonzero(greedy_out != alt_out))
    path_cells = int(sum(len(cells) for cells in greedy_cells))
    return {
        "greedy_out": greedy_out,
        "alt_out": alt_out,
        "diff_cells": diff_cells,
        "path_cells": path_cells,
        **greedy_stats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles on a path where each collision chooses the longer turn.

    Input structure:
    - The top-left area contains a row of 2x2 solid blocks that encode a repeating paint cycle.
    - One or two isolated pixels in template colors sit on the left edge as launch markers inside the path.
    - A separate 2x2 solid block of another color marks the collision/end color.

    Rule:
    - Each marker starts moving along the adjacent path, recoloring visited path cells by advancing through the template cycle.
    - When the next forward cell is not path, that collision cell is marked with the end color if it is background.
    - Then test both 90-degree turn options; choose the side whose straight continuation stays on path for more cells.
    - If neither side is path, the particle stops; if both straight runs are equal, use the left turn as the deterministic tie-break.

    Key insight:
    - The hard part is no longer turn parity but local branch comparison: the solver must evaluate both candidate corridors at each collision and greedily commit to the longer straight run.
    """
    out, _, _ = _simulate(grid, mode="greedy")
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
    """Carve a background fjord into the left side of the band."""
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


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.72 else 1

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

    try:
        metrics = _behavior_metrics(g)
    except Exception:
        return None

    if metrics["path_cells"] < 15:
        return None
    if metrics["total_turns"] < 2:
        return None
    if metrics["strict_greedy_decisions"] < 1:
        return None
    if metrics["diff_cells"] < 4:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
