# Variant of 195c6913
# Dimension: D2 — 碰撞转弯规则：严格交替左右转 (path)
# Change: 碰撞时第1/3/5次左转(CCW)，第2/4/6次右转(CW)，严格交替；若计划转向方向不是path则停止（无后备方案）... → 始终同方向转弯

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


# Design analysis for the D2 variant (fixed-left turning).
# a) Concrete behavior:
#    The input objects stay the same: a thick anti-diagonal path band, 2x2 cycle
#    blocks in the top-left, 1-2 single-pixel markers on col 0, and one 2x2 end
#    block in the bottom-right background region. The visible change only happens
#    when a particle collides with a non-path cell. In the original task, the
#    first two valid turns from a right-moving marker were Left then Right, so a
#    corridor could zig-zag. Here the first two valid turns are Left then Left,
#    so the same corridor bends into a spiral-like walk. Example: if a marker
#    goes Right into a wall, both tasks turn Up/Down according to CCW from the
#    incoming direction on the first hit; on the second hit the original turns
#    the opposite way while this variant keeps turning CCW again, so the traced
#    colored strip diverges from that corner onward.
# b) Complexity calibration:
#    The change is subtle if a sample has only 0-1 successful turns, because
#    fixed-left and alternating behave identically until the second turn. The
#    generator therefore rejects trivial samples and keeps only inputs where at
#    least one marker makes >=2 successful turns and the fixed-left output differs
#    from the old alternating output on multiple cells.
# c) Old assumptions that break:
#    The original visited-state key stored hit parity because future behavior
#    depended on odd/even collisions. That is no longer true. Keeping parity in
#    the state would allow loops that revisit the same position/direction/color
#    phase with a different hit count, so the new simulation removes parity from
#    the fixed-turn state key. The rest of the parsing logic still relies on the
#    same visible invariants: template in the top-left, markers isolated, path as
#    the largest non-bg color, and one non-template 2x2 end block.
# d) Placement legality:
#    Template blocks still occupy bg-only cells in rows 1-3 and cols {1,4,7,10};
#    markers still overwrite path cells on col 0 while preserving path 4-connectivity;
#    the end block still sits in an isolated bottom-right bg pocket; separators
#    may carve the path band but never split it into multiple path components.
# e) Visual observability:
#    The generator measures divergence against the original alternating rule and
#    keeps only samples with clear output differences. This avoids invisible
#    cases where the new rule is technically present but indistinguishable by eye.
# f) Visual cleanliness:
#    The scene layout is unchanged, so object roles remain easy to parse: template
#    in the top-left, path band across the board, markers on the left edge, and
#    end block in the lower-right. No new overlay objects are added.
# g) Color inference:
#    Colors still encode only the cycle, the path, the background, and the end
#    color. No ordering by color id is introduced. The cycle is still read from
#    left-to-right 2x2 blocks, exactly as in the original task.
# h) New solver assumptions audited:
#    main() now assumes the fixed turn direction is globally CCW for every marker.
#    generate_input() enforces this by filtering for samples that are valid under
#    that rule, have nontrivial turning, and remain disjoint across markers.

_TURN_POLICY_FIXED_LEFT = "fixed_left"
_TURN_POLICY_ALTERNATING = "alternating"


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color must stay one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


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


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
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
            if v == template_bg or v == path_color:
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

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = v
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
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), v))

    _assert_marker_embedded_in_path(g, markers, path_color)
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


def _turn_left(dr: int, dc: int) -> tuple[int, int]:
    return -int(dc), int(dr)


def _turn_right(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), -int(dr)


def _simulate_from_parsed(parsed: dict[str, object], turn_policy: str) -> dict[str, object]:
    g = np.array(parsed["grid"], dtype=int)
    H = int(parsed["H"])
    W = int(parsed["W"])
    template_bg = int(parsed["template_bg"])
    path_color = int(parsed["path_color"])
    pattern = list(parsed["pattern"])
    pat_cells = list(parsed["pat_cells"])
    end_color = parsed["end_color"]
    end_pos = parsed["end_pos"]
    markers = list(parsed["markers"])

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = template_bg

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    per_marker_cells: list[set[tuple[int, int]]] = []
    marker_stats: list[dict[str, int]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            marker_stats.append({"steps": 0, "turns": 0, "corner_marks": 0})
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        corner_marks = 0
        my_cells: set[tuple[int, int]] = set()
        visited: set[tuple[int, ...]] = set()

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

            if int(g[nr, nc]) == path_color:
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == template_bg and end_color is not None:
                out[nr, nc] = int(end_color)
                corner_marks += 1

            if turn_policy == _TURN_POLICY_ALTERNATING:
                tdr, tdc = _turn_left(dr, dc) if (int(hits) % 2 == 0) else _turn_right(dr, dc)
            else:
                tdr, tdc = _turn_left(dr, dc)
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)
        marker_stats.append(
            {"steps": len(my_cells), "turns": int(hits), "corner_marks": int(corner_marks)}
        )

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return {
        "output": out,
        "per_marker_cells": per_marker_cells,
        "marker_stats": marker_stats,
        "total_turns": int(sum(s["turns"] for s in marker_stats)),
        "max_turns": int(max((s["turns"] for s in marker_stats), default=0)),
        "total_steps": int(sum(s["steps"] for s in marker_stats)),
    }


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    parsed = _parse_scene(grid)
    fixed = _simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT)
    alt = _simulate_from_parsed(parsed, _TURN_POLICY_ALTERNATING)
    return {
        "n_markers": len(parsed["markers"]),
        "fixed_total_turns": int(fixed["total_turns"]),
        "fixed_max_turns": int(fixed["max_turns"]),
        "fixed_total_steps": int(fixed["total_steps"]),
        "diff_cells_vs_alternating": int(
            np.count_nonzero(np.array(fixed["output"]) != np.array(alt["output"]))
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace particles through a path band using one fixed turning direction at every collision.

    Input structure:
    - The top-left contains a row of 2x2 solid-color blocks whose left-to-right order defines a repeating color cycle.
    - One or two isolated single pixels on the left edge reuse a cycle color and act as launch markers beside the path.
    - A separate 2x2 block of another color marks the collision-highlight color source.

    Rule:
    - Each marker starts a particle that moves straight through adjacent path cells, repainting them with the cycle colors in order.
    - When the next cell is not path, that collision cell is colored with the end color if it is background.
    - Every successful turn is the same 90-degree counterclockwise turn; there is no left/right alternation memory.
    - If that fixed turn would not enter a path cell, the particle stops immediately.

    Key insight:
    - The template still defines only the paint cycle; once the turn rule is fixed, the motion state no longer depends on collision parity, so loops are determined just by position, heading, and cycle phase.
    """
    parsed = _parse_scene(grid)
    return np.array(_simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT)["output"], dtype=int)


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


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.65 else 1

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
        parsed = _parse_scene(g)
        fixed = _simulate_from_parsed(parsed, _TURN_POLICY_FIXED_LEFT)
        alt = _simulate_from_parsed(parsed, _TURN_POLICY_ALTERNATING)
    except Exception:
        return None

    diff_cells = int(np.count_nonzero(np.array(fixed["output"]) != np.array(alt["output"])))
    if fixed["max_turns"] < 2:
        return None
    if fixed["total_turns"] < 2:
        return None
    if fixed["total_steps"] < 12:
        return None
    if diff_cells < 4:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 600 attempts")
