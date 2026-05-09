# Variant of 195c6913
# Dimension: D3 — 轨迹涂色方式：颜色循环 (trajectory)
# Change: 粒子沿路径前进时用模板定义的颜色循环（2-4色）依次涂色，每步推进一个循环位；涂色连续不间断。... → 间歇涂色（跳格涂）
#
# Design analysis
# a) Concrete behavior and trace:
#    - Input layout stays the same: a top-left row of 2x2 template blocks, 1-2
#      left-edge markers embedded in the path band, and one 2x2 end block in the
#      opposite bg region.
#    - Only the trajectory painting cadence changes. A particle still walks the
#      same path cells, uses the same initial direction preference, and turns on
#      collisions with the same CCW/CW alternation.
#    - Old local trace with cycle [A,B,C] and marker color A:
#      move1->paint B, move2->paint C, move3->paint A, move4->paint B ...
#    - New local trace:
#      move1->paint B, move2->leave as path_color, move3->paint C,
#      move4->leave as path_color, move5->paint A ...
#      The visible path becomes a dashed line; collision cells are still marked
#      by the end color exactly as before.
# b) Complexity calibration:
#    - The reference task 4a21e3da implements intermittent drawing by leaving
#      some traversed cells untouched. This variant mirrors that level of change:
#      intermittent output is visible, but the scene grammar is unchanged.
#    - The skip period is fixed at 2. Allowing many possible N values would make
#      the rule hard to infer from 2-4 ARC examples.
# c) Hidden assumptions audit:
#    - Original `idx = (idx + 1) % len(pattern); out[nr,nc] = pattern[idx]`
#      assumes every traversed path cell is recolored. That is no longer valid.
#      We must decouple "move" from "paint".
#    - Original loop-state memoization omits stride phase because continuous
#      painting has no phase. The dashed variant does, so state now includes
#      `steps_taken % paint_period`.
#    - Overlap checks must still use all traversed path cells, not only painted
#      ones, otherwise two particles could silently share the same skipped cells.
# d) Placement legality:
#    - No new objects are introduced, so template/end/marker placement rules
#      remain valid. The generator only needs to guarantee enough trajectory
#      length that both painted and skipped cells appear.
# e) Visual observability:
#    - The output difference is directly visible without relying on color ids:
#      every other path cell stays in the original path color, producing a
#      regular dash-gap rhythm along the same corridor.
#    - The generator rejects samples where dashed output would equal the old
#      continuous output or where the dash gaps are too sparse.
# f) Visual cleanliness:
#    - Skipped cells remain the original path color, so the dashed track reads as
#      a clean interruption inside one existing band rather than as overlaid
#      clutter from extra symbols.
# g) Color inferability:
#    - The template still explains the order of painted dash colors.
#    - To avoid washed-out cases, the generator enforces at least two distinct
#      template colors, giving human solvers repeated evidence that spacing and
#      color-order are separate parts of the rule.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker in col 0 sits inside a vertical path run."""
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


def _scan_scene(
    g: np.ndarray,
) -> tuple[int, int, list[int], list[tuple[int, int]], int | None, tuple[int, int] | None, list[tuple[int, int, int]]]:
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
    return template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def _simulate(
    grid: np.ndarray, paint_period: int = 2
) -> tuple[np.ndarray, list[set[tuple[int, int]]], list[dict[str, int]], int]:
    assert paint_period >= 1
    g = np.array(grid, dtype=int)
    H, W = g.shape
    (
        template_bg,
        path_color,
        pattern,
        pat_cells,
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

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    per_marker_cells: list[set[tuple[int, int]]] = []
    stats: list[dict[str, int]] = []

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
            stats.append(
                {
                    "traversed_steps": 0,
                    "painted_steps": 0,
                    "skipped_steps": 0,
                    "turns": 0,
                    "collision_marks": 0,
                }
            )
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        steps_taken = 0
        visited: set[tuple[int, int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()
        painted_steps = 0
        skipped_steps = 0
        collision_marks = 0

        for _ in range(H * W * 4):
            state = (
                int(r),
                int(c),
                int(dr),
                int(dc),
                int(idx),
                int(hits & 1),
                int(steps_taken % paint_period),
            )
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == int(path_color):
                steps_taken += 1
                my_cells.add((int(nr), int(nc)))
                if ((steps_taken - 1) % paint_period) == 0:
                    idx = int((idx + 1) % len(pattern))
                    out[nr, nc] = int(pattern[idx])
                    painted_steps += 1
                else:
                    skipped_steps += 1
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
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
                continue
            break

        per_marker_cells.append(my_cells)
        stats.append(
            {
                "traversed_steps": int(steps_taken),
                "painted_steps": int(painted_steps),
                "skipped_steps": int(skipped_steps),
                "turns": int(hits),
                "collision_marks": int(collision_marks),
            }
        )

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, per_marker_cells, stats, int(path_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate path-following particles whose visible trail is dashed instead of solid.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks encodes the cyclic paint order.
    - Isolated edge pixels in template colors: launch markers embedded in the left path rim.
    - A separate 2x2 monochrome block: the collision-marker color source.

    Rule:
    - Each marker launches a particle that advances straight through path cells and only paints every other move, using the template cycle in order on painted moves.
    - When a forward step leaves the path, that collision cell is marked with the end color and the particle tries a 90° turn.
    - Turn direction alternates strictly CCW, CW, CCW, CW ... across successive collisions.
    - If the scheduled turn is not a path cell, the particle stops; template blocks and the 2x2 end block are erased in the output.

    Key insight: motion is unchanged from the base task, but the trail is sampled at a fixed cadence, so the solver must separate path traversal from paint application.
    """
    out, _, _, _ = _simulate(grid, paint_period=2)
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


def _outputs_differ_on_dashed_cells(g: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    dashed_out, _, dashed_stats, path_color = _simulate(g, paint_period=2)
    solid_out, _, _, _ = _simulate(g, paint_period=1)
    dash_gap_cells = int(np.count_nonzero((dashed_out == path_color) & (solid_out != path_color)))
    return dash_gap_cells, dashed_stats


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
    if len(set(cycle_colors)) < 2:
        cycle_colors[random.randrange(n_blocks)] = cycle_pool[1]
    if len(set(cycle_colors)) < 2:
        return None

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
        _ = main(g)
    except Exception:
        return None

    try:
        dash_gap_cells, dashed_stats = _outputs_differ_on_dashed_cells(g)
    except Exception:
        return None

    active_stats = [s for s in dashed_stats if s["traversed_steps"] > 0]
    if len(active_stats) != n_markers:
        return None
    if dash_gap_cells < 6:
        return None
    if any(s["painted_steps"] < 3 or s["skipped_steps"] < 3 for s in active_stats):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
