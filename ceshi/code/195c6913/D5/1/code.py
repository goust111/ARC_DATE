# Variant of 195c6913
# Dimension: D5 — 终止条件 (dst)
# Change: 粒子在计划转向方向不是path时停止（collision_stop）；也在离开网格边界时停止。... → 固定步数停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for this D5 variant
# a) Concrete behavior:
#    The top-left row of 2x2 monochrome blocks still defines the color cycle,
#    but now the same template also encodes distance: every colored template
#    cell counts as one move, so n blocks => 4*n steps. Markers still launch
#    particles from col 0, and collisions still create end_color corner marks
#    plus alternating CCW/CW turns. The visible change is that trajectories are
#    truncated after a fixed prefix instead of running until the first blocked
#    turn/out-of-grid event. Example: with 3 blocks the old rule could paint 20+
#    path cells before a blocked turn; the new rule paints exactly 12 path cells
#    and then stops even if the path continues.
# b) Complexity calibration:
#    Step counts are limited to {8, 12, 16}. That is small enough to infer from
#    2-4 examples, but large enough that the output differs visibly from the old
#    "run until blocked" behavior. Keeping the alternating-turn path means the
#    task is still about following the band rather than just drawing a short ray.
# c) Old assumptions that no longer hold:
#    The original main loop assumed "stop when the scheduled turn is not path"
#    was part of the rule, so it used that branch as the semantic endpoint.
#    Here that condition is only a malformed-input safeguard. Another old
#    assumption was that every marker's natural trajectory length could vary
#    freely; now generator and solver must agree on a shared fixed budget taken
#    from the template area. We therefore parse the template before simulating
#    and assert exact-budget completion in generated instances.
# d) Placement legality:
#    The generator still reserves template space in rows 1-3, keeps the path a
#    single 4-connected component, embeds markers in a col-0 path lobe, and
#    places the 2x2 end block inside bottom-right background. New legality
#    checks ensure each marker can complete the encoded number of steps without
#    hitting an invalid turn early.
# e) Visual observability:
#    The step budget depends only on visible block count/area, never on color id
#    or hidden ordering. Across samples, longer templates produce longer painted
#    prefixes while the rest of the path remains untouched, making the D5 change
#    readable from input/output pairs.
# f) Visual cleanliness:
#    The template, path, markers, and end block remain spatially separated as in
#    the source task. We keep disjoint marker trajectories so recolored prefixes
#    do not tangle into ambiguous overlaps.
# g) Color-change inference:
#    Colors still follow the left-to-right template cycle exactly as before.
#    The only new inference is distance counting from the same template cells,
#    which is structurally tied to a single compact object rather than scattered
#    cues.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
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


def _detect_pattern(
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
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


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


def _trace_particle(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    end_color: int | None,
    start: tuple[int, int, int],
    step_budget: int | None,
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
    if init is None or not pattern:
        return {"cells": set(), "steps": 0, "turns": 0, "blocked": True}

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    steps = 0
    turns = 0
    my_cells: set[tuple[int, int]] = set()
    visited: set[tuple[int, int, int, int, int, int]] = set()

    while True:
        if step_budget is not None and int(steps) >= int(step_budget):
            break
        if step_budget is None:
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            return {"cells": my_cells, "steps": steps, "turns": turns, "blocked": True}

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            if out is not None:
                out[nr, nc] = int(pattern[idx])
            my_cells.add((int(nr), int(nc)))
            r, c = nr, nc
            steps += 1
            continue

        if out is not None and int(g[nr, nc]) == int(template_bg) and end_color is not None:
            out[nr, nc] = int(end_color)

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
        return {"cells": my_cells, "steps": steps, "turns": turns, "blocked": True}

    return {"cells": my_cells, "steps": steps, "turns": turns, "blocked": False}


def _template_step_budget(pat_cells: list[tuple[int, int]]) -> int:
    return int(len(pat_cells))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles along a path, but stop after a template-encoded number of painted steps.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the repeating paint cycle and also a visible step count.
    - Isolated single pixels of the first cycle color sit on the left edge as start markers embedded in the path.
    - A distinct 2x2 block supplies the corner-mark color that appears where turns are triggered.

    Rule:
    - Each start marker follows the path with the same alternating turn rule as the source task: collisions schedule CCW, CW, CCW, CW...
    - Every successful move onto a path cell advances the cycle and consumes one step from the shared budget.
    - The budget equals the number of colored cells inside the top-left template blocks; after exactly that many path cells, the particle stops.

    Key insight: The template now has two jobs at once: read it left-to-right for the color cycle, and count its colored squares to know how far each particle travels.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _detect_pattern(g, template_bg, path_color)
    step_budget = _template_step_budget(pat_cells)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _detect_end_block(g, template_bg, path_color, pat_set)
    markers = _detect_markers(g, path_color, pat_set)
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for marker in markers:
        trace = _trace_particle(
            g, pattern, int(path_color), end_color, marker, int(step_budget), out=out
        )
        assert int(trace["steps"]) == int(step_budget), (
            f"marker at {marker[:2]} could not complete encoded step budget "
            f"{step_budget} (got {trace['steps']})"
        )
        assert not bool(trace["blocked"]), f"marker at {marker[:2]} stopped before budget"
        per_marker_cells.append(set(trace["cells"]))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
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
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.choice([2, 3, 3, 4, 4])
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, k=n_blocks)
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

    for r in marker_rows:
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

    markers = [(int(r), 0, int(cycle_colors[0])) for r in marker_rows]
    fixed_traces = []
    total_turns = 0
    visibly_truncated = False
    for marker in markers:
        fixed = _trace_particle(g, cycle_colors, path_color, end_color, marker, step_budget)
        old = _trace_particle(g, cycle_colors, path_color, end_color, marker, None)
        if int(fixed["steps"]) != step_budget or bool(fixed["blocked"]):
            return None
        if int(old["steps"]) < step_budget + 3:
            return None
        if int(fixed["turns"]) < 1:
            return None
        fixed_traces.append(set(fixed["cells"]))
        total_turns += int(fixed["turns"])
        visibly_truncated = visibly_truncated or (int(old["steps"]) > step_budget)

    if total_turns < n_markers:
        return None
    for i in range(len(fixed_traces)):
        for j in range(i + 1, len(fixed_traces)):
            if fixed_traces[i] & fixed_traces[j]:
                return None
    if not visibly_truncated:
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
