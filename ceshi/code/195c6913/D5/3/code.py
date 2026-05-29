# Variant of 195c6913
# Dimension: D5 — 终止条件 (dst)
# Change: 粒子在计划转向方向不是path时停止（collision_stop）；也在离开网格边界时停止。... → 两个粒子相遇时同时停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D5 variant:
# variant_spec:
#   requested_source_task: e12f9a14
#   source_solver_status: >-
#     No local solver file for e12f9a14 exists in the provided workspace. I
#     searched the repo and only found its ARC json plus task notes, so the
#     implementation baseline below uses the original 195c6913 solver/spec and
#     the provided "collision_detection" description.
#   a_concretization:
#     input_objects:
#       - The top-left still contains 2x2 monochrome blocks whose left-to-right
#         order defines the paint cycle.
#       - The path is still one thick anti-diagonal band.
#       - The generator always places exactly two start markers on column 0 so
#         the interaction is visible in every sample.
#       - A shallow background fjord cuts between the two markers near the left
#         edge, forcing two early channels that later merge back into one band.
#     behavior_trace:
#       - Old logic: marker A and marker B are simulated independently, so even
#         if both eventually use the same corridor, each keeps painting until it
#         hits its own local stop condition.
#       - New logic: both particles advance in lockstep while sharing an owner
#         map of already-painted path cells. When one particle plans to enter a
#         cell that the other particle painted on an earlier tick, both freeze
#         immediately and neither applies its tick's move.
#       - Visual difference: the merged corridor is only partially painted; one
#         trail reaches the other and both stop, leaving remaining path cells
#         blank where the original task would have continued.
#   b_complexity_calibration:
#     - The variant uses exactly two markers because one marker cannot expose
#       the new stopping rule.
#     - The path still has 2-4 cycle colors and several wall-collision corner
#       stamps, so the original cyclic painting logic remains legible.
#     - The generator rejects samples where the new output matches the old
#       independent-trace output, preventing silent degeneration.
#   c_implicit_assumptions_audit:
#     - Original per-marker loop assumed trajectories are independent; that is
#       no longer valid, so the solver now plans all active particles together.
#     - The original final assert required pairwise-disjoint trajectories; that
#       assumption is inverted here and replaced by a required merge in the
#       generator.
#     - Writing directly into out during one particle's move was harmless before
#       because paths were disjoint; with interaction that becomes order-biased,
#       so the new solver computes one tick of plans first and then applies the
#       non-conflicting moves simultaneously.
#     - Turn decisions still consult the immutable input path `g`, not the
#       painted output, because recoloring does not change traversability.
#   d_placement_legality:
#     - Template blocks stay in the original top-left slots and must sit on bg.
#     - The end block stays in bottom-right bg separated from the path.
#     - Markers remain embedded in a col-0 path segment with path neighbors
#       above and below.
#     - The fjord carve never disconnects the path and stops early so the two
#       channels merge later; the generator checks this with the new simulator.
#   e_visual_observability:
#     - Every generated puzzle has two markers and at least one interaction
#       event, so the new rule is visible from 2-3 examples.
#     - The generator also checks that the new solver's output differs from an
#       "old-rule" independent simulation on the same input.
#   f_visual_cleanliness:
#     - The scene keeps the original object vocabulary: one band, one template
#       row, one end block, two marker dots.
#     - Only one shallow fjord is carved, which is enough to split then merge
#       the routes without cluttering the grid.
#   g_color_inference:
#     - Colors are still explained purely by visible structure: the template
#       defines the cycle, wall hits use the end color, and no color-id-based
#       ordering or hidden numeric rule is introduced.


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker sits inside a col-0 path segment."""
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


def _extract_scene(grid: np.ndarray) -> dict:
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
            rc = (int(r), int(c))
            if rc in seen:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(v)
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    pat_set = set(pattern)
    end_color = None
    end_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                end_color = v
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in DIR_ORDER:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), v))

    markers.sort()
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


def _make_base_output(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    for r, c in scene["pat_cells"]:
        out[r, c] = int(scene["template_bg"])
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[er : er + 2, ec : ec + 2] = int(scene["template_bg"])
    return out


def _initial_direction(g: np.ndarray, path_color: int, sr: int, sc: int) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in DIR_ORDER:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def _build_particles(scene: dict) -> list[dict]:
    g = scene["grid"]
    pattern = scene["pattern"]
    path_color = int(scene["path_color"])
    particles: list[dict] = []
    for sr, sc, sv in scene["markers"]:
        init = _initial_direction(g, path_color, sr, sc)
        idx = int(pattern.index(int(sv))) if pattern and int(sv) in pattern else 0
        particles.append(
            {
                "r": int(sr),
                "c": int(sc),
                "dr": int(init[0]) if init is not None else 0,
                "dc": int(init[1]) if init is not None else 0,
                "idx": idx,
                "hits": 0,
                "active": init is not None and bool(pattern),
            }
        )
    return particles


def _simulate_variant(scene: dict) -> tuple[np.ndarray, dict]:
    g = scene["grid"]
    H = int(scene["H"])
    W = int(scene["W"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    end_color = scene["end_color"]
    template_bg = int(scene["template_bg"])

    out = _make_base_output(scene)
    particles = _build_particles(scene)
    owner = np.full((H, W), -1, dtype=int)
    visited_states: list[set[tuple[int, int, int, int, int, int]]] = [set() for _ in particles]
    painted_cells: list[set[tuple[int, int]]] = [set() for _ in particles]
    stop_reasons = ["inactive" if not p["active"] else "" for p in particles]
    turns = [0 for _ in particles]
    meeting_events: list[tuple[int, int, tuple[int, int]]] = []

    for _ in range(H * W * 8):
        active_ids = [i for i, p in enumerate(particles) if p["active"]]
        if not active_ids:
            break

        plans: dict[int, dict] = {}
        for i in active_ids:
            p = particles[i]
            state = (p["r"], p["c"], p["dr"], p["dc"], p["idx"], p["hits"] & 1)
            if state in visited_states[i]:
                plans[i] = {"type": "stop", "reason": "loop"}
                continue
            visited_states[i].add(state)

            nr = int(p["r"] + p["dr"])
            nc = int(p["c"] + p["dc"])
            if not (0 <= nr < H and 0 <= nc < W):
                plans[i] = {"type": "stop", "reason": "boundary"}
                continue

            if int(g[nr, nc]) == path_color:
                prev_owner = int(owner[nr, nc])
                if prev_owner >= 0 and prev_owner != i:
                    plans[i] = {
                        "type": "meet",
                        "other": prev_owner,
                        "target": (int(nr), int(nc)),
                    }
                else:
                    plans[i] = {"type": "step", "target": (int(nr), int(nc))}
                continue

            left = (-int(p["dc"]), int(p["dr"]))
            right = (int(p["dc"]), -int(p["dr"]))
            tdr, tdc = left if (int(p["hits"]) % 2 == 0) else right
            rr = int(p["r"] + tdr)
            cc = int(p["c"] + tdc)
            mark_cell = (int(nr), int(nc)) if int(g[nr, nc]) == template_bg and end_color is not None else None
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                plans[i] = {"type": "turn", "dir": (int(tdr), int(tdc)), "mark": mark_cell}
            else:
                plans[i] = {"type": "stop", "reason": "blocked_turn", "mark": mark_cell}

        stop_ids: set[int] = set()
        seen_pairs: set[tuple[int, int, int, int]] = set()
        for i, plan in plans.items():
            if plan["type"] != "meet":
                continue
            j = int(plan["other"])
            pair = tuple(sorted((int(i), j)))
            key = (pair[0], pair[1], int(plan["target"][0]), int(plan["target"][1]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                meeting_events.append((pair[0], pair[1], (int(plan["target"][0]), int(plan["target"][1]))))
            stop_ids.add(int(i))
            if 0 <= j < len(particles):
                stop_ids.add(j)

        targets: dict[tuple[int, int], list[int]] = {}
        for i, plan in plans.items():
            if plan["type"] == "step":
                targets.setdefault(plan["target"], []).append(int(i))
        for target, ids in targets.items():
            if len(ids) >= 2:
                for i in ids:
                    stop_ids.add(i)
                if len(ids) >= 2:
                    pair = tuple(sorted(ids[:2]))
                    meeting_events.append((pair[0], pair[1], (int(target[0]), int(target[1]))))

        for i in stop_ids:
            if 0 <= i < len(particles) and particles[i]["active"]:
                particles[i]["active"] = False
                stop_reasons[i] = "met_particle"

        for i in active_ids:
            if i in stop_ids:
                continue
            plan = plans[i]
            p = particles[i]
            if plan["type"] == "step":
                nr, nc = plan["target"]
                p["idx"] = int((p["idx"] + 1) % len(pattern))
                out[nr, nc] = int(pattern[p["idx"]])
                owner[nr, nc] = int(i)
                painted_cells[i].add((int(nr), int(nc)))
                p["r"], p["c"] = int(nr), int(nc)
            elif plan["type"] == "turn":
                if plan["mark"] is not None:
                    mr, mc = plan["mark"]
                    out[mr, mc] = int(end_color)
                p["dr"], p["dc"] = plan["dir"]
                p["hits"] += 1
                turns[i] += 1
            else:
                if plan.get("mark") is not None:
                    mr, mc = plan["mark"]
                    out[mr, mc] = int(end_color)
                p["active"] = False
                stop_reasons[i] = str(plan["reason"])

    for i, p in enumerate(particles):
        if p["active"]:
            p["active"] = False
            if not stop_reasons[i]:
                stop_reasons[i] = "exhausted"
        elif not stop_reasons[i]:
            stop_reasons[i] = "inactive"

    stats = {
        "meeting_events": meeting_events,
        "meeting_count": len(meeting_events),
        "painted_cells": painted_cells,
        "steps": [len(cells) for cells in painted_cells],
        "turns": turns,
        "stop_reasons": stop_reasons,
    }
    return out, stats


def _simulate_original_independent(scene: dict) -> tuple[np.ndarray, dict]:
    g = scene["grid"]
    H = int(scene["H"])
    W = int(scene["W"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    end_color = scene["end_color"]
    template_bg = int(scene["template_bg"])

    out = _make_base_output(scene)
    per_marker_cells: list[set[tuple[int, int]]] = []
    turns: list[int] = []

    for sr, sc, sv in scene["markers"]:
        init = _initial_direction(g, path_color, sr, sc)
        if init is None or not pattern:
            per_marker_cells.append(set())
            turns.append(0)
            continue

        r, c = int(sr), int(sc)
        dr, dc = int(init[0]), int(init[1])
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()
        my_turns = 0

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
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

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                my_turns += 1
                continue
            break

        per_marker_cells.append(my_cells)
        turns.append(my_turns)

    return out, {"painted_cells": per_marker_cells, "steps": [len(c) for c in per_marker_cells], "turns": turns}


def _behavior_metrics(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    variant_out, variant_stats = _simulate_variant(scene)
    original_out, original_stats = _simulate_original_independent(scene)
    return {
        "variant_out": variant_out,
        "variant_stats": variant_stats,
        "original_out": original_out,
        "original_stats": original_stats,
        "diff_from_original": not np.array_equal(variant_out, original_out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Run two alternating-turn particles on one path and stop both when one reaches the other's trail.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the repeating paint cycle.
    - Left edge: two isolated marker pixels, colored with the first cycle color, are embedded in the path.
    - Elsewhere: one thick anti-diagonal path region plus a separate 2x2 block for the end color.

    Rule:
    - Each marker starts with the original local motion: choose the first adjacent path cell in R,D,L,U order, move straight, and repaint path cells by advancing through the cycle.
    - On each non-path collision, stamp that collision cell with the end color and alternate the forced turn direction CCW, CW, CCW, CW...; if the scheduled turn is unavailable, that particle halts.
    - The two particles are simulated in lockstep, and if either tries to enter a path cell already painted by the other particle, both particles stop immediately before that tick's move is applied.

    Key insight: the template still explains the color cycle, but the decisive new cue is the shared corridor where one trail catches the other and truncates both traces at once.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    out, _stats = _simulate_variant(scene)
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal band similar to the original task."""
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
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            center = centers[r] + cc_jit[r]
            c_left[r] = int(round(center - width / 2))
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


def _carve_merge_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a shallow fjord that splits the left side but leaves a later merge."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    cur_top = random.randint(sr_min, sr_max)

    working = mask.copy()
    snapshots: list[np.ndarray] = []
    for k in range(W - 1):
        trial = working.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        working = trial
        snapshots.append(working.copy())
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1

    if len(snapshots) < 4:
        return False

    max_keep = min(len(snapshots) - 1, max(4, W // 2))
    if max_keep < 4:
        return False
    keep = random.randint(4, max_keep)
    np.copyto(mask, snapshots[keep - 1])
    return True


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

    r1 = random.randint(H // 3, H // 2)
    r2 = random.randint(max(r1 + 6, 2 * H // 3), H - 4)
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

    trial = mask.copy()
    if not _carve_merge_fjord(trial, r1, r2):
        return None
    _, n_path_after_carve = label_mask(trial, connectivity=4)
    if n_path_after_carve != 1:
        return None
    for r in marker_rows:
        if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
            return None
    mask = trial

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr = template_row + dr
                cc = bc + dc
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
            r0, r1b = max(0, r - 1), min(H, r + 3)
            c0, c1b = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1b, c0:c1b] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in marker_rows:
        g[r, 0] = cycle_colors[0]

    _, n_path_after_markers = label_mask(g == path_color, connectivity=4)
    if n_path_after_markers != 1:
        return None
    if int(g[0, 0]) != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or int(non_bg[0][1]) != path_color:
        return None

    try:
        metrics = _behavior_metrics(g)
    except Exception:
        return None

    variant_stats = metrics["variant_stats"]
    original_stats = metrics["original_stats"]
    if len(variant_stats["steps"]) != 2:
        return None
    if variant_stats["meeting_count"] < 1:
        return None
    if min(variant_stats["steps"]) < 6:
        return None
    if sum(variant_stats["turns"]) < 2:
        return None
    if variant_stats["stop_reasons"] != ["met_particle", "met_particle"]:
        return None
    if sum(original_stats["steps"]) <= sum(variant_stats["steps"]):
        return None
    if not metrics["diff_from_original"]:
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
