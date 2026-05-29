# Combo variant of 195c6913 (index 12)
# combo_spec: D5/3+D9/1+D15/1
# Name: 粒子相遇停止+轨迹允许交叉+碰撞后发射新粒子

from __future__ import annotations

# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - The input vocabulary stays close to 195c6913: one thick anti-diagonal
#      path band, one top-left row of 2x2 template blocks defining a cycle,
#      one 2x2 end-color block, and exactly two col-0 markers.
#    - The fused behavior is not a literal sum of the three single variants.
#      D5 alone would stop when one particle reaches the other's painted trail,
#      while D9 explicitly allows trail crossing. The coherent merge is:
#      active particles only stop when they meet on the current tick; previously
#      painted cells are traversable and later arrivals overwrite earlier paint.
#    - Small trace:
#      base task: A and B paint independently, never overlap, and blocked turns
#      simply terminate each trace.
#      combo task:
#        1. A and B start together and take the original path/turn dynamics.
#        2. A hits a bg fjord wall, cannot make its scheduled turn, stamps that
#           impact with end_color, and emits a straight bridge particle.
#        3. That bridge crosses the fjord, re-enters path, and resumes cycle
#           painting from the landing point.
#        4. B later reuses one of A's old path cells; this is allowed, and B's
#           later cycle color overwrites A's older one.
#        5. The two active particles eventually target the same cell in the
#           shared corridor, so both stop before that tick is applied.
#      The visible difference is therefore a chain-reaction trace with both
#      end-color bridges and overwritten path cells, truncated by a final
#      particle-meeting event.
# b) Complexity calibration
#    - The generator always uses exactly two markers; otherwise D5/D9 would be
#      invisible.
#    - The bridge rule is constrained to one palette role: end_color remains the
#      only color used for impact dots and bridge segments, so the extra stage is
#      visually learnable.
#    - Crossing is constrained to "later write wins" rather than a large mix
#      table. That keeps the output readable and avoids hidden color mappings.
#    - Accepted samples must show all three signals: at least one bridge spawn
#      and re-entry, at least one cross-lineage overwrite, and at least one
#      active-particle meeting that changes the output versus a no-meeting
#      baseline.
# c) Implicit-assumption audit
#    - The original per-marker independent loop is invalid: particles now share
#      time and may interact, so the simulator plans all active particles in
#      lockstep each tick.
#    - The original disjoint-trajectory assert is inverted. Overlap is no longer
#      illegal; it is required in accepted samples and implemented as overwrite.
#    - The D5 single-variant owner-map stop rule is relaxed but not removed.
#      Old paint is still traversable and visibly overwritten, but if one live
#      lineage steps into the other live lineage's already-painted trail while
#      both are still active, that trail meeting halts the active set.
#    - The D15 spawn rule introduces a new mode ("bridge"), so visited-state
#      loop detection must include the mode as well as direction, cycle index,
#      and turn parity.
#    - Traversability must still be read from the immutable input grid `g`, not
#      from the partially painted output, because recoloring does not change
#      which cells are path or background.
# d) Placement legality
#    - Template blocks stay in the original reserved top-left bg pocket and the
#      end block stays in a bottom-right bg pocket separated from path.
#    - Both markers remain embedded on col 0 with path above and below so the
#      original initialization logic still applies.
#    - The generator carves fjords/islands only when the path remains a single
#      4-connected component and re-validates marker embedding after every carve.
#    - Any extra gap used to encourage bridge re-entry is restored near col 0 so
#      the left-edge marker lobe is not destroyed.
# e) Visual observability
#    - A solver seeing 2-4 examples can infer:
#      * template blocks still define the cycle,
#      * dead-end impacts no longer just stop; they launch end-color bridges,
#      * old paint can be overwritten by the other lineage,
#      * the final shared corridor is cut short because the two active particles
#        meet and both halt.
#    - No decision depends on raw color ids or non-visual ordering.
# f) Visual cleanliness
#    - Only three paint roles appear in output: cycle colors on path, end_color
#      on impact/bridge cells, and untouched path_color where nothing visited.
#    - Overlap uses overwrite rather than a fourth mixed symbol, so shared
#      regions remain readable.
#    - The generator requires each lineage to keep exclusive path length in
#      addition to shared behavior, preventing collapse into a single ambiguous
#      scribble.
# g) Color-change inferability
#    - The template still explains the cycle colors.
#    - end_color has one structural meaning across all samples: collision and
#      chain-reaction emission.
#    - Overwrite uses the arriving particle's ordinary cycle/end color rather
#      than a hidden color algebra, so humans can infer the rule from chronology
#      and geometry alone.
# Solver/generator contract
#    - main() assumes exactly one path component, a detectable template cycle,
#      exactly one end-color 2x2 block, and two embedded markers; generate_input()
#      constructs and checks those invariants.
#    - main() additionally assumes any visible overlap is resolved by time order,
#      while a live trail catch or same-tick occupancy meeting halts the active
#      set. generate_input() validates that its accepted examples exercise the
#      overwrite-vs-stop distinction.

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import label_mask


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
_CACHE: list[np.ndarray] = []


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker sits inside a vertical path segment."""
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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
            cell = (int(r), int(c))
            if cell in seen:
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
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
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


def _make_base_output(scene: dict[str, object]) -> np.ndarray:
    out = np.array(scene["grid"], copy=True)
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(scene["template_bg"])
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(scene["template_bg"])
    return out


def _initial_direction(g: np.ndarray, path_color: int, sr: int, sc: int) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in DIR_ORDER:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def _build_particles(scene: dict[str, object]) -> list[dict[str, object]]:
    g = np.array(scene["grid"], dtype=int)
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    particles: list[dict[str, object]] = []
    for lineage, (sr, sc, sv) in enumerate(scene["markers"]):
        init = _initial_direction(g, path_color, sr, sc)
        idx = int(pattern.index(int(sv))) if pattern and int(sv) in pattern else 0
        particles.append(
            {
                "lineage": int(lineage),
                "r": int(sr),
                "c": int(sc),
                "dr": int(init[0]) if init is not None else 0,
                "dc": int(init[1]) if init is not None else 0,
                "idx": idx,
                "hits": 0,
                "mode": "path",
                "active": init is not None and bool(pattern),
                "chain_segments": 1 if init is not None and bool(pattern) else 0,
            }
        )
    return particles


def _plan_particle(
    g: np.ndarray,
    particle: dict[str, object],
    path_color: int,
    template_bg: int,
    pattern: list[int],
    end_color: int | None,
    visited: set[tuple[str, int, int, int, int, int, int]],
) -> dict[str, object]:
    H, W = g.shape
    mode = str(particle["mode"])
    r = int(particle["r"])
    c = int(particle["c"])
    dr = int(particle["dr"])
    dc = int(particle["dc"])
    idx = int(particle["idx"])
    hits = int(particle["hits"])

    state = (mode, r, c, dr, dc, idx, hits & 1)
    if state in visited:
        return {"kind": "stop", "reason": "loop", "writes": [], "move_to": None}
    visited.add(state)

    nr = int(r + dr)
    nc = int(c + dc)
    if not (0 <= nr < H and 0 <= nc < W):
        return {"kind": "stop", "reason": "boundary", "writes": [], "move_to": None}

    if mode == "path":
        if int(g[nr, nc]) == int(path_color):
            next_idx = int((idx + 1) % len(pattern))
            return {
                "kind": "step",
                "reason": "path",
                "writes": [((int(nr), int(nc)), int(pattern[next_idx]))],
                "move_to": (int(nr), int(nc)),
                "next_idx": next_idx,
            }

        writes: list[tuple[tuple[int, int], int]] = []
        if int(g[nr, nc]) == int(template_bg) and end_color is not None:
            writes.append(((int(nr), int(nc)), int(end_color)))

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (hits % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return {
                "kind": "turn",
                "reason": "turn",
                "writes": writes,
                "move_to": None,
                "dir": (int(tdr), int(tdc)),
            }

        if int(g[nr, nc]) == int(template_bg):
            return {
                "kind": "spawn",
                "reason": "spawn_bridge",
                "writes": writes,
                "move_to": (int(nr), int(nc)),
            }

        return {"kind": "stop", "reason": "blocked_turn", "writes": writes, "move_to": None}

    if int(g[nr, nc]) == int(template_bg):
        writes = [((int(nr), int(nc)), int(end_color))] if end_color is not None else []
        return {
            "kind": "bridge_step",
            "reason": "bridge_bg",
            "writes": writes,
            "move_to": (int(nr), int(nc)),
        }

    if int(g[nr, nc]) == int(path_color):
        writes = [((int(nr), int(nc)), int(end_color))] if end_color is not None else []
        return {
            "kind": "reenter",
            "reason": "bridge_reenter",
            "writes": writes,
            "move_to": (int(nr), int(nc)),
        }

    writes = [((int(nr), int(nc)), int(end_color))] if end_color is not None else []
    return {"kind": "stop", "reason": "bridge_blocked", "writes": writes, "move_to": None}


def _plans_meet(
    particles: list[dict[str, object]],
    plans: dict[int, dict[str, object]],
) -> tuple[bool, str]:
    active_ids = list(plans.keys())
    for i in range(len(active_ids)):
        a = int(active_ids[i])
        for j in range(i + 1, len(active_ids)):
            b = int(active_ids[j])
            pa = plans[a]
            pb = plans[b]
            ta = pa.get("move_to")
            tb = pb.get("move_to")
            ar, ac = int(particles[a]["r"]), int(particles[a]["c"])
            br, bc = int(particles[b]["r"]), int(particles[b]["c"])

            if ta is not None and tb is not None and tuple(ta) == tuple(tb):
                return True, "same_target"
            if ta is not None and tb is not None and tuple(ta) == (br, bc) and tuple(tb) == (ar, ac):
                return True, "swap"
            if ta is not None and pb["kind"] == "turn" and tuple(ta) == (br, bc):
                return True, "enter_turning_particle"
            if tb is not None and pa["kind"] == "turn" and tuple(tb) == (ar, ac):
                return True, "enter_turning_particle"
    return False, ""


def _trail_meet_kind(
    active_ids: list[int],
    plans: dict[int, dict[str, object]],
    ever_written: list[set[tuple[int, int]]],
) -> str:
    """Detect post-apply meetings where a live particle enters the other live trail."""
    if len(active_ids) < 2:
        return ""
    for i in active_ids:
        target = plans[i].get("move_to")
        if target is None:
            continue
        for j in active_ids:
            if i == j:
                continue
            if tuple(target) in ever_written[j]:
                return "trail_catch"
    return ""


def _write_cell(
    out: np.ndarray,
    g: np.ndarray,
    write_owner: np.ndarray,
    cell: tuple[int, int],
    color: int,
    lineage: int,
    stats: dict[str, object],
) -> None:
    r, c = cell
    prev_owner = int(write_owner[r, c])
    if prev_owner >= 0 and prev_owner != int(lineage):
        stats["cross_lineage_overwrites"] += 1
        if int(g[r, c]) == int(stats["path_color"]):
            stats["cross_lineage_path_overwrites"] += 1
    elif prev_owner == int(lineage):
        stats["self_overwrites"] += 1
    out[r, c] = int(color)
    write_owner[r, c] = int(lineage)


def _simulate_combo(
    scene: dict[str, object],
    *,
    stop_on_meet: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(scene["grid"], dtype=int)
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    end_color = scene["end_color"]

    out = _make_base_output(scene)
    write_owner = np.full((H, W), -1, dtype=int)
    particles = _build_particles(scene)
    visited_states: list[set[tuple[str, int, int, int, int, int, int]]] = [set() for _ in particles]
    ever_written: list[set[tuple[int, int]]] = [set() for _ in particles]

    stats: dict[str, object] = {
        "path_color": path_color,
        "meeting_count": 0,
        "meeting_kind": "",
        "spawn_count": 0,
        "reentry_count": 0,
        "bridge_cells": 0,
        "cross_lineage_overwrites": 0,
        "cross_lineage_path_overwrites": 0,
        "self_overwrites": 0,
        "turns": [0 for _ in particles],
        "path_steps": [0 for _ in particles],
        "bridge_steps": [0 for _ in particles],
        "stop_reasons": ["inactive" if not p["active"] else "" for p in particles],
        "chain_segments": [int(p["chain_segments"]) for p in particles],
    }

    for _ in range(H * W * 12):
        active_ids = [i for i, p in enumerate(particles) if bool(p["active"])]
        if not active_ids:
            break

        plans: dict[int, dict[str, object]] = {}
        for i in active_ids:
            plans[i] = _plan_particle(
                g,
                particles[i],
                path_color,
                template_bg,
                pattern,
                end_color,
                visited_states[i],
            )

        if stop_on_meet:
            met, kind = _plans_meet(particles, plans)
            if met:
                stats["meeting_count"] += 1
                stats["meeting_kind"] = str(kind)
                for i in active_ids:
                    particles[i]["active"] = False
                    stats["stop_reasons"][i] = "met_particle"
                break

        for i in active_ids:
            particle = particles[i]
            plan = plans[i]
            for cell, color in plan["writes"]:
                _write_cell(out, g, write_owner, cell, int(color), int(i), stats)
                ever_written[i].add((int(cell[0]), int(cell[1])))

            kind = str(plan["kind"])
            if kind == "step":
                nr, nc = plan["move_to"]
                particle["idx"] = int(plan["next_idx"])
                particle["r"] = int(nr)
                particle["c"] = int(nc)
                stats["path_steps"][i] += 1
            elif kind == "turn":
                tdr, tdc = plan["dir"]
                particle["dr"] = int(tdr)
                particle["dc"] = int(tdc)
                particle["hits"] = int(particle["hits"]) + 1
                stats["turns"][i] += 1
            elif kind == "spawn":
                nr, nc = plan["move_to"]
                particle["mode"] = "bridge"
                particle["r"] = int(nr)
                particle["c"] = int(nc)
                particle["chain_segments"] = int(particle["chain_segments"]) + 1
                stats["chain_segments"][i] = int(particle["chain_segments"])
                stats["spawn_count"] += 1
            elif kind == "bridge_step":
                nr, nc = plan["move_to"]
                particle["r"] = int(nr)
                particle["c"] = int(nc)
                stats["bridge_steps"][i] += 1
                stats["bridge_cells"] += 1
            elif kind == "reenter":
                nr, nc = plan["move_to"]
                particle["mode"] = "path"
                particle["r"] = int(nr)
                particle["c"] = int(nc)
                particle["chain_segments"] = int(particle["chain_segments"]) + 1
                stats["chain_segments"][i] = int(particle["chain_segments"])
                stats["bridge_steps"][i] += 1
                stats["bridge_cells"] += 1
                stats["reentry_count"] += 1
            else:
                particle["active"] = False
                stats["stop_reasons"][i] = str(plan["reason"])

        if stop_on_meet:
            kind = _trail_meet_kind(active_ids, plans, ever_written)
            if kind:
                stats["meeting_count"] += 1
                stats["meeting_kind"] = str(kind)
                for i in active_ids:
                    if bool(particles[i]["active"]):
                        particles[i]["active"] = False
                        stats["stop_reasons"][i] = "met_particle"
                break

    for i, particle in enumerate(particles):
        if bool(particle["active"]):
            particle["active"] = False
            if not stats["stop_reasons"][i]:
                stats["stop_reasons"][i] = "exhausted"
        elif not stats["stop_reasons"][i]:
            stats["stop_reasons"][i] = "inactive"

    stats["max_chain_segments"] = max(stats["chain_segments"], default=0)
    stats["total_writes"] = int(np.count_nonzero(write_owner >= 0))
    return out, stats


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    scene = _extract_scene(np.array(grid, dtype=int))
    variant_out, variant_stats = _simulate_combo(scene, stop_on_meet=True)
    baseline_out, baseline_stats = _simulate_combo(scene, stop_on_meet=False)
    return {
        "variant_out": variant_out,
        "variant_stats": variant_stats,
        "baseline_out": baseline_out,
        "baseline_stats": baseline_stats,
        "diff_from_no_meet": not np.array_equal(variant_out, baseline_out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate two alternating-turn particle lineages whose old trails can be reused but whose live meeting ends the chain.

    Input structure:
    - The upper-left still holds a row of monochrome 2x2 blocks whose left-to-right order defines the repeating cycle.
    - Exactly two isolated marker pixels sit on the left-edge path lobe and use a cycle color.
    - A separate monochrome 2x2 block gives the terminal color used for impacts and bridge segments, while the main path is one thick anti-diagonal band.

    Rule:
    - The two markers are simulated in lockstep. In path mode they follow the original straight-ahead motion and strict alternating turn rule, repainting path cells by advancing through the template cycle.
    - When a path particle hits background, it stamps that impact cell with the terminal color. If the scheduled alternating turn exists, it turns in place; otherwise that stop point emits a new straight bridge particle that continues through background and, on hitting path again, restarts path painting from the landing cell.
    - Old paint does not immediately block motion: a later particle may reuse and overwrite an earlier lineage's trail. But if two active particles target the same place on the same tick, or one live particle steps into the other live lineage's already-painted trail, the active set stops at that meeting.

    Key insight:
    - The puzzle is no longer independent path tracing. The decisive cue is the combination of three visible effects: bridge-style chain reactions at dead ends, later-overwrites-earlier crossings on reused trails, and a final shared-corridor truncation where the two live particles meet and freeze together.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    out, _stats = _simulate_combo(scene, stop_on_meet=True)
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal band that embeds markers on col 0."""
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


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg fjord from col 0 between two rows without disconnecting path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    cur_top = random.randint(sr_min, sr_max)

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


def _carve_merge_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a shallow fjord that splits early channels but leaves a later merge."""
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

    keep_max = min(len(snapshots) - 1, max(4, W // 2))
    if keep_max < 4:
        return False
    keep = random.randint(4, keep_max)
    np.copyto(mask, snapshots[keep - 1])
    return True


def _apply_gap_variant(mask: np.ndarray, markers: list[int]) -> np.ndarray | None:
    """Create a bridge-friendly gap or island adapted from the D15 single variant."""
    trial = mask.copy()
    if len(markers) == 1:
        mr = int(markers[0])
        r1 = max(2, mr - random.randint(3, 5))
        r2 = min(mask.shape[0] - 3, mr + random.randint(3, 5))
        if r2 - r1 < 5:
            r2 = min(mask.shape[0] - 3, r1 + 5)
        if not _carve_fjord(trial, r1, r2):
            return None
        restore = random.randint(2, 4)
        pre_carve = mask & ~trial
        for c in range(min(restore, mask.shape[1])):
            trial[:, c] |= pre_carve[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        return trial

    if not _carve_fjord(trial, int(markers[0]), int(markers[1])):
        return None
    if random.random() < 0.2:
        restore = random.randint(2, 4)
        pre_carve = mask & ~trial
        for c in range(min(restore, mask.shape[1])):
            trial[:, c] |= pre_carve[:, c]
    _, nc = label_mask(trial, connectivity=4)
    if nc != 1:
        return None
    return trial


def _apply_combo_gaps(mask: np.ndarray, markers: list[int]) -> np.ndarray | None:
    """Blend a bridge-friendly gap with a later shared corridor when possible."""
    base = _apply_gap_variant(mask, markers)
    if base is None:
        return None
    if len(markers) != 2 or random.random() < 0.5:
        return base

    merged = base.copy()
    if not _carve_merge_fjord(merged, int(markers[0]), int(markers[1])):
        return base
    _, nc = label_mask(merged, connectivity=4)
    if nc != 1:
        return base
    for r in markers:
        if not (merged[int(r), 0] and merged[int(r) - 1, 0] and merged[int(r) + 1, 0]):
            return base
    return merged


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    marker_r1 = random.randint(H // 3, H // 2)
    marker_r2 = random.randint(max(marker_r1 + 6, 2 * H // 3), H - 4)
    marker_rows = [marker_r1, marker_r2]

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, k=n_blocks)

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

    gap_mask = _apply_combo_gaps(mask, marker_rows)
    if gap_mask is None:
        return None
    _, n_gap_path = label_mask(gap_mask, connectivity=4)
    if n_gap_path != 1:
        return None
    if not gap_mask[0, :].any() or not gap_mask[:, W - 1].any():
        return None
    if not gap_mask[H - 1, :].any() or not gap_mask[:, 0].any():
        return None
    for r in marker_rows:
        if not (gap_mask[r, 0] and gap_mask[r - 1, 0] and gap_mask[r + 1, 0]):
            return None
    mask = gap_mask

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
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in marker_rows:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if int(g[0, 0]) != int(bg):
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        scene = _extract_scene(g)
        if len(scene["markers"]) != 2:
            return None
        metrics = _behavior_metrics(g)
    except Exception:
        return None

    variant = metrics["variant_stats"]
    baseline = metrics["baseline_stats"]

    if variant["meeting_count"] < 1:
        return None
    if variant["spawn_count"] < 1 or variant["reentry_count"] < 1:
        return None
    if variant["cross_lineage_path_overwrites"] < 1:
        return None
    if variant["max_chain_segments"] < 3:
        return None
    if min(variant["path_steps"]) < 8:
        return None
    if sum(variant["turns"]) < 2:
        return None
    if variant["stop_reasons"] != ["met_particle", "met_particle"]:
        return None
    if not metrics["diff_from_no_meet"]:
        return None
    if baseline["total_writes"] <= variant["total_writes"]:
        return None

    return g


def _fill_cache(target: int = 3, max_attempts: int = 6000) -> None:
    if len(_CACHE) >= target:
        return

    preferred_by_role: dict[str, np.ndarray] = {}
    extras: list[np.ndarray] = []
    attempts = 0
    while attempts < max_attempts:
        if len(preferred_by_role) + len(extras) >= target and {"lineage0_chain", "lineage1_chain"} <= set(preferred_by_role):
            break
        attempts += 1
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            try:
                chain_segments = list(_behavior_metrics(g)["variant_stats"]["chain_segments"])
            except Exception:
                continue
            key = ""
            if len(chain_segments) >= 2:
                if int(chain_segments[0]) > int(chain_segments[1]):
                    key = "lineage0_chain"
                elif int(chain_segments[1]) > int(chain_segments[0]):
                    key = "lineage1_chain"
            if key and key not in preferred_by_role:
                preferred_by_role[key] = np.array(g, copy=True)
            elif len(extras) < target:
                extras.append(np.array(g, copy=True))
    ordered: list[np.ndarray] = []
    for key in ("lineage0_chain", "lineage1_chain"):
        if key in preferred_by_role:
            ordered.append(preferred_by_role[key])
    for g in extras:
        if len(ordered) >= target:
            break
        ordered.append(g)
    if not ordered:
        raise RuntimeError(f"generate_input: failed after {max_attempts} attempts")
    _CACHE.extend(ordered[: max(0, target - len(_CACHE))])


def generate_input() -> np.ndarray:
    _fill_cache()
    choice = random.choice(_CACHE)
    return np.array(choice, copy=True)
