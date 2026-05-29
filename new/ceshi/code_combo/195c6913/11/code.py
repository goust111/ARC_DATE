# Combo variant of 195c6913 (index 11)
# combo_spec: D2/3+D9/3
# Name: 碰撞选较长路径+单标记多轮发射

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The visible vocabulary stays close to the base task: a top-left row of 2x2
#    monochrome template blocks defines the cycle, one isolated marker on col 0
#    launches the particle, and a distinct 2x2 block provides the collision dot
#    color. The combo changes two interacting stages at once:
#    - At each collision inside a launch, the particle compares the left/right
#      90-degree options by immediate straight-run length on still-unused path
#      cells, then greedily takes the longer one (left breaks ties).
#    - After a launch stops, the same source relaunches from that stop cell,
#      searching directions in cyclic preference order R,D,L,U and skipping
#      blocked directions until some fresh path cell exists.
#    Small trace on the generated corridor tree:
#      round 1 launches Right, reaches a collision with Up-run=2 and Down-run=5,
#      so it goes Down; the next collision has Left-run=3 and Right-run=20, so
#      it goes Right and stops on the grid edge. round 2 restarts from that edge
#      tip, first finds a Down launch, later takes a longer Down branch over a
#      short Up spur, then stops on the bottom edge. round 3 relaunches again
#      and consumes the remaining large branch. The output therefore shows one
#      visible marker producing several separated painted runs plus several
#      end-color collision dots.
# b) Complexity calibration:
#    The path is not a broad band anymore; it is a single connected thin tree
#    with three launch rounds baked into its geometry. This keeps the combo
#    legible: humans can see one marker, several restart tips on the boundary,
#    and local short-vs-long branch choices. The generator rejects samples with
#    too few rounds or too few strict greedy decisions so the combo does not
#    collapse into a plain single walk or a plain multi-round path with only
#    forced turns.
# c) Hidden assumptions reviewed against the original main():
#    - Original code iterated over 1-2 markers on a static path. Here the path
#      must become consumable state (`avail`) and there must be exactly one
#      visible marker.
#    - Original alternating-turn parity is no longer the control state. Greedy
#      comparison depends on current unused geometry, so the visited-state key
#      should not depend on hit parity.
#    - A crucial interaction appears after combining D2 and D9: if a greedy
#      launch stops at a normal collision, then both side turns were blocked, so
#      no fresh adjacent path remains for relaunch. Therefore the generator must
#      make launches stop mainly at grid edges, where out-of-bounds ends the
#      round before side branches are tested. Those boundary tips are what allow
#      later relaunches.
#    - Because later rounds depend on the path cells consumed by earlier rounds,
#      the generator must oracle-check the full combo behavior, not just the
#      local collision rule.
# d) Placement legality:
#    The generator keeps all objects disjoint and in-bounds: template blocks stay
#    in the top-left background zone, the end block is a 2x2 background block in
#    the bottom-right region with a one-cell path buffer, and the marker still
#    overwrites a col-0 path cell whose vertical neighbors remain path. A 3x2
#    start pad prevents the marker overwrite from disconnecting the path.
# e) Visual observability:
#    The combo is visible from examples because both ingredients leave evidence.
#    Greedy turns produce asymmetric short-vs-long choices at collisions, while
#    relaunchs produce multiple spatially separated painted rounds even though
#    the input shows only one marker. The generator requires both a nontrivial
#    difference from the alternating multi-round ablation and a nontrivial
#    difference from the greedy single-round ablation.
# f) Visual cleanliness:
#    The corridor tree is intentionally sparse. Unused short spurs get isolated
#    once the trunk cells are consumed, so the output does not become an
#    overpainted knot. Template, marker, end block, and the three launch rounds
#    remain easy to separate visually.
# g) Color inferability:
#    Colors keep their original structural meaning. The left-to-right template
#    blocks define the repeating cycle, the single marker uses the first cycle
#    color, and the distinct 2x2 block supplies the collision/end color. No
#    behavior depends on numeric color ids.
# New solver-side assumptions and generator guarantees:
#    - Exactly one isolated marker is present.
#    - Relaunchs are enabled by boundary stop cells; accepted samples contain at
#      least two edge-stop rounds.
#    - Accepted samples contain several strict left-vs-right length comparisons,
#      so the greedy rule is visually necessary rather than incidental.


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    H, _ = g.shape
    for r, c, _ in markers:
        assert c == 0, f"marker must be on col 0, got ({r},{c})"
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor must be path_color {path_color}"
            )


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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

    pat_set = set(map(int, pattern))
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
            for dr, dc in DIR_ORDER:
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


def _straight_run_length(avail: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    H, W = avail.shape
    length = 0
    nr = int(r + dr)
    nc = int(c + dc)
    while 0 <= nr < H and 0 <= nc < W and bool(avail[nr, nc]):
        length += 1
        nr += int(dr)
        nc += int(dc)
    return int(length)


def _pick_greedy_turn(
    avail: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[tuple[int, int] | None, dict[str, int | bool]]:
    H, W = avail.shape
    left = (-int(dc), int(dr))
    right = (int(dc), -int(dr))
    options: list[tuple[int, int, int, int]] = []
    for side_rank, (tdr, tdc) in enumerate([left, right]):
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and bool(avail[rr, cc]):
            run = _straight_run_length(avail, r, c, tdr, tdc)
            options.append((int(run), int(-side_rank), int(tdr), int(tdc)))

    info: dict[str, int | bool] = {
        "both_open": len(options) == 2,
        "strict_preference": False,
        "tie": False,
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
        info["tie"] = int(chosen_run) == int(other_run)
    info["chosen_run"] = int(chosen_run)
    return (int(tdr), int(tdc)), info


def _pick_alternating_turn(
    avail: np.ndarray, r: int, c: int, dr: int, dc: int, hits: int
) -> tuple[tuple[int, int] | None, dict[str, int | bool]]:
    H, W = avail.shape
    left = (-int(dc), int(dr))
    right = (int(dc), -int(dr))
    tdr, tdc = left if (int(hits) % 2 == 0) else right
    rr = int(r + tdr)
    cc = int(c + tdc)
    info: dict[str, int | bool] = {
        "both_open": False,
        "strict_preference": False,
        "tie": False,
        "chosen_run": 0,
        "other_run": 0,
    }
    if 0 <= rr < H and 0 <= cc < W and bool(avail[rr, cc]):
        info["chosen_run"] = _straight_run_length(avail, r, c, tdr, tdc)
        return (int(tdr), int(tdc)), info
    return None, info


def _simulate_single_source(
    scene: dict[str, object],
    turn_mode: str,
    relaunch: bool,
) -> tuple[np.ndarray, dict[str, int], list[set[tuple[int, int]]]]:
    g = np.array(scene["grid"], dtype=int)
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    pat_cells = list(scene["pat_cells"])
    end_color = scene["end_color"]
    end_pos = scene["end_pos"]
    markers = list(scene["markers"])

    assert len(markers) == 1, f"variant requires exactly one marker, got {len(markers)}"

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    if not pattern:
        return out, {
            "rounds": 0,
            "nonempty_rounds": 0,
            "used_cells": 0,
            "turns": 0,
            "both_open_collisions": 0,
            "strict_greedy_decisions": 0,
            "tie_collisions": 0,
            "forced_turns": 0,
            "edge_stops": 0,
            "blocked_stops": 0,
        }, []

    avail = g == path_color
    sr, sc, sv = markers[0]
    current = (int(sr), int(sc))
    base_idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    next_pref = 0

    per_round_cells: list[set[tuple[int, int]]] = []
    stats = {
        "rounds": 0,
        "nonempty_rounds": 0,
        "used_cells": 0,
        "turns": 0,
        "both_open_collisions": 0,
        "strict_greedy_decisions": 0,
        "tie_collisions": 0,
        "forced_turns": 0,
        "edge_stops": 0,
        "blocked_stops": 0,
    }

    for _ in range(H * W):
        launch = None
        for off in range(4):
            k = int((next_pref + off) % 4)
            dr, dc = DIR_ORDER[k]
            nr = int(current[0] + dr)
            nc = int(current[1] + dc)
            if 0 <= nr < H and 0 <= nc < W and bool(avail[nr, nc]):
                launch = (k, int(dr), int(dc))
                break
        if launch is None:
            break

        launch_k, dr, dc = launch
        next_pref = int((launch_k + 1) % 4)
        r, c = current
        idx = int(base_idx)
        hits = 0
        my_cells: set[tuple[int, int]] = set()

        for _step in range(H * W * 4):
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                current = (int(r), int(c))
                stats["edge_stops"] += 1
                break

            if bool(avail[nr, nc]):
                avail[nr, nc] = False
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                stats["used_cells"] += 1
                r, c = int(nr), int(nc)
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            if turn_mode == "greedy":
                turn, info = _pick_greedy_turn(avail, r, c, dr, dc)
                if bool(info["both_open"]):
                    stats["both_open_collisions"] += 1
                    if bool(info["strict_preference"]):
                        stats["strict_greedy_decisions"] += 1
                    elif bool(info["tie"]):
                        stats["tie_collisions"] += 1
                elif turn is not None:
                    stats["forced_turns"] += 1
            else:
                turn, _info = _pick_alternating_turn(avail, r, c, dr, dc, hits)

            if turn is None:
                current = (int(r), int(c))
                stats["blocked_stops"] += 1
                break

            dr, dc = turn
            stats["turns"] += 1
            if turn_mode == "alternate":
                hits += 1
        else:
            current = (int(r), int(c))

        per_round_cells.append(my_cells)
        stats["rounds"] += 1
        if my_cells:
            stats["nonempty_rounds"] += 1
        if not relaunch:
            break

    return out, stats, per_round_cells


def _combo_metrics(grid: np.ndarray) -> dict[str, int | np.ndarray]:
    scene = _extract_scene(grid)
    combo_out, combo_stats, _ = _simulate_single_source(scene, turn_mode="greedy", relaunch=True)
    alt_out, alt_stats, _ = _simulate_single_source(scene, turn_mode="alternate", relaunch=True)
    single_out, single_stats, _ = _simulate_single_source(scene, turn_mode="greedy", relaunch=False)
    return {
        "n_markers": len(scene["markers"]),
        "combo_out": combo_out,
        "alt_out": alt_out,
        "single_out": single_out,
        "diff_vs_alt": int(np.count_nonzero(combo_out != alt_out)),
        "diff_vs_single": int(np.count_nonzero(combo_out != single_out)),
        "alt_rounds": int(alt_stats["rounds"]),
        "single_rounds": int(single_stats["rounds"]),
        **combo_stats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Consume one path with greedy collision choices, relaunching the same marker after each stop.

    Input structure:
    - The top-left area contains a row of 2x2 solid blocks that define the repeating paint cycle.
    - Exactly one isolated cycle-colored marker sits on col 0 inside the path.
    - A separate 2x2 solid block of another color provides the collision/end color.

    Rule:
    - Erase the template blocks and the 2x2 end block in the output, then launch from the lone marker.
    - During a launch, move only on still-unused path cells and paint them by advancing through the template cycle.
    - When forward movement is blocked, mark the collision cell with the end color if it is background, then compare the left and right turn options by how many unused path cells they can continue straight through; take the longer one, breaking ties to the left.
    - When a launch stops, restart from that stop cell using cyclic launch preference R,D,L,U and skipping blocked directions; stop only when no adjacent unused path remains.

    Key insight:
    - The combo couples two stateful effects: greedy branch choice decides which local corridor survives each collision, and consumed path cells make those choices irreversible across later relaunchs from the same source.
    """
    scene = _extract_scene(grid)
    assert len(scene["markers"]) == 1, "combo variant expects exactly one visible marker"
    out, _stats, _rounds = _simulate_single_source(scene, turn_mode="greedy", relaunch=True)
    return out


def _add_segment(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        for c in range(lo, hi + 1):
            cells.add((int(r1), int(c)))
    else:
        assert c1 == c2
        lo, hi = sorted((r1, r2))
        for r in range(lo, hi + 1):
            cells.add((int(r), int(c1)))


def _build_path_cells(H: int, W: int, template_row: int) -> tuple[set[tuple[int, int]], int] | None:
    for _ in range(80):
        try:
            cA = random.randint(7, max(7, W // 4 + 1))
            cE = random.randint(4, cA - 3)
            cB = random.randint(cE + 1, cA - 2)
            cFr = random.randint(cE + 2, min(cA - 1, cE + 3))
            cD = random.randint(max(cA + 5, W // 3), W - 6)

            rs = random.randint(max(template_row + 8, H // 2 - 3), min(H // 2 + 1, H - 15))
            rAu = random.randint(template_row + 4, rs - 2)
            rF = random.randint(rs + 2, rs + 4)
            rB = random.randint(rF + 1, min(rF + 2, H - 10))
            rDu = random.randint(rB + 2, min(rB + 3, H - 7))
            rC = random.randint(rDu + 2, min(rDu + 3, H - 6))
        except ValueError:
            continue

        cells: set[tuple[int, int]] = set()
        for rr in range(rs - 1, rs + 2):
            for cc in range(0, 2):
                cells.add((int(rr), int(cc)))

        segments = [
            ((rs, 1), (rs, cA)),          # launch corridor
            ((rs, cA), (rAu, cA)),        # short up spur at first collision
            ((rs, cA), (rB, cA)),         # long down branch at first collision
            ((rB, cB), (rB, cA)),         # short left spur at second collision
            ((rB, cA), (rB, W - 1)),      # long right branch to first edge stop
            ((rB, W - 1), (rC, W - 1)),   # round-2 launch down the right edge
            ((rC, cD), (rC, W - 1)),      # forced left turn from the right edge
            ((rDu, cD), (rC, cD)),        # short up spur at round-2 greedy node
            ((rC, cD), (H - 1, cD)),      # long down branch to second edge stop
            ((H - 1, cE), (H - 1, cD)),   # round-3 launch left along bottom edge
            ((rF, cE), (H - 1, cE)),      # forced up turn from the bottom edge
            ((rF, 0), (rF, cE)),          # long left branch to third edge stop
            ((rF, cE), (rF, cFr)),        # short right spur at round-3 greedy node
        ]
        for a, b in segments:
            _add_segment(cells, a, b)

        if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
            continue
        if any(r <= template_row + 1 and c <= 11 for r, c in cells):
            continue
        if (0, 0) in cells:
            continue
        return cells, int(rs)
    return None


def _place_end_block(g: np.ndarray, bg: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(24, 30)
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    path_color = int(palette[1])
    end_color = int(palette[2])
    n_blocks = random.randint(2, 4)
    cycle_colors = [int(col) for col in random.sample(palette[3:], k=n_blocks)]

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None

    built = _build_path_cells(H, W, template_row)
    if built is None:
        return None
    cells, sr = built

    g = np.full((H, W), bg, dtype=int)
    for r, c in cells:
        g[int(r), int(c)] = int(path_color)

    for i, bc in enumerate(block_cols):
        if (g[template_row : template_row + 2, bc : bc + 2] == path_color).any():
            return None
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_pos = _place_end_block(g, bg, path_color)
    if end_pos is None:
        return None
    er, ec = end_pos
    g[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(end_color)

    if not (
        int(g[sr - 1, 0]) == int(path_color)
        and int(g[sr + 1, 0]) == int(path_color)
        and int(g[sr, 1]) == int(path_color)
    ):
        return None
    g[int(sr), 0] = int(cycle_colors[0])

    _assert_path_single_component(g, int(path_color))
    if int(g[0, 0]) != int(bg):
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        metrics = _combo_metrics(g)
    except Exception:
        return None

    if int(metrics["n_markers"]) != 1:
        return None
    if int(metrics["rounds"]) < 3 or int(metrics["nonempty_rounds"]) < 3:
        return None
    if int(metrics["used_cells"]) < 45:
        return None
    if int(metrics["edge_stops"]) < 2:
        return None
    if int(metrics["strict_greedy_decisions"]) < 3:
        return None
    if int(metrics["both_open_collisions"]) < 3:
        return None
    if int(metrics["diff_vs_alt"]) < 8:
        return None
    if int(metrics["diff_vs_single"]) < 12:
        return None
    if np.array_equal(metrics["combo_out"], g):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
