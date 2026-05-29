# Combo variant of 195c6913 (index 3)
# combo_spec: D2/1+D6/1+D14/0
# Name: 碰撞色决定转向+碰撞次序涂角+无终止块

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import label_mask

"""
design_analysis:
  a_specific_behavior:
    input_objects:
      - top-left still has a row of 2x2 monochrome template blocks; left-to-right order is the cycle
      - col 0 still has 1-2 isolated start pixels in cycle[0], vertically embedded in the path band
      - the old bottom-right 2x2 end block is removed entirely
      - new non-path "turn cue" cells are placed exactly at forward collision cells; they reuse template colors
        and are given one same-colored orthogonal neighbor so they are visible blockers rather than fake markers
    behavior_trace:
      - suppose cycle = [A, B, C] with A designated as the only left-turn color and {B, C} meaning right-turn
      - a marker starts at color A; path repaint still goes B, C, A, ...
      - if the particle faces a blocker cell colored A, it paints that collision corner with cycle[0] on hit 1
        and attempts a CCW turn
      - if later it faces a blocker cell colored C, it paints that collision corner with cycle[1] on hit 2
        and attempts a CW turn
      - under the original task the same two bends would have been controlled by alternating parity and both
        corner pixels would have shared one end_color from a separate 2x2 block
    combined_difference:
      - D2 changes the cause of turning from collision parity to visible blocker color
      - D6 changes the corner output from one constant color to hit-order colors cycle[k]
      - D14 removes the end block, so every collision clue now comes from template colors only
  b_complexity_calibration:
    - keep the path geometry, marker placement, and fjord/island separator from the original task so the
      puzzle still reads as "follow a path with bends", not as a new object-inventory task
    - use 3-4 distinct cycle colors so one color can repeatedly mean left while multiple "other" colors
      repeatedly mean right; this makes the mapping visible without exploding the vocabulary
    - reject samples unless both left-turn and right-turn cue colors actually control successful turns
    - reject samples unless hit-order corner coloring visibly differs from the simpler "current phase" coloring
  c_implicit_assumption_review:
    - the original solver assumed an end block existed and could be erased; that search must be removed
    - the original visited-state only needed turn parity; now future visible output depends on hit_count mod
      len(cycle), so the loop state must carry that residue instead
    - marker detection treats isolated template-colored cells adjacent to path as starts; collision cues
      therefore cannot be isolated single pixels of a template color, so the generator pairs each cue cell
      with a same-colored support cell
    - collision painting can no longer be limited to bg-only cells conceptually, because the turn-determining
      blocker color already occupies the collision cell and must be overwritten in the output
    - color-driven turning introduces a new assumption: the designated left-turn color is cycle[0] and every
      other template color means right; the generator fixes the cycle order so this semantic is recoverable
  d_placement_legality:
    - template blocks remain in the reserved top-left bg region and never overlap the band
    - markers still overwrite path cells in col 0 only after the band is built
    - each blocker support cell must be in-bounds, non-path, currently plain bg, and distinct from every
      collision cell, marker, and previously claimed support cell
    - the generator reparses the finished grid and rejects it if blockers accidentally create extra markers
      or corrupt template detection
  e_visual_observability:
    - a human can compare several examples and see that turns happen at colored non-path blockers rather than
      by odd/even bend count, because the same blocker color causes the same turn orientation in multiple places
    - corner marks use cycle[0], cycle[1], cycle[2], ... regardless of the blocker's own color, so input
      blocker color and output corner color are decoupled in a structured way that is still template-driven
    - with no end block, every relevant color already appears in the template, avoiding hidden palettes
  f_visual_cleanliness:
    - blockers are sparse 2-cell mini-objects on bg cells near bends, so they remain readable and do not
      visually merge with the thick path band
    - trajectories stay disjoint across markers, as in the original task
    - generator rejects repeated corner-cell hits so the collision-order code is not muddied by overwrites
  g_color_inference:
    - the only left-turn color is cycle[0]; all other template colors are right-turn cues
    - hit-order corner colors also come from the template order, so the template simultaneously explains
      path repaint, turn cues, and collision-order marks
    - because the rule is positional inside the template rather than based on raw color ids, it remains
      human-visible across random palette permutations
  solver_generator_contract:
    - main() assumes the path is a single 4-connected component, markers are embedded in col 0 path, the
      template cycle is non-empty, and cycle[0] is the left-turn cue color
    - generate_input() guarantees these conditions, guarantees blocker cues are non-isolated, and filters out
      samples where the combo degenerates to the alternating-turn rule or to current-phase corner coloring
"""


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker is vertically embedded in the path."""
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
    return pattern, pat_cells


def _find_markers(
    g: np.ndarray, pat_set: set[int], path_color: int
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), v))
    return markers


def _parse_scene(
    g: np.ndarray,
) -> tuple[int, int, list[int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            (cnt, int(col))
            for col, cnt in counts.items()
            if int(col) != int(template_bg)
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _find_pattern(g, template_bg, path_color)
    pat_set = set(pattern)
    markers = _find_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)
    return template_bg, path_color, pattern, pat_cells, markers


def _initial_direction(
    g: np.ndarray, path_color: int, sr: int, sc: int
) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
            return int(dr), int(dc)
    return None


def _run_simulation(
    grid: np.ndarray,
    *,
    turn_mode: str,
    corner_mode: str,
    collect_info: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg, path_color, pattern, pat_cells, markers = _parse_scene(g)
    assert pattern, "template cycle must be non-empty"

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = template_bg

    left_turn_color = int(pattern[0])
    per_marker_cells: list[set[tuple[int, int]]] = []
    collision_events: list[dict[str, object]] = []
    turns_per_marker: list[int] = []

    for marker_index, (sr, sc, sv) in enumerate(markers):
        init = _initial_direction(g, path_color, int(sr), int(sc))
        if init is None:
            per_marker_cells.append(set())
            turns_per_marker.append(0)
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (
                int(r),
                int(c),
                int(dr),
                int(dc),
                int(idx),
                int(hits % len(pattern)),
            )
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

            if corner_mode == "hit_order":
                corner_color = int(pattern[hits % len(pattern)])
            elif corner_mode == "current_phase":
                corner_color = int(pattern[idx])
            else:
                raise ValueError(f"unknown corner_mode: {corner_mode}")
            out[nr, nc] = corner_color

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            left_valid = False
            right_valid = False
            lrr = int(r + left[0])
            lcc = int(c + left[1])
            rrr = int(r + right[0])
            rcc = int(c + right[1])
            if 0 <= lrr < H and 0 <= lcc < W and int(g[lrr, lcc]) == path_color:
                left_valid = True
            if 0 <= rrr < H and 0 <= rcc < W and int(g[rrr, rcc]) == path_color:
                right_valid = True

            if turn_mode == "combo":
                turn_is_left = int(g[nr, nc]) == left_turn_color
            elif turn_mode == "alternating":
                turn_is_left = (int(hits) % 2 == 0)
            elif turn_mode == "always_left":
                turn_is_left = True
            elif turn_mode == "always_right":
                turn_is_left = False
            else:
                raise ValueError(f"unknown turn_mode: {turn_mode}")

            tdr, tdc = left if turn_is_left else right
            tr = int(r + tdr)
            tc = int(c + tdc)
            turn_valid = 0 <= tr < H and 0 <= tc < W and int(g[tr, tc]) == path_color

            collision_events.append(
                {
                    "marker": int(marker_index),
                    "ordinal": int(hits + 1),
                    "cell": (int(nr), int(nc)),
                    "blocker_color": int(g[nr, nc]),
                    "corner_color": int(corner_color),
                    "turn": "L" if turn_is_left else "R",
                    "turn_valid": bool(turn_valid),
                    "left_valid": bool(left_valid),
                    "right_valid": bool(right_valid),
                }
            )

            if turn_valid:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)
        turns_per_marker.append(int(hits))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    info = {
        "pattern": pattern,
        "markers": markers,
        "collision_events": collision_events,
        "turn_count": int(sum(turns_per_marker)),
        "turns_per_marker": turns_per_marker,
        "unique_corner_cells": len({event["cell"] for event in collision_events}),
        "distinct_corner_colors": len({event["corner_color"] for event in collision_events}),
        "left_turn_successes": sum(
            1 for event in collision_events if event["turn"] == "L" and event["turn_valid"]
        ),
        "right_turn_successes": sum(
            1 for event in collision_events if event["turn"] == "R" and event["turn_valid"]
        ),
    }
    return (out, info) if collect_info else (out, {})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate path-following particles whose bend direction is read from visible collision colors.

    Input structure:
    - The top-left area contains a row of monochrome 2x2 template blocks whose left-to-right order defines the cycle.
    - One or two isolated pixels on col 0 use the first cycle color and are embedded inside the path band.
    - There is no separate terminal 2x2 block; instead, some non-path cells near bends are precolored with template colors as turn cues.

    Rule:
    - Each marker launches a particle in the first adjacent path direction using R,D,L,U priority and repaints path cells by advancing through the cycle.
    - When the next step is a non-path cell, that cell's input color decides the turn: cycle[0] means turn left (CCW), every other color means turn right (CW).
    - The collision cell itself is overwritten with the color for that collision number: 1st hit uses cycle[0], 2nd hit uses cycle[1], and so on with wraparound.
    - If the direction prescribed by that collision color is not a path cell, the particle stops.

    Key insight:
    - The template now drives three linked roles at once: traversal repaint order, which collision colors trigger left versus right bends, and the visible hit-order coding on corner cells.
    """
    out, _ = _run_simulation(
        grid, turn_mode="combo", corner_mode="hit_order", collect_info=False
    )
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
            c_right[r] = int(
                round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2)
            )
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
    """Carve a bg fjord from col 0 without disconnecting the band."""
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


def _select_planned_turn(
    left_valid: bool,
    right_valid: bool,
    planned_left: int,
    planned_right: int,
    global_collision_index: int,
) -> str | None:
    if left_valid and right_valid:
        if planned_left == 0:
            return "L"
        if planned_right == 0:
            return "R"
        return "L" if (global_collision_index % 2 == 0) else "R"
    if left_valid:
        return "L"
    if right_valid:
        return "R"
    return None


def _plan_blockers(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, int]] | None:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg, path_color, pattern, _pat_cells, markers = _parse_scene(g)
    if len(pattern) < 2:
        return None

    planned: list[dict[str, object]] = []
    planned_left = 0
    planned_right = 0
    global_collision_index = 0

    for marker_index, (sr, sc, sv) in enumerate(markers):
        init = _initial_direction(g, path_color, int(sr), int(sc))
        if init is None:
            return None

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx))
            if state in visited:
                return None
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == path_color:
                idx = int((idx + 1) % len(pattern))
                r, c = nr, nc
                continue

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            lrr = int(r + left[0])
            lcc = int(c + left[1])
            rrr = int(r + right[0])
            rcc = int(c + right[1])
            left_valid = 0 <= lrr < H and 0 <= lcc < W and int(g[lrr, lcc]) == path_color
            right_valid = 0 <= rrr < H and 0 <= rcc < W and int(g[rrr, rcc]) == path_color

            turn = _select_planned_turn(
                left_valid, right_valid, planned_left, planned_right, global_collision_index
            )
            planned.append(
                {
                    "marker": int(marker_index),
                    "ordinal": int(hits + 1),
                    "cell": (int(nr), int(nc)),
                    "from": (int(r), int(c)),
                    "move": (int(dr), int(dc)),
                    "turn": turn,
                    "left_valid": bool(left_valid),
                    "right_valid": bool(right_valid),
                }
            )

            if turn is None:
                break

            if turn == "L":
                dr, dc = int(left[0]), int(left[1])
                planned_left += 1
            else:
                dr, dc = int(right[0]), int(right[1])
                planned_right += 1
            hits += 1
            global_collision_index += 1

    summary = {
        "planned_left": int(planned_left),
        "planned_right": int(planned_right),
        "total_collisions": int(len(planned)),
    }
    return planned, summary


def _find_support_cell(
    g: np.ndarray,
    *,
    current: tuple[int, int],
    collision: tuple[int, int],
    move: tuple[int, int],
    template_bg: int,
    path_color: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    H, W = g.shape
    r, c = collision
    dr, dc = move
    candidates = [
        (int(r + dr), int(c + dc)),
        (int(r - dc), int(c + dr)),
        (int(r + dc), int(c - dr)),
    ]
    for rr, cc in candidates:
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        if (int(rr), int(cc)) == current:
            continue
        if (int(rr), int(cc)) in blocked:
            continue
        if int(g[rr, cc]) == int(path_color):
            continue
        if int(g[rr, cc]) != int(template_bg):
            continue
        return int(rr), int(cc)
    return None


def _apply_blockers(grid: np.ndarray) -> np.ndarray | None:
    g = np.array(grid, dtype=int)
    template_bg, path_color, pattern, _pat_cells, markers = _parse_scene(g)
    planned_result = _plan_blockers(g)
    if planned_result is None:
        return None
    planned, summary = planned_result
    if summary["planned_left"] < 1 or summary["planned_right"] < 1:
        return None

    blocked: set[tuple[int, int]] = {(int(r), int(c)) for r, c, _ in markers}
    collision_cells = {tuple(map(int, event["cell"])) for event in planned if event["turn"] is not None}
    blocked |= collision_cells

    right_colors = pattern[1:]
    right_idx = 0
    for event in planned:
        turn = event["turn"]
        if turn is None:
            continue
        cr, cc = map(int, event["cell"])
        current = tuple(map(int, event["from"]))
        move = tuple(map(int, event["move"]))
        if int(g[cr, cc]) != int(template_bg):
            return None
        support = _find_support_cell(
            g,
            current=current,
            collision=(cr, cc),
            move=move,
            template_bg=template_bg,
            path_color=path_color,
            blocked=blocked,
        )
        if support is None:
            return None

        if turn == "L":
            blocker_color = int(pattern[0])
        else:
            blocker_color = int(right_colors[right_idx % len(right_colors)])
            right_idx += 1

        g[cr, cc] = blocker_color
        g[support[0], support[1]] = blocker_color
        blocked.add(tuple(map(int, support)))

    return g


def _combo_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    combo_out, combo_info = _run_simulation(
        grid, turn_mode="combo", corner_mode="hit_order", collect_info=True
    )
    alt_out, _ = _run_simulation(
        grid, turn_mode="alternating", corner_mode="hit_order", collect_info=True
    )
    phase_out, _ = _run_simulation(
        grid, turn_mode="combo", corner_mode="current_phase", collect_info=True
    )
    right_out, _ = _run_simulation(
        grid, turn_mode="always_right", corner_mode="hit_order", collect_info=True
    )
    events = combo_info["collision_events"]
    successful_events = [event for event in events if bool(event["turn_valid"])]
    blocker_colors = {int(event["blocker_color"]) for event in successful_events}
    return {
        "collision_count": len(events),
        "successful_turns": len(successful_events),
        "left_turn_successes": int(combo_info["left_turn_successes"]),
        "right_turn_successes": int(combo_info["right_turn_successes"]),
        "unique_corner_cells": int(combo_info["unique_corner_cells"]),
        "distinct_corner_colors": int(combo_info["distinct_corner_colors"]),
        "distinct_success_blocker_colors": len(blocker_colors),
        "diff_from_alternating": not np.array_equal(combo_out, alt_out),
        "diff_from_current_phase": not np.array_equal(combo_out, phase_out),
        "diff_from_all_right": not np.array_equal(combo_out, right_out),
    }


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    n_blocks = random.randint(3, 4)
    cycle_colors = random.sample(palette[2:], k=n_blocks)

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
                rr = template_row + dr
                cc = bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color
    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]
    for r in markers:
        g[r, 0] = cycle_colors[0]

    if g[0, 0] != bg:
        return None
    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted(((c, col) for col, c in cnt.items() if col != bg), reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    g = _apply_blockers(g)
    if g is None:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted(((c, col) for col, c in cnt.items() if col != bg), reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        template_bg, parsed_path, pattern, _pat_cells, parsed_markers = _parse_scene(g)
    except Exception:
        return None
    if template_bg != bg or parsed_path != path_color:
        return None
    if pattern != cycle_colors:
        return None
    if sorted((int(r), int(c), int(v)) for r, c, v in parsed_markers) != sorted(
        (int(r), 0, int(cycle_colors[0])) for r in markers
    ):
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    metrics = _combo_metrics(g)
    if int(metrics["collision_count"]) < 3:
        return None
    if int(metrics["successful_turns"]) < 2:
        return None
    if int(metrics["left_turn_successes"]) < 1:
        return None
    if int(metrics["right_turn_successes"]) < 1:
        return None
    if int(metrics["unique_corner_cells"]) != int(metrics["collision_count"]):
        return None
    if int(metrics["distinct_corner_colors"]) < 2:
        return None
    if int(metrics["distinct_success_blocker_colors"]) < 2:
        return None
    if not bool(metrics["diff_from_alternating"]):
        return None
    if not bool(metrics["diff_from_current_phase"]):
        return None
    if not bool(metrics["diff_from_all_right"]):
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
