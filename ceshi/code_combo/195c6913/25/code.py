# Combo variant of 195c6913 (index 25)
# combo_spec: D1/2+D5/0+D10/1
# Name: 走廊跟随+到达目标停止+网格状path

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import label_mask

# Design analysis for the combo variant
# a) Concrete behavior:
#    - The thick anti-diagonal band is replaced by a thin street network: several horizontal
#      roads intersect several vertical roads, producing visible crossroads. Markers still sit
#      on col 0 and still use the first template color.
#    - Motion is no longer "ray until collision". The particle advances one corridor cell at a
#      time. If the corridor only offers one forward continuation (including a forced bend), it
#      follows that continuation automatically. At a junction with multiple non-backtracking
#      choices, it prefers alternating side branches: first eligible junction takes the left
#      branch, next eligible junction takes the right branch, then left again, etc.
#    - A single finish-colored pixel is embedded on the route for each marker. Stepping onto
#      that pixel ends the trace immediately, so the visible stop reason is arrival at a target,
#      not failed collision turning.
#    - Small trace example:
#      old rule: start -> long straight run -> forward blocked by bg -> turn -> maybe stop.
#      new rule: start on a row road -> enter an intersection -> if both forward and "left"
#      branch exist, take left immediately inside the passable network -> later at another
#      intersection take right -> stop exactly when the trace reaches the finish-colored cell.
# b) Complexity calibration:
#    - The network stays single-color and width 1, so the new topology is readable. The only
#      added cue is the isolated finish pixel, reusing the same color as the removable 2x2
#      legend block.
#    - The grid uses 4-6 horizontal roads and 4-6 vertical roads, which is enough to create
#      several candidate turns without flooding the board with arbitrary branches.
#    - Marker count remains 1 or 2. Two-marker cases are kept only when their pre-target
#      prefixes are disjoint, preventing visually confusing overlaps.
# c) Old assumptions that must change:
#    - The original solver assumed every traversable cell had color path_color. That is false
#      now because the finish-colored target pixel is also traversable.
#    - The original "turn only when forward is blocked" logic is invalid. Here the key decision
#      can happen while forward is still open, so the state update must inspect the local
#      corridor topology on every step.
#    - Collision dots disappear from the output. end_color now means "legend + destination",
#      not "painted collision corner".
#    - A single connected path-color band is no longer the right structural description; the
#      passable region is a graph-like network. main() therefore validates the component on
#      path_color U targets U markers rather than on path_color alone.
# d) Placement legality:
#    - Template blocks stay in rows 1-3 and cols 1/4/7/10, while the corridor network starts
#      lower in the board so objects never overlap.
#    - Markers are chosen from interior horizontal roads and a col-0 vertical entry segment is
#      added so each marker keeps path-colored vertical neighbors, matching the original cue.
#    - Targets are selected from actual simulated trajectories, filtered for isolation, and for
#      two-marker instances the chosen prefixes must be pairwise disjoint.
# e) Visual observability:
#    - Humans can see the path is a road grid rather than a diagonal blob, and the trace turns
#      inside the network at crossroads before reaching the finish pixel.
#    - The finish color appears both as a removable 2x2 legend block and as the in-network stop
#      pixel, making the new termination rule legible from a few examples.
# f) Visual cleanliness:
#    - The template is separated from the road network by empty rows, and the removable legend
#      block is pushed into the lower-right background corner away from the active streets.
#    - Targets are chosen away from the dense start area and usually after at least one true
#      junction-side turn, so the output highlights the intended route rather than cluttering
#      the left edge.
# g) Inferability and solver/generator contract:
#    - No hidden color ordering is introduced. The template still explains the repaint cycle,
#      the marker color still matches the first template block, and end_color still has a single
#      visible meaning: destination.
#    - main() assumes each finish-colored isolated pixel lies on the passable network, there is
#      one such target per marker, marker cells themselves are launchable passable cells, and the
#      chosen target is reached before the trace can merge with another marker's target prefix.
#      generate_input() enforces those assumptions.


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
_NEXT_MARKER_COUNT = 2


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
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(int(v))
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                return int(v), (int(r), int(c))
    return None, None


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
            for dr, dc in DIR_ORDER:
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


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == int(path_color), (
                f"marker at ({r},{c}) needs path above on col 0"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == int(path_color), (
                f"marker at ({r},{c}) needs path below on col 0"
            )


def _assert_targets_embedded_in_path(
    g: np.ndarray, targets: list[tuple[int, int]], path_color: int
) -> None:
    H, W = g.shape
    for r, c in targets:
        same = 0
        path_neighbors = 0
        for dr, dc in DIR_ORDER:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                if int(g[rr, cc]) == int(g[r, c]):
                    same += 1
                if int(g[rr, cc]) == int(path_color):
                    path_neighbors += 1
        assert same == 0, f"target at ({r},{c}) must be isolated"
        assert path_neighbors >= 1, f"target at ({r},{c}) must touch the corridor network"


def _assert_passable_single_component(passable_mask: np.ndarray) -> None:
    _, n_path = label_mask(passable_mask, connectivity=4)
    assert n_path == 1, f"passable region must be a single 4-connected component, got {n_path}"


def _passable_mask(
    g: np.ndarray,
    path_color: int,
    target_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    mask = g == int(path_color)
    for r, c in target_cells:
        mask[int(r), int(c)] = True
    if marker_cells is not None:
        for r, c in marker_cells:
            mask[int(r), int(c)] = True
    return mask


def _choose_next_direction(
    passable: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    branch_turns: int,
) -> tuple[tuple[int, int] | None, int, bool]:
    H, W = passable.shape
    forward = (int(dr), int(dc))
    left = (-int(dc), int(dr))
    right = (int(dc), -int(dr))
    back = (-int(dr), -int(dc))
    dirs = [forward, left, right, back]
    open_dirs: dict[tuple[int, int], bool] = {}
    for ddr, ddc in dirs:
        rr = int(r + ddr)
        cc = int(c + ddc)
        open_dirs[(int(ddr), int(ddc))] = 0 <= rr < H and 0 <= cc < W and bool(passable[rr, cc])

    candidates = [d for d in (forward, left, right) if open_dirs[d]]
    if not candidates:
        return None, int(branch_turns), False
    if len(candidates) == 1:
        return candidates[0], int(branch_turns), False

    scheduled = left if (int(branch_turns) % 2 == 0) else right
    if open_dirs[scheduled]:
        took_open_side = bool(open_dirs[forward])
        return scheduled, int(branch_turns) + 1, took_open_side
    if open_dirs[forward]:
        return forward, int(branch_turns), False
    if open_dirs[left]:
        return left, int(branch_turns), False
    if open_dirs[right]:
        return right, int(branch_turns), False
    if open_dirs[back]:
        return back, int(branch_turns), False
    return None, int(branch_turns), False


def _trace_particle(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    target_cells: set[tuple[int, int]],
    marker: tuple[int, int, int],
    out: np.ndarray | None = None,
    end_color: int | None = None,
) -> dict[str, object]:
    H, W = g.shape
    sr, sc, sv = marker
    passable = _passable_mask(g, int(path_color), target_cells, {(int(sr), int(sc))})

    init = None
    for dr, dc in DIR_ORDER:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(passable[rr, cc]):
            init = (int(dr), int(dc))
            break
    if init is None or not pattern:
        return {
            "cells": [],
            "branch_counts": [],
            "open_side_turn_counts": [],
            "reached_target": False,
            "target": None,
            "open_side_turns": 0,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    branch_turns = 0
    open_side_turns = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    cells: list[tuple[int, int]] = []
    branch_counts: list[int] = []
    open_side_turn_counts: list[int] = []

    # The marker first launches into the corridor using the original R,D,L,U
    # preference. The col-0 vertical embedding is only structural support for
    # marker detection and should not count as the first junction choice.
    nr = int(r + dr)
    nc = int(c + dc)
    if not (0 <= nr < H and 0 <= nc < W and bool(passable[nr, nc])):
        return {
            "cells": [],
            "branch_counts": [],
            "open_side_turn_counts": [],
            "reached_target": False,
            "target": None,
            "open_side_turns": 0,
        }
    if (int(nr), int(nc)) in target_cells:
        if out is not None and end_color is not None:
            out[nr, nc] = int(end_color)
        return {
            "cells": [(int(nr), int(nc))],
            "branch_counts": [0],
            "open_side_turn_counts": [0],
            "reached_target": True,
            "target": (int(nr), int(nc)),
            "open_side_turns": 0,
        }
    idx = int((idx + 1) % len(pattern))
    if out is not None:
        out[nr, nc] = int(pattern[idx])
    cells.append((int(nr), int(nc)))
    branch_counts.append(0)
    open_side_turn_counts.append(0)
    r, c = nr, nc

    for _ in range(H * W * 8):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(branch_turns & 1))
        if state in visited:
            break
        visited.add(state)

        next_dir, new_branch_turns, took_open_side = _choose_next_direction(
            passable, int(r), int(c), int(dr), int(dc), int(branch_turns)
        )
        if next_dir is None:
            break
        dr, dc = int(next_dir[0]), int(next_dir[1])
        branch_turns = int(new_branch_turns)
        if took_open_side:
            open_side_turns += 1

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if (int(nr), int(nc)) in target_cells:
            if out is not None and end_color is not None:
                out[nr, nc] = int(end_color)
            cells.append((int(nr), int(nc)))
            branch_counts.append(int(branch_turns))
            open_side_turn_counts.append(int(open_side_turns))
            return {
                "cells": cells,
                "branch_counts": branch_counts,
                "open_side_turn_counts": open_side_turn_counts,
                "reached_target": True,
                "target": (int(nr), int(nc)),
                "open_side_turns": int(open_side_turns),
            }

        assert int(g[nr, nc]) == int(path_color), "non-target passable cells must be path_color"
        idx = int((idx + 1) % len(pattern))
        if out is not None:
            out[nr, nc] = int(pattern[idx])
        cells.append((int(nr), int(nc)))
        branch_counts.append(int(branch_turns))
        open_side_turn_counts.append(int(open_side_turns))
        r, c = nr, nc

    return {
        "cells": cells,
        "branch_counts": branch_counts,
        "open_side_turn_counts": open_side_turn_counts,
        "reached_target": False,
        "target": None,
        "open_side_turns": int(open_side_turns),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint corridor traces on a road-grid path until they reach explicit finish pixels.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks still encodes the repeating paint cycle.
    - Left edge: one or two isolated marker pixels in the first cycle color sit inside a col-0 corridor segment.
    - Path area: a thin network of horizontal and vertical corridors with visible intersections.
    - Finish cue: one removable 2x2 legend block of a distinct color plus isolated pixels of that same color embedded on the corridor network.

    Rule:
    - Each marker launches a particle with the original direction preference R, D, L, U.
    - The particle advances one corridor cell at a time. If the corridor forces a bend, it follows the bend automatically.
    - At a junction with multiple forward continuations, the particle prefers alternating side branches: first eligible choice takes the left branch, next eligible choice takes the right branch, then left again, and so on. If the scheduled side branch does not exist, it keeps following the available corridor.
    - Stepping onto a finish-colored isolated target pixel stops that particle immediately. The template blocks and the 2x2 legend block are erased to background in the output.

    Key insight: The route is determined by local corridor topology, not by ray collisions. The solver must read crossroads as decision points and use the alternating left/right side preference to pick a path through the grid toward the explicit target.
    """
    g = np.array(grid, dtype=int)

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )

    pattern, pat_cells = _find_pattern(g, int(template_bg), int(path_color))
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, int(template_bg), int(path_color), pat_set)
    markers = _find_markers(g, pat_set, int(path_color))
    targets = _find_targets(g, end_color, int(path_color))
    target_cells = set((int(r), int(c)) for r, c in targets)
    marker_cells = set((int(r), int(c)) for r, c, _ in markers)

    assert pattern, "template pattern must be present"
    assert end_color is not None and end_pos is not None, "finish legend block must be present"
    assert markers, "at least one marker must be present"
    assert targets and len(targets) == len(markers), "generator contract: one isolated target per marker"
    _assert_marker_embedded_in_path(g, markers, int(path_color))
    _assert_targets_embedded_in_path(g, targets, int(path_color))
    _assert_passable_single_component(
        _passable_mask(g, int(path_color), target_cells, marker_cells)
    )

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    er, ec = end_pos
    out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for marker in markers:
        trace = _trace_particle(
            g,
            pattern,
            int(path_color),
            target_cells,
            marker,
            out=out,
            end_color=int(end_color),
        )
        assert bool(trace["reached_target"]), "each particle must reach a target"
        my_cells = set(trace["cells"])  # type: ignore[arg-type]
        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    if count <= 0:
        return []
    for _ in range(80):
        random.shuffle(vals)
        picked: list[int] = []
        for v in vals:
            if all(abs(int(v) - int(p)) >= int(min_gap) for p in picked):
                picked.append(int(v))
                if len(picked) == count:
                    return sorted(picked)
    return None


def _build_grid_network(
    H: int,
    W: int,
    row_lines: list[int],
    col_lines: list[int],
    marker_rows: list[int],
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    row_top = int(min(row_lines))
    row_bottom = int(max(row_lines))
    col_right = int(max(col_lines) + random.randint(2, 4))
    col_right = min(col_right, W - 5)

    for r in row_lines:
        mask[int(r), 0 : int(col_right) + 1] = True
    for c in col_lines:
        mask[int(row_top) : int(row_bottom) + 1, int(c)] = True

    entry_top = max(0, min(marker_rows) - 1)
    entry_bottom = min(H - 1, max(marker_rows) + 1)
    mask[int(entry_top) : int(entry_bottom) + 1, 0] = True

    # A few short stubs make the network look less like a perfect rectangle while
    # keeping the topology readable and still clearly grid-like.
    for c in random.sample(col_lines, k=max(1, len(col_lines) // 2)):
        if random.random() < 0.5 and row_top - 2 >= 0:
            stub = random.randint(1, min(2, row_top))
            mask[int(row_top) - int(stub) : int(row_top), int(c)] = True
        if random.random() < 0.5 and row_bottom + 2 < H:
            stub = random.randint(1, min(2, H - 1 - row_bottom))
            mask[int(row_bottom) + 1 : int(row_bottom) + int(stub) + 1, int(c)] = True
    return mask


def _network_is_grid_like(mask: np.ndarray) -> bool:
    H, W = mask.shape
    intersections = 0
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue
            deg = 0
            for dr, dc in DIR_ORDER:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
                    deg += 1
            if deg >= 3:
                intersections += 1
    return intersections >= 4


def _path_neighbor_count(g: np.ndarray, r: int, c: int, path_color: int) -> int:
    H, W = g.shape
    total = 0
    for dr, dc in DIR_ORDER:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            total += 1
    return total


def _target_candidates(
    g: np.ndarray, trace: dict[str, object], path_color: int
) -> list[tuple[int, int, int]]:
    cells = trace["cells"]  # type: ignore[assignment]
    branch_counts = trace["branch_counts"]  # type: ignore[assignment]
    open_side_turn_counts = trace["open_side_turn_counts"]  # type: ignore[assignment]
    assert isinstance(cells, list)
    assert isinstance(branch_counts, list)
    assert isinstance(open_side_turn_counts, list)
    if len(cells) < 8:
        return []

    later_start = max(5, len(cells) // 2)
    candidates: list[tuple[int, int, int]] = []
    for idx in range(len(cells) - 1, later_start - 1, -1):
        r, c = cells[idx]
        if c <= 1:
            continue
        if _path_neighbor_count(g, int(r), int(c), int(path_color)) < 2:
            continue
        if int(open_side_turn_counts[idx]) < 1:
            continue
        candidates.append((int(r), int(c), int(idx)))
    return candidates


def _targets_are_legal(
    g: np.ndarray,
    path_color: int,
    end_color: int,
    targets: list[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> bool:
    if len(set(targets)) != len(targets):
        return False

    trial = g.copy()
    for r, c in targets:
        trial[int(r), int(c)] = int(end_color)

    for r, c in targets:
        same = 0
        has_path = False
        for dr, dc in DIR_ORDER:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < trial.shape[0] and 0 <= cc < trial.shape[1]:
                if int(trial[rr, cc]) == int(end_color):
                    same += 1
                if int(trial[rr, cc]) == int(path_color):
                    has_path = True
        if same != 0 or not has_path:
            return False

    passable = _passable_mask(
        trial,
        int(path_color),
        set((int(r), int(c)) for r, c in targets),
        marker_cells,
    )
    _, n_pass = label_mask(passable, connectivity=4)
    return n_pass == 1


def _prefix_disjoint(
    traces: list[dict[str, object]], chosen: list[tuple[int, int, int]]
) -> bool:
    prefixes: list[set[tuple[int, int]]] = []
    targets = {(int(r), int(c)) for r, c, _ in chosen}
    if len(targets) != len(chosen):
        return False
    for trace, (_, _, idx) in zip(traces, chosen):
        cells = trace["cells"]  # type: ignore[assignment]
        assert isinstance(cells, list)
        prefixes.append(set((int(r), int(c)) for r, c in cells[: int(idx) + 1]))
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            if prefixes[i] & prefixes[j]:
                return False
    return True


def _choose_targets(
    g: np.ndarray,
    traces: list[dict[str, object]],
    path_color: int,
    end_color: int,
    marker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidate_lists = [_target_candidates(g, trace, int(path_color)) for trace in traces]
    if any(not cand for cand in candidate_lists):
        return None

    if len(candidate_lists) == 1:
        for r, c, _idx in candidate_lists[0][:16]:
            targets = [(int(r), int(c))]
            if _targets_are_legal(
                g, int(path_color), int(end_color), targets, marker_cells
            ):
                return targets
        return None

    left = candidate_lists[0][:18]
    right = candidate_lists[1][:18]
    for cand1 in left:
        for cand2 in right:
            chosen = [cand1, cand2]
            targets = [(int(chosen[0][0]), int(chosen[0][1])), (int(chosen[1][0]), int(chosen[1][1]))]
            if not _prefix_disjoint(traces, chosen):
                continue
            if _targets_are_legal(
                g, int(path_color), int(end_color), targets, marker_cells
            ):
                return targets
    return None


def _gen_attempt(n_markers: int | None = None) -> np.ndarray | None:
    H = W = random.randint(22, 30)
    if n_markers is None:
        n_markers = 2 if random.random() < 0.6 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    max_rows = 1 + ((H - 6) - 6) // 3
    max_cols = 1 + ((W - 8) - 3) // 3
    if max_rows < 4 or max_cols < 4:
        return None
    n_rows = random.randint(4, min(6, max_rows))
    n_cols = random.randint(4, min(6, max_cols))
    row_lines = _sample_spaced_positions(6, H - 6, n_rows, 3)
    col_lines = _sample_spaced_positions(3, W - 8, n_cols, 3)
    if row_lines is None or col_lines is None:
        return None

    interior_rows = row_lines[1:-1]
    if len(interior_rows) < n_markers:
        return None
    if n_markers == 1:
        marker_rows = [random.choice(interior_rows)]
    else:
        candidates = interior_rows[:]
        random.shuffle(candidates)
        marker_rows = sorted(candidates[:2])
        if len(marker_rows) != 2 or abs(marker_rows[0] - marker_rows[1]) < 3:
            return None

    mask = _build_grid_network(H, W, row_lines, col_lines, marker_rows)
    if mask[0, 0]:
        return None
    if not _network_is_grid_like(mask):
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    for r in marker_rows:
        if not (mask[int(r), 0] and mask[int(r) - 1, 0] and mask[int(r) + 1, 0]):
            return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr = int(template_row + dr)
                cc = int(bc + dc)
                if rr >= H or cc >= W or bool(mask[rr, cc]):
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color
    for i, bc in enumerate(block_cols):
        g[int(template_row) : int(template_row) + 2, int(bc) : int(bc) + 2] = int(cycle_colors[i])

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == int(bg)).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == int(path_color)).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[int(er) : int(er) + 2, int(ec) : int(ec) + 2] = int(end_color)

    markers: list[tuple[int, int, int]] = []
    for r in marker_rows:
        g[int(r), 0] = int(cycle_colors[0])
        markers.append((int(r), 0, int(cycle_colors[0])))
    marker_cells = set((int(r), 0) for r in marker_rows)

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if int(col) != int(bg)], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    traces = [
        _trace_particle(g, cycle_colors, int(path_color), set(), marker)
        for marker in markers
    ]
    if any(len(trace["cells"]) < 8 for trace in traces):  # type: ignore[arg-type]
        return None
    if any(int(trace["open_side_turns"]) < 1 for trace in traces):
        return None

    targets = _choose_targets(
        g, traces, int(path_color), int(end_color), marker_cells
    )
    if targets is None or len(targets) != len(markers):
        return None
    for tr, tc in targets:
        g[int(tr), int(tc)] = int(end_color)

    if not _targets_are_legal(
        g, int(path_color), int(end_color), targets, marker_cells
    ):
        return None

    try:
        _ = main(g)
    except Exception:
        return None
    return g


def generate_input() -> np.ndarray:
    global _NEXT_MARKER_COUNT

    preferred = int(_NEXT_MARKER_COUNT)
    fallback = 1 if preferred == 2 else 2
    for marker_count, attempts in ((preferred, 240), (fallback, 160)):
        for _ in range(attempts):
            try:
                g = _gen_attempt(marker_count)
            except Exception:
                g = None
            if g is not None:
                _NEXT_MARKER_COUNT = fallback
                return g
    raise RuntimeError("generate_input: failed after 400 attempts")
