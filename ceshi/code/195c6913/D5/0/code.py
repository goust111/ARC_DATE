# Variant of 195c6913
# Dimension: D5 — 终止条件 (dst)
# Change: 粒子在计划转向方向不是path时停止（collision_stop）；也在离开网格边界时停止。... → 到达目标像素时停止

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import label_mask

# Design analysis for the D5 variant
# a) Concrete behavior:
#    - Keep the original anti-diagonal path band, top-left 2x2 template blocks, left-edge markers,
#      and a distinct 2x2 legend block for the terminal color.
#    - Add 1 isolated in-path target pixel per marker. Each target uses the same color as the legend
#      block, so the 2x2 block still tells the solver "this is the finish color", but the actual stop
#      point is now a single cell embedded in the path.
#    - Small trace example:
#      old rule: marker visits ... -> P -> Q, then looks ahead into bg, paints a collision dot,
#      planned turn is blocked, so it stops after the failed collision logic.
#      new rule: recolor Q in the input to the finish color and declare it a target. The particle
#      still visits ... -> P, then steps onto Q and stops immediately there; the blocked-turn stop
#      is never used on generated instances.
# b) Complexity calibration:
#    - Only one new object type is introduced: isolated finish pixels. Their count matches the marker
#      count (1 or 2), and they always reuse the single end_color already present in the task.
#    - This avoids ambiguous multi-color destination systems while making the D5 change visible: the
#      path now contains a finish-colored cell that survives in the output and acts like a finish line.
# c) Old assumptions that must change:
#    - The original solver assumed every traversable cell had color path_color. That is no longer true:
#      target cells are traversable even though they have end_color.
#    - The original path-component assertion on (g == path_color) alone is too strict for the variant,
#      so main() checks connectivity on the passable mask = path_color U targets.
#    - Turn checks and initial direction checks must accept either a path cell or a target cell.
#    - end_color is no longer confined to the 2x2 legend block, so target detection must look for
#      isolated end_color pixels adjacent to the path, not assume every end_color cell belongs to the block.
# d) Placement legality:
#    - Targets are chosen only from actual cells on a marker's precomputed trajectory, so they are
#      reachable under the alternating-turn dynamics.
#    - Target choices are filtered so they stay isolated (no 4-neighbor end_color partner), remain
#      embedded in the path neighborhood, and do not break the path_color component when recolored.
#    - When there are two markers, target cells are chosen on disjoint channels and cannot be 4-adjacent.
# e) Visual observability:
#    - Across examples, the same color appears both as a removable 2x2 legend block and as the actual
#      finish pixel inside the path. In the output, trails end exactly at those in-path finish pixels.
#    - Because the finish pixel lies on the traversed route, the D5 change is visible without needing
#      hidden ordering conventions or color-id arithmetic.
# f) Visual cleanliness:
#    - The generator places targets in the later half of each trajectory, away from the template area
#      and away from the dense col-0 marker region. This prevents clutter near the object cues.
#    - The original separator logic for 2-marker cases is preserved so the two traces remain visually distinct.
# g) Color inferability and new solver assumptions:
#    - No new color mapping is added: end_color keeps one semantic role, "finish/collision marker".
#    - main() assumes there is exactly one reachable target per marker, each target is an isolated
#      end_color pixel adjacent to path cells, and the only 2x2 monochrome block of that color is the
#      removable legend block. generate_input() explicitly enforces these assumptions.


def _assert_passable_single_component(passable_mask: np.ndarray) -> None:
    """Variant invariant: path cells plus in-path targets form one 4-connected component."""
    _, n_path = label_mask(passable_mask, connectivity=4)
    assert n_path == 1, f"passable region must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Markers stay embedded in the original col-0 path segment."""
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


def _assert_targets_embedded_in_path(
    g: np.ndarray, targets: list[tuple[int, int]], path_color: int
) -> None:
    """Targets are isolated finish-color pixels sitting inside the path neighborhood."""
    H, W = g.shape
    for r, c in targets:
        path_neighbors = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                path_neighbors += 1
        assert path_neighbors >= 1, f"target at ({r},{c}) must touch the path"


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
                pattern.append(int(v))
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


def _trace_particle(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
    target_cells: set[tuple[int, int]],
    marker: tuple[int, int, int],
    out: np.ndarray | None = None,
) -> dict[str, object]:
    H, W = g.shape
    sr, sc, sv = marker
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    def is_passable(rr: int, cc: int) -> bool:
        return 0 <= rr < H and 0 <= cc < W and (
            int(g[rr, cc]) == int(path_color) or (int(rr), int(cc)) in target_cells
        )

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if is_passable(rr, cc):
            init = (int(dr), int(dc))
            break

    if init is None or not pattern:
        return {
            "cells": [],
            "hit_counts": [],
            "turns": 0,
            "reached_target": False,
            "target": None,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    cells: list[tuple[int, int]] = []
    hit_counts: list[int] = []
    reached_target = False
    target_hit = None

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        if state in visited:
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if (int(nr), int(nc)) in target_cells:
            if out is not None and end_color is not None:
                out[nr, nc] = int(end_color)
            cells.append((int(nr), int(nc)))
            hit_counts.append(int(hits))
            reached_target = True
            target_hit = (int(nr), int(nc))
            break

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            if out is not None:
                out[nr, nc] = int(pattern[idx])
            cells.append((int(nr), int(nc)))
            hit_counts.append(int(hits))
            r, c = nr, nc
            continue

        if out is not None and int(g[nr, nc]) == int(template_bg) and end_color is not None:
            out[nr, nc] = int(end_color)

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if is_passable(rr, cc):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            continue
        break

    return {
        "cells": cells,
        "hit_counts": hit_counts,
        "turns": int(hits),
        "reached_target": bool(reached_target),
        "target": target_hit,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles along a path that now ends at explicit in-path targets.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks encodes the repeating paint cycle.
    - Left edge: isolated marker pixels of a cycle color sit inside a col-0 path segment.
    - Finish cue: one 2x2 legend block of a distinct color, plus isolated pixels of that color placed on the path.

    Rule:
    - Each marker launches a particle that advances straight and paints path cells with successive cycle colors.
    - On hitting a non-passable cell, the particle paints that collision cell with the finish color and alternates turns CCW, CW, CCW, ...
    - A finish-colored isolated pixel inside the path is traversable and acts as a destination: stepping onto it stops that particle immediately.

    Key insight: The alternating turn pattern still determines the route, but termination is no longer inferred from a failed turn; the destination is an explicit finish pixel embedded on the route.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )

    pattern, pat_cells = _find_pattern(g, int(template_bg), int(path_color))
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, int(template_bg), int(path_color), pat_set)
    markers = _find_markers(g, pat_set, int(path_color))
    targets = _find_targets(g, end_color, int(path_color))

    _assert_marker_embedded_in_path(g, markers, int(path_color))
    _assert_targets_embedded_in_path(g, targets, int(path_color))
    assert pattern, "template pattern must be present"
    assert end_color is not None and end_pos is not None, "finish legend block must be present"
    assert targets, "at least one finish target must be present"
    assert len(targets) == len(markers), "generator contract: one finish target per marker"

    passable = g == int(path_color)
    for r, c in targets:
        passable[int(r), int(c)] = True
    _assert_passable_single_component(passable)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    er, ec = end_pos
    out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    target_cells = set((int(r), int(c)) for r, c in targets)
    for marker in markers:
        trace = _trace_particle(
            g,
            pattern,
            int(path_color),
            int(template_bg),
            end_color,
            target_cells,
            marker,
            out=out,
        )
        assert bool(trace["reached_target"]), "each particle must terminate at a finish target"
        per_marker_cells.append(set(trace["cells"]))  # type: ignore[arg-type]

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array([
        0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
        for r in range(H)
    ])

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
    """Carve a bg fjord from the left side without disconnecting the path band."""
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
    cells = trace["cells"]  # type: ignore[assignment]
    hit_counts = trace["hit_counts"]  # type: ignore[assignment]
    assert isinstance(cells, list)
    assert isinstance(hit_counts, list)
    if len(cells) < 6:
        return []

    later_start = max(4, len(cells) // 2)
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for idx in range(len(cells) - 1, later_start - 1, -1):
        r, c = cells[idx]
        if _path_neighbor_count(g, int(r), int(c), int(path_color)) < 2:
            continue
        cell = (int(r), int(c))
        if int(hit_counts[idx]) >= 1:
            preferred.append(cell)
        else:
            fallback.append(cell)
    return preferred + fallback


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

    _, n_path = label_mask(trial == int(path_color), connectivity=4)
    if n_path != 1:
        return False

    passable = trial == int(path_color)
    for r, c in targets:
        passable[int(r), int(c)] = True
    _, n_pass = label_mask(passable, connectivity=4)
    return n_pass == 1


def _choose_targets(
    g: np.ndarray, traces: list[dict[str, object]], path_color: int, end_color: int
) -> list[tuple[int, int]] | None:
    candidate_lists = [_target_candidates(g, trace, int(path_color)) for trace in traces]
    if any(not candidates for candidates in candidate_lists):
        return None

    if len(candidate_lists) == 1:
        for target in candidate_lists[0][:14]:
            if _targets_are_legal(g, int(path_color), int(end_color), [target]):
                return [target]
        return None

    left = candidate_lists[0][:14]
    right = candidate_lists[1][:14]
    for t1 in left:
        for t2 in right:
            targets = [t1, t2]
            if _targets_are_legal(g, int(path_color), int(end_color), targets):
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

    marker_info: list[tuple[int, int, int]] = []
    for r in markers:
        g[r, 0] = cycle_colors[0]
        marker_info.append((int(r), 0, int(cycle_colors[0])))

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    traces = [
        _trace_particle(
            g,
            cycle_colors,
            int(path_color),
            int(bg),
            int(end_color),
            set(),
            marker,
        )
        for marker in marker_info
    ]
    if any(len(trace["cells"]) < 6 for trace in traces):  # type: ignore[arg-type]
        return None

    targets = _choose_targets(g, traces, int(path_color), int(end_color))
    if targets is None or len(targets) != len(marker_info):
        return None
    for tr, tc in targets:
        g[int(tr), int(tc)] = int(end_color)

    if not _targets_are_legal(g, int(path_color), int(end_color), targets):
        return None

    try:
        _ = main(g)
    except Exception:
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
