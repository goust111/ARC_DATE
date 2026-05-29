# Combo variant of 195c6913 (index 17)
# combo_spec: D10/0+D16/1
# Name: 窄path带+小网格

# Design analysis for the combo variant
# ------------------------------------
# concretization_and_trace:
#   - The original task runs particles inside a broad anti-diagonal band on a
#     20-30 square grid. Here the same color-cycle logic is kept, but the whole
#     board is compressed to a 10-15 square and the path is only 2-3 cells wide.
#   - Concrete trace difference:
#       old layout: a marker on col 0 could move right for a long horizontal run
#       before the first wall hit, so the left/right alternation was sparse.
#       fused layout: the marker still enters to the right first, but on a width-2
#       or width-3 row it meets a wall after 1-2 forward moves, turns, then meets
#       another wall again a couple of cells later. The visible effect is a short
#       zig-zag trace with dense corner marks.
#   - The path object still spans from the top-right side to the bottom-left side,
#     so solver parsing remains based on the same dominant path-color region.
# complexity_calibration:
#   - Width is restricted to the small visible set {2, 3}; that is narrow enough
#     to be unmistakable, but avoids degenerating into a 1-pixel skeleton.
#   - The small-grid change is implemented as H=W in [10,15], but template block
#     count is clipped by width so each sample still has a legal top-left template
#     with repeated 2x2 evidence rather than an overpacked corner.
#   - The solver remains compatible with one or two markers, and the generator
#     now covers both branches explicitly. One-marker scenes use the full 10-15
#     range, while two-marker scenes use a dedicated 15x15 staircase corridor
#     that keeps the separator readable instead of letting the tiny grid blur
#     both traces together.
# implicit_assumptions_review:
#   - `main()` assumes `g[0,0]` is background and the path color is the largest
#     non-background population. The generator keeps the narrow band dominant over
#     the template plus end block even on the small board.
#   - `main()` finds template blocks by scanning the top-left rows. The generator
#     therefore clears rows 1-3 up to the last chosen template block.
#   - `main()` identifies markers as isolated cycle-colored cells adjacent to path.
#     The generator still paints markers on col 0 and preserves path above/below
#     each marker so the embedded-marker assert stays valid.
#   - `main()` uses first-adjacent-path priority R,D,L,U. For left-edge markers,
#     this means `(r,1)` must be path, so every marker row belongs to the corridor.
#   - `main()` never searches for alternate turns: if the scheduled CCW/CW turn is
#     blocked, the particle stops. The generator therefore rejects paths whose
#     traced runs are too short or too straight after compression.
# placement_legality:
#   - Template blocks use the original column rhythm {1,4,7,10}, but only those
#     that still leave at least two columns for the path on template rows.
#   - The path remains one 4-connected component touching the top-right and the
#     bottom-left extremes, while staying out of the top-left template rectangle.
#   - The end block is still placed as an isolated 2x2 background block in the
#     bottom-right region with a one-cell moat from the path.
# visual_observability:
#   - The fusion is inferable by eye because both changed dimensions are global:
#     every example is visibly smaller than the original spec and every path row
#     is visibly creek-like rather than broad.
#   - Collision density becomes a visible regularity in the outputs: short painted
#     runs and many turn/corner marks appear within a compact area.
# visual_cleanliness:
#   - The template stays in a reserved top-left strip, markers stay on the left
#     edge, the end block stays in bottom-right background, and the thin path is
#     the only large non-rectangular object. That keeps object roles legible.
#   - Two-marker scenes use a fixed shallow fjord profile on the left edge. That
#     keeps the two traces legible on a 15x15 board without introducing tiny
#     interior holes that would read as noise.
# color_inference:
#   - Color semantics are unchanged and fully visible: template blocks define the
#     cycle, markers reuse the first template color, and collision pixels reuse
#     the end-block color. The fusion only changes geometry and scale.
# new_main_assumption_audit:
#   - `_parse_scene()` now tolerates fewer template blocks because the small grid
#     may only fit 2 or 3; generator guarantees at least 2 such blocks exist.
#   - `_simulate()` still assumes trajectories are disjoint when there are two
#     markers. The generator therefore uses a hand-shaped 15x15 two-marker band
#     whose oracle traces stay disjoint under the same alternating-turn physics.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each left-edge marker sits inside a col-0 path run."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != template_bg])[1]
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
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    pat_set = set(map(int, pattern))
    end_color = None
    end_pos = None
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


def _simulate(grid: np.ndarray, *, with_stats: bool = False):
    scene = _parse_scene(grid)
    g = scene["grid"]
    H, W = scene["H"], scene["W"]
    template_bg = scene["template_bg"]
    path_color = scene["path_color"]
    pattern = scene["pattern"]
    end_color = scene["end_color"]
    end_pos = scene["end_pos"]
    markers = scene["markers"]

    out = g.copy()
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    per_marker_cells: list[set[tuple[int, int]]] = []
    turn_counts: list[int] = []
    step_counts: list[int] = []
    collision_counts: list[int] = []

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
            turn_counts.append(0)
            step_counts.append(0)
            collision_counts.append(0)
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        steps = 0
        collisions = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
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
                steps += 1
                r, c = nr, nc
                continue

            collisions += 1
            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

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
        turn_counts.append(int(hits))
        step_counts.append(int(steps))
        collision_counts.append(int(collisions))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    if with_stats:
        stats = {
            "per_marker_cells": per_marker_cells,
            "turn_counts": turn_counts,
            "step_counts": step_counts,
            "collision_counts": collision_counts,
            "path_color": path_color,
            "markers": markers,
            "pattern_len": len(pattern),
        }
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace alternating-turn particles through a compact thin anti-diagonal corridor.

    Input structure:
    - A small square board contains one dominant 2-3-cell-wide path color, plus a top-left row of monochrome 2x2 template blocks that encode a paint cycle.
    - One or two isolated cycle-colored pixels on the left edge are start markers embedded in that thin path.
    - A distinct 2x2 block in bottom-right background provides the terminal/collision color.

    Rule:
    - Erase the template blocks and the end block in the output, then launch each marker along its first adjacent path cell using R,D,L,U priority.
    - Every time the particle advances onto a path cell, repaint that cell with the next color in the template cycle.
    - When straight-ahead motion is blocked, paint the collision cell with the end color if it is background, then attempt exactly one 90-degree turn: CCW on the first successful turn, CW on the next, alternating forever; if that scheduled turn is not path, stop.

    Key insight:
    - The rule itself is still "alternate left and right turns on collisions", but the fused small-grid thin-band geometry makes those collisions happen after only a few steps, so the visible output is a short, bend-dense trace rather than a long run through a wide river.
    """
    return _simulate(grid, with_stats=False)


def _available_block_cols(W: int) -> list[int]:
    return [c for c in (1, 4, 7, 10) if c <= W - 4]


def _overlap_bounds(prev_left: int, prev_w: int, next_w: int, W: int) -> tuple[int, int]:
    lo = max(0, int(prev_left - next_w + 1))
    hi = min(int(W - next_w), int(prev_left + prev_w - 1))
    return lo, hi


def _sample_markers(H: int, W: int, n_markers: int) -> list[int] | None:
    upper_need = max(6, (W + 1) // 2)
    if n_markers == 1:
        lo = max(upper_need, H // 2)
        hi = H - 3
        if lo > hi:
            return None
        return [int(random.randint(lo, hi))]

    r1_lo = max(upper_need, H // 2 - 1)
    r1_hi = H - 8
    if r1_lo > r1_hi:
        return None
    r1 = random.randint(r1_lo, r1_hi)
    r2_lo = max(r1 + 4, H - 4)
    r2_hi = H - 3
    if r2_lo > r2_hi:
        return None
    r2 = random.randint(r2_lo, r2_hi)
    return [int(r1), int(r2)]


def _build_desired_lefts(
    H: int,
    W: int,
    marker_rows: list[int],
    widths: list[int],
    template_clear: int,
) -> np.ndarray | None:
    left_start = marker_rows[0] - 1
    if left_start < 4:
        return None

    max_drop = sum(widths[r] - 1 for r in range(1, left_start + 1))
    if max_drop < W - widths[0]:
        return None

    left = np.zeros(H, dtype=int)
    left[0] = W - widths[0]
    target = np.linspace(left[0], 0, left_start + 1)

    future_drop = [0] * H
    acc = 0
    for r in range(left_start - 1, -1, -1):
        acc += widths[r + 1] - 1
        future_drop[r] = acc

    for r in range(1, left_start + 1):
        lo, hi = _overlap_bounds(int(left[r - 1]), int(widths[r - 1]), int(widths[r]), W)
        hi = min(hi, int(future_drop[r]))
        if 1 <= r <= 3:
            lo = max(lo, template_clear)
        if lo > hi:
            return None
        desired = int(round(target[r])) + random.choice([-1, 0, 0, 1])
        desired = max(lo, min(hi, desired))
        choices = list(range(lo, hi + 1))
        choices.sort(key=lambda x: (abs(x - desired), abs(x - int(round(target[r]))), random.random()))
        left[r] = int(choices[0])

    for r in range(left_start + 1, H):
        left[r] = 0

    if len(marker_rows) == 2:
        r1, r2 = marker_rows
        gap_rows = list(range(r1 + 2, r2 - 1))
        if len(gap_rows) < 2:
            return None

        profile: list[int] = []
        for i in range(len(gap_rows)):
            if len(gap_rows) >= 3 and abs(i - (len(gap_rows) - 1) / 2.0) <= 0.75:
                profile.append(2)
            else:
                profile.append(1)

        future_gap_drop = {}
        acc = 0
        for r in range(r2 - 2, r1 + 1, -1):
            acc += widths[r + 1] - 1
            future_gap_drop[r] = acc

        prev_row = r1 + 1
        for desired, r in zip(profile, gap_rows):
            lo, hi = _overlap_bounds(int(left[prev_row]), int(widths[prev_row]), int(widths[r]), W)
            hi = min(hi, int(future_gap_drop.get(r, 0)))
            lo = max(lo, 1)
            if lo > hi:
                return None
            left[r] = int(max(lo, min(hi, desired)))
            prev_row = r

        lo, hi = _overlap_bounds(
            int(left[gap_rows[-1]]),
            int(widths[gap_rows[-1]]),
            int(widths[r2 - 1]),
            W,
        )
        if not (lo <= 0 <= hi):
            return None
        left[r1 - 1 : r1 + 2] = 0
        left[r2 - 1 :] = 0

    return left


def _build_thin_band(H: int, W: int, marker_rows: list[int], template_clear: int) -> np.ndarray | None:
    left_start = marker_rows[0] - 1
    widths = [2] * H
    for r in range(H):
        if r <= left_start or random.random() < 0.35:
            widths[r] = 3
    widths[0] = random.choice([2, 3])
    widths[H - 1] = random.choice([2, 3])

    need_drop = W - widths[0]
    if sum(widths[r] - 1 for r in range(1, left_start + 1)) < need_drop:
        for r in range(1, left_start + 1):
            widths[r] = 3
        if sum(widths[r] - 1 for r in range(1, left_start + 1)) < need_drop:
            return None

    left = _build_desired_lefts(H, W, marker_rows, widths, template_clear)
    if left is None:
        return None

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, int(left[r]) : int(left[r] + widths[r])] = True

    if mask[0, 0]:
        return None
    if not mask[0, W - 1]:
        return None
    if not mask[H - 1, 0]:
        return None
    if not mask[:, 0].any():
        return None

    row_widths = mask.sum(axis=1)
    if int(row_widths.min()) < 2 or int(row_widths.max()) > 3:
        return None

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    return mask


def _build_two_marker_band() -> np.ndarray:
    """Stable 15x15 narrow-band layout with a readable left-edge separator.

    The top half is a strict anti-diagonal staircase; the lower half hugs the
    left edge with a shallow fjord between the two marker rows. This keeps the
    path 2-3 cells wide everywhere, preserves a single path component, and
    yields disjoint traces for the two embedded markers under the solver's
    strict alternating-turn rule.
    """
    lefts = [12, 10, 8, 6, 4, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    widths = [3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2]
    mask = np.zeros((15, 15), dtype=bool)
    for r, (left, width) in enumerate(zip(lefts, widths)):
        mask[r, left : left + width] = True
    return mask


def _place_end_block(g: np.ndarray, bg: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    end_candidates = []
    for r in range(max(H // 2, 4), H - 1):
        for c in range(max(W // 2, 4), W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    return random.choice(end_candidates)


def _variant_metrics(g: np.ndarray, stats: dict) -> dict:
    scene = _parse_scene(g)
    path_color = int(stats["path_color"])
    path_mask = g == path_color
    for r, c, _ in stats["markers"]:
        path_mask[int(r), int(c)] = True
    row_widths = path_mask.sum(axis=1)
    _, n_bg = label_mask(~path_mask, connectivity=4)
    total_steps = int(sum(stats["step_counts"]))
    total_collisions = int(sum(stats["collision_counts"]))
    payload_cells = 4 * len(scene["pattern"]) + (4 if scene["end_pos"] is not None else 0)
    return {
        "H": int(g.shape[0]),
        "W": int(g.shape[1]),
        "max_width": int(row_widths.max()),
        "min_width": int(row_widths.min()),
        "avg_width": float(row_widths.mean()),
        "total_turns": int(sum(stats["turn_counts"])),
        "per_marker_turns": list(map(int, stats["turn_counts"])),
        "per_marker_steps": list(map(int, stats["step_counts"])),
        "per_marker_collisions": list(map(int, stats["collision_counts"])),
        "bg_components": int(n_bg),
        "collision_stride": float(total_steps / max(1, total_collisions)),
        "payload_ratio": float(payload_cells / float(g.shape[0] * g.shape[1])),
    }


def _gen_attempt(target_markers: int | None = None) -> np.ndarray | None:
    n_markers = int(target_markers) if target_markers in (1, 2) else (2 if random.random() < 0.5 else 1)
    if n_markers == 2:
        H = W = 15
        block_cols = [1, 4]
        template_clear = 6
        markers = [7, 12]
    else:
        H = W = random.choices([10, 11, 12, 13, 14, 15], weights=[1, 1, 2, 2, 3, 3])[0]
        block_options = _available_block_cols(W)
        if len(block_options) < 2:
            return None
        min_blocks = 3 if W >= 14 and len(block_options) >= 3 else 2
        n_blocks = random.randint(min_blocks, len(block_options))
        block_cols = block_options[:n_blocks]
        template_clear = block_cols[-1] + 2
        markers = _sample_markers(H, W, n_markers)
    if markers is None:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    cycle_colors = random.sample(palette[3:], k=len(block_cols))

    mask = _build_two_marker_band() if n_markers == 2 else _build_thin_band(H, W, markers, template_clear)
    if mask is None or mask[0, 0]:
        return None

    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_markers == 1 and n_bg < 2:
        return None
    if n_markers == 2 and n_bg < 3:
        return None

    template_row = random.choice([1, 2])
    for bc in block_cols:
        if mask[template_row : template_row + 2, bc : bc + 2].any():
            return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_pos = _place_end_block(g, bg, path_color)
    if end_pos is None:
        return None
    er, ec = end_pos
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
        _, stats = _simulate(g, with_stats=True)
    except Exception:
        return None

    metrics = _variant_metrics(g, stats)
    if metrics["H"] < 10 or metrics["H"] > 15 or metrics["W"] < 10 or metrics["W"] > 15:
        return None
    if metrics["max_width"] > 3 or metrics["min_width"] < 2:
        return None
    if min(metrics["per_marker_steps"], default=0) < 5:
        return None
    if max(metrics["per_marker_steps"], default=0) > 24:
        return None
    if metrics["total_turns"] < 2:
        return None
    if metrics["collision_stride"] > 3.4:
        return None
    if metrics["payload_ratio"] < 0.05:
        return None
    return g


_next_marker_count = 1


def generate_input() -> np.ndarray:
    global _next_marker_count
    target_markers = _next_marker_count
    _next_marker_count = 1 if _next_marker_count == 2 else 2
    for _ in range(800):
        try:
            g = _gen_attempt(target_markers=target_markers)
        except Exception:
            g = None
        if g is not None:
            return g
    for _ in range(200):
        try:
            g = _gen_attempt(target_markers=None)
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 1000 attempts")
