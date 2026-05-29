# Variant of 195c6913
# Dimension: D10 — path区域形态：厚蜿蜒反对角带 (obstacle)
# Change: path为单个4连通的厚蜿蜒反对角带（宽4-12格），从右上延伸到左下，将bg分成2-4个区域；n_markers=2时... → path为薄蜿蜒带（宽2-3格）

# Design analysis for the D10 variant
# ----------------------------------
# concretization_and_trace:
#   - The old task used a broad anti-diagonal river. Here every row of the path
#     is only 2 or 3 cells wide, so the same particle rule now behaves like a
#     bead sliding inside a creek. The visible change is not a new color rule;
#     it is the corridor geometry: narrow rows, tighter overlaps, and more
#     frequent wall hits.
#   - Example trace at one marker row:
#       old band row width ~= 8: marker moves right for many cells before the
#       first collision, so turns are sparse.
#       new band row width = 2 or 3: the same marker hits the row boundary
#       quickly, turns upward, then soon hits the column boundary of the next
#       overlap. The alternating CCW/CW rule is unchanged, but collisions are
#       denser because the channel is thin.
# complexity_calibration:
#   - Width is restricted to the small visible set {2, 3}. That is narrow enough
#     to be obvious by eye, but not so thin that the path degenerates into an
#     ambiguous 1-cell polyline.
#   - For n_markers=2 the generator keeps a left-edge fjord between the two
#     entrances. I intentionally drop the old interior-island branch: on a
#     2-3-cell corridor it becomes either visually muddy or topologically fragile.
# implicit_assumptions_review:
#   - main() assumes path_color is the largest non-template_bg color. The thin
#     generator keeps the path area safely larger than the template/end blocks.
#   - main() assumes markers live on col 0 and their vertical neighbors are path.
#     The generator explicitly forces 3-row col-0 runs around each marker.
#   - main() detects the initial direction by priority R,D,L,U. For left-edge
#     markers that still needs a path cell at (r,1), so the generator always
#     makes the marker row itself part of the thin band.
#   - main() only turns if the scheduled 90-degree direction is path. Thin bands
#     can dead-end too early, so the generator rejects samples whose traced
#     trajectories are too short or have too few successful turns.
# placement_legality:
#   - The path touches top/right at row 0 and bottom/left at row H-1.
#   - Rows 1-3 keep the left side clear (c_left >= 12), preserving the original
#     template placement scheme at cols 1,4,7,10.
#   - The end block is still placed in a bottom-right background patch with a
#     1-cell moat from the path, so solver parsing remains unambiguous.
# visual_observability:
#   - The change is inferable from a few examples because the band itself is the
#     dominant object and its thickness is globally different on every row.
#   - No rule depends on hidden color ids. Colors only label the cycle/template,
#     end marker, background, and path in the same visible way as before.
# visual_cleanliness:
#   - Objects stay separated: template in the cleared top-left region, end block
#     in bottom-right background, markers on the left edge, path as one thin band.
#   - Restricting separator shape to one fjord branch avoids noisy tiny islands
#     inside an already narrow corridor.
# color_inference:
#   - Color behavior is unchanged from the original task: template blocks define
#     the cycle, markers use the first cycle color, collision pixels use the end
#     color. The only changed signal is geometric width, which is directly visible.

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
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each left-edge marker is inside a col-0 path segment."""
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
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
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
        }
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles inside a thin anti-diagonal creek.

    Input structure:
    - Top-left: a horizontal row of monochrome 2x2 blocks gives the repeating paint cycle.
    - Left edge: one or two isolated cycle-colored marker pixels are embedded in the thin path.
    - Elsewhere: one distinct 2x2 end block sits in background opposite the path.

    Rule:
    - Each marker launches a particle along the path, coloring visited path cells by advancing through the cycle.
    - When the next forward cell is not path, that collision cell is painted with the end color if it is background.
    - The turn schedule is strict and alternating: first successful turn is CCW, next is CW, then CCW again; if the scheduled turn is not path, the particle stops.

    Key insight:
    - The motion rule is unchanged from the original task, but the thin 2-3-cell corridor makes the alternating wall-bounce pattern visible much more often than in a broad band.
    """
    return _simulate(grid, with_stats=False)


def _overlap_bounds(prev_left: int, prev_w: int, next_w: int, W: int) -> tuple[int, int]:
    lo = max(0, int(prev_left - next_w + 1))
    hi = min(int(W - next_w), int(prev_left + prev_w - 1))
    return lo, hi


def _build_desired_lefts(H: int, W: int, marker_rows: list[int], widths: list[int]) -> np.ndarray | None:
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
            lo = max(lo, 12)
        if lo > hi:
            return None
        desired = int(round(target[r] + random.choice([-1, 0, 1])))
        desired = max(lo, min(hi, desired))
        candidates = list(range(lo, hi + 1))
        candidates.sort(key=lambda x: (abs(x - desired), abs(x - int(round(target[r]))), random.random()))
        left[r] = int(candidates[0])

    for r in range(left_start + 1, H):
        left[r] = 0

    if len(marker_rows) == 2:
        r1, r2 = marker_rows
        gap_rows = list(range(r1 + 2, r2 - 1))
        if len(gap_rows) < 2:
            return None
        peak = 2 if len(gap_rows) >= 4 and random.random() < 0.7 else 1
        profile: list[int] = []
        for i in range(len(gap_rows)):
            if peak == 1:
                profile.append(1)
            else:
                mid = (len(gap_rows) - 1) / 2.0
                profile.append(2 if abs(i - mid) <= max(0.75, len(gap_rows) / 4.0) else 1)

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
            choose = max(lo, min(hi, desired))
            left[r] = int(choose)
            prev_row = r

        if r2 - 1 < H:
            lo, hi = _overlap_bounds(int(left[gap_rows[-1]]), int(widths[gap_rows[-1]]), int(widths[r2 - 1]), W)
            if not (lo <= 0 <= hi):
                return None
            left[r2 - 1 :] = 0

        left[r1 - 1 : r1 + 2] = 0
        left[r2 - 1 :] = 0

    return left


def _build_thin_band(H: int, W: int, marker_rows: list[int]) -> np.ndarray | None:
    left_start = marker_rows[0] - 1
    widths = [3] * H
    for r in range(H):
        if r > left_start + 2 and random.random() < 0.55:
            widths[r] = 2
        elif r <= left_start + 2 and random.random() < 0.25:
            widths[r] = 2
    widths[0] = random.choice([2, 3])
    widths[H - 1] = random.choice([2, 3])

    max_drop = sum(widths[r] - 1 for r in range(1, left_start + 1))
    need_drop = W - widths[0]
    if max_drop < need_drop:
        for r in range(1, left_start + 1):
            widths[r] = 3
        if sum(widths[r] - 1 for r in range(1, left_start + 1)) < need_drop:
            return None

    left = _build_desired_lefts(H, W, marker_rows, widths)
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

    row_widths = mask.sum(axis=1)
    if int(row_widths.min()) < 2 or int(row_widths.max()) > 3:
        return None

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    return mask


def _place_end_block(g: np.ndarray, bg: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
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


def _sample_markers(H: int, n_markers: int) -> list[int] | None:
    if n_markers == 1:
        r1 = random.randint(max(5, H // 2), H - 4)
        return [int(r1)]

    r1_min = max(8, (H - 3 + 1) // 2 + 1)
    r1_max = H - 10
    if r1_min > r1_max:
        return None
    r1 = random.randint(r1_min, r1_max)
    r2_min = max(r1 + 6, 2 * H // 3)
    if r2_min > H - 4:
        return None
    r2 = random.randint(r2_min, H - 4)
    return [int(r1), int(r2)]


def _variant_metrics(g: np.ndarray, stats: dict) -> dict:
    path_color = int(stats["path_color"])
    path_mask = g == path_color
    for r, c, _ in stats["markers"]:
        path_mask[int(r), int(c)] = True
    row_widths = path_mask.sum(axis=1)
    _, n_bg = label_mask(~path_mask, connectivity=4)
    return {
        "max_width": int(row_widths.max()),
        "min_width": int(row_widths.min()),
        "avg_width": float(row_widths.mean()),
        "total_turns": int(sum(stats["turn_counts"])),
        "per_marker_turns": list(map(int, stats["turn_counts"])),
        "per_marker_steps": list(map(int, stats["step_counts"])),
        "bg_components": int(n_bg),
    }


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.9 else 1
    markers = _sample_markers(H, n_markers)
    if markers is None:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, k=n_blocks)

    mask = _build_thin_band(H, W, markers)
    if mask is None:
        return None
    if mask[0, 0]:
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
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
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
    if metrics["max_width"] > 3 or metrics["min_width"] < 2:
        return None
    if min(metrics["per_marker_steps"], default=0) < 8:
        return None
    turn_floor = 3 if n_markers == 1 else 4
    if metrics["total_turns"] < turn_floor:
        return None
    if n_markers == 2 and metrics["bg_components"] < 3:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
