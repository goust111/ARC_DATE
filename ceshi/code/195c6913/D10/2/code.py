# Variant of 195c6913
# Dimension: D10 - path区域形态：厚蜿蜒反对角带 (obstacle)
# Change: path为单个4连通的厚蜿蜒反对角带（宽4-12格），从右上延伸到左下，将bg分成2-4个区域；n_markers=2时... -> path包含内部障碍物（非bg非path像素）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D10 obstacle variant
# a) Concrete behavior and trace:
#    - Keep the original visual grammar: top-left 2x2 template blocks define the
#      color cycle, col-0 markers launch particles, and a thick anti-diagonal
#      path band separates the template side from the end-block side.
#    - Add 1-2 isolated obstacle pixels of one extra color inside the path band.
#      Each obstacle sits strictly inside the corridor: it is off the grid border
#      and all four orthogonal neighbors are path cells.
#    - Local trace: suppose a marker is moving right through path cells and the
#      next cell is replaced by an obstacle O. In the original task the particle
#      would enter that cell and keep moving straight. In this variant the
#      particle stays on the previous cell, treats O as a collision, applies the
#      scheduled CCW/CW turn, and continues only if the turned direction is path.
#      The obstacle cell itself remains O in the output; only collisions against
#      background cells receive the end color.
# b) Complexity calibration:
#    - Obstacles are few (1-2) and share one color, so they are easy to spot and
#      do not compete with the existing template/end-block cues.
#    - Generator only accepts samples where every placed obstacle is actually hit
#      by some particle, so the change is behavioral rather than decorative.
# c) Hidden-assumption audit:
#    - Path detection via "most frequent non-bg color" still works because the
#      obstacle count is tiny relative to the band area.
#    - Pattern/end detection scans non-path colors, so obstacles must not form a
#      2x2 block and must use a color distinct from cycle/end colors.
#    - The original collision branch already turned on any non-path cell but only
#      painted background cells. That behavior is now made explicit and validated.
# d) Placement legality:
#    - Obstacles are isolated, off-border, non-adjacent to one another, and
#      placed only where removing the path cell preserves a single 4-connected
#      path component.
#    - After obstacle placement the standard oracle still checks marker embedding
#      and pairwise-disjoint trajectories.
# e) Visual observability:
#    - In the input, the obstacle pixels are visible as foreign-colored dots
#      embedded in the band interior rather than sitting on its outline.
#    - In the output, the recolored trajectories visibly bend around those dots
#      while the dots themselves remain unchanged.
# f) Visual cleanliness:
#    - Obstacles are isolated single pixels, which avoids accidental confusion
#      with template blocks, markers, or the end block.
# g) Color inference:
#    - Only one extra obstacle color is introduced per sample, keeping the
#      palette readable from a few examples.
# 9) New solver assumptions guaranteed by the generator:
#    - At least one obstacle exists.
#    - Every obstacle has four orthogonal path neighbors in the input.
#    - Every generated obstacle is hit by some trajectory.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each col-0 marker must sit inside a vertical path segment."""
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
            if (r, c) in seen:
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
                    seen.add((rr, cc))
                    pat_cells.append((rr, cc))
    return pattern, pat_cells


def _detect_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
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
                return v, (r, c)
    return None, None


def _detect_markers(
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
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((r, c, v))
    return markers


def _collect_obstacles(
    g: np.ndarray,
    template_bg: int,
    path_color: int,
    pat_cells: list[tuple[int, int]],
    end_pos: tuple[int, int] | None,
    markers: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    reserved = set(pat_cells)
    reserved.update((r, c) for r, c, _ in markers)
    if end_pos is not None:
        er, ec = end_pos
        for rr in range(er, er + 2):
            for cc in range(ec, ec + 2):
                reserved.add((rr, cc))

    H, W = g.shape
    obstacles: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if (r, c) in reserved:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            obstacles.append((r, c))
    return obstacles


def _assert_internal_obstacles(
    g: np.ndarray, obstacle_cells: list[tuple[int, int]], path_color: int
) -> None:
    assert obstacle_cells, "variant requires at least one internal obstacle"
    H, W = g.shape
    for r, c in obstacle_cells:
        assert 0 < r < H - 1 and 0 < c < W - 1, (
            f"obstacle at ({r},{c}) must stay off the outer border"
        )
        orth = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        assert all(int(g[rr, cc]) == path_color for rr, cc in orth), (
            f"obstacle at ({r},{c}) must be surrounded orthogonally by path"
        )
        win = g[r - 1 : r + 2, c - 1 : c + 2]
        assert int(np.count_nonzero(win == path_color)) >= 6, (
            f"obstacle at ({r},{c}) is too close to the band boundary"
        )


def _parse_instance(grid: np.ndarray, *, require_obstacles: bool) -> dict:
    g = np.array(grid, dtype=int)
    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            (cnt, int(col))
            for col, cnt in counts.items()
            if int(col) != template_bg
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _detect_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _detect_end_block(g, template_bg, path_color, pat_set)
    markers = _detect_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)

    obstacle_cells = _collect_obstacles(g, template_bg, path_color, pat_cells, end_pos, markers)
    if require_obstacles:
        _assert_internal_obstacles(g, obstacle_cells, path_color)

    return {
        "grid": g,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
        "obstacle_cells": obstacle_cells,
    }


def _trace_particle(
    g: np.ndarray,
    out: np.ndarray | None,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int | None,
    obstacle_cells: set[tuple[int, int]],
    marker: tuple[int, int, int],
) -> dict:
    H, W = g.shape
    sr, sc, sv = marker
    dir_order = ((0, 1), (1, 0), (0, -1), (-1, 0))

    init = None
    for dr, dc in dir_order:
        rr = sr + dr
        cc = sc + dc
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
            init = (dr, dc)
            break

    stats = {
        "visited_cells": set(),
        "path_order": [],
        "obstacle_hit_cells": set(),
        "obstacle_hits": 0,
        "bg_hits": 0,
        "turns": 0,
    }
    if init is None or not pattern:
        return stats

    r, c = sr, sc
    dr, dc = init
    idx = int(pattern.index(sv)) if sv in pattern else 0
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(H * W * 4):
        state = (r, c, dr, dc, idx, hits & 1)
        if state in visited:
            break
        visited.add(state)

        nr = r + dr
        nc = c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if int(g[nr, nc]) == path_color:
            idx = (idx + 1) % len(pattern)
            if out is not None:
                out[nr, nc] = int(pattern[idx])
            stats["visited_cells"].add((nr, nc))
            stats["path_order"].append((nr, nc))
            r, c = nr, nc
            continue

        if (nr, nc) in obstacle_cells:
            stats["obstacle_hits"] += 1
            stats["obstacle_hit_cells"].add((nr, nc))
        elif int(g[nr, nc]) == template_bg and end_color is not None:
            stats["bg_hits"] += 1
            if out is not None:
                out[nr, nc] = int(end_color)

        left = (-dc, dr)
        right = (dc, -dr)
        tdr, tdc = left if (hits % 2 == 0) else right
        rr = r + tdr
        cc = c + tdc
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
            dr, dc = tdr, tdc
            hits += 1
            stats["turns"] += 1
            continue
        break

    return stats


def _simulate_instance(parsed: dict, *, paint: bool) -> tuple[np.ndarray | None, list[dict]]:
    g = parsed["grid"]
    out = g.copy() if paint else None
    if out is not None:
        for r, c in parsed["pat_cells"]:
            out[r, c] = parsed["template_bg"]
        if parsed["end_pos"] is not None:
            er, ec = parsed["end_pos"]
            out[er : er + 2, ec : ec + 2] = parsed["template_bg"]

    per_marker_cells: list[set[tuple[int, int]]] = []
    marker_stats: list[dict] = []
    obstacle_set = set(parsed["obstacle_cells"])
    for marker in parsed["markers"]:
        stats = _trace_particle(
            g,
            out,
            parsed["path_color"],
            parsed["pattern"],
            parsed["template_bg"],
            parsed["end_color"],
            obstacle_set,
            marker,
        )
        per_marker_cells.append(stats["visited_cells"])
        marker_stats.append(stats)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out, marker_stats


def _variant_metrics(grid: np.ndarray, *, require_obstacles: bool) -> dict:
    parsed = _parse_instance(grid, require_obstacles=require_obstacles)
    _, stats = _simulate_instance(parsed, paint=False)
    hit_cells: set[tuple[int, int]] = set()
    total_obstacle_hits = 0
    total_turns = 0
    for st in stats:
        hit_cells |= st["obstacle_hit_cells"]
        total_obstacle_hits += int(st["obstacle_hits"])
        total_turns += int(st["turns"])
    return {
        "path_color": parsed["path_color"],
        "obstacle_cells": list(parsed["obstacle_cells"]),
        "obstacle_count": len(parsed["obstacle_cells"]),
        "obstacle_hit_cells": hit_cells,
        "obstacle_hit_count": total_obstacle_hits,
        "total_turns": total_turns,
        "path_orders": [list(st["path_order"]) for st in stats],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles inside a thick anti-diagonal path band with internal blockers.

    Input structure:
    - Top-left: 2x2 monochrome blocks define the repeating paint cycle from left to right.
    - Col 0 contains 1-2 isolated start markers colored with the first cycle color.
    - A distinct 2x2 end block sits in the opposite background region.
    - The path band also contains a few isolated non-path obstacle pixels embedded inside it.

    Rule:
    - Each marker launches a particle that advances through path cells and cycles through the template colors.
    - A step into background or into an internal obstacle counts as a collision.
    - Only background collisions are marked with the end color; obstacle pixels stay unchanged.
    - After each collision, the turn direction alternates CCW then CW; if the scheduled turn is not path, the particle stops.

    Key insight:
    The path is no longer defined only by its outer boundary: interior obstacle pixels act as miniature walls, so the solver must treat every non-path cell inside the band as a hard turn trigger while preserving those blocker pixels.
    """
    parsed = _parse_instance(grid, require_obstacles=True)
    out, _ = _simulate_instance(parsed, paint=True)
    assert out is not None
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
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - width / 2))
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
    """Carve a background fjord from col 0 without disconnecting the band."""
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


def _is_obstacle_site(g: np.ndarray, path_color: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    H, W = g.shape
    if not (0 < r < H - 1 and 0 < c < W - 1):
        return False
    if int(g[r, c]) != path_color:
        return False
    orth = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    if any(int(g[rr, cc]) != path_color for rr, cc in orth):
        return False
    win = g[r - 1 : r + 2, c - 1 : c + 2]
    return int(np.count_nonzero(win == path_color)) >= 7


def _place_obstacles(g: np.ndarray, obstacle_color: int) -> np.ndarray | None:
    g_cur = np.array(g, dtype=int)
    placed: list[tuple[int, int]] = []
    target = 2 if random.random() < 0.6 else 1

    for _ in range(target):
        metrics = _variant_metrics(g_cur, require_obstacles=False)
        path_color = int(metrics["path_color"])

        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for order in metrics["path_orders"]:
            usable = order[2:-2] if len(order) > 6 else order[1:-1]
            for cell in usable:
                if cell in seen:
                    continue
                seen.add(cell)
                if _is_obstacle_site(g_cur, path_color, cell):
                    candidates.append(cell)
        random.shuffle(candidates)

        accepted = False
        for r, c in candidates[:80]:
            if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in placed):
                continue

            trial = g_cur.copy()
            trial[r, c] = obstacle_color

            _, n_path = label_mask(trial == path_color, connectivity=4)
            if n_path != 1:
                continue

            counts = Counter(map(int, trial.flatten()))
            non_bg = sorted(
                ((cnt, col) for col, cnt in counts.items() if col != int(trial[0, 0])),
                reverse=True,
            )
            if not non_bg or int(non_bg[0][1]) != path_color:
                continue

            try:
                _ = main(trial)
                trial_metrics = _variant_metrics(trial, require_obstacles=True)
            except Exception:
                continue

            needed = set(placed)
            needed.add((r, c))
            if not needed.issubset(trial_metrics["obstacle_hit_cells"]):
                continue

            g_cur = trial
            placed.append((r, c))
            accepted = True
            break

        if not accepted:
            break

    if not placed:
        return None

    final_metrics = _variant_metrics(g_cur, require_obstacles=True)
    if not set(placed).issubset(final_metrics["obstacle_hit_cells"]):
        return None
    return g_cur


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
    obstacle_choices = [
        col
        for col in palette
        if col not in {bg, path_color, end_color, *cycle_colors}
    ]
    if not obstacle_choices:
        return None
    obstacle_color = obstacle_choices[0]

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
            restore_cols = random.randint(2, 4)
            carved = mask & ~trial
            for c in range(min(restore_cols, W)):
                trial[:, c] |= carved[:, c]
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

    end_candidates: list[tuple[int, int]] = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1c = max(0, r - 1), min(H, r + 3)
            c0, c1c = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1c, c0:c1c] == path_color).any():
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

    counts = Counter(map(int, g.flatten()))
    non_bg = sorted(((cnt, col) for col, cnt in counts.items() if col != bg), reverse=True)
    if not non_bg or int(non_bg[0][1]) != path_color:
        return None

    g = _place_obstacles(g, obstacle_color)
    if g is None:
        return None

    try:
        _ = main(g)
        metrics = _variant_metrics(g, require_obstacles=True)
    except Exception:
        return None
    if metrics["obstacle_count"] < 1:
        return None
    if not set(metrics["obstacle_cells"]).issubset(metrics["obstacle_hit_cells"]):
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
