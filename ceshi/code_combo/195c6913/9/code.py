# Combo variant of 195c6913 (index 9)
# combo_spec: D6/2+D10/2
# Name: 碰撞十字形标记+path含内部障碍物

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the fused D6/2 + D10/2 variant
# a) Concrete behavior and trace:
#    - Keep the original scene grammar: top-left 2x2 template blocks encode a
#      color cycle; col-0 marker pixels launch particles into a thick
#      anti-diagonal corridor; a 2x2 end block sits in the opposite bg region.
#    - Add 1-2 internal obstacle pixels of one extra color inside the corridor.
#      A particle treats an obstacle exactly like a wall: it does not enter that
#      cell, counts a collision, and applies the scheduled alternating turn.
#    - Upgrade collision marking from one bg pixel to a larger local stamp. We
#      choose a bg-aware 3x3 stamp rather than a thin plus because obstacle
#      sites are orthogonally surrounded by path; with a pure plus, obstacle
#      hits would often leave only a one-pixel mark. The fused stamp paints the
#      obstacle center itself when hit and also paints any bg cells in its 3x3
#      neighborhood, making obstacle-triggered turns visible.
#    - Example trace: if a particle moving right sees a blocker at (r,c+1), the
#      original task would have advanced through that cell if it were path and
#      only marked single-pixel bg corners elsewhere. Here the particle stays at
#      (r,c), stamps a 3x3 collision patch around (r,c+1), turns CCW/CW by the
#      alternating schedule, and stops if that turned step is not path.
# b) Complexity calibration:
#    - Only one extra obstacle color is introduced, and there are only 1-2 such
#      pixels, so the scene stays readable.
#    - The larger collision stamp is limited to the 3x3 neighborhood and only
#      affects bg plus the struck obstacle center, so it is salient without
#      flooding the corridor.
#    - Generator requires every obstacle to be hit and requires at least one
#      multi-cell stamp, so the fusion is behaviorally active rather than
#      decorative.
# c) Hidden-assumption audit:
#    - `path_color` remains "most frequent non-bg color", so obstacle count must
#      stay tiny relative to the band area.
#    - Pattern/end-block detection scans non-path colors; therefore obstacles may
#      not form 2x2 blocks and must use a color distinct from template/end.
#    - Marker detection still relies on isolated template-colored singles next to
#      path, so the generator keeps obstacles away from the template row and does
#      not add same-colored clutter.
#    - Collision painting can no longer assume only bg collisions matter; solver
#      must explicitly classify obstacle collisions and stamp them too.
# d) Placement legality:
#    - Obstacles are off the outer border, not adjacent to each other, and each
#      chosen site was path beforehand with all four orthogonal neighbors still
#      path after removal.
#    - Obstacle placement preserves a single 4-connected path component and does
#      not overlap markers, template blocks, or the end block.
# e) Visual observability:
#    - In the input, obstacle pixels are foreign-colored dots embedded in the
#      path interior.
#    - In the output, collision events become broad end-colored patches rather
#      than isolated corner dots, and at least one obstacle hit creates a stamp
#      larger than one cell. That makes the fused rule inferable from a few
#      examples.
# f) Visual cleanliness:
#    - The stamp paints only local non-path space, so path recoloring remains
#      legible and obstacle markers do not smear across the whole band.
#    - Obstacles are sparse and separated by at least one cell in Chebyshev
#      distance, which avoids ambiguous merged symbols.
# g) Change inferability:
#    - The output still uses the original cycle colors for trajectory paint and a
#      single end color for all collision stamps. No new color mapping needs to
#      be inferred; only the shape/extent of the collision mark changes.
# 9) New solver assumptions guaranteed by the generator:
#    - At least one internal obstacle exists and every obstacle is hit.
#    - Every obstacle has four orthogonal path neighbors in the input and at
#      least one diagonal bg neighbor so its collision stamp is visually larger
#      than a solitary center pixel.
#    - No sample relies on collisions into template/end decorations, so humans
#      only need to infer two collision sources: ordinary bg walls and internal
#      obstacle pixels.
#    - At least one collision stamp paints multiple cells, and the fused output
#      differs from the baseline single-pixel-corner behavior on every sample.


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
        diag = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
        bg_diag = sum(int(g[rr, cc]) == int(g[0, 0]) for rr, cc in diag)
        assert bg_diag >= 1, (
            f"obstacle at ({r},{c}) must have a diagonal bg neighbor for a visible stamp"
        )
        win = g[r - 1 : r + 2, c - 1 : c + 2]
        assert int(np.count_nonzero(win == path_color)) >= 4, (
            f"obstacle at ({r},{c}) is too exposed to the outside"
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


def _paint_collision_stamp(
    out: np.ndarray,
    g: np.ndarray,
    center: tuple[int, int],
    template_bg: int,
    path_color: int,
    end_color: int | None,
    obstacle_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if end_color is None:
        return set()
    H, W = g.shape
    r, c = center
    painted: set[tuple[int, int]] = set()
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            cell = (rr, cc)
            v = int(g[rr, cc])
            if v == path_color:
                continue
            if v == template_bg or cell == center or cell in obstacle_cells:
                out[rr, cc] = int(end_color)
                painted.add(cell)
    return painted


def _paint_baseline_collision(
    out: np.ndarray,
    g: np.ndarray,
    center: tuple[int, int],
    template_bg: int,
    end_color: int | None,
) -> set[tuple[int, int]]:
    if end_color is None:
        return set()
    r, c = center
    if int(g[r, c]) != template_bg:
        return set()
    out[r, c] = int(end_color)
    return {center}


def _trace_particle(
    g: np.ndarray,
    out: np.ndarray | None,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int | None,
    obstacle_cells: set[tuple[int, int]],
    marker: tuple[int, int, int],
    *,
    expanded_stamp: bool,
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
        "collision_centers": [],
        "collision_kinds": [],
        "collision_stamp_sizes": [],
        "stamp_cells": set(),
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

        center = (nr, nc)
        if center in obstacle_cells:
            kind = "obstacle"
            stats["obstacle_hits"] += 1
            stats["obstacle_hit_cells"].add(center)
        elif int(g[nr, nc]) == template_bg:
            kind = "bg"
            stats["bg_hits"] += 1
        else:
            kind = "other"

        stamp_cells: set[tuple[int, int]] = set()
        if out is not None:
            if expanded_stamp:
                stamp_cells = _paint_collision_stamp(
                    out, g, center, template_bg, path_color, end_color, obstacle_cells
                )
            else:
                stamp_cells = _paint_baseline_collision(out, g, center, template_bg, end_color)
        stats["collision_centers"].append(center)
        stats["collision_kinds"].append(kind)
        stats["collision_stamp_sizes"].append(len(stamp_cells))
        stats["stamp_cells"].update(stamp_cells)

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


def _simulate_instance(parsed: dict, *, paint: bool, expanded_stamp: bool) -> tuple[np.ndarray | None, list[dict]]:
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
            expanded_stamp=expanded_stamp,
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
    fused_out, fused_stats = _simulate_instance(parsed, paint=True, expanded_stamp=True)
    baseline_out, baseline_stats = _simulate_instance(parsed, paint=True, expanded_stamp=False)
    assert fused_out is not None and baseline_out is not None

    obstacle_hit_cells: set[tuple[int, int]] = set()
    stamp_cells: set[tuple[int, int]] = set()
    max_stamp_size = 0
    max_obstacle_stamp_size = 0
    total_turns = 0
    collision_count = 0
    other_collision_count = 0
    for st in fused_stats:
        obstacle_hit_cells |= st["obstacle_hit_cells"]
        stamp_cells |= st["stamp_cells"]
        total_turns += int(st["turns"])
        collision_count += len(st["collision_centers"])
        for kind, size in zip(st["collision_kinds"], st["collision_stamp_sizes"]):
            max_stamp_size = max(max_stamp_size, int(size))
            if kind == "obstacle":
                max_obstacle_stamp_size = max(max_obstacle_stamp_size, int(size))
            elif kind == "other":
                other_collision_count += 1

    diff_cells = int(np.count_nonzero(fused_out != baseline_out))
    return {
        "path_color": parsed["path_color"],
        "obstacle_cells": list(parsed["obstacle_cells"]),
        "obstacle_count": len(parsed["obstacle_cells"]),
        "obstacle_hit_cells": obstacle_hit_cells,
        "obstacle_hit_count": len(obstacle_hit_cells),
        "total_turns": total_turns,
        "collision_count": collision_count,
        "other_collision_count": other_collision_count,
        "total_stamp_cells": len(stamp_cells),
        "max_stamp_size": max_stamp_size,
        "max_obstacle_stamp_size": max_obstacle_stamp_size,
        "baseline_diff_cells": diff_cells,
        "path_orders": [list(st["path_order"]) for st in fused_stats],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles in a thick path band with internal blockers and enlarged collision stamps.

    Input structure:
    - The upper-left area contains 2x2 monochrome blocks whose left-to-right order defines the repeating paint cycle.
    - One or two isolated marker pixels on col 0 use the first cycle color and sit inside the path band.
    - A separate 2x2 block gives the end color.
    - The path band also contains a few isolated obstacle pixels of another color embedded inside it.

    Rule:
    - Each marker launches a particle that advances through path cells and cycles through the template colors.
    - Any attempted step into a non-path cell is a collision, including both ordinary background and internal obstacle pixels.
    - After each collision, paint a local end-colored 3x3 stamp over nearby non-path space, then apply the strict alternating turn schedule CCW, CW, CCW, CW...
    - If the scheduled turn does not lead to a path cell, that particle stops; template blocks and the end block are erased to background in the output.

    Key insight:
    The useful state is not just the path boundary anymore: foreign-colored pixels inside the corridor are also hard walls, and every collision leaves a visibly enlarged local mark, so obstacle-triggered turns become part of the observable output pattern.
    """
    parsed = _parse_instance(grid, require_obstacles=True)
    out, _ = _simulate_instance(parsed, paint=True, expanded_stamp=True)
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


def _is_obstacle_site(g: np.ndarray, path_color: int, cell: tuple[int, int], bg: int) -> bool:
    r, c = cell
    H, W = g.shape
    if not (0 < r < H - 1 and 0 < c < W - 1):
        return False
    if int(g[r, c]) != path_color:
        return False
    orth = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    if any(int(g[rr, cc]) != path_color for rr, cc in orth):
        return False
    diag = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
    bg_diag = sum(int(g[rr, cc]) == bg for rr, cc in diag)
    if bg_diag < 1:
        return False
    win = g[r - 1 : r + 2, c - 1 : c + 2]
    return int(np.count_nonzero(win == path_color)) >= 5


def _place_obstacles(g: np.ndarray, obstacle_color: int) -> np.ndarray | None:
    g_cur = np.array(g, dtype=int)
    placed: list[tuple[int, int]] = []
    target = 2 if random.random() < 0.7 else 1

    for _ in range(target):
        metrics = _variant_metrics(g_cur, require_obstacles=False)
        path_color = int(metrics["path_color"])
        bg = int(g_cur[0, 0])

        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for order in metrics["path_orders"]:
            usable = order[2:-2] if len(order) > 6 else order[1:-1]
            for cell in usable:
                if cell in seen:
                    continue
                seen.add(cell)
                if _is_obstacle_site(g_cur, path_color, cell, bg):
                    candidates.append(cell)
        random.shuffle(candidates)

        accepted = False
        for r, c in candidates[:100]:
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
                trial_metrics = _variant_metrics(trial, require_obstacles=True)
            except Exception:
                continue

            needed = set(placed)
            needed.add((r, c))
            if not needed.issubset(trial_metrics["obstacle_hit_cells"]):
                continue
            if trial_metrics["other_collision_count"] != 0:
                continue
            if trial_metrics["max_obstacle_stamp_size"] < 2:
                continue
            if trial_metrics["max_stamp_size"] < 3:
                continue
            if trial_metrics["baseline_diff_cells"] < len(needed) + 2:
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
    if final_metrics["other_collision_count"] != 0:
        return None
    if final_metrics["max_stamp_size"] < 3:
        return None
    if final_metrics["baseline_diff_cells"] < len(placed) + 2:
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
    if metrics["other_collision_count"] != 0:
        return None
    if metrics["max_stamp_size"] < 3:
        return None
    if metrics["baseline_diff_cells"] < metrics["obstacle_count"] + 2:
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
