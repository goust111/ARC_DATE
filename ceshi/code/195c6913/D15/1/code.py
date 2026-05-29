# Variant of 195c6913
# Dimension: D15 — 多步处理/迭代 (multi_step)
# Change: 所有标记独立且同时模拟（单步处理），无顺序依赖或迭代。... → 碰撞后发射新粒子（链式反应）
#
from __future__ import annotations

# Design analysis
# a) Concrete behavior:
#    The original particle only painted path cells and left a single end-color
#    dot at each collision. In this variant, only the "would stop here" case is
#    changed: when a path particle collides with background and its scheduled
#    alternating turn is blocked, the collision cell becomes an end-color spark
#    and launches a straight end-color bridge across the background gap. If that
#    bridge hits path again, a new path particle restarts there in the same
#    direction and the template cycle resumes. Small example:
#      original: path ... -> dead-end collision -> one end-color corner dot -> stop
#      variant : path ... -> dead-end collision -> end-color bridge over fjord
#               -> hit opposite path wall -> continue cycling on path
#    This changes the processing stage, not the object vocabulary: template,
#    markers, path band, and end block are still read the same way.
# b) Complexity calibration:
#    A single extra dot would be too weak and could visually collapse back to
#    the original task. The bridge must span at least 2 background cells and
#    re-enter path at least once, so the output visibly contains a path-colored
#    segment, then an end-color bridge, then more path recoloring.
# c) Hidden assumptions review:
#    - Path detection still assumes the path is the most frequent non-bg color.
#      Generator keeps that invariant.
#    - Marker detection still assumes isolated pattern-colored pixels touching
#      path; generator keeps markers on col 0 and embedded vertically.
#    - Original code assumed a blocked turn means termination. That line must be
#      replaced: blocked turns now hand control to a bridge particle.
#    - Original disjointness only tracked path cells. Here overlaps could also
#      happen on bridge cells, so the solver tracks each marker's full painted
#      footprint (path, impact dots, and bridge cells) and generator rejects
#      overlapping cases.
# d) Placement legality:
#    The bridge is only allowed to travel on true background cells and may
#    re-enter only on path cells. Generator therefore keeps template blocks in
#    the top-left pocket and end block in the bottom-right pocket, away from the
#    bridge corridors, and uses fjords/islands inside the path band as the
#    intended jump gaps.
# e) Visual observability:
#    The new rule is visible without hidden color-id logic: dead-ends no longer
#    simply stop. They create an end-color line across a gap and often restart
#    coloring on another path segment. A human can infer this from 2-3 examples.
# f) Visual cleanliness:
#    The generator keeps trajectories disjoint and requires at least one clean
#    bridge/re-entry event, avoiding noisy piles of overlapping marks.
# g) Color inference:
#    The template still teaches the cycle colors. The end color now has a clear
#    structural meaning: collision dots and bridge rays. No sorting or behavior
#    depends on numeric color ids.
# Solver/generator contract:
#    - Every accepted sample must have at least one spawned bridge and at least
#      one bridge-to-path re-entry.
#    - The bridge moves only through background, so generator must provide a bg
#      fjord or enclosed bg island with path on both sides.

from collections import Counter
import sys
import random
import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker cell's vertical col-0 neighbors are path."""
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


def _parse_problem(
    g: np.ndarray,
) -> tuple[int, int, list[int], list[tuple[int, int]], int | None, tuple[int, int] | None, list[tuple[int, int, int]]]:
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
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(int(v))
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
    return int(template_bg), int(path_color), pattern, pat_cells, end_color, end_pos, markers


def _simulate_variant(
    g: np.ndarray,
    template_bg: int,
    path_color: int,
    pattern: list[int],
    pat_cells: list[tuple[int, int]],
    end_color: int | None,
    end_pos: tuple[int, int] | None,
    markers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[str, int], list[set[tuple[int, int]]]]:
    H, W = g.shape
    out = g.copy()

    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    stats = {
        "path_steps": 0,
        "spawn_count": 0,
        "bridge_cells": 0,
        "reentry_count": 0,
        "chain_segments": 0,
        "max_chain_segments": 0,
    }
    per_marker_cells: list[set[tuple[int, int]]] = []

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
            continue

        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        r, c = int(sr), int(sc)
        dr, dc = init
        mode = "path"
        marker_cells: set[tuple[int, int]] = set()
        visited: set[tuple[str, int, int, int, int, int, int]] = set()
        chain_segments = 1
        stats["chain_segments"] += 1
        steps_left = H * W * 12

        while steps_left > 0:
            steps_left -= 1
            state = (mode, int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if mode == "path":
                if int(g[nr, nc]) == int(path_color):
                    idx = int((idx + 1) % len(pattern))
                    out[nr, nc] = int(pattern[idx])
                    marker_cells.add((int(nr), int(nc)))
                    stats["path_steps"] += 1
                    r, c = int(nr), int(nc)
                    continue

                if end_color is not None:
                    out[nr, nc] = int(end_color)
                    marker_cells.add((int(nr), int(nc)))

                left = (-int(dc), int(dr))
                right = (int(dc), -int(dr))
                tdr, tdc = left if (int(hits) % 2 == 0) else right
                rr = int(r + tdr)
                cc = int(c + tdc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                    dr, dc = int(tdr), int(tdc)
                    hits += 1
                    continue

                if int(g[nr, nc]) == int(template_bg):
                    stats["spawn_count"] += 1
                    chain_segments += 1
                    stats["chain_segments"] += 1
                    mode = "bridge"
                    r, c = int(nr), int(nc)
                    continue
                break

            if int(g[nr, nc]) == int(template_bg):
                if end_color is not None:
                    out[nr, nc] = int(end_color)
                marker_cells.add((int(nr), int(nc)))
                stats["bridge_cells"] += 1
                r, c = int(nr), int(nc)
                continue

            if int(g[nr, nc]) == int(path_color):
                if end_color is not None:
                    out[nr, nc] = int(end_color)
                marker_cells.add((int(nr), int(nc)))
                stats["bridge_cells"] += 1
                stats["reentry_count"] += 1
                chain_segments += 1
                stats["chain_segments"] += 1
                mode = "path"
                r, c = int(nr), int(nc)
                continue

            if end_color is not None:
                out[nr, nc] = int(end_color)
            marker_cells.add((int(nr), int(nc)))
            stats["bridge_cells"] += 1
            break

        stats["max_chain_segments"] = max(stats["max_chain_segments"], int(chain_segments))
        per_marker_cells.append(marker_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, stats, per_marker_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate template-colored path walkers whose dead-end collisions emit bridge particles.

    Input structure:
    - Top-left: a row of monochrome 2x2 blocks encodes the repeating path-paint cycle.
    - Isolated single pixels of a cycle color sit on the left-edge path lobe as start markers.
    - A separate monochrome 2x2 block marks the terminal color used for impacts and bridge rays.

    Rule:
    - Each marker walks on path cells, recoloring them by advancing through the template cycle.
    - On each non-path hit, paint the impact cell with the terminal color and try the strict alternating turn rule (CCW, then CW, then CCW...).
    - If that scheduled turn is blocked, the path particle stops and launches a new terminal-color particle from the impact cell; this bridge moves straight through background cells, and if it strikes path it restarts the path walk from there in the same direction.

    Key insight:
    - The task is no longer a one-pass stop-at-corner simulation: dead-end collisions now create iterative bridge-and-restart events, so the visible output is a chain reaction of cycle-colored path segments linked by terminal-color rays.
    """
    g = np.array(grid, dtype=int)
    parsed = _parse_problem(g)
    out, _stats, _cells = _simulate_variant(g, *parsed)
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
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
    """Carve a bg fjord from col 0 between two marker rows."""
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


def _apply_gap_variant(mask: np.ndarray, markers: list[int]) -> np.ndarray | None:
    """Create a bg gap that can support a bridge-and-reentry event."""
    trial = mask.copy()
    if len(markers) == 1:
        mr = markers[0]
        r1 = max(2, mr - random.randint(3, 5))
        r2 = min(mask.shape[0] - 3, mr + random.randint(3, 5))
        if r2 - r1 < 5:
            r2 = min(mask.shape[0] - 3, r1 + 5)
        if not _carve_fjord(trial, r1, r2):
            return None
        restore = random.randint(2, 4)
        pre_carve_cells = mask & ~trial
        for c in range(min(restore, mask.shape[1])):
            trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        return trial

    if not _carve_fjord(trial, markers[0], markers[1]):
        return None
    if random.random() < 0.2:
        restore = random.randint(2, 4)
        pre_carve_cells = mask & ~trial
        for c in range(min(restore, mask.shape[1])):
            trial[:, c] |= pre_carve_cells[:, c]
    _, nc = label_mask(trial, connectivity=4)
    if nc != 1:
        return None
    return trial


def _variant_stats(g: np.ndarray) -> dict[str, int]:
    parsed = _parse_problem(np.array(g, dtype=int))
    _out, stats, _cells = _simulate_variant(np.array(g, dtype=int), *parsed)
    return stats


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2

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

    gap_mask = _apply_gap_variant(mask, markers)
    if gap_mask is None:
        return None
    for r in markers:
        if not (gap_mask[r, 0] and gap_mask[r - 1, 0] and gap_mask[r + 1, 0]):
            return None
    if not gap_mask[0, :].any() or not gap_mask[:, W - 1].any():
        return None
    if not gap_mask[H - 1, :].any() or not gap_mask[:, 0].any():
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
        stats = _variant_stats(g)
    except Exception:
        return None

    if stats["spawn_count"] < 1:
        return None
    if stats["reentry_count"] < 1:
        return None
    if stats["bridge_cells"] < 2:
        return None
    if stats["max_chain_segments"] < 3:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 800 attempts")
