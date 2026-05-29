# Variant of 195c6913
# Dimension: D5 — 终止条件 (dst)
# Change: 粒子在计划转向方向不是path时停止（collision_stop）；也在离开网格边界时停止。... → 完成一个完整颜色循环后停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D5 variant
# a) Concrete behavior and trace:
#    The top-left template still shows the color order as a row of 2x2 blocks.
#    In this variant the same row also encodes K: the number of blocks. If the
#    template is [A][B][C][D], then the particle paints the cycle
#    B,C,D,A | B,C,D,A | B,C,D,A | B,C,D,A and stops after exactly 16 path
#    cells, even if the path still continues. Collision dots in the end color
#    still appear whenever the particle probes into background, but those dots
#    are steering markers only; they no longer determine termination.
# b) Complexity calibration:
#    K is kept in a small visible range by sampling 3-4 blocks. That makes the
#    quota 9 or 16 painted cells, long enough to be seen and short enough that
#    the fixed-length stop is noticeably different from the old dead-end stop.
#    Cycle colors are sampled distinctly so humans can count the template row
#    without ambiguity.
# c) Old-code assumptions that break under D5:
#    The original loop used `break` on out-of-bounds and on blocked scheduled
#    turns. That was harmless because stopping at the first impossible move was
#    the intended rule. Under D5 we instead track painted-step progress and only
#    accept generated instances where every marker can reach the quota. The
#    visited-state key now includes progress when a paint limit is active, so a
#    legitimate small loop can still be traversed multiple times up to the quota.
# d) Placement legality:
#    The generator still reserves the top-left template area, embeds markers in
#    a col-0 path segment, places the end-color 2x2 block in isolated bottom-
#    right background, and keeps the path as one 4-connected component. In
#    addition it simulates each marker to guarantee quota completion and
#    pairwise-disjoint painted path cells.
# e) Visual observability:
#    Accepted samples must allow each marker to paint at least one extra cell
#    beyond the quota under the old motion rule. Therefore the output trails end
#    early because of the new progress-bound stop, not because the geometry ran
#    out in the same place as before.
# f) Visual cleanliness:
#    The path band and separator structure are preserved from the base task, so
#    template blocks, marker pixels, collision dots, and painted trails stay
#    spatially separated rather than piling up in one corner.
# g) Color inferability:
#    No sorting or behavior depends on numeric color ids. The only new count is
#    the visible number of template blocks, which humans can read directly from
#    shape repetition.


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


def _extract_pattern(
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
                return int(v), (int(r), int(c))
    return None, None


def _find_markers(g: np.ndarray, pat_set: set[int], path_color: int) -> list[tuple[int, int, int]]:
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


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _extract_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, template_bg, path_color, pat_set)
    markers = _find_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)

    return {
        "grid": g,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "pat_set": pat_set,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _initial_direction(g: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # R, D, L, U
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def _quota_steps(pattern: list[int]) -> int:
    return int(len(pattern) * len(pattern))


def _trace_marker(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
    marker: tuple[int, int, int],
    paint_limit: int | None,
    out: np.ndarray | None = None,
) -> dict[str, object]:
    sr, sc, sv = marker
    init = _initial_direction(g, sr, sc, path_color)
    info: dict[str, object] = {
        "init_found": init is not None,
        "cells": set(),
        "hits": 0,
        "steps": 0,
        "limit_complete": False,
        "stop_reason": "no_init" if init is None else "empty_pattern",
        "final_pos": (int(sr), int(sc)),
    }
    if init is None or not pattern:
        return info

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    steps = 0
    visited: set[tuple[int, ...]] = set()
    H, W = g.shape

    while True:
        if paint_limit is not None and steps >= paint_limit:
            info["limit_complete"] = True
            info["stop_reason"] = "quota"
            break

        state: tuple[int, ...]
        if paint_limit is None:
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        else:
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1), int(steps))
        if state in visited:
            info["stop_reason"] = "loop"
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            info["stop_reason"] = "edge"
            break

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            if out is not None:
                out[nr, nc] = int(pattern[idx])
            cast_cells = info["cells"]
            assert isinstance(cast_cells, set)
            cast_cells.add((int(nr), int(nc)))
            steps += 1
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(template_bg) and end_color is not None and out is not None:
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

        info["stop_reason"] = "blocked_turn"
        break

    info["hits"] = int(hits)
    info["steps"] = int(steps)
    info["final_pos"] = (int(r), int(c))
    return info


def _collect_traces(grid: np.ndarray, paint_limit: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    pattern = scene["pattern"]
    assert isinstance(g, np.ndarray)
    assert isinstance(pattern, list)
    traces = [
        _trace_marker(
            g=g,
            pattern=pattern,
            path_color=int(scene["path_color"]),
            template_bg=int(scene["template_bg"]),
            end_color=None if scene["end_color"] is None else int(scene["end_color"]),
            marker=marker,
            paint_limit=paint_limit,
            out=None,
        )
        for marker in scene["markers"]
    ]
    return scene, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each marker's trail for a template-coded number of full color loops.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks lists the cycle order, and the number of blocks also gives K.
    - Isolated cycle-colored pixels mark particle starts on the path band's left edge.
    - A separate 2x2 monochrome block provides the collision-dot color.

    Rule:
    - Each marker advances along the path, repainting visited path cells by stepping through the cycle.
    - When forward motion hits a non-path cell, the particle still drops an end-colored dot there and turns by the same strict alternating CCW/CW schedule.
    - The walk ends only after exactly K full passes through the cycle have been painted; collision geometry steers the particle but does not define the stopping point.

    Key insight:
    - The template length is used twice: once as the cycle length and once as the repeat count, so the visible trail length is fixed by color progress rather than by the first dead-end.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    pattern = scene["pattern"]
    pat_cells = scene["pat_cells"]
    end_pos = scene["end_pos"]
    template_bg = int(scene["template_bg"])
    quota = _quota_steps(pattern)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for marker in scene["markers"]:
        trace = _trace_marker(
            g=g,
            pattern=pattern,
            path_color=int(scene["path_color"]),
            template_bg=template_bg,
            end_color=None if scene["end_color"] is None else int(scene["end_color"]),
            marker=marker,
            paint_limit=quota,
            out=out,
        )
        assert bool(trace["init_found"]), f"marker {marker} must touch the path"
        assert bool(trace["limit_complete"]), (
            f"marker {marker} stopped before completing the template-coded quota: "
            f"{trace['stop_reason']}"
        )
        cells = trace["cells"]
        assert isinstance(cells, set)
        per_marker_cells.append(cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick meandering anti-diagonal path band."""
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
    """Carve a narrow background fjord from the left edge without disconnecting the path."""
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


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(3, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

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
            keep_cols = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(keep_cols, W)):
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

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        rr0, rr1 = template_row, template_row + 2
        cc0, cc1 = bc, bc + 2
        if (g[rr0:rr1, cc0:cc1] != bg).any():
            return None
        g[rr0:rr1, cc0:cc1] = cycle_colors[i]

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
        scene, quota_traces = _collect_traces(g, paint_limit=_quota_steps(cycle_colors))
        _, beyond_traces = _collect_traces(g, paint_limit=_quota_steps(cycle_colors) + 1)
        _ = main(g)
    except Exception:
        return None

    markers_found = scene["markers"]
    assert isinstance(markers_found, list)
    if len(markers_found) != n_markers:
        return None
    if not quota_traces or not beyond_traces:
        return None
    if not all(bool(t["limit_complete"]) for t in quota_traces):
        return None
    if not all(bool(t["limit_complete"]) for t in beyond_traces):
        return None
    if sum(int(t["hits"]) for t in quota_traces) < 1:
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
