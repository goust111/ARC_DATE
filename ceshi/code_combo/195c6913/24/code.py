# Combo variant of 195c6913 (index 24)
# combo_spec: D5/2+D13/1
# Name: 完成循环后停止+模板变色为终止循环色

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import label_mask

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    We freeze the scene to one left-edge marker and a 3-4 block template row.
#    Example: template [A][B][C], marker color A. The old task would erase the
#    template, walk until the first blocked scheduled turn, and leave only the
#    trail plus corner dots. Here K is the visible template length, so the
#    marker paints exactly K*K=9 path cells:
#      B,C,A | B,C,A | B,C,A
#    then stops even if the path still continues. The last cycle color is
#    therefore A again, so instead of erasing the template we repaint every
#    template block to A. The end block is still erased and collision dots still
#    appear during steering.
# b) Complexity calibration:
#    Using exactly one marker removes template↔marker ambiguity from D13, while
#    keeping 3-4 distinct template blocks preserves a long enough quota (9 or
#    16 cells) for D5 to be visually obvious. Two-block templates would be too
#    weak because the final color would often make the template look unchanged.
# c) Hidden assumptions checked against main():
#    - Quota stopping breaks the old assumption that edge/blocked-turn always
#      terminates the walk, so trace state now includes step count when a quota
#      is active.
#    - Template recolor must use the last cycle color written on path cells, not
#      the end-color collision dot.
#    - Recoloring all template blocks assumes a single marker; the generator
#      enforces that explicitly.
#    - The solver still assumes the detected path is the dominant non-bg color,
#      the template occupies isolated 2x2 blocks, and the marker sits inside a
#      col-0 vertical path segment; the generator preserves all of these.
# d) Placement legality:
#    The generator keeps the base top-left template lane clear, embeds the lone
#    marker in the col-0 lobe of a single 4-connected path band, and places the
#    end block in isolated bottom-right background so no objects overlap.
# e) Visual observability:
#    A human can count the template blocks to recover K, see that the trail ends
#    after exactly K full color loops rather than at the first dead-end, and see
#    the entire template row become the marker's terminal cycle color.
# f) Visual cleanliness:
#    Only one marker is active, so the trail, collision dots, template summary,
#    and erased end block remain easy to separate spatially.
# g) Color inferability:
#    The final template color is not chosen by numeric ids. It is the visibly
#    repeated terminal cycle phase of the lone trajectory, and multiple template
#    blocks changing to the same color provide redundant evidence.


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
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1, c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1, c])} "
                f"must be path_color {path_color}"
            )


def _extract_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    pattern: list[int] = []
    template_blocks: list[tuple[int, int, int]] = []
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
                template_blocks.append((int(r), int(c), int(v)))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, template_blocks, pat_cells


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

    pattern, template_blocks, pat_cells = _extract_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, template_bg, path_color, pat_set)
    markers = _find_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)

    return {
        "grid": g,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "template_blocks": template_blocks,
        "pat_cells": pat_cells,
        "pat_set": pat_set,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _initial_direction(g: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
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
        "last_cycle_color": int(sv),
    }
    if init is None or not pattern:
        return info

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    steps = 0
    last_cycle_color = int(sv)
    visited: set[tuple[int, ...]] = set()
    H, W = g.shape

    while True:
        if paint_limit is not None and steps >= paint_limit:
            info["limit_complete"] = True
            info["stop_reason"] = "quota"
            break

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
            last_cycle_color = int(pattern[idx])
            if out is not None:
                out[nr, nc] = int(last_cycle_color)
            cells = info["cells"]
            assert isinstance(cells, set)
            cells.add((int(nr), int(nc)))
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
    info["last_cycle_color"] = int(last_cycle_color)
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
    Paint a left-edge particle for a template-coded number of full color loops, then summarize its terminal cycle phase in the template.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks lists the cycle order, and the number of blocks also serves as the loop-count code K.
    - Left edge: one isolated cycle-colored marker sits inside a col-0 path segment and launches the particle.
    - Elsewhere: a distinct 2x2 end block provides the collision-dot color while the dominant non-background band provides the walkable path.

    Rule:
    - The marker advances along the path, recoloring visited path cells by stepping through the template cycle.
    - On each non-path probe into background, that collision cell still receives the end color, and turning still alternates strictly CCW, CW, CCW, ...
    - The walk stops only after exactly K complete passes through the cycle have been painted, even if the path could continue.
    - The end block is erased, but the template blocks are kept and repainted to the last cycle color written by the trajectory.

    Key insight:
    - The same visible template row controls both when the particle stops and what color the template becomes: counting its blocks gives the paint quota, and completing whole cycles makes the terminal template color predictable.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    pattern = scene["pattern"]
    template_blocks = scene["template_blocks"]
    end_pos = scene["end_pos"]
    template_bg = int(scene["template_bg"])
    markers = scene["markers"]
    quota = _quota_steps(pattern)

    assert len(markers) == 1, f"combo generator expects exactly one marker, got {len(markers)}"

    out = g.copy()
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    final_template_color = None
    for marker in markers:
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
        final_template_color = int(trace["last_cycle_color"])

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    assert final_template_color is not None
    for r, c, _v in template_blocks:
        out[int(r) : int(r + 2), int(c) : int(c + 2)] = int(final_template_color)

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


_GEN_CALL_COUNT = 0


def _gen_attempt(target_blocks: int | None = None) -> np.ndarray | None:
    H = W = random.randint(20, 30)
    marker_row = random.randint(H // 2, H - 4)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = int(target_blocks) if target_blocks is not None else random.randint(3, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    mask = _build_band(H, W, lobe_rows=[marker_row])
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    if not (mask[marker_row, 0] and mask[marker_row - 1, 0] and mask[marker_row + 1, 0]):
        return None

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

    start_idx = random.randrange(n_blocks)
    start_color = int(cycle_colors[start_idx])
    g[marker_row, 0] = start_color

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if int(col) != int(bg)], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    quota = _quota_steps(cycle_colors)
    try:
        scene, quota_traces = _collect_traces(g, paint_limit=quota)
        _, beyond_traces = _collect_traces(g, paint_limit=quota + 1)
        out = main(g)
    except Exception:
        return None

    markers_found = scene["markers"]
    assert isinstance(markers_found, list)
    if len(markers_found) != 1:
        return None
    if not quota_traces or not beyond_traces:
        return None

    trace = quota_traces[0]
    extra_trace = beyond_traces[0]
    if not bool(trace["limit_complete"]):
        return None
    if not bool(extra_trace["limit_complete"]):
        return None
    if int(trace["hits"]) < 1:
        return None
    if int(trace["last_cycle_color"]) != start_color:
        return None

    changed_blocks = 0
    for bc, orig_color in zip(block_cols, cycle_colors):
        block = out[template_row : template_row + 2, bc : bc + 2]
        if not np.all(block == start_color):
            return None
        if int(orig_color) != start_color:
            changed_blocks += 1
    if changed_blocks < 2:
        return None

    return g


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_blocks = 3 if (_GEN_CALL_COUNT % 2 == 0) else 4
    _GEN_CALL_COUNT += 1

    for _ in range(180):
        try:
            g = _gen_attempt(target_blocks=preferred_blocks)
        except Exception:
            g = None
        if g is not None:
            return g

    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
