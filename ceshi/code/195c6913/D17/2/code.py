# Variant of 195c6913
# Dimension: D17 — bg分隔条/峡湾形态 (obstacle)
# Change: n_markers=2时，两个标记的轨迹由bg分隔条分开：~75%为从col 0向右上延伸的峡湾，~25%为path内部... → 分隔条为可穿越的桥梁

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D17 bridge variant
# a) Concrete behavior:
#    The original two-marker instances use a thin bg separator to keep the
#    upper and lower marker channels completely apart. The variant keeps the
#    same elongated obstacle silhouette, but restores one full separator column
#    in its middle. Because the separator is 2-3 cells thick, restoring that
#    one-column slice creates a visibly narrow bridge through the obstacle.
#    Small trace example:
#      original fjord: upper marker moving right hits the fjord wall, paints an
#      end-color collision pixel, then must turn along its own side.
#      variant bridge: the same wall has a one-column opening farther along, so
#      a marker that reaches that row before its next forced turn continues
#      straight through the gap and recolors cells on the opposite side.
#      difference: the change is geometric/topological, not a hidden color rule.
# b) Complexity calibration:
#    Only one bridge gap is added, always narrow and always near the separator's
#    middle. This is enough to be visually salient without introducing multiple
#    arbitrary crossings. Generated samples are fixed to the two-marker case so
#    the bridge is always relevant; fjord vs interior-obstacle ratios stay close
#    to the source task so D17 is the only meaningful change.
# c) Old implicit assumptions that break:
#    - The original solver asserted pairwise-disjoint trajectories. That is no
#      longer a task invariant because the two channels are conditionally
#      connected. The solver must still simulate correctly when the opening
#      exists, so the disjointness assertion is removed.
#    - Collision handling must remain keyed to "next cell is non-path". The
#      bridge is therefore encoded as restored path cells; no extra symbol or
#      branch is introduced in the solver.
#    - Marker detection, template detection, end-block detection, and path
#      single-component checks all remain valid and are intentionally preserved.
# d) Placement legality:
#    The generator still builds a single 4-connected anti-diagonal path band,
#    carves a legal separator without disconnecting the path, optionally detaches
#    the separator from col 0 to get the interior case, and only then restores
#    one central separator slice as the bridge. Template blocks stay in the
#    cleared top-left bg zone; the end block stays in the bottom-right bg zone.
# e) Visual observability:
#    The bridge is a visible 1-column notch in the separator, and the generator
#    accepts only cases where that notch changes the simulated output relative to
#    the same mask without the notch. This prevents silent degeneration back to
#    the source task.
# f) Visual cleanliness:
#    The bridge is constrained away from the separator ends so the obstacle still
#    reads as a single long divider with a central break instead of scattered bg
#    clutter. The generator also keeps trajectories disjoint in accepted samples
#    so recolored outputs remain easy to read.
# g) Color-change inferability:
#    No new color semantics are introduced. The cycle blocks still define the
#    recolor order, the end block still defines the collision color, and the
#    bridge effect is inferred entirely from the obstacle geometry.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: col-0 markers sit inside a vertical path segment."""
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


def _parse_problem(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
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
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(pattern)
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
                end_color = v
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
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
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), v))

    _assert_marker_embedded_in_path(g, markers, path_color)

    return {
        "grid": g,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _trace_particles(info: dict[str, object]) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    g = np.array(info["grid"], dtype=int)
    H, W = g.shape
    template_bg = int(info["template_bg"])
    path_color = int(info["path_color"])
    pattern = list(info["pattern"])
    pat_cells = list(info["pat_cells"])
    end_color = info["end_color"]
    end_pos = info["end_pos"]
    markers = list(info["markers"])

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = template_bg

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    per_marker_cells: list[set[tuple[int, int]]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
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

            if int(g[nr, nc]) == path_color:
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == template_bg and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (hits % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)

    return out, per_marker_cells


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object], list[set[tuple[int, int]]]]:
    info = _parse_problem(grid)
    out, traces = _trace_particles(info)
    return out, info, traces


def _traces_disjoint(traces: list[set[tuple[int, int]]]) -> bool:
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            if traces[i] & traces[j]:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate cycle-colored particles on a single path band whose separator may contain a narrow bridge.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks defines the repeating paint cycle.
    - Left edge: one or two isolated marker pixels of a cycle color sit inside a col-0 path segment.
    - Elsewhere: a 2x2 block of a distinct color marks the collision color, and in two-marker cases the bg separator may have a one-column opening.

    Rule:
    - Each marker launches a particle along the first adjacent path cell in R,D,L,U priority, recoloring path cells by advancing through the cycle.
    - Whenever the next step is non-path, paint that collision cell with the end color and apply the strict alternating turn schedule CCW, CW, CCW, ...
    - If the scheduled turn is not a path cell, the particle stops; if a separator has a bridge opening, the particle simply keeps going straight through it because no collision occurs there.

    Key insight:
    - The bridge is encoded only by geometry: the deciding event is whether a would-be separator collision still exists or has been replaced by a tiny path opening.
    """
    out, _info, _traces = _solve_core(grid)
    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Build the thick anti-diagonal path band used by the source task."""
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


def _carve_separator(mask: np.ndarray, r1: int, r2: int) -> list[tuple[int, int, int]] | None:
    """Carve a thin elongated separator and return its per-column slices."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return None
    sr = random.randint(sr_min, sr_max)

    slices: list[tuple[int, int, int]] = []
    cur_top = sr
    for k in range(W - 1):
        trial = mask.copy()
        rows = []
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
                rows.append(rr)
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        slices.append((k, cur_top, sep_h))
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1

    if len(slices) < 6:
        return None
    return slices


def _restore_slices(mask: np.ndarray, slices: list[tuple[int, int, int]]) -> None:
    for c, top, h in slices:
        for dr in range(h):
            rr = top + dr
            if 0 <= rr < mask.shape[0]:
                mask[rr, c] = True


def _paint_grid(
    mask: np.ndarray,
    bg: int,
    path_color: int,
    template_row: int,
    block_cols: list[int],
    cycle_colors: list[int],
    marker_rows: list[int],
    *,
    end_color: int | None = None,
    end_pos: tuple[int, int] | None = None,
) -> np.ndarray:
    g = np.full(mask.shape, bg, dtype=int)
    g[mask] = path_color
    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]
    for r in marker_rows:
        g[r, 0] = cycle_colors[0]
    if end_color is not None and end_pos is not None:
        er, ec = end_pos
        g[er : er + 2, ec : ec + 2] = end_color
    return g


def _find_end_block(g: np.ndarray, bg: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _bridge_candidates(n_slices: int) -> list[int]:
    if n_slices < 3:
        return []
    center = (n_slices - 1) / 2
    lo = max(1, n_slices // 4)
    hi = min(n_slices - 2, (3 * n_slices) // 4)
    idxs = list(range(lo, hi + 1))
    if not idxs:
        idxs = list(range(1, n_slices - 1))
    idxs.sort(key=lambda i: (abs(i - center), random.random()))
    return idxs


def _search_bridge(
    mask_no_bridge: np.ndarray,
    active_slices: list[tuple[int, int, int]],
    build_grid,
) -> tuple[np.ndarray, dict[str, object]] | None:
    try:
        base_grid = build_grid(mask_no_bridge)
        base_out, _base_info, base_traces = _solve_core(base_grid)
    except Exception:
        return None
    if not _traces_disjoint(base_traces):
        return None

    for idx in _bridge_candidates(len(active_slices)):
        bridge_slice = active_slices[idx]
        bridge_mask = mask_no_bridge.copy()
        _restore_slices(bridge_mask, [bridge_slice])
        try:
            out, _info, traces = _solve_core(build_grid(bridge_mask))
        except Exception:
            continue
        if not _traces_disjoint(traces):
            continue
        bridge_cells = {
            (bridge_slice[1] + dr, bridge_slice[0])
            for dr in range(bridge_slice[2])
            if 0 <= bridge_slice[1] + dr < bridge_mask.shape[0]
        }
        if not any(trace & bridge_cells for trace in traces):
            continue

        extra_nonbridge = 0
        for i, trace in enumerate(traces):
            extra_nonbridge = max(extra_nonbridge, len((trace - base_traces[i]) - bridge_cells))
        output_delta = int(np.count_nonzero(out != base_out))
        if extra_nonbridge < 3 or output_delta < len(bridge_cells) + 3:
            continue

        return bridge_mask, {
            "bridge_slice": bridge_slice,
            "bridge_cells": bridge_cells,
            "bridge_extra_nonbridge": extra_nonbridge,
            "bridge_output_delta": output_delta,
        }

    return None


def _gen_attempt(return_meta: bool = False):
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
        marker_rows = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        marker_rows = [r1, r2]

    mask = _build_band(H, W, lobe_rows=marker_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in marker_rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    separator_mode = "none"
    bridge_meta = {
        "bridge_slice": None,
        "bridge_cells": set(),
        "bridge_extra_nonbridge": 0,
        "bridge_output_delta": 0,
    }
    if n_markers == 2:
        carved = mask.copy()
        slices = _carve_separator(carved, r1, r2)
        if slices is None:
            return None
        separator_mode = "fjord"
        active_slices = list(slices)

        if random.random() < 0.25:
            max_prefix = min(4, len(slices) - 5)
            if max_prefix < 2:
                return None
            prefix = random.randint(2, max_prefix)
            _restore_slices(carved, slices[:prefix])
            active_slices = slices[prefix:]
            separator_mode = "island"

        _, n_after_carve = label_mask(carved, connectivity=4)
        if n_after_carve != 1:
            return None
        for r in marker_rows:
            if not (carved[r, 0] and carved[r - 1, 0] and carved[r + 1, 0]):
                return None

        template_row = random.choice([1, 2])
        block_cols = [1, 4, 7, 10][:n_blocks]
        if block_cols[-1] + 1 >= W:
            return None
        for bc in block_cols:
            for dr in range(2):
                for dc in range(2):
                    rr, cc = template_row + dr, bc + dc
                    if rr >= H or cc >= W or carved[rr, cc]:
                        return None

        def build_partial(candidate_mask: np.ndarray) -> np.ndarray:
            g0 = _paint_grid(
                candidate_mask,
                bg,
                path_color,
                template_row,
                block_cols,
                cycle_colors,
                marker_rows,
            )
            _, n_path_after = label_mask(g0 == path_color, connectivity=4)
            if n_path_after != 1:
                raise ValueError("marker overwrite disconnected path")
            return g0

        search = _search_bridge(carved, active_slices, build_partial)
        if search is None:
            return None
        mask, bridge_meta = search
    else:
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

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    g = _paint_grid(mask, bg, path_color, template_row, block_cols, cycle_colors, marker_rows)
    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    end_pos = _find_end_block(g, bg, path_color)
    if end_pos is None:
        return None
    g = _paint_grid(
        mask,
        bg,
        path_color,
        template_row,
        block_cols,
        cycle_colors,
        marker_rows,
        end_color=end_color,
        end_pos=end_pos,
    )

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        out, _info, traces = _solve_core(g)
    except Exception:
        return None
    if n_markers == 2 and not _traces_disjoint(traces):
        return None

    if return_meta:
        return g, {
            "n_markers": n_markers,
            "separator_mode": separator_mode,
            "bridge_slice": bridge_meta["bridge_slice"],
            "bridge_extra_nonbridge": bridge_meta["bridge_extra_nonbridge"],
            "bridge_output_delta": bridge_meta["bridge_output_delta"],
            "used_bridge": bool(bridge_meta["bridge_cells"]),
            "output_shape": out.shape,
        }
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
