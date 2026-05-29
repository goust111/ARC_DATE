# Variant of 195c6913
# Dimension: D4 — 源对象：标记像素定义 (src)
# Change: 1-2个孤立的单像素标记，颜色=循环色[0]，位于col 0（左网格边缘），嵌入在path区段中；初始方向优先级为R,D... → 箭头形状标记编码初始方向

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete marker geometry and behavior trace
# - Original task: a marker was a single cycle-colored pixel on col 0. The solver
#   recovered the initial direction by probing path neighbors in fixed priority
#   R,D,L,U, which effectively meant "go right first" on all legal inputs.
# - Variant: a marker is now a 3-pixel arrowhead component of cycle_colors[0].
#   We use the four visually obvious 8-connected shapes `> < ^ v`:
#     R: tip (r,c), wings (r-1,c-1),(r+1,c-1)
#     L: tip (r,c), wings (r-1,c+1),(r+1,c+1)
#     U: tip (r,c), wings (r+1,c-1),(r+1,c+1)
#     D: tip (r,c), wings (r-1,c-1),(r-1,c+1)
#   The tip cell is the particle's start cell; the decoded arrow direction is the
#   initial motion direction. Example: if a `^` marker sits at tip (15,4), the
#   first step must target (14,4), even if the old R,D,L,U probe would have chosen
#   (15,5). That is the key behavioral difference from the original solver.
# b) Complexity calibration
# - A 3-cell 2x2 L-tromino is only a corner and does not naturally distinguish
#   between its two arm directions, so it is visually ambiguous for ARC. The
#   arrowhead shapes above keep the D4 change obvious while adding only one local
#   decoding rule.
# - Marker color remains cycle_colors[0], so the first template block still
#   explains which cycle phase the trajectory starts from.
# c) Hidden-assumption audit against the original main()
# - `_assert_marker_embedded_in_path(...)` depended on markers always living on
#   col 0 and having vertical path neighbors. That is no longer valid and is
#   replaced by `_assert_arrow_markers_valid(...)`, which checks shape decodability
#   and that the tip's forward cell is path.
# - The old marker-detection loop assumed size-1 components (`same == 0`) and only
#   needed a path-adjacency witness. That would miss all arrow markers, so the
#   variant detects 8-connected size-3 components in cycle colors and decodes them.
# - `dir_order = [R,D,L,U]` was harmless before because the left-edge inputs made
#   "right" the intended launch. In the variant that would overwrite the shape
#   signal, so the initial direction is taken exclusively from the decoded arrow.
# d) Placement legality
# - Generator markers are placed only on ambient path cells, away from template and
#   end blocks, with all 3 arrow cells inside bounds and the tip's forward cell
#   still path-colored. Markers of the same color are kept Chebyshev-separated so
#   they remain distinct 8-connected components.
# - After marker painting, the remaining path_color cells must still form one
#   4-connected component, matching the solver's invariant.
# e) Visual observability
# - The new rule is visible from input alone: the source objects are no longer
#   edge pixels but arrowheads whose orientations vary across samples. The solver
#   leaves the arrowheads in place in the output and launches the painted trail
#   from the tip, so the mapping "shape -> first travel direction" is readable.
# - To prevent regression to the original rule, the generator requires at least one
#   marker per grid whose decoded direction differs from the old R,D,L,U neighbor
#   priority at the same tip cell.
# f) Visual cleanliness
# - Markers are still concentrated in the left-side path lobe created by the
#   original generator, so they stay spatially grouped and do not clutter the
#   template or end-block regions. Two-marker cases retain the fjord/island
#   separator so the two trajectories are usually visually disentangled.
# g) Color inference
# - No color-id arithmetic or invisible ordering is introduced. The only color rule
#   remains the original one: template blocks define the cycle, the marker uses the
#   first cycle color, and collisions stamp end_color.
# New solver assumptions introduced by this variant
# - Each marker must be an 8-connected component of exactly 3 cells in a template
#   color and must match exactly one arrowhead orientation.
# - The arrow tip must have a forward ambient path cell, because motion now depends
#   on the decoded shape rather than a fallback adjacency search.
# - The generator guarantees those assumptions directly when sampling markers.

DIR4 = {
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
    "U": (-1, 0),
}
DIR_ORDER = ["R", "D", "L", "U"]
ARROW_SHAPES = {
    "R": ((0, 0), (-1, -1), (1, -1)),
    "L": ((0, 0), (-1, 1), (1, 1)),
    "U": ((0, 0), (1, -1), (1, 1)),
    "D": ((0, 0), (-1, -1), (-1, 1)),
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _arrow_cells(tip_r: int, tip_c: int, direction: str) -> list[tuple[int, int]]:
    return [
        (int(tip_r + dr), int(tip_c + dc))
        for dr, dc in ARROW_SHAPES[str(direction)]
    ]


def _decode_arrow_component(coords: np.ndarray) -> tuple[int, int, str] | None:
    """Return (tip_r, tip_c, direction) for a 3-cell arrowhead component."""
    if int(coords.shape[0]) != 3:
        return None
    comp = {(int(r), int(c)) for r, c in coords}
    matches: list[tuple[int, int, str]] = []
    for direction in DIR_ORDER:
        for tip_r, tip_c in comp:
            if set(_arrow_cells(int(tip_r), int(tip_c), direction)) == comp:
                matches.append((int(tip_r), int(tip_c), direction))
    if len(matches) != 1:
        return None
    return matches[0]


def _find_arrow_markers(
    g: np.ndarray, pattern: list[int], path_color: int
) -> list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]]:
    """Find 3-cell arrow markers in pattern colors."""
    H, W = g.shape
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]] = []
    for color in sorted(set(map(int, pattern))):
        lab, n_comp = label_mask(g == int(color), connectivity=8)
        for k in range(1, int(n_comp) + 1):
            coords = np.argwhere(lab == k)
            decoded = _decode_arrow_component(coords)
            if decoded is None:
                continue
            tip_r, tip_c, direction = decoded
            dr, dc = DIR4[direction]
            nr, nc = int(tip_r + dr), int(tip_c + dc)
            assert 0 <= nr < H and 0 <= nc < W, (
                f"marker tip at ({tip_r},{tip_c}) points {direction} out of bounds"
            )
            assert int(g[nr, nc]) == int(path_color), (
                f"marker tip at ({tip_r},{tip_c}) points to ({nr},{nc})="
                f"{int(g[nr, nc])}, expected path_color {path_color}"
            )
            cells = tuple(sorted((int(r), int(c)) for r, c in coords.tolist()))
            markers.append((int(tip_r), int(tip_c), int(color), direction, cells))
    markers.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
    return markers


def _assert_arrow_markers_valid(
    g: np.ndarray,
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]],
    path_color: int,
) -> None:
    H, W = g.shape
    assert 1 <= len(markers) <= 2, f"expected 1-2 arrow markers, got {len(markers)}"
    used: set[tuple[int, int]] = set()
    for tip_r, tip_c, color, direction, cells in markers:
        assert len(cells) == 3
        assert tuple(sorted(_arrow_cells(tip_r, tip_c, direction))) == cells, (
            f"marker at tip ({tip_r},{tip_c}) does not match decoded direction {direction}"
        )
        dr, dc = DIR4[direction]
        nr, nc = int(tip_r + dr), int(tip_c + dc)
        assert 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color)
        for r, c in cells:
            assert int(g[r, c]) == int(color)
            assert (int(r), int(c)) not in used, "marker cells must be disjoint"
            used.add((int(r), int(c)))


def _trace_marker(
    g: np.ndarray,
    out: np.ndarray,
    sr: int,
    sc: int,
    sv: int,
    init_dir: str,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
) -> set[tuple[int, int]]:
    """Run one particle trace and paint `out` in place."""
    H, W = g.shape
    r, c = int(sr), int(sc)
    dr, dc = DIR4[str(init_dir)]
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

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            out[nr, nc] = int(pattern[idx])
            my_cells.add((int(nr), int(nc)))
            r, c = nr, nc
            continue

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

    return my_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles on the path, but read each particle's launch direction from an arrow marker.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks defines the repeating paint cycle.
    - Start markers are 3-cell arrowheads in the first cycle color, embedded in the path region.
    - A separate 2x2 monochrome block gives the collision/end color.

    Rule:
    - Each arrowhead launches one particle from its tip, moving in the direction encoded by the shape.
    - While the next cell is path, the particle advances and paints cycle colors in order.
    - On a non-path collision, that cell is marked with the end color; the particle then alternates
      strict CCW and CW turns, and stops immediately if the scheduled turn is blocked.

    Key insight: The path dynamics are unchanged; the new challenge is to recover the initial heading
    from local marker geometry instead of using the old fixed neighbor-priority launch.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [
                (cnt, int(col))
                for col, cnt in counts.items()
                if int(col) != int(template_bg)
            ]
        )[1]
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
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
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

    markers = _find_arrow_markers(g, pattern, path_color)
    _assert_arrow_markers_valid(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv, init_dir, _cells in markers:
        if not pattern:
            per_marker_cells.append(set())
            continue
        per_marker_cells.append(
            _trace_marker(
                g,
                out,
                int(sr),
                int(sc),
                int(sv),
                str(init_dir),
                pattern,
                int(path_color),
                int(template_bg),
                end_color,
            )
        )

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


# --------------- generator (arrow-markers variant) ---------------
# Build the same meandering path / separator family as the original task, then
# replace the col-0 single-pixel markers with interior arrowhead markers whose
# directions are decoded by the solver.


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
    """Carve a bg separator from the left side without disconnecting the band."""
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    cur_top = random.randint(sr_min, sr_max)
    made_carve = False
    for k in range(mask.shape[1] - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = int(cur_top + dr)
            if 0 <= rr < mask.shape[0]:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _priority_dir(mask: np.ndarray, r: int, c: int) -> str | None:
    H, W = mask.shape
    for direction in DIR_ORDER:
        dr, dc = DIR4[direction]
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
            return direction
    return None


def _chebyshev_separated(
    cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]
) -> bool:
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return False
    return True


def _collect_arrow_candidates(
    mask: np.ndarray, target_row: int
) -> list[dict[str, object]]:
    H, W = mask.shape
    col_hi = max(5, W // 3)
    candidates: list[dict[str, object]] = []
    row_ranges = [
        range(max(1, target_row - 3), min(H - 1, target_row + 4)),
        range(1, H - 1),
    ]
    for row_range in row_ranges:
        candidates.clear()
        for direction in DIR_ORDER:
            dr, dc = DIR4[direction]
            for r in row_range:
                for c in range(1, min(W - 1, col_hi + 1)):
                    cells = _arrow_cells(int(r), int(c), direction)
                    if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in cells):
                        continue
                    if not all(bool(mask[rr, cc]) for rr, cc in cells):
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc])):
                        continue
                    local = mask[
                        max(0, r - 1) : min(H, r + 2),
                        max(0, c - 1) : min(W, c + 2),
                    ]
                    if int(local.sum()) < 5:
                        continue
                    priority = _priority_dir(mask, int(r), int(c))
                    candidates.append(
                        {
                            "tip": (int(r), int(c)),
                            "direction": direction,
                            "cells": tuple(sorted(cells)),
                            "interesting": priority != direction,
                            "row_gap": abs(int(r) - int(target_row)),
                            "col": int(c),
                        }
                    )
        if candidates:
            random.shuffle(candidates)
            candidates.sort(
                key=lambda cand: (
                    0 if bool(cand["interesting"]) else 1,
                    int(cand["row_gap"]),
                    int(cand["col"]),
                )
            )
            return candidates[:60]
    return []


def _place_arrow_markers(
    g: np.ndarray, mask: np.ndarray, target_rows: list[int], marker_color: int
) -> list[dict[str, object]] | None:
    cand_lists = [_collect_arrow_candidates(mask, int(row)) for row in target_rows]
    if any(not cands for cands in cand_lists):
        return None

    if len(cand_lists) == 1:
        singles = [cand for cand in cand_lists[0] if bool(cand["interesting"])]
        if not singles:
            return None
        choice = random.choice(singles[: min(20, len(singles))])
        for r, c in choice["cells"]:
            g[int(r), int(c)] = int(marker_color)
        return [choice]

    pairs: list[dict[str, object]] = []
    for cand_a in cand_lists[0]:
        for cand_b in cand_lists[1]:
            if not _chebyshev_separated(cand_a["cells"], cand_b["cells"]):
                continue
            interesting_count = int(bool(cand_a["interesting"])) + int(bool(cand_b["interesting"]))
            if interesting_count == 0:
                continue
            pair_dirs = {str(cand_a["direction"]), str(cand_b["direction"])}
            pairs.append(
                {
                    "a": cand_a,
                    "b": cand_b,
                    "interesting_count": interesting_count,
                    "has_r": "R" in pair_dirs,
                    "row_gap": int(cand_a["row_gap"]) + int(cand_b["row_gap"]),
                    "col_sum": int(cand_a["col"]) + int(cand_b["col"]),
                }
            )
    if not pairs:
        return None
    random.shuffle(pairs)
    pairs.sort(
        key=lambda item: (
            -int(item["interesting_count"]),
            int(item["row_gap"]),
            int(item["col_sum"]),
        )
    )
    r_pairs = [pair for pair in pairs if bool(pair["has_r"])]
    top_pool = pairs[: min(12, len(pairs))]
    if r_pairs and random.random() < 0.35:
        top_pool = r_pairs[: min(12, len(r_pairs))]
    pick = random.choice(top_pool)
    cand_a = pick["a"]
    cand_b = pick["b"]
    for cand in [cand_a, cand_b]:
        for r, c in cand["cells"]:
            g[int(r), int(c)] = int(marker_color)
    return [cand_a, cand_b]


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
        target_rows = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        target_rows = [r1, r2]

    mask = _build_band(H, W, lobe_rows=target_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
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
                rr = int(template_row + dr)
                cc = int(bc + dc)
                if rr >= H or cc >= W or bool(mask[rr, cc]):
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = int(end_color)

    placed = _place_arrow_markers(g, mask, target_rows, int(cycle_colors[0]))
    if placed is None:
        return None

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    if not any(bool(cand["interesting"]) for cand in placed):
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
