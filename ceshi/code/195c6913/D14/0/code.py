# Variant of 195c6913
# Dimension: D14 — 终止块结构 (meta)
# Change: 终止块为单个2x2单色块（end_color），位于右下bg区域，所有碰撞角标记都使用此颜色。... → 无终止块——碰撞角色由循环色决定

# Design analysis for the D14 variant
# a) Concrete behavior trace:
#    Input now contains only three visible ingredients: the path band, the top-left
#    row of 2x2 template blocks, and 1-2 isolated start markers on col 0. The old
#    right-bottom 2x2 end block is removed entirely, so there is no dedicated
#    "corner color object" to discover. The path walk stays the same except at a
#    collision: previously a marker walking with cycle [A,B,C] would paint path
#    cells B,C,A,... and every in-bounds corner cell would become a fixed end color E.
#    In the variant, if the particle has just painted C and then faces a wall, the
#    corner cell is also painted C. The turn direction sequence is still CCW/CW/CCW...
#    and stopping still depends on whether the scheduled turn leads into path.
# b) Complexity calibration:
#    Removing the end block simplifies object inventory, so the generator compensates
#    by making cycle block colors distinct and by rejecting samples with no visible
#    corner paint. This keeps the new rule inferable: the only colors available for
#    corner marks are already shown in the template cycle, and corner colors vary with
#    the walk phase instead of a separate constant object.
# c) Hidden assumptions from the original solver:
#    - `end_color/end_pos` search assumed a unique non-template 2x2 object existed.
#      That is false now, so the search and erase logic must be deleted.
#    - `out[nr,nc] = end_color` was harmless before because all corners shared one
#      constant color. In the variant this must depend on the current cycle index,
#      otherwise repeated unconditional assignment would destroy the phase signal.
#    - Marker detection still assumes markers are isolated template-colored single
#      pixels adjacent to path; the generator keeps this unchanged.
#    - Path detection still assumes `g[0,0]` is the template-region background and
#      path is the most frequent non-bg color; the generator still guarantees both.
# d) Placement legality:
#    Template blocks remain in top-left bg, separated by bg columns exactly as before.
#    Start markers still overwrite col-0 path cells whose vertical neighbors are path.
#    Because the end block is removed, the generator only has to ensure the path does
#    not invade template space and that trajectories/corner paints remain valid under
#    the solver's assertions.
# e) Visual observability:
#    Across examples, solvers can no longer rely on spotting a distinct 2x2 end block.
#    Instead they see that collision pixels use template colors and that these colors
#    can differ between corners. The generator therefore rejects samples with zero
#    corner paints so the new mechanism is present in the training signal.
# f) Visual cleanliness:
#    The band, template, and markers still occupy separated regions. Corner paints are
#    sparse single pixels on bg cells near bends, so removing the end block actually
#    makes the canvas cleaner rather than noisier.
# g) Color inferability:
#    Corner colors come only from the small template palette, never from raw color-id
#    ordering or a hidden lookup. Distinct template colors avoid ambiguous repeated
#    blocks and make "corner uses current cycle phase" visually recoverable.
# New solver assumptions to audit against the generator:
#    - At least one 2x2 template block exists and is detectable in the top-left search
#      window. The generator keeps 2-4 blocks at rows 1/2 and cols 1,4,7,10.
#    - There are no other non-path 2x2 monochrome distractors that would be mistaken
#      for template blocks. The generator only places the template as 2x2 objects.
#    - Corner paints are only written onto bg cells. The generator keeps template
#      distant from the traversed band, so collision cells are ordinary bg corners.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker sits inside a vertical path segment."""
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


def _solve_core(
    grid: np.ndarray, *, collect_info: bool = False
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _find_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    markers = _find_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    per_marker_cells: list[set[tuple[int, int]]] = []
    corner_cells: list[tuple[int, int]] = []
    corner_colors: list[int] = []
    turns_per_marker: list[int] = []

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
            turns_per_marker.append(0)
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

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg):
                out[nr, nc] = int(pattern[idx])
                corner_cells.append((int(nr), int(nc)))
                corner_colors.append(int(pattern[idx]))

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
        turns_per_marker.append(int(hits))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    info = {
        "pattern": pattern,
        "markers": markers,
        "corner_cells": corner_cells,
        "corner_colors": corner_colors,
        "corner_count": len(corner_cells),
        "corner_color_count": len(set(corner_colors)),
        "turn_count": sum(turns_per_marker),
        "turns_per_marker": turns_per_marker,
    }
    return (out, info) if collect_info else (out, {})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles whose corner marks inherit the cycle phase.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the repeating color cycle.
    - Col 0: 1-2 isolated template-colored start pixels are embedded in the path band.
    - No separate end block exists; the remaining non-template mass is the path region.

    Rule:
    - Each start pixel launches a particle in the first adjacent path direction (R,D,L,U priority).
    - Path cells are recolored by advancing through the template cycle.
    - When the next cell is non-path, the collision cell is painted with the particle's current cycle color, then the particle attempts the scheduled 90-degree turn.
    - Turn directions alternate strictly CCW, CW, CCW, CW; if the scheduled turn is not path, the particle stops.

    Key insight:
    - The geometry alone is not enough: without a dedicated end color block, the solver must track the cycle phase so corner pixels reuse the same phase color that the particle is currently carrying.
    """
    out, _ = _solve_core(grid, collect_info=False)
    return out


# --------------- generator (markers-first SAv4_mine, no end block) ---------------
# Markers-first construction: pick markers, build band with lobe to embed them,
# optionally carve fjord→island to create an interior separator, place only the
# template blocks, then run the variant solver as an oracle and reject samples
# with no visible phase-colored corner paints.


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
            c_right[r] = int(
                round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2)
            )
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
    """Carve a bg fjord from col 0 without disconnecting the band."""
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
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[2:], n_blocks)

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
        use_interior = random.random() < 0.25
        if use_interior:
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
        _, info = _solve_core(g, collect_info=True)
    except Exception:
        return None

    if int(info["corner_count"]) < 1:
        return None
    if int(info["turn_count"]) < max(1, n_markers):
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
