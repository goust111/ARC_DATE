# Variant of 195c6913
# Dimension: D6 — 碰撞变换：终止色涂角标记 (transform)
# Change: 碰撞非path格时（且该格为bg），用终止色（end_color）涂该格作为角标记；同时擦除模板2x2块和终止2x2块。... → 碰撞时用碰撞次数对应的循环色涂角

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

from arc_utils import label_mask

"""
design_analysis:
  a_specific_behavior:
    input_objects:
      - top-left has 2-4 monochrome 2x2 blocks; left-to-right order is the cycle
      - col 0 contains 1-2 isolated start pixels using cycle[0], each embedded in the path band
      - bottom-right has one 2x2 end block whose only job is still "identify and erase this block"
      - the path is a single thick anti-diagonal component
    output_change:
      - path traversal coloring is unchanged: each forward path step advances through the cycle
      - collision coloring changes: the kth collision corner uses cycle[(k-1) mod len(cycle)]
        instead of the uniform end_color
      - example trace:
        cycle = [2, 7, 4]
        a marker of color 2 walks along path cells and paints 7, then 4, then 2, ...
        at its first wall hit, the bg corner cell is painted 2
        after the forced turn, the second wall hit paints 7
        under the original rule both corners would have been the same end_color
    affected_stage:
      - solver collision painting
      - solver loop-state encoding, because collision order now affects future visible output
      - generator observability checks, because 0-1 collisions or repeated cycle colors would hide D6
  b_complexity_calibration:
    - keep the geometry, marker counts, fjord/island separator, and alternating CCW/CW rule unchanged
    - enforce distinct cycle colors so corner-order coding is visually legible
    - require at least 2 collision events and at least 2 distinct corner colors in generated samples
    - keep cycle length small (2-4) so humans can infer the repeating corner-color schedule
  c_implicit_assumption_review:
    - original visited state stored only hit parity; that was sufficient when every corner was end_color
    - in this variant, two states with same position/direction/path-paint index/parity but different
      hit-count modulo cycle length would paint different future corners, so visited state must also
      track hit_count mod len(cycle)
    - original constant-color corners made repeated corner-cell overwrites harmless; now they would erase
      sequence information, so the generator rejects samples where two collision events target one cell
    - erasing the template and end block still happens before simulation; this remains valid because
      movement decisions still read from the untouched input grid, not from the evolving output
  d_placement_legality:
    - template blocks stay in the reserved top-left bg area
    - the end block stays in the bottom-right bg region with a 1-cell moat from the path
    - markers remain on col 0 and overwrite path cells only after the path mask is built
    - generator uses the solver as oracle and additionally checks collision metrics before accepting
  e_visual_observability:
    - the human-visible novelty is "corner colors encode hit order"
    - this is only learnable if multiple corners exist, so generation rejects samples with <2 collisions
    - corner colors come directly from the visible template order, never from hidden color-id sorting
  f_visual_cleanliness:
    - disjoint path trajectories are preserved from the original task
    - generator also keeps collision corner cells unique, preventing messy multi-event overwrites
    - template and end blocks are erased in the output, so only path paint and corner markers remain
  g_color_inference:
    - the same small cycle controls both along-path repainting and corner-order colors
    - using distinct template colors provides repeated evidence for the mapping from block order to
      collision order without introducing extra color vocabularies
  solver_generator_contract:
    - main() assumes the path is one component, markers are embedded in col 0 path, and the template
      cycle is non-empty; generate_input() explicitly constructs and verifies those conditions
    - main() can tolerate repeated corner-cell hits by overwriting, but generate_input() filters them out
      because they would weaken the D6 signal
"""


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker is vertically embedded in the path."""
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


def _parse_scene(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    list[int],
    list[tuple[int, int]],
    int | None,
    tuple[int, int] | None,
    list[tuple[int, int, int]],
]:
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            (cnt, int(col))
            for col, cnt in counts.items()
            if int(col) != int(template_bg)
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
    return template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def _run_simulation(
    grid: np.ndarray,
    *,
    corner_mode: str,
) -> tuple[
    np.ndarray,
    list[set[tuple[int, int]]],
    list[tuple[int, int, int, int, int]],
    list[int],
    int | None,
    list[tuple[int, int, int]],
]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    (
        template_bg,
        path_color,
        pattern,
        pat_cells,
        end_color,
        end_pos,
        markers,
    ) = _parse_scene(g)

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = template_bg

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    per_marker_cells: list[set[tuple[int, int]]] = []
    collision_events: list[tuple[int, int, int, int, int]] = []

    for marker_index, (sr, sc, sv) in enumerate(markers):
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
        visited: set[tuple[int, int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (
                int(r),
                int(c),
                int(dr),
                int(dc),
                int(idx),
                int(hits & 1),
                int(hits % len(pattern)),
            )
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

            if int(g[nr, nc]) == int(template_bg):
                if corner_mode == "variant":
                    corner_color = int(pattern[hits % len(pattern)])
                elif corner_mode == "original" and end_color is not None:
                    corner_color = int(end_color)
                else:
                    corner_color = None
                if corner_color is not None:
                    out[nr, nc] = corner_color
                    collision_events.append(
                        (int(marker_index), int(hits + 1), int(nr), int(nc), int(corner_color))
                    )

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out, per_marker_cells, collision_events, pattern, end_color, markers


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    variant_out, _, collision_events, pattern, _end_color, markers = _run_simulation(
        grid, corner_mode="variant"
    )
    original_out, _, _, _, _, _ = _run_simulation(grid, corner_mode="original")
    unique_corner_cells = {(r, c) for _m, _k, r, c, _color in collision_events}
    distinct_corner_colors = {color for _m, _k, _r, _c, color in collision_events}
    return {
        "n_markers": len(markers),
        "cycle_len": len(pattern),
        "total_collisions": len(collision_events),
        "unique_corner_cells": len(unique_corner_cells),
        "distinct_corner_colors": len(distinct_corner_colors),
        "outputs_differ_from_original": not np.array_equal(variant_out, original_out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles that traverse one path while alternating turn direction at impacts.

    Input structure:
    - The upper-left area contains a row of monochrome 2x2 blocks whose left-to-right order is a cycle.
    - One or two isolated pixels on col 0 use the first cycle color and sit inside the path band as launch markers.
    - A separate 2x2 block of another color marks the removable terminal block.

    Rule:
    - Each marker moves straight along adjacent path cells, repainting visited path cells by advancing through the cycle.
    - Whenever the next step would enter a non-path cell, that corner cell is marked only if it is background.
    - The 1st, 2nd, 3rd, ... collision corners use cycle[0], cycle[1], cycle[2], ... with wraparound, rather than a fixed end color.
    - Turn direction still alternates strictly CCW, CW, CCW, CW; if the scheduled turn is not path, the particle stops.

    Key insight:
    - The geometry still comes from alternating left/right turns, but the collision corner colors now encode hit order using the visible template cycle.
    """
    out, _, _, _, _, _ = _run_simulation(grid, corner_mode="variant")
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
    """Carve a bg fjord into the band without disconnecting the path."""
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
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, k=n_blocks)

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
    non_bg = sorted(((c, col) for col, c in cnt.items() if col != bg), reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    metrics = _variant_metrics(g)
    if int(metrics["total_collisions"]) < 2:
        return None
    if int(metrics["distinct_corner_colors"]) < 2:
        return None
    if int(metrics["unique_corner_cells"]) != int(metrics["total_collisions"]):
        return None
    if not bool(metrics["outputs_differ_from_original"]):
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
