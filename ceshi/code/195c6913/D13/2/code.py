# Variant of 195c6913
# Dimension: D13 — 模板和终止块的擦除行为 (transform)
# Change: 输出中不再简单擦除终止2x2块；模板2x2块仍被擦除，但输入里的终止块会被“搬运”到每个粒子的最终停止位置旁边。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D13 variant
# -----------------------------------
# a) Concrete behavior:
#    The original task erased the unique 2x2 end-color block from its input
#    location and only left single-pixel end-color corner marks at collisions.
#    In this variant, the 2x2 end block becomes a movable object: after each
#    particle finishes, a copy of that 2x2 block is placed beside the final path
#    cell where the particle stops. "Beside" is made geometric and visible:
#    when the particle attempts its scheduled turn and that turn is blocked, the
#    2x2 block is placed on the blocked-turn side of the final path cell, aligned
#    with the incoming motion. Example: if a particle is moving right, hits a bg
#    cell ahead, and its scheduled CCW turn upward is also blocked, then the
#    final path cell is the cell just left of that corner and the moved 2x2
#    block appears directly above that final cell. The collision pixel ahead is
#    still painted end_color, so the output shows both the old single-pixel
#    corner language and the moved 2x2 object.
# b) Complexity calibration:
#    Only one new transform mechanism is introduced: moved 2x2 end blocks.
#    Their count matches the number of markers (1 or 2), so the rule is repeated
#    enough to be learnable but not overloaded. The input still has one unique
#    end block color, avoiding ambiguity with the cycle template colors.
# c) Hidden-assumption audit against the original main():
#    - The original unconditional erasure of the end block relied on that block
#      not carrying information into the output. This no longer holds.
#    - The original simulation only needed visited path cells. The variant also
#      needs the final heading, collision parity, and stop reason to derive the
#      new 2x2 placement.
#    - The original early break on out-of-bounds was harmless because no moved
#      object depended on the stopping geometry. Here it would make the final
#      placement undefined, so the generator rejects such samples and guarantees
#      every marker stops via a blocked scheduled turn inside the grid.
#    - The original end-color writes were all single cells; now 2x2 writes could
#      overlap if endpoints are too close. The generator explicitly enforces
#      pairwise-disjoint moved-block placements.
# d) Placement legality:
#    The generator simulates every marker with the same helper used by main(),
#    computes each moved-block anchor, and rejects any sample where a 2x2 moved
#    block would leave the grid, overlap the path, overlap the template area,
#    overlap another moved block, or overlap a marker cell.
# e) Visual observability:
#    Compared with the original task, outputs now visibly contain 1 or 2 intact
#    2x2 end-color blocks near trajectory endpoints. This is a strong visual cue
#    that the end block is not erased but relocated. The rule does not depend on
#    numeric color ids or hidden ordering.
# f) Visual cleanliness:
#    The moved block is always placed in a bg pocket adjacent to the final stop,
#    never on top of the painted trajectory. The original top-left template is
#    still erased, so the upper-left region does not become cluttered.
# g) Color inference:
#    No new color mapping is introduced. The existing end_color already labels
#    both collision corners and the input 2x2 end block, so reusing that same
#    color for the moved 2x2 blocks is directly inferable from the examples.
# h) Variant-specific success metric:
#    For generated samples, the key metric is the number of moved 2x2 end-color
#    blocks in the output that are not at the original input location. It must
#    equal the number of markers, and the variant output should differ from the
#    original solver output on those same inputs.
# i) New solver assumptions checked by the generator:
#    - The input contains exactly one detectable 2x2 end block.
#    - Every marker has a non-empty path simulation and terminates by blocked
#      scheduled turn, not by leaving the board.
#    - The derived moved-block anchors are valid 2x2 bg placements and are
#      pairwise disjoint.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker at col 0 is embedded in a vertical path run."""
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


def _detect_pattern_and_end(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]], int | None, tuple[int, int] | None]:
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
    return pattern, pat_cells, end_color, end_pos


def _detect_markers(
    g: np.ndarray, path_color: int, pattern: list[int]
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    pat_set = set(map(int, pattern))
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


def _moved_block_anchor(
    r: int, c: int, dr: int, dc: int, tdr: int, tdc: int
) -> tuple[int, int]:
    """Anchor the moved 2x2 end block on the blocked-turn side of the stop cell."""
    if (tdr, tdc) == (-1, 0):  # blocked turn is up
        return int(r - 2), int(c if dc >= 0 else c - 1)
    if (tdr, tdc) == (1, 0):  # blocked turn is down
        return int(r + 1), int(c if dc >= 0 else c - 1)
    if (tdr, tdc) == (0, -1):  # blocked turn is left
        return int(r if dr >= 0 else r - 1), int(c - 2)
    if (tdr, tdc) == (0, 1):  # blocked turn is right
        return int(r if dr >= 0 else r - 1), int(c + 1)
    raise ValueError(f"invalid turn direction {(tdr, tdc)}")


def _simulate_marker(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    end_color: int | None,
    marker: tuple[int, int, int],
) -> dict:
    H, W = g.shape
    sr, sc, sv = marker
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break
    if init is None or not pattern:
        return {
            "visited_cells": set(),
            "paint_steps": [],
            "collision_marks": [],
            "stop": "no_init",
            "block_anchor": None,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()
    paint_steps: list[tuple[int, int, int]] = []
    collision_marks: list[tuple[int, int]] = []

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        if state in visited:
            return {
                "visited_cells": my_cells,
                "paint_steps": paint_steps,
                "collision_marks": collision_marks,
                "stop": "loop",
                "block_anchor": None,
            }
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            return {
                "visited_cells": my_cells,
                "paint_steps": paint_steps,
                "collision_marks": collision_marks,
                "stop": "out_of_bounds",
                "block_anchor": None,
            }

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            paint_steps.append((int(nr), int(nc), int(pattern[idx])))
            my_cells.add((int(nr), int(nc)))
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(g[0, 0]) and end_color is not None:
            collision_marks.append((int(nr), int(nc)))

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            continue

        return {
            "visited_cells": my_cells,
            "paint_steps": paint_steps,
            "collision_marks": collision_marks,
            "stop": "blocked_turn",
            "block_anchor": _moved_block_anchor(int(r), int(c), int(dr), int(dc), int(tdr), int(tdc)),
        }

    return {
        "visited_cells": my_cells,
        "paint_steps": paint_steps,
        "collision_marks": collision_marks,
        "stop": "loop_cap",
        "block_anchor": None,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles on a path while carrying the terminal 2x2 block to their endpoints.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks defines the repeating paint cycle.
    - Path-embedded isolated pixels of cycle colors are the launch markers.
    - One separate 2x2 monochrome block supplies the shared terminal color/object.

    Rule:
    - Each marker advances along the path, repainting visited path cells by cycling colors.
    - Every collision with a bg cell still stamps a single end-color corner mark.
    - Collision turns still alternate CCW, CW, CCW, CW; if the scheduled turn is blocked, the particle stops.
    - The template blocks are erased, and the input terminal 2x2 block is removed from its source location and re-placed beside each particle's final stop.

    Key insight: The motion rule is unchanged; the new work is to recover each particle's stopping geometry and use that geometry to relocate the terminal 2x2 object instead of erasing it.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells, end_color, end_pos = _detect_pattern_and_end(g, int(template_bg), int(path_color))
    markers = _detect_markers(g, int(path_color), pattern)
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    moved_anchors: list[tuple[int, int]] = []
    for marker in markers:
        sim = _simulate_marker(g, pattern, int(path_color), end_color, marker)
        per_marker_cells.append(set(sim["visited_cells"]))

        for rr, cc, color in sim["paint_steps"]:
            out[int(rr), int(cc)] = int(color)
        if end_color is not None:
            for rr, cc in sim["collision_marks"]:
                out[int(rr), int(cc)] = int(end_color)

        anchor = sim["block_anchor"]
        if end_color is not None and sim["stop"] == "blocked_turn" and anchor is not None:
            ar, ac = anchor
            if 0 <= ar and 0 <= ac and ar + 1 < H and ac + 1 < W:
                out[int(ar) : int(ar + 2), int(ac) : int(ac + 2)] = int(end_color)
                moved_anchors.append((int(ar), int(ac)))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    if end_color is not None:
        assert len(moved_anchors) == len(markers), "every marker must receive one moved end block"
        assert len(set(moved_anchors)) == len(moved_anchors), "moved end blocks must not overlap"

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
    """Carve a bg fjord into the band while preserving path connectivity."""
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


def _moved_blocks_are_legal(
    g: np.ndarray,
    path_color: int,
    markers: list[tuple[int, int, int]],
    sims: list[dict],
    template_row: int,
    block_cols: list[int],
    end_pos: tuple[int, int],
) -> bool:
    H, W = g.shape
    occupied = np.zeros((H, W), dtype=bool)
    occupied[g == path_color] = True
    for r in range(template_row, template_row + 2):
        for bc in block_cols:
            occupied[int(r), int(bc) : int(bc + 2)] = True
    er, ec = end_pos
    occupied[int(er) : int(er + 2), int(ec) : int(ec + 2)] = True
    for r, c, _ in markers:
        occupied[int(r), int(c)] = True

    used: set[tuple[int, int]] = set()
    for sim in sims:
        if sim["stop"] != "blocked_turn" or sim["block_anchor"] is None:
            return False
        ar, ac = sim["block_anchor"]
        if not (0 <= ar and 0 <= ac and ar + 1 < H and ac + 1 < W):
            return False
        if (ar, ac) in used:
            return False
        used.add((int(ar), int(ac)))
        if occupied[int(ar) : int(ar + 2), int(ac) : int(ac + 2)].any():
            return False
        occupied[int(ar) : int(ar + 2), int(ac) : int(ac + 2)] = True
    return True


def _gen_attempt(n_markers: int | None = None) -> np.ndarray | None:
    H = W = random.randint(20, 30)
    if n_markers is None:
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
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    pattern, _, found_end_color, found_end_pos = _detect_pattern_and_end(g, bg, path_color)
    if pattern != cycle_colors or found_end_color != end_color or found_end_pos != (er, ec):
        return None
    marker_triplets = _detect_markers(g, path_color, pattern)
    _assert_marker_embedded_in_path(g, marker_triplets, path_color)
    sims = [_simulate_marker(g, pattern, path_color, end_color, marker) for marker in marker_triplets]

    for sim in sims:
        if sim["stop"] != "blocked_turn":
            return None

    per_marker_cells = [set(sim["visited_cells"]) for sim in sims]
    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            if per_marker_cells[i] & per_marker_cells[j]:
                return None

    if not _moved_blocks_are_legal(g, path_color, marker_triplets, sims, template_row, block_cols, (er, ec)):
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    target_n = 2 if random.random() < 0.5 else 1
    for _ in range(200):
        try:
            g = _gen_attempt(target_n)
        except Exception:
            g = None
        if g is not None:
            return g
    for _ in range(600):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
