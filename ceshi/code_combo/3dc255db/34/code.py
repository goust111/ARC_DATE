# Combo variant of 3dc255db (index 34)
# combo_spec: D2/1+D14/1
# Name: 密度决定方向+垂直弹射

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


# Design analysis for combo D2/1 + D14/1
# --------------------------------------
# a) Concretization and behavior trace
#    - Keep the original cup grammar: one hollow Y/funnel frame per border color, with a single-pixel
#      spike on one bbox edge and the mouth open on the opposite edge.
#    - D14/1 changes the emission axis: the line no longer follows the spike/tail axis. It must go
#      perpendicular to that axis.
#    - D2/1 chooses which perpendicular branch to use by content density, not by the frame geometry.
#      We realize "more concentrated side" as:
#          side density = (# content pixels on that strict mouth half) / (# mouth cells on that half)
#      where the mouth is split by the perpendicular coordinate around the mouth's own center.
#    - Concrete example: tail=U cup, so the old solver would emit upward. Suppose the mouth's left strict
#      half has 2 cells and the right strict half has 6 cells. If input content occupies all 2 left cells
#      plus 3 right cells, then left density = 2/2 = 1.0 while right density = 3/6 = 0.5.
#      Fused output must emit LEFT: not UP (original task) and not necessarily toward the content centroid
#      or larger raw count side.
#    - The visible difference from the original task is therefore twofold:
#      1. the emitted line is orthogonal to the cup axis;
#      2. among the two orthogonal branches, the chosen one is the denser occupied side of the mouth.
#
# b) Complexity calibration
#    - The object vocabulary is unchanged: same cups, same single content color per cup, same stray pixel
#      grammar. That keeps recognition difficulty near the source task.
#    - Branch choice remains binary per cup, so the new rule is still learnable from 2-4 examples.
#    - To keep D2 behavior from collapsing into the D14 single-variant centroid heuristic, the generator
#      prefers examples where both mouth halves contain content and the density-winning side often differs
#      from the raw-count side and sometimes from the centroid side.
#
# c) Implicit-assumption audit against original main()
#    - "Flooding from the side opposite a candidate spike finds the mouth interior": still valid.
#    - "The unique bbox-edge singleton identifies the cup axis": still valid and still needed, because the
#      axis tells us which coordinate is perpendicular.
#    - "Ejection direction equals tail_dir": broken by D14/1 and replaced with a perpendicular direction.
#    - "Perpendicular branch can be read from centroid offset": intentionally broken by D2/1; solver must
#      compare side densities instead.
#    - New assumption introduced here: both strict mouth halves must exist and their densities must be
#      unequal. The generator guarantees that by filtering tiny ambiguous cups and by constructing a
#      side-biased content pattern.
#
# d) Placement legality
#    - Generation stays output-first.
#    - Path reservation must use the chosen perpendicular eject direction, not the tail direction.
#    - Cup bboxes still keep a 1-cell gap, paths must still remain background in the input, and overflow
#      past the canvas edge is still clipped exactly as before.
#
# e) Visual observability
#    - A human can compare input and output locally: contents disappear from inside the cup and reappear as
#      a sideward line.
#    - The "denser side wins" cue is made visible because both halves are often occupied, with one side
#      more tightly packed. That is easier to infer than a hidden numeric score.
#
# f) Visual cleanliness
#    - No extra markers or color remapping are added.
#    - Paths are reserved and cups remain separated, so the new side-spray does not visually collide with
#      other objects.
#
# g) Change inferability
#    - Colors still only identify border vs content. All new behavior is geometric and occupancy-based.
#      There is no hidden dependency on color ids or ordering.
#
# 9) Post-change audit of new main() assumptions
#    - Every processed border component must still have exactly one valid spike candidate after the mouth
#      check; the generator validates this by oracle-testing `main(generate_input())`.
#    - Density comparison only uses content cells inside the flooded mouth, so stray pixels outside cup
#      bboxes cannot interfere.
#    - Equal-density ties are treated as ambiguous and skipped by the solver; the generator avoids them.


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
NEG_PERP_DIR = {"U": "L", "D": "L", "L": "U", "R": "U"}
POS_PERP_DIR = {"U": "R", "D": "R", "L": "D", "R": "D"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-neighbor flood within bbox, seeded from the given bbox edge and blocked by border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:  # "R"
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _perp_index(tail_dir: str) -> int:
    return 1 if tail_dir in ("U", "D") else 0


def _split_mouth_by_perp_side(
    mouth_cells: list[tuple[int, int]] | set[tuple[int, int]], tail_dir: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split mouth cells into strict negative/positive halves on the axis-perpendicular coordinate."""
    mouth_list = list(mouth_cells)
    idx = _perp_index(tail_dir)
    coord_sum = sum(cell[idx] for cell in mouth_list)
    n = len(mouth_list)
    negative = [cell for cell in mouth_list if n * cell[idx] < coord_sum]
    positive = [cell for cell in mouth_list if n * cell[idx] > coord_sum]
    return negative, positive


def _density_eject_dir(
    tail_dir: str,
    mouth_cells: list[tuple[int, int]] | set[tuple[int, int]],
    content_cells: list[tuple[int, int]],
) -> str | None:
    """Choose the perpendicular eject direction by comparing content density on the two mouth halves."""
    negative, positive = _split_mouth_by_perp_side(mouth_cells, tail_dir)
    if not negative or not positive or not content_cells:
        return None

    content_set = set(content_cells)
    neg_count = sum(cell in content_set for cell in negative)
    pos_count = sum(cell in content_set for cell in positive)

    neg_scaled = neg_count * len(positive)
    pos_scaled = pos_count * len(negative)
    if neg_scaled > pos_scaled:
        return NEG_PERP_DIR[tail_dir]
    if pos_scaled > neg_scaled:
        return POS_PERP_DIR[tail_dir]
    return None


def _centroid_perp_dir(
    tail_dir: str,
    mouth_cells: list[tuple[int, int]] | set[tuple[int, int]],
    content_cells: list[tuple[int, int]],
) -> str | None:
    """Legacy D14-style helper used only for generator-side non-degeneracy checks."""
    mouth_list = list(mouth_cells)
    if not mouth_list or not content_cells:
        return None
    idx = _perp_index(tail_dir)
    mouth_sum = sum(cell[idx] for cell in mouth_list)
    content_sum = sum(cell[idx] for cell in content_cells)
    score = len(mouth_list) * content_sum - len(content_cells) * mouth_sum
    if score < 0:
        return NEG_PERP_DIR[tail_dir]
    if score > 0:
        return POS_PERP_DIR[tail_dir]
    return None


def _raw_count_perp_dir(tail_dir: str, content_cells: list[tuple[int, int]], mouth_cells: list[tuple[int, int]]) -> str | None:
    """Side winner by raw counts only; generator uses this to prefer non-degenerate samples."""
    negative, positive = _split_mouth_by_perp_side(mouth_cells, tail_dir)
    if not negative or not positive:
        return None
    content_set = set(content_cells)
    neg_count = sum(cell in content_set for cell in negative)
    pos_count = sum(cell in content_set for cell in positive)
    if neg_count > pos_count:
        return NEG_PERP_DIR[tail_dir]
    if pos_count > neg_count:
        return POS_PERP_DIR[tail_dir]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels sideways relative to the cup axis, toward the denser occupied mouth half.

    Solver outline:
    1. Detect each cup exactly as in the original task by finding a valid spike candidate.
    2. Flood the mouth interior from the side opposite that spike.
    3. Identify the cup's content color inside the mouth.
    4. Split the mouth into the two strict perpendicular halves and compare side densities.
    5. Erase the contents and emit a straight line from the spike in the winning perpendicular direction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (tail_dir, cells[0]) for tail_dir, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                eject_dir = _density_eject_dir(tail_dir, mouth, content_cells)
                if eject_dir is None:
                    continue
                passing.append((spike, content_col, content_cells, eject_dir))

            if len(passing) != 1:
                continue
            spike, content_col, content_cells, eject_dir = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[eject_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
CUP_BUILD_TRIES = 40


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP (spike at top, row 0)."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from the given side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:  # "R"
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical tail=U patch to the requested tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters that stay within the original bbox bounds after rotation."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    eject_dir: str, content_count: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a valid placement whose perpendicular ejection path remains clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[eject_dir]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _perp_distance_key(
    cell: tuple[int, int], tail_dir: str, mouth_size: int, mouth_coord_sum: int,
) -> int:
    idx = _perp_index(tail_dir)
    return abs(mouth_size * cell[idx] - mouth_coord_sum)


def _choose_density_pattern(
    rng: np.random.Generator, mouth: list[tuple[int, int]], tail_dir: str,
) -> tuple[str, list[tuple[int, int]], dict] | None:
    """
    Choose fixed content cells whose winning branch is determined by side density.

    The search prefers examples where:
    - both mouth halves are occupied;
    - the density winner differs from the raw-count winner;
    - the density winner differs from the centroid winner when geometry allows it.
    """
    negative, positive = _split_mouth_by_perp_side(mouth, tail_dir)
    if len(negative) < 2 or len(positive) < 2:
        return None

    mouth_size = len(mouth)
    if mouth_size < 5:
        return None
    idx = _perp_index(tail_dir)
    mouth_coord_sum = sum(cell[idx] for cell in mouth)

    neg_center = sorted(negative, key=lambda cell: _perp_distance_key(cell, tail_dir, mouth_size, mouth_coord_sum))
    pos_center = sorted(positive, key=lambda cell: _perp_distance_key(cell, tail_dir, mouth_size, mouth_coord_sum))
    neg_extreme = list(reversed(neg_center))
    pos_extreme = list(reversed(pos_center))

    candidates = []
    for dense_tag in ("negative", "positive"):
        if dense_tag == "negative":
            dense_cells = negative
            sparse_cells = positive
            dense_pick = neg_center
            sparse_pick = pos_extreme
            dense_dir = NEG_PERP_DIR[tail_dir]
        else:
            dense_cells = positive
            sparse_cells = negative
            dense_pick = pos_center
            sparse_pick = neg_extreme
            dense_dir = POS_PERP_DIR[tail_dir]

        max_dense = min(len(dense_cells), CONTENT_COUNT_MAX - 1)
        max_sparse = min(len(sparse_cells), CONTENT_COUNT_MAX - 1)
        for nd in range(1, max_dense + 1):
            for ns in range(1, min(max_sparse, CONTENT_COUNT_MAX - nd) + 1):
                if nd * len(sparse_cells) <= ns * len(dense_cells):
                    continue
                selected = dense_pick[:nd] + sparse_pick[:ns]
                density_dir = _density_eject_dir(tail_dir, mouth, selected)
                if density_dir != dense_dir:
                    continue

                centroid_dir = _centroid_perp_dir(tail_dir, mouth, selected)
                raw_dir = _raw_count_perp_dir(tail_dir, selected, mouth)
                score = (
                    int(centroid_dir != dense_dir),
                    int(raw_dir != dense_dir),
                    int(ns > nd),
                    nd + ns,
                    min(nd, ns),
                    -abs(nd * len(sparse_cells) - ns * len(dense_cells)),
                )
                candidates.append((score, dense_dir, list(selected), {
                    "dense_tag": dense_tag,
                    "dense_count": nd,
                    "sparse_count": ns,
                    "centroid_dir": centroid_dir,
                    "raw_dir": raw_dir,
                    "neg_size": len(negative),
                    "pos_size": len(positive),
                }))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    _score, eject_dir, cells, stats = best[int(rng.integers(0, len(best)))]
    return eject_dir, cells, stats


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of the bbox, unchanged from the original task."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:  # "U"
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally place one isolated stray pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_color: int,
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup and a density-disambiguating content layout, then commit it."""
    for _ in range(CUP_BUILD_TRIES):
        direction = str(rng.choice(["U", "D", "L", "R"]))
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue
        L_arm, R_arm, hL, hR = shape

        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        if not mouth_canon:
            continue

        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        choice = _choose_density_pattern(rng, mouth, direction)
        if choice is None:
            continue
        eject_dir, content_cells_rel, density_stats = choice

        placement = _try_place_cup(
            rng, H, W, patch, spike, eject_dir, len(content_cells_rel), occ_cup, occ_path,
        )
        if placement is None:
            continue
        r0, c0, path_cells, ejection_cells = placement

        ph, pw = patch.shape
        patch_mask = patch != 0
        canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        occ_cup[r0:r0 + ph, c0:c0 + pw] = True
        for r, c in ejection_cells:
            canvas_out[r, c] = content_color
        for r, c in path_cells:
            occ_path[r, c] = True

        content_cells_abs = [(r0 + r, c0 + c) for r, c in content_cells_rel]
        return {
            "border_color": border_color,
            "content_color": content_color,
            "direction": direction,
            "eject_dir": eject_dir,
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
            "content_cells_abs": content_cells_abs,
            "content_count": len(content_cells_abs),
            "path_cells": path_cells,
            "density_stats": density_stats,
        }
    return None


def _scatter_content(canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Commit the preselected density-bearing content cells into the input canvas."""
    for meta in cups_meta:
        if not meta["content_cells_abs"]:
            return False
        for r, c in meta["content_cells_abs"]:
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single end-to-end generation attempt for the fused density + perpendicular-ejection variant."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
