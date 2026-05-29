# Combo variant of 3dc255db (index 10)
# combo_spec: D3/0+D10/0
# Name: 多色内容+杯口面积决定长度
#
# Design analysis
# ===============
# Fusion summary:
# - Preserve the original cup detection, mouth flood, and tail-direction ejection geometry.
# - Apply D3/0 by allowing 2-3 content colors inside one cup, ordered by visible mouth slices
#   (left->right for U/D tails, top->bottom for L/R tails).
# - Apply D10/0 by making the total ejection length equal the full mouth area rather than the
#   number of colored content pixels.
# - The fused rule must not simply concatenate the two single variants. The total line length is
#   area-driven, but the color block order comes from the content colors. To keep those reasoning
#   steps decoupled and still human-visible, each color owns one contiguous mouth slice, and the
#   output allocates that color a block whose length equals the area of its slice, not the count
#   of visible colored pixels.
#
# a) Concretization and behavioral trace:
# - Original task:
#   A cup with 3 red content pixels ejects a 3-cell red line.
# - D3/0 alone:
#   A cup with 2 red pixels on the left slice and 3 blue pixels on the right slice ejects
#   [red, red, blue, blue, blue].
# - D10/0 alone:
#   A cup with mouth area 6 and only 2 red pixels ejects 6 red cells.
# - Fused variant here:
#   Suppose a cup has mouth area 6. Its left slice has area 4 and contains sparse red pixels.
#   Its right slice has area 2 and contains sparse blue pixels. Input may show only
#   2 red + 1 blue pixels. Output erases those 3 pixels, then ejects [red x4, blue x2].
#   Difference from the original: total length is 6, not 3.
#   Difference from D3-only: block lengths come from slice area, not content counts.
#   Difference from D10-only: the 6-cell line is multicolor, grouped by mouth slice order.
#
# b) Complexity calibration:
# - Limit to 2-3 content colors per cup.
# - Keep mouth area in 4..8 so area can be estimated by eye and fully displayed.
# - Require path length >= mouth area so area-driven length is visible instead of clipped.
# - Require total visible content count <= mouth_area - 2 so the task does not collapse back to
#   the original count-driven behavior.
# - Use only contiguous slice partitions over the mouth-order axis; no hidden color-id ordering.
#
# c) Implicit-assumption audit against the original main():
# - `Counter(...).most_common(1)` assumed a single content color. Invalid now.
# - `content_cells = [...] == content_col` assumed only one color needed erasing. Invalid now.
# - `len(content_cells)` previously drove both "which cells to erase" and "how long to eject".
#   In the combo, erase still depends on actual colored cells, but ejection length depends on
#   full mouth area and per-color slice areas.
# - The tail disambiguation logic still works because a valid candidate is still the one whose
#   opposite-side flood reaches colored content in the mouth.
# - New solver assumption: every mouth-order projection slice assigned to a color contains at
#   least one content pixel of that color, so the sparse input reveals the full slice ownership.
#   The generator guarantees this explicitly.
#
# d) Placement legality:
# - Cups keep a 1-cell gap via dilated bbox checks.
# - Tail paths remain obstacle-free and pairwise non-overlapping.
# - Stray pixels stay outside all cup bboxes and tail paths, on the mouth side only.
# - Output-first generation is preserved: place cups and expected ejection first, reconstruct
#   sparse multicolor input second, then oracle-check with `main`.
#
# e) Visual observability:
# - A human can see two independent cues:
#   1. The total output length matches the cavity area, not the number of content dots.
#   2. The output color blocks follow the left->right or top->bottom order of the mouth colors.
# - The generator ensures every color appears on every projection slice it owns, so the slice
#   boundaries are visible from the sparse content and can be mapped to the output blocks.
#
# f) Visual cleanliness:
# - At most 3 content colors per cup.
# - A small shared content palette prevents color explosion.
# - Sparse content remains inside the cup; the output line stays straight and isolated.
#
# g) Inference quality:
# - Color ordering is geometric, never based on color ids.
# - Per-color output lengths are also geometric: they equal the area of the slice indicated by
#   that color, not the number of dots of that color.
# - This makes the fused rule recoverable from 2-4 examples without relying on invisible facts.
#
# Post-edit solver assumption audit:
# - `_content_groups_in_mouth` assumes each color occupies one contiguous range of mouth-order
#   projections. The generator constructs exactly that.
# - `_group_eject_sequence` assumes the union of those projection ranges covers the whole mouth,
#   so per-group areas sum to the full mouth area. The generator partitions all mouth projections.
# - `_scatter_content` assumes each group has preselected anchor cells, one per owned projection,
#   so solver-visible slice coverage is complete. The generator stores and uses those anchors.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


def _flood_from_bbox_side(border_set: set, bbox: tuple, side: str) -> set:
    """4-connected flood within bbox, seeded from a chosen bbox side."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _mouth_projection(cell: tuple[int, int], tail_dir: str) -> int:
    r, c = cell
    return int(c if tail_dir in ("U", "D") else r)


def _content_groups_in_mouth(
    g: np.ndarray, mouth: set[tuple[int, int]], bg: int, border_col: int, tail_dir: str,
) -> list[dict]:
    """Group mouth content by color and sort groups by visible mouth slice order."""
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in mouth:
        value = int(g[r, c])
        if value != bg and value != border_col:
            by_color[value].append((int(r), int(c)))
    if not by_color:
        return []

    groups = []
    for color, cells in by_color.items():
        projections = sorted({_mouth_projection(cell, tail_dir) for cell in cells})
        other_axis = [cell[0] if tail_dir in ("U", "D") else cell[1] for cell in cells]
        groups.append(
            {
                "color": int(color),
                "cells": cells,
                "projections": projections,
                "order_key": (
                    min(projections),
                    sum(projections) / len(projections),
                    min(other_axis) * 100 + int(color),
                ),
            }
        )
    groups.sort(key=lambda item: item["order_key"])
    return groups


def _group_eject_sequence(
    mouth: set[tuple[int, int]], tail_dir: str, groups: list[dict],
) -> list[int]:
    """Expand ordered color groups so each group length equals the area of its mouth slice."""
    area_by_projection: dict[int, int] = defaultdict(int)
    for cell in mouth:
        area_by_projection[_mouth_projection(cell, tail_dir)] += 1

    sequence: list[int] = []
    for group in groups:
        block_len = sum(area_by_projection[p] for p in group["projections"])
        sequence.extend([int(group["color"])] * int(block_len))
    return sequence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect cups exactly as in the original task.
    - The mouth may contain multiple content colors occupying contiguous mouth slices.
    - Erase all content pixels from the mouth.
    - Eject a straight line whose total length equals the full mouth area.
    - Partition that line into contiguous color blocks in mouth slice order, where each block
      length equals the area of the slice indicated by that color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    dir4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}

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
                (direction, cells[0]) for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, opposite[direction])
                groups = _content_groups_in_mouth(g, mouth, bg, int(border_col), direction)
                if not groups:
                    continue
                eject_sequence = _group_eject_sequence(mouth, direction, groups)
                if not eject_sequence:
                    continue
                passing.append((direction, spike, groups, eject_sequence))

            if len(passing) != 1:
                continue
            tail_dir, spike, groups, eject_sequence = passing[0]

            for group in groups:
                for r, c in group["cells"]:
                    out[r, c] = bg

            dr, dc = dir4[tail_dir]
            sr, sc = spike
            for step, color in enumerate(eject_sequence, start=1):
                rr = sr + dr * step
                cc = sc + dc * step
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = int(color)

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

MIN_MOUTH_AREA = 4
MAX_MOUTH_AREA = 8
MIN_AREA_COUNT_GAP = 2
CONTENT_GROUP_COUNT_RANGE = (2, 3)
CONTENT_COUNT_MAX = 7
GLOBAL_CONTENT_POOL_SIZE = 4

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical funnel with tail pointing up."""
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
    """Flood-fill the cup interior from the open side."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical cup data to the requested tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample canonical cup parameters while staying within the original bbox bounds."""
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


def _split_sorted_values(values: list[int], groups: int, rng: np.random.Generator) -> list[list[int]] | None:
    """Split sorted values into contiguous, non-empty groups."""
    if len(values) < groups:
        return None
    if groups == 1:
        return [list(values)]
    cut_points = sorted(int(x) for x in rng.choice(np.arange(1, len(values)), size=groups - 1, replace=False))
    out: list[list[int]] = []
    start = 0
    for stop in [*cut_points, len(values)]:
        out.append(list(values[start:stop]))
        start = stop
    return out


def _choose_anchor_cells(
    rng: np.random.Generator, zone: list[tuple[int, int]], direction: str, projections: list[int],
) -> list[tuple[int, int]]:
    """Pick one visible content cell per owned mouth projection slice."""
    anchors: list[tuple[int, int]] = []
    for projection in projections:
        candidates = [cell for cell in zone if _mouth_projection(cell, direction) == projection]
        if not candidates:
            return []
        idx = int(rng.integers(0, len(candidates)))
        anchors.append(candidates[idx])
    return anchors


def _choose_content_groups(
    rng: np.random.Generator, mouth: list[tuple[int, int]], direction: str, available_colors: list[int],
) -> list[dict] | None:
    """
    Partition the mouth into contiguous projection slices, assign each slice a color, and choose
    sparse content while guaranteeing that every owned projection slice is visibly represented.
    """
    mouth_area = len(mouth)
    if mouth_area < MIN_MOUTH_AREA or mouth_area > MAX_MOUTH_AREA:
        return None

    unique_proj = sorted({_mouth_projection(cell, direction) for cell in mouth})
    min_required = len(unique_proj)
    max_total_content = min(CONTENT_COUNT_MAX, mouth_area - MIN_AREA_COUNT_GAP)
    if max_total_content < min_required:
        return None

    max_groups = min(CONTENT_GROUP_COUNT_RANGE[1], len(unique_proj), len(available_colors))
    if max_groups < CONTENT_GROUP_COUNT_RANGE[0]:
        return None
    num_groups = int(rng.integers(CONTENT_GROUP_COUNT_RANGE[0], max_groups + 1))

    proj_groups = _split_sorted_values(unique_proj, num_groups, rng)
    if proj_groups is None:
        return None

    zones: list[dict] = []
    for proj_values in proj_groups:
        proj_set = set(proj_values)
        zone = [cell for cell in mouth if _mouth_projection(cell, direction) in proj_set]
        zone = sorted(zone, key=lambda rc: (_mouth_projection(rc, direction), rc[0], rc[1]))
        if not zone:
            return None
        anchors = _choose_anchor_cells(rng, zone, direction, proj_values)
        if len(anchors) != len(proj_values):
            return None
        anchor_set = set(anchors)
        extra_pool = [cell for cell in zone if cell not in anchor_set]
        zones.append(
            {
                "projections": list(proj_values),
                "zone": zone,
                "anchors": anchors,
                "extra_pool": extra_pool,
            }
        )

    colors = [int(v) for v in rng.choice(np.array(available_colors, dtype=int), size=num_groups, replace=False)]
    rng.shuffle(colors)

    counts = [len(zone["anchors"]) for zone in zones]
    capacities = [len(zone["extra_pool"]) for zone in zones]
    min_total = max(sum(counts), 3)
    if max_total_content < min_total:
        min_total = sum(counts)
    if max_total_content < min_total:
        return None
    total_count = int(rng.integers(min_total, max_total_content + 1))

    remaining = total_count - sum(counts)
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        idx = int(rng.choice(np.array(choices, dtype=int)))
        counts[idx] += 1
        capacities[idx] -= 1
        remaining -= 1

    groups = []
    for i, zone in enumerate(zones):
        groups.append(
            {
                "color": colors[i],
                "count": counts[i],
                "projections": zone["projections"],
                "zone": zone["zone"],
                "anchors": zone["anchors"],
                "extra_pool": zone["extra_pool"],
                "eject_len": len(zone["zone"]),
            }
        )
    return groups


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, eject_len: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup with a full visible tail path for the area-driven ejection."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)

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
        if len(path_cells) < eject_len:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Mask of cells on the mouth side of a cup, within its transverse extent."""
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
    """Optionally place one isolated stray pixel using a border or content-group color."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    content_colors = [group["color"] for group in meta["content_groups"]]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = meta["border_color"]
    else:
        stray_color = int(rng.choice(np.array(content_colors, dtype=int)))

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
    for item in cups_meta:
        br0, bc0, br1, bc1 = item["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in item["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_palette: list[int],
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, assign ordered multicolor slices, and commit its expected output."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    mouth_area = len(mouth)
    available_content_colors = [color for color in content_palette if color != border_color]
    content_groups = _choose_content_groups(rng, mouth, direction, available_content_colors)
    if content_groups is None:
        return None

    eject_sequence = [group["color"] for group in content_groups for _ in range(group["eject_len"])]
    if len(eject_sequence) != mouth_area:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, len(eject_sequence), occ_cup, occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for (r, c), color in zip(ejection_cells, eject_sequence):
        canvas_out[r, c] = int(color)
    for r, c in path_cells:
        occ_path[r, c] = True

    content_groups_abs = []
    for group in content_groups:
        content_groups_abs.append(
            {
                "color": int(group["color"]),
                "count": int(group["count"]),
                "eject_len": int(group["eject_len"]),
                "zone_abs": [(r0 + r, c0 + c) for r, c in group["zone"]],
                "anchors_abs": [(r0 + r, c0 + c) for r, c in group["anchors"]],
                "extra_pool_abs": [(r0 + r, c0 + c) for r, c in group["extra_pool"]],
            }
        )

    return {
        "border_color": border_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_area": mouth_area,
        "content_groups": content_groups_abs,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter sparse content while preserving one visible pixel per owned projection slice."""
    for meta in cups_meta:
        for group in meta["content_groups"]:
            anchors_abs = group["anchors_abs"]
            extra_pool_abs = group["extra_pool_abs"]
            count = int(group["count"])
            if count < len(anchors_abs):
                return False
            for r, c in anchors_abs:
                canvas_in[r, c] = int(group["color"])
            extra_needed = count - len(anchors_abs)
            if extra_needed > len(extra_pool_abs):
                return False
            if extra_needed > 0:
                idxs = rng.choice(len(extra_pool_abs), size=extra_needed, replace=False)
                for idx in idxs:
                    r, c = extra_pool_abs[int(idx)]
                    canvas_in[r, c] = int(group["color"])
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single output-first generation attempt with oracle checking."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    border_palette = list(range(1, 10))
    rng.shuffle(border_palette)
    content_palette = [int(v) for v in rng.choice(np.arange(1, 10), size=GLOBAL_CONTENT_POOL_SIZE, replace=False)]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if not border_palette:
            return None
        border_color = int(border_palette.pop())
        meta = _add_cup(
            rng, H, W, border_color, content_palette, canvas_in, canvas_out, occ_cup, occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
