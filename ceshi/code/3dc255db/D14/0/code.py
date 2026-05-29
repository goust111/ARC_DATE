# Variant of 3dc255db
# Dimension: D14 — 弹射方向与杯口关系 (path)
# Change: 弹射方向指向尾部(远离杯口)——内容从杯口'倒出'到尾部方向... → 弹射方向指向杯口外侧

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


# Design analysis for the D14 variant
# -----------------------------------
# source_task:
#   present: false
#   note: no separate reference task was provided, so the variant is derived directly from
#     the original solver/spec.
# variant_spec:
#   core_change:
#     original: a cup's content is erased from the mouth interior, then re-drawn as a straight
#       ray starting at the spike and extending farther in the tail direction.
#     variant: keep tail detection exactly the same, but flip the travel direction so the
#       straight ray extends outward from the mouth side instead. The spike remains part of the
#       border; the emitted pixels now appear outside the open side of the cup.
#   concrete_pixel_behavior:
#     cup_detection:
#       unchanged: identify a candidate tail as the unique non-corner border pixel on a bbox edge.
#       unchanged: flood from the opposite bbox side to find the mouth interior and its content.
#     new_ray_anchor:
#       rule: project the spike across the cup bbox onto the opposite edge. That projected edge
#         cell lies on the cup mouth centerline in this generator family.
#       example: if the spike is the only border cell on the top edge at column c, then the ray
#         starts just below the bottom edge at column c.
#     output_effect:
#       erase the content pixels inside the mouth interior.
#       draw len(content) pixels in the mouth-outward direction, starting one step outside the
#         mouth edge; truncate at the grid edge.
#   traced_example:
#     canonical_tail_up:
#       input_patch: spike at (0,2), mouth opens downward, 3 content pixels inside.
#       original_output: pixels appear at (- notional) (1 step above spike), i.e. rows -1,-2,-3
#         relative to the spike direction, clipped by the grid if needed.
#       variant_output: content is removed from the mouth interior, then appears below the mouth
#         at rows h,h+1,h+2 on the same center column.
#       difference: recognition of the cup and counting of content stay the same; only the path
#         anchor and direction are inverted.
#   complexity_calibration:
#     kept_from_original:
#       - 1..3 cups per grid
#       - asymmetric cup shapes
#       - single-color content per cup
#       - optional stray pixel
#     why_not_simpler:
#       If the ray merely started from an arbitrary mouth cell, the solver would need extra
#       tie-breaking logic not visible from 2-4 examples. Projecting the spike to the opposite
#       bbox side gives a stable, visible "same axis, opposite side" rule.
#     visual_support:
#       Cups still appear in multiple orientations, so the reversal is observable across examples
#       without introducing unrelated dimensions.
#   implicit_assumption_audit_main:
#     preserved_assumptions:
#       - A true cup still has exactly one useful tail candidate after disambiguation by mouth
#         flood and content detection.
#       - The content color is still the dominant non-bg, non-border color inside that mouth.
#     changed_lines_and_new_requirements:
#       - The original solver used the spike itself as the ray anchor. That silently assumed the
#         old tail-outward behavior.
#       - The new solver computes a mouth-edge anchor by projecting the spike to the opposite bbox
#         side. This requires the generator to keep the cup axis straight so that the projected
#         mouth-edge cell is part of the opening rather than a wall.
#       - Path clearance must now be guaranteed on the mouth side, not on the tail side.
#   placement_legality:
#     generator_updates:
#       - when placing a cup, reserve the outward half-line starting just outside the mouth side;
#         this path must avoid every cup bbox and every other reserved path.
#       - the cup must have at least one cell of exterior runway on its mouth side, otherwise the
#         sample is rejected.
#       - stray pixels may still appear on a mouth-side half-plane, but never on a reserved path
#         and never adjacent to existing non-bg cells.
#   visual_observability:
#     thought_experiment:
#       Across 2-4 examples, a human still sees the cup opening by its hollow side, but the line
#       no longer shoots from the spike. Instead it appears outside the opening on the same axis
#       as the spike, which is a strong, consistent inversion cue.
#   cleanliness:
#     note: because both the new ray and the optional stray pixel live on the mouth side, the
#       generator explicitly forbids stray/path overlap so the line stays visually readable.
#   color_reasoning:
#     note: no color mapping is introduced. Colors keep their original role, so D14 remains a
#       spatial-logic change only.
#   post_change_solver_assumptions_to_enforce:
#     - Generated cups come from the canonical funnel family where the spike-to-mouth projection
#       is well-defined.
#     - The projected mouth axis has at least one free outside cell.
#     - No other cup/path occupies that mouth-side runway.


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, seeded from the chosen edge and blocked by the border."""
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


def _project_spike_to_mouth(
    bbox: tuple[int, int, int, int], spike: tuple[int, int], tail_dir: str,
) -> tuple[tuple[int, int], str]:
    """Project the spike to the opposite bbox edge. Returns (mouth_edge_cell, mouth_out_dir)."""
    r0, c0, r1, c1 = bbox
    sr, sc = spike
    if tail_dir == "U":
        return (r1, sc), "D"
    if tail_dir == "D":
        return (r0, sc), "U"
    if tail_dir == "L":
        return (sr, c1), "R"
    return (sr, c0), "L"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward from the cup mouth.

    Recognition stays identical to the original task:
    - detect the tail spike as the unique non-corner edge cell of the border component;
    - flood from the opposite bbox side to locate the mouth interior and its content.

    Variant action:
    - erase the content inside the mouth interior;
    - project the spike to the opposite bbox edge and emit the straight line outward from that
      mouth side, rather than outward from the spike.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

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
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            mouth_edge, path_dir = _project_spike_to_mouth(bbox, spike, tail_dir)
            dr, dc = DIR4[path_dir]
            sr, sc = mouth_edge
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

# Cup bbox limits stay the same as the original task family.
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


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
    """Flood-fill the cup interior from the mouth side."""
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
    """Rotate a canonical tail-up cup to the target tail direction."""
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
    """Sample canonical cup parameters that still fit the original bbox constraints."""
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


def _path_cells_from_mouth(
    H: int, W: int, bbox: tuple[int, int, int, int], spike_abs: tuple[int, int], tail_dir: str,
) -> list[tuple[int, int]]:
    """All cells on the mouth-outward half-line, starting just outside the cup."""
    mouth_edge, path_dir = _project_spike_to_mouth(bbox, spike_abs, tail_dir)
    dr, dc = DIR4[path_dir]
    rr, cc = mouth_edge[0] + dr, mouth_edge[1] + dc
    path_cells = []
    while 0 <= rr < H and 0 <= cc < W:
        path_cells.append((rr, cc))
        rr += dr
        cc += dc
    return path_cells


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, content_count: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a placement with a clear mouth-side ejection runway."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        spike_abs = (r0 + spike[0], c0 + spike[1])
        path_cells = _path_cells_from_mouth(H, W, bbox, spike_abs, direction)
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of the cup bbox, within the perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally add one isolated stray pixel on the mouth side of a cup."""
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
    """Build, place, and commit one cup using the mouth-side ejection variant."""
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
    ph, pw = patch.shape

    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))
    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one valid input by building the target output first."""
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
