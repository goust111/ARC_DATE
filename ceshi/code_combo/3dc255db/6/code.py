# Combo variant of 3dc255db (index 6)
# combo_spec: D1/2+D13/1
# Name: 重力流动+杯口宽度射线
#
# Design analysis
# ---------------
# spec:
#   fused_rule:
#     - Object recognition stays the same as the original task: detect each hollow cup/funnel,
#       find its unique actionable spike candidate by flooding from the spike-opposite bbox
#       side, and read the content color/count from the mouth interior.
#     - Change 1 (D1/2): motion is no longer along the tail direction. After erasing the mouth
#       content, output always flows straight downward under gravity.
#     - Change 2 (D13/1): the falling output is no longer 1px wide. Its width equals the cup
#       opening width measured on the mouth side of the bbox. The solver therefore must first
#       locate the opening edge cells, reduce them to a width scalar, and then paint a vertical
#       stack of horizontal bands of that width.
#   a_concretization_and_trace:
#     - Small example: suppose a cup has 3 content pixels, its mouth opening width is 4, and
#       its spike is at column 6.
#     - Original task:
#         1. detect tail direction, say tail=R;
#         2. erase the 3 mouth pixels;
#         3. paint 3 single cells at (spike + 1*R), (spike + 2*R), (spike + 3*R).
#     - Fused task here:
#         1. detect the same cup and same 3 mouth pixels;
#         2. measure that the mouth opening spans width 4;
#         3. erase the 3 mouth pixels;
#         4. paint 3 downward bands, each 4 cells wide, centered deterministically on the spike
#            column, at rows spike_row+1, +2, +3.
#       Result: a 4x3 waterfall instead of a 1x3 directional ray.
#     - The key difference is not only "down instead of tail direction", but also that one
#       erased content pixel now contributes one whole horizontal band rather than one point.
#   b_complexity_calibration:
#     - I keep the original cup variety, scattered mouth content, optional stray pixel, and
#       multiple-cup grids, so the task is not reduced to a template.
#     - I deliberately keep the widened flow as a single fixed-width rectangle per cup. This
#       makes the fusion visually legible from 2-4 examples without introducing extra state
#       like sideways splitting or irregular fluid simulation.
#     - The generator explicitly requires at least one non-downward cup to demonstrate gravity,
#       and at least one cup with mouth_width >= 2 and visible multi-row output to demonstrate
#       the width dimension.
#   c_hidden_assumption_audit_of_original_main:
#     - `_flood_from_bbox_side(...)` still works: mouth detection is unchanged.
#     - Original candidate-spike logic still works: the cup geometry itself is unchanged.
#     - Original `content_cells` counting still works, but it no longer determines a 1px path;
#       instead it determines the number of downward bands.
#     - Original `dr, dc = DIR4[tail_dir]` and the straight-ray loop are invalid because motion
#       is no longer shape-directed. They are replaced by a downward band generator.
#     - Original in-place one-cup-at-a-time drawing could create order dependence once widened
#       gravity regions exist. The solver therefore first erases all actionable content, freezes
#       the remaining obstacles, and only then paints all waterfalls.
#   d_placement_legality:
#     - Cup bboxes still keep a one-cell gap via dilated bbox occupancy.
#     - A candidate placement is rejected if the widened gravity rectangle would overlap another
#       cup bbox, another reserved waterfall, or go out of horizontal bounds.
#     - Static blocking for each waterfall includes already placed cup borders plus the new cup's
#       own border cells, so the reserved path is faithful to solver-time obstacles.
#   e_visual_observability:
#     - Gravity is visible because at least one actionable cup has tail_dir != D yet still emits
#       a downward output.
#     - Width is visible because at least one actionable cup produces a waterfall wider than 1px.
#     - The two controls are geometric and human-visible; they do not depend on color ordering.
#   f_visual_cleanliness:
#     - Waterfalls are reserved as whole rectangles, so different cups do not interpenetrate.
#     - Stray pixels are excluded from all cup bboxes and all reserved waterfall corridors.
#     - The result preserves three readable layers: border frame, erased mouth interior, and
#       external falling band(s).
#   g_color_inference:
#     - Colors are unchanged from the original task: one border color plus one content color per
#       cup. The transformation is driven entirely by geometry.
#   h_quantifiable_variant_signal:
#     - Post-write validation will measure:
#         1. share of generated samples containing a cup with direction != D and visible output;
#         2. share containing a cup with mouth_width >= 2 and visible output depth >= 2;
#         3. share where this fused solver's output differs from the original straight-ray solver.
#       High values on all three confirm that generation is not collapsing back to the source task.
#   i_new_solver_assumptions_and_generator_guarantees:
#     - Assumption: opening edge cells on the mouth side form one contiguous interval. The cup
#       generator preserves the original single-opening funnel structure, so this holds.
#     - Assumption: widened bands are centered on the spike column and fully fit in-bounds. The
#       generator checks that the entire width fits before accepting a placement.
#     - Assumption: actionable cups do not need waterfall tie-breaking. The generator reserves
#       disjoint output corridors, so the solver can paint in any order after the erase phase.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from non-border cells on the given bbox edge."""
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


def _opening_edge_cells(
    mouth: set[tuple[int, int]] | list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    """Cells of the mouth that lie on the cup opening edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return sorted((r, c) for r, c in mouth if r == r0)
    if side == "D":
        return sorted((r, c) for r, c in mouth if r == r1)
    if side == "L":
        return sorted((r, c) for r, c in mouth if c == c0)
    return sorted((r, c) for r, c in mouth if c == c1)


def _opening_width(edge_cells: list[tuple[int, int]], side: str) -> int:
    """Width of the cup opening measured on the axis perpendicular to the mouth side."""
    if not edge_cells:
        return 0
    if side in ("U", "D"):
        vals = [c for _, c in edge_cells]
    else:
        vals = [r for r, _ in edge_cells]
    return max(vals) - min(vals) + 1


def _centered_band_span(center_col: int, width: int, W: int) -> tuple[int, int] | None:
    """Deterministic horizontal span for a widened falling band."""
    if width <= 0:
        return None
    left = (width - 1) // 2
    right = width - 1 - left
    lo = center_col - left
    hi = center_col + right
    if lo < 0 or hi >= W:
        return None
    return lo, hi


def _gravity_bands(
    spike: tuple[int, int],
    width: int,
    H: int,
    W: int,
    blocked: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Consecutive downward horizontal bands until the first obstacle or the bottom edge."""
    sr, sc = spike
    span = _centered_band_span(sc, width, W)
    if span is None:
        return []
    lo, hi = span
    bands: list[list[tuple[int, int]]] = []
    for rr in range(sr + 1, H):
        band = [(rr, cc) for cc in range(lo, hi + 1)]
        if any(bool(blocked[r, c]) for r, c in band):
            break
        bands.append(band)
    return bands


def _flatten_bands(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for band in bands for cell in band]


def _detect_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Detect actionable cups and derive the fused-rule metadata needed by the solver."""
    cups: list[dict] = []
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
            for tail_dir, spike in candidates:
                mouth_side = OPPOSITE[tail_dir]
                mouth = _flood_from_bbox_side(border_set, bbox, mouth_side)
                edge_cells = _opening_edge_cells(mouth, bbox, mouth_side)
                mouth_width = _opening_width(edge_cells, mouth_side)
                if mouth_width <= 0:
                    continue

                cnt = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        cnt[value] += 1
                if not cnt:
                    continue

                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((tail_dir, spike, mouth_width, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, mouth_width, content_col, content_cells = passing[0]
            cups.append(
                {
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "mouth_width": mouth_width,
                    "content_color": content_col,
                    "content_cells": content_cells,
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - detect cups exactly as in the original task;
    - erase the cup content;
    - measure the cup-mouth opening width;
    - generate a downward waterfall from the spike, one horizontal band per erased content pixel,
      with band width equal to the measured mouth width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    cups = _detect_cups(g, bg)

    out = g.copy()
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    blocked = out != bg
    for cup in cups:
        bands = _gravity_bands(cup["spike"], int(cup["mouth_width"]), H, W, blocked)
        for band in bands[: len(cup["content_cells"])]:
            for r, c in band:
                out[r, c] = int(cup["content_color"])

    return out


# --------------- generator ---------------

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

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
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


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the cup mouth interior from its open side."""
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
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical tail-UP geometry to the requested tail direction."""
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
    """Sample canonical cup parameters that fit the original bbox limits after rotation."""
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


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, matching the original stray semantics."""
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


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth_width: int,
    content_count: int,
    occ_cup: np.ndarray,
    occ_block: np.ndarray,
    occ_drop: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int] | None:
    """Place a cup so its widened gravity flow is legal and visually separated."""
    ph, pw = patch.shape
    patch_mask = patch != 0
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_drop[r0:r0 + ph, c0:c0 + pw].any():
            continue

        blocked = occ_block | occ_drop
        blocked = blocked.copy()
        blocked[r0:r0 + ph, c0:c0 + pw] |= patch_mask

        spike_abs = (r0 + spike[0], c0 + spike[1])
        path_bands = _gravity_bands(spike_abs, mouth_width, H, W, blocked)
        if not path_bands:
            continue

        path_cells = _flatten_bands(path_bands)
        if any(bool(occ_cup[r, c]) for r, c in path_cells):
            continue

        visible_depth = min(content_count, len(path_bands))
        if visible_depth <= 0:
            continue
        ejection_cells = _flatten_bands(path_bands[:visible_depth])
        if any(bool(occ_drop[r, c]) for r, c in ejection_cells):
            continue

        return r0, c0, path_cells, ejection_cells, visible_depth

    return None


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally add one isolated pixel away from cup bboxes and waterfall corridors."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return

    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_block: np.ndarray,
    occ_drop: np.ndarray,
) -> dict | None:
    """Build one cup, reserve its gravity waterfall, and commit it to the canvases."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    open_side = OPPOSITE[direction]
    opening = _opening_edge_cells(mouth, (0, 0, ph - 1, pw - 1), open_side)
    mouth_width = _opening_width(opening, open_side)
    if mouth_width <= 0:
        return None

    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        mouth_width,
        content_count,
        occ_cup,
        occ_block,
        occ_drop,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells, visible_depth = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_block[r0:r0 + ph, c0:c0 + pw] |= patch_mask
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
        occ_drop[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "drop_cells": ejection_cells,
        "drop_depth": visible_depth,
        "mouth_width": mouth_width,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _variant_signal_ok(cups_meta: list[dict]) -> bool:
    """Require visible evidence for both gravity and widened-flow dimensions."""
    has_gravity_case = any(meta["direction"] != "D" and meta["drop_depth"] >= 1 for meta in cups_meta)
    has_width_case = any(meta["mouth_width"] >= 2 and meta["drop_depth"] >= 2 for meta in cups_meta)
    return has_gravity_case and has_width_case


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one full grid and verify it with the fused solver."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_block = np.zeros((H, W), dtype=bool)
    occ_drop = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng,
            H,
            W,
            border_color,
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_block,
            occ_drop,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _variant_signal_ok(cups_meta):
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
