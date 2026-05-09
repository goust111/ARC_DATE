# Variant of 3dc255db
# Dimension: D1 — 路径类型 (path)
# Change: straight_ray — 内容像素从尖刺沿尾部方向发射为一条直线... → 重力流动
#
# Design analysis
# ----------------
# variant_spec:
#   source_review:
#     - The prompt asks to read source task 28a6681f first. In this workspace I could find
#       /ARC-data/ARC-AGI-2/data/evaluation/28a6681f.json plus mechanism notes saying
#       "flow along neighboring bg cells to left/right/down (like water)", but no local
#       solver source file named 28a6681f.py in the allowed repo paths. I therefore use
#       the available local description only as a baseline for the concept "gravity-driven
#       downward motion" and deliberately do not import any extra branching/sideways flow.
#   a_concretization_and_trace:
#     - Objects stay the same: 1-3 hollow cup/Y/funnel borders, each with one spike cell on
#       one bbox edge and one content color scattered in the mouth interior.
#     - Old behavior: after detecting the spike and counting the content pixels, the solver
#       erased the mouth content and painted the same number of pixels along the spike's own
#       tail direction, e.g. a left-pointing cup emitted a leftward ray.
#     - New behavior: after erasing the mouth content, the pixels "pour out" from the spike
#       under gravity. Regardless of the cup orientation, they occupy consecutive cells in
#       the same column directly below the spike, stopping early at the first non-bg cell or
#       at the bottom edge. One erased content pixel still corresponds to at most one placed
#       output pixel, so content count remains visible and meaningful.
#     - Small example:
#         input: left-pointing cup, 3 content pixels in the mouth.
#         old output: 3 pixels extend horizontally left from the spike.
#         new output: 3 pixels stack vertically below the spike.
#       The object-recognition stage is unchanged; only the output path geometry changes.
#   b_complexity_calibration:
#     - I keep the original cup variety, content scattering, color freedoms, and optional
#       stray pixel. This preserves enough diversity that the task is not reduced to trivial
#       "find a marker and draw downward".
#     - I do not add sideways splitting or iterative liquid simulation because that would
#       create extra path subcases not supported by only 2-4 ARC examples. The key concept
#       is simply that motion direction is now globally "down" instead of shape-derived.
#     - To make the change visible, the generator rejects attempts where every actionable cup
#       points downward; at least one cup must have tail_dir in {U, L, R} and produce a
#       gravity column of length >= 2.
#   c_hidden_assumption_audit_of_original_main:
#     - `_flood_from_bbox_side(...)`: depends only on border cells enclosing the mouth and on
#       the spike-opposite side corresponding to the mouth. This remains valid.
#     - Candidate spike detection on bbox edges: still valid because the cup geometry is not
#       changed.
#     - `passing.append((d, spike, content_col, content_cells))`: still valid; tail_dir is
#       still useful for identifying which bbox side is the mouth, even though it no longer
#       controls the painted trajectory.
#     - Original erase-then-ray block:
#           dr, dc = DIR4[tail_dir]
#           for t in range(1, len(content_cells) + 1): ...
#       This is exactly the old-path assumption that breaks. The path is no longer derived
#       from `tail_dir`, so this code must be replaced with a vertical scan below the spike.
#     - Original one-cup-at-a-time mutation of `out` was harmless because rays were generated
#       in independent directions and the generator guaranteed clear paths to the edge. With
#       gravity, two cups could in principle share a column, so the solver now first collects
#       all actionable cups and erases all their contents before drawing any gravity columns.
#       The generator also enforces non-overlapping gravity outputs to avoid hidden order
#       dependence.
#   d_placement_legality:
#     - Cup bboxes still keep a 1-cell gap.
#     - A new cup cannot overlap existing gravity columns.
#     - Its gravity path is computed against static obstacles: existing cup borders plus the
#       new cup's own border. The path may travel through the cup's own hollow interior, but
#       it may not pass through another cup's bbox, which keeps visuals clean and prevents
#       accidental interaction with another cup's mouth.
#     - Gravity output cells from different cups are forbidden to overlap.
#   e_visual_observability:
#     - The decisive visual clue is that non-downward cups still produce vertical downward
#       output. This is easy to infer from 2-3 examples and does not rely on color identity.
#     - Some cups naturally pour through their own cavity (tail U/L/R), which further makes
#       the "gravity, not tail direction" rule visible.
#   f_visual_cleanliness:
#     - No cup is placed on top of another gravity column.
#     - Gravity paths are prevented from entering other cup bboxes and from crossing other
#       gravity outputs.
#     - Stray pixels are kept away from all cup bboxes and from full gravity corridors so the
#       scene remains legible.
#   g_color_inference:
#     - Colors are unchanged: each cup still has one border color and one content color.
#       The transformation depends on geometry, not on palette value ordering.
#   h_quantifiable_variant_signal:
#     - For post-write validation I will measure:
#         1. fraction of generated samples containing at least one actionable cup with
#            tail_dir != D and gravity output length >= 2;
#         2. fraction of generated samples whose new output differs from the original
#            straight-ray solver on the same generated input.
#       High values on both indicate that the generator is expressing the new dimension.
#   i_new_solver_assumptions_and_generator_guarantees:
#     - Assumption: actionable cups are still unambiguous under the original spike-detection
#       rule. Generator guarantee: unchanged cup-construction logic.
#     - Assumption: gravity outputs for different cups do not need tie-breaking. Generator
#       guarantee: ejection cell sets are disjoint.
#     - Assumption: a cup has at least one free cell below its spike to show any motion.
#       Generator guarantee: every placed cup has gravity capacity >= 1.

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


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from non-border cells on one bbox edge."""
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


def _detect_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Return actionable cups found by the original spike/mouth logic."""
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
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
                mouth = _flood_from_bbox_side(border_set, bbox, opposite[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        cnt[value] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "border_color": int(border_col),
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_color": content_col,
                    "content_cells": content_cells,
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect the same cup objects and their mouth content as in the original task.
    - Erase the mouth content.
    - Starting from each spike, let the content fall straight DOWN under gravity,
      regardless of the cup's tail direction.
    - Place at most one falling pixel per erased content pixel, stopping at the grid bottom
      or at the first non-background obstacle.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    cups = _detect_cups(g, bg)

    out = g.copy()
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    for cup in cups:
        sr, sc = cup["spike"]
        content_col = int(cup["content_color"])
        rr = sr + 1
        remaining = len(cup["content_cells"])
        while remaining > 0 and 0 <= rr < h and 0 <= sc < w and int(out[rr, sc]) == bg:
            out[rr, sc] = content_col
            rr += 1
            remaining -= 1

    return out


# --------------- generator ---------------

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
# ==================================

_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical cup with tail pointing UP."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    width = L + R + 1
    height = max(L + hL, R + hR) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, width - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the hollow mouth-interior of a cup patch."""
    height, width = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(width) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(height - 1, c) for c in range(width) if not border_mask[height - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(height) if not border_mask[r, 0]]
    else:  # "R"
        seeds = [(r, width - 1) for r in range(height) if not border_mask[r, width - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited and not border_mask[nr, nc]:
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
    height, width = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (height - 1 - sr, sc), [(height - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (width - 1 - sc, sr), [(width - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, height - 1 - sr), [(c, height - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample a cup shape that remains within the original bbox bounds after rotation."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    left_arm = int(rng.integers(lo, hi + 1))
    right_arm = int(rng.integers(lo, hi + 1))
    while left_arm + right_arm + 1 > max_w_canon:
        if left_arm >= right_arm:
            left_arm -= 1
        else:
            right_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(left_arm + hL, right_arm + hR) + 1 > max_h_canon:
        if left_arm + hL >= right_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return left_arm, right_arm, hL, hR


def _gravity_run(h: int, spike_abs: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    """Cells directly below the spike until the first static obstacle or the bottom edge."""
    sr, sc = spike_abs
    cells = []
    rr = sr + 1
    while rr < h and not bool(blocked[rr, sc]):
        cells.append((rr, sc))
        rr += 1
    return cells


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    content_count: int,
    occ_cup: np.ndarray,
    occ_block: np.ndarray,
    occ_drop: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so its gravity output is valid and visually separated."""
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

        blocked = occ_block.copy()
        blocked[r0:r0 + ph, c0:c0 + pw] |= patch_mask

        spike_abs = (r0 + spike[0], c0 + spike[1])
        path_cells = _gravity_run(H, spike_abs, blocked)
        if not path_cells:
            continue
        if any(bool(occ_cup[r, c]) for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells))
        ejection_cells = path_cells[:eject_len]
        if any(bool(occ_drop[r, c]) for r, c in ejection_cells):
            continue
        return r0, c0, path_cells, ejection_cells

    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, keeping the original stray semantics."""
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


def _try_place_stray(rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]) -> None:
    """Optionally add an isolated stray pixel, away from all cup regions and gravity corridors."""
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
    """Build one cup, place it, and commit its static border plus gravity output."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    left_arm, right_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(left_arm, right_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, content_count, occ_cup, occ_block, occ_drop)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

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
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "drop_cells": ejection_cells,
        "gravity_capacity": len(path_cells),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter content pixels inside each cup mouth."""
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
    """Require at least one clearly non-original gravity case in each generated grid."""
    return any(meta["direction"] != "D" and len(meta["drop_cells"]) >= 2 for meta in cups_meta)


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one full input grid and verify it by running the solver."""
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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
