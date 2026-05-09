# Variant of 3dc255db
# Dimension: D5 — 轨迹涂色模式 (trajectory)
# Change: 连续单色线段——弹射线为内容像素颜色的连续直线... → 间歇式轨迹
#
# Design analysis
# ===============
# spec:
#   variant_goal:
#     keep: cup/Y/funnel detection, interior-content identification, tail-direction inference,
#       content erasure, axis-aligned outward launch, optional stray pixels, output-first generator.
#     change: the launched trail is no longer painted on every step. Instead, erased content pixels
#       map to odd offsets from the spike: 1, 3, 5, ...; even offsets stay background.
#   concrete_behavior:
#     appearance:
#       - Input still shows 1-3 hollow cups with scattered interior content.
#       - Output still removes the interior content and launches it from the spike outward.
#       - The visible difference is along the launch line: colored, blank, colored, blank...
#     traced_example:
#       - Suppose a tail points right and the spike is at (5, 4).
#       - Original with 3 content pixels paints (5,5), (5,6), (5,7).
#       - Variant with 3 content pixels paints (5,5), leaves (5,6) blank, paints (5,7),
#         leaves (5,8) blank, paints (5,9), dropping any step beyond the grid edge.
#       - The only semantic change is the mapping from "k-th erased payload pixel" to
#         "2*k+1 cells away from the spike" instead of "k+1 cells away".
#   complexity_calibration:
#     kept_simple:
#       - Only one intermittent mode exists globally: fixed paint-gap-paint-gap starting with paint.
#       - No extra color mapping, no multiple dash lengths, no per-cup mode switches.
#     made_observable:
#       - Generator forces every cup to have at least 2 interior payload pixels.
#       - Placement also forces at least 2 odd-offset launch cells in-bounds, so at least one
#         visible gap exists per launched trail; this avoids samples that look identical to the original.
#   implicit_assumption_review:
#     unaffected_solver_parts:
#       - Border-component discovery only assumes cup borders are 8-connected and large enough.
#       - Tail-candidate detection only assumes one bbox edge contains the lone spike.
#       - Flooding from the opposite bbox side only assumes the mouth interior is reachable there.
#       - Content color selection still assumes one dominant non-bg, non-border color inside the mouth.
#     changed_solver_part:
#       - Original code used `for t in range(1, len(content_cells)+1)` and painted every step.
#       - That loop implicitly assumed "every traveled cell is colored". This is false in the variant.
#       - The fix is to paint only odd offsets `t = 1, 3, 5, ...`; even offsets must remain untouched
#         or the solver would silently collapse back to the original continuous-line behavior.
#   placement_legality:
#     - Cup bboxes still keep a 1-cell gap via dilated occupancy.
#     - The full launch corridor to the edge is still reserved in `occ_path`, including future blank
#       gap cells, so another cup or stray pixel cannot sit inside the dashed trajectory.
#     - Output paints only the odd-indexed cells of that corridor.
#   visual_observability:
#     - Humans can infer the rule from 2-4 examples because the cup identification logic is unchanged
#       and only the launched line differs systematically in the same way each time.
#     - The generator avoids one-pixel launches, which would hide the alternating pattern.
#   cleanliness:
#     - No other object is allowed on the launch corridor, including skipped cells.
#     - This keeps the blank cells interpretable as intentional gaps rather than occlusion.
#   color_inference:
#     - Colors keep the original meaning: border color marks the cup, content color marks payload.
#     - The variant changes only spatial occupancy along the trajectory, so no new color reasoning is required.
#   new_solver_assumptions_and_generator_support:
#     - main() now assumes that if a cup is intended to demonstrate the variant, the corridor has enough
#       in-bounds odd positions for the alternating pattern to be visible.
#     - generate_input() guarantees this by requiring at least two odd-offset corridor cells per cup.

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


def _flood_from_bbox_side(border_set: set, bbox: tuple, side: str) -> set:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge,
    stopping at border cells. Returns the set of reached cells (includes seeds)."""
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


def _odd_offset_cells(
    spike: tuple[int, int], direction: str, count: int, H: int, W: int
) -> list[tuple[int, int]]:
    """Return the odd-offset cells 1,3,5,... from the spike, clipped by the grid edge."""
    dr, dc = DIR4[direction]
    sr, sc = spike
    cells: list[tuple[int, int]] = []
    for k in range(int(count)):
        t = 2 * k + 1
        rr = sr + dr * t
        cc = sc + dc * t
        if not (0 <= rr < H and 0 <= cc < W):
            break
        cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Variant rule:
    - The launched trail is intermittent, not continuous.
    - If a cup has N payload pixels, erase them from the mouth and paint the payload color at
      odd offsets 1,3,5,... from the spike, stopping at the grid edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            # Candidate tail spikes: single border cell on a bbox edge, not at a corner.
            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            # Each candidate: flood from opposite bbox side; keep candidates whose flood
            # reaches content (non-bg, non-border) cells.
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

            # Only act on cups with an unambiguous tail.
            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            # Erase content, then eject on odd offsets only.
            for r, c in content_cells:
                out[r, c] = bg

            for rr, cc in _odd_offset_cells(spike, tail_dir, len(content_cells), H, W):
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)  # (max_h, max_w) when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # (max_h, max_w) when tail is L or R

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_COUNT_MIN = 2
MIN_VISIBLE_TRAIL_PIXELS = 2

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(
    L: int, R: int, hL: int, hR: int, color: int
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
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from cells on the given bbox edge, stopping at
    the border. Returns the sorted list of interior coordinates (seeds included)."""
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch, its spike and mouth coords to the target tail direction."""
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
    """Pick (L, R, hL, hR) parameters for a canonical cup, clamped so the patch fits the bbox spec."""
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
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a valid placement offset such that the cup's bbox has a 1-cell gap from other cups
    and the intermittent ejection corridor is clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
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

        odd_path_cells = path_cells[::2]
        visible_count = min(content_count, len(odd_path_cells))
        if visible_count < MIN_VISIBLE_TRAIL_PIXELS:
            continue
        return r0, c0, path_cells, odd_path_cells[:visible_count]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean (H, W) mask of cells on the mouth side of ``bbox`` (opposite the tail direction),
    restricted to the cup's perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:  # "U"
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    """With probability ``STRAY_PROB``, place an isolated pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for (r, c) in m["path_cells"]:
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
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup (shape + placement + ejection line) and commit it to the canvases."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    content_count = int(rng.integers(CONTENT_COUNT_MIN, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    for (r, c) in ejection_cells:
        canvas_out[r, c] = content_color
    for (r, c) in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "visible_trail_count": len(ejection_cells),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content_count pixels at random inside its mouth interior."""
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
    """Single end-to-end generation attempt. Returns a valid input grid or None on failure."""
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

        meta = _add_cup(rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path)
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
