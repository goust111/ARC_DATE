# Combo variant of 3dc255db (index 40)
# combo_spec: D9/1+D14/1
# Name: L形框架+垂直弹射

"""
design_analysis:
  concretization:
    - Each object is a 1-pixel-thick axis-aligned L with exactly one elbow and two unequal legs.
    - The longer leg is the L's main axis. The spike is the endpoint of that longer leg, not the
      shorter one. The other endpoint is the mouth end.
    - Content occupies the open concave rectangle inside the L, exactly as implied by the two legs.
    - Ejection starts from the spike but does not continue the main axis. Instead it goes
      perpendicular to the main axis, toward the side where the short arm lies.
    - Canonical example:
        border = left column + bottom row
        spike = top-left
        main axis = downward left column
        short arm side = right
        content interior = rows [top..bottom-1] x cols [left+1..right]
        output line = horizontal rightward ray from the spike
  behavior_trace:
    - Original task:
        1. detect a funnel/Y spike on a bbox edge
        2. flood the opposite side to get the mouth
        3. erase content
        4. eject along the tail axis
    - D9 alone replaced the cup with an L and usually made the spike the short-leg endpoint,
      so the output still followed the short leg's outward continuation.
    - This fused variant changes both the recognizable object grammar and the direction logic:
        1. detect a simple 4-connected L path
        2. orient that path so it starts at the endpoint of the strictly longer leg
        3. reconstruct the concave content rectangle from the oriented path
        4. erase the content
        5. eject from the spike along the short-arm direction, which is perpendicular to the
           main axis
    - Concrete mini-example:
        input border:
          4 0 0 0
          4 0 0 0
          4 0 0 0
          4 7 7 0
        content inside:
          rows 0..2, cols 1..2 may contain color 2
        old-L behavior would continue upward/downward depending on which endpoint was treated as
        the spike; this fused behavior always sprays rightward because the long leg is vertical
        and the short arm is on the right side.
  complexity_calibration:
    - The rule is still learnable from a few ARC examples because each cup has only one elbow,
      two endpoints, and one binary asymmetry (which leg is longer).
    - The fused change is visually obvious: objects are Ls rather than funnels, and the output
      ray is orthogonal to the visually dominant leg.
    - The generator enforces strict leg-length inequality, so "main axis" is always recoverable.
    - Rotations are limited to four axis-aligned cases, which keeps the hypothesis space small.
  implicit_assumption_audit:
    - Original main() assumed:
        * the spike is a unique non-corner bbox-edge cell: false for Ls
        * the mouth lies opposite the spike across the bbox: false for Ls
        * eject_dir == tail_dir: false in the fused task
    - New solver assumptions:
        * each border component is a simple 4-connected path with exactly two endpoints
        * that path has exactly one 90-degree turn
        * the two run lengths are unequal, so the longer run defines the main axis
        * the implied interior rectangle is non-empty and contains the cup's content color
      The generator explicitly constructs every cup to satisfy these conditions.
    - Because the output ray now follows the short-arm side, the adjacent cell from the spike in
      that direction must be empty. The canonical L and all its rotations satisfy this.
  placement_legality:
    - Generator remains output-first.
    - Cup bboxes keep a 1-cell gap through dilated occupancy checks.
    - Reserved ejection paths use the perpendicular fused direction, not the original tail axis.
    - Content is only sampled from the reconstructed concave rectangle, so solver and generator
      agree on which pixels are erased and counted.
    - Optional stray pixels are kept outside all cup bboxes and reserved paths.
  visual_observability:
    - A human can read the longer leg as the dominant axis and see that the output goes sideways
      relative to it.
    - Because the side choice comes from the short arm's location, not from hidden color ids or
      tie-breaking order, the mapping is visible directly in the shape.
    - Rotated examples expose all four axis-aligned cases without requiring any invisible state.
  cleanliness:
    - Each cup owns one bbox, one content color, and one straight output ray.
    - No object needs overlapping marks to encode direction.
    - Stray pixels remain isolated distractors rather than competing structure.
  color_reasoning:
    - Colors keep the original semantics: one border color and one content color per cup.
    - No new color remapping is introduced, so the fusion is purely geometric.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
VEC_TO_DIR = {v: k for k, v in DIR4.items()}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
ROTATE_CCW_DIR = {"U": "L", "L": "D", "D": "R", "R": "U"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _neighbors4_in_set(
    cell: tuple[int, int], cell_set: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4.values() if (r + dr, c + dc) in cell_set]


def _run_lengths(path: list[tuple[int, int]]) -> list[tuple[tuple[int, int], int]]:
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    if not dirs or any(d not in VEC_TO_DIR for d in dirs):
        return []

    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    return runs


def _extract_l_path(border_set: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Return an ordered simple 4-connected L path, or None."""
    if not border_set:
        return None

    neighbors = {cell: _neighbors4_in_set(cell, border_set) for cell in border_set}
    endpoints = [cell for cell, ns in neighbors.items() if len(ns) == 1]
    if len(endpoints) != 2 or any(len(ns) not in (1, 2) for ns in neighbors.values()):
        return None

    start, goal = endpoints
    path = [start]
    prev = None
    cur = start
    while cur != goal:
        nxts = [nxt for nxt in neighbors[cur] if nxt != prev]
        if len(nxts) != 1:
            return None
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
        if len(path) > len(border_set):
            return None

    if len(path) != len(border_set):
        return None

    runs = _run_lengths(path)
    if len(runs) != 2:
        return None
    d1, d2 = runs[0][0], runs[1][0]
    if d1[0] * d2[0] + d1[1] * d2[1] != 0:
        return None
    return path


def _axis_values_excluding_end(a: int, b: int) -> list[int]:
    step = 1 if b > a else -1
    return list(range(a, b, step))


def _axis_values_excluding_start(a: int, b: int) -> list[int]:
    step = 1 if b > a else -1
    return list(range(a + step, b + step, step))


def _orient_path_from_long_leg(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Orient the L so path[0] is the endpoint of the strictly longer leg."""
    runs = _run_lengths(path)
    if len(runs) != 2:
        return None

    first_len = runs[0][1]
    second_len = runs[1][1]
    if first_len == second_len:
        return None
    if first_len > second_len:
        return path
    return list(reversed(path))


def _candidate_from_path(
    path: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]], str, str] | None:
    """Interpret the path as spike -> long leg -> elbow -> short leg -> mouth."""
    runs = _run_lengths(path)
    if len(runs) != 2:
        return None

    d1, first_len = runs[0]
    d2, _ = runs[1]
    spike = path[0]
    elbow = path[first_len]
    mouth = path[-1]

    if d1[0] != 0:
        row_vals = _axis_values_excluding_end(spike[0], elbow[0])
        col_vals = _axis_values_excluding_start(elbow[1], mouth[1])
    else:
        row_vals = _axis_values_excluding_start(elbow[0], mouth[0])
        col_vals = _axis_values_excluding_end(spike[1], elbow[1])

    if not row_vals or not col_vals:
        return None

    interior = [(r, c) for r in row_vals for c in col_vals]
    main_axis_dir = VEC_TO_DIR[d1]
    eject_dir = VEC_TO_DIR[d2]
    return spike, interior, main_axis_dir, eject_dir


def _detect_l_cup(
    g: np.ndarray, border_coords: np.ndarray, border_col: int, bg: int
) -> tuple[tuple[int, int], list[tuple[int, int]], str, str, int, list[tuple[int, int]]] | None:
    border_set = {(int(r), int(c)) for r, c in border_coords}
    path = _extract_l_path(border_set)
    if path is None:
        return None

    ordered_path = _orient_path_from_long_leg(path)
    if ordered_path is None:
        return None

    candidate = _candidate_from_path(ordered_path)
    if candidate is None:
        return None
    spike, interior, main_axis_dir, eject_dir = candidate

    cnt = Counter()
    for r, c in interior:
        v = int(g[r, c])
        if v != bg and v != int(border_col):
            cnt[v] += 1
    if not cnt:
        return None

    top = cnt.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    content_col = int(top[0][0])
    content_cells = [cell for cell in interior if int(g[cell[0], cell[1]]) == content_col]
    return spike, interior, main_axis_dir, eject_dir, content_col, content_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from L frames perpendicular to each L's main axis.

    The main axis is the longer leg of the L. The spike is the endpoint of that long leg.
    The output line leaves the spike on the side occupied by the shorter arm.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            detected = _detect_l_cup(g, border_coords, int(border_col), bg)
            if detected is None:
                continue
            spike, _interior, _axis_dir, eject_dir, content_col, content_cells = detected

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


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
CUP_BBOX_MIN = (4, 3)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300


def _rotate_dir(direction: str, k: int) -> str:
    out = direction
    for _ in range(k % 4):
        out = ROTATE_CCW_DIR[out]
    return out


def _build_canonical_cup(
    height: int, width: int, color: int
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Canonical fused L: long vertical leg on the left, short horizontal arm at bottom."""
    assert height >= 4 and width >= 3 and height > width
    g = np.zeros((height, width), dtype=int)
    g[:, 0] = color
    g[-1, :] = color
    mouth = [(r, c) for r in range(height - 1) for c in range(1, width)]
    return g, (0, 0), mouth


def _rotate_patch_k(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], k: int
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate patch/cells by k quarter turns CCW."""
    k %= 4
    out_patch = np.rot90(patch, k=k).copy()
    cur_h, cur_w = patch.shape
    sr, sc = spike
    cells = list(mouth)

    for _ in range(k):
        sr, sc = cur_w - 1 - sc, sr
        cells = [(cur_w - 1 - c, r) for r, c in cells]
        cur_h, cur_w = cur_w, cur_h

    return out_patch, (sr, sc), cells


def _sample_cup_shape(rng: np.random.Generator, rotation_k: int) -> tuple[int, int]:
    """Pick canonical dimensions with a strict main-axis/short-arm asymmetry."""
    if rotation_k % 2 == 0:
        max_h, max_w = CUP_BBOX_MAX_UD
    else:
        max_h, max_w = CUP_BBOX_MAX_LR[1], CUP_BBOX_MAX_LR[0]

    min_h, min_w = CUP_BBOX_MIN
    while True:
        height = int(rng.integers(min_h, max_h + 1))
        width_hi = min(max_w, height - 1)
        if width_hi < min_w:
            continue
        width = int(rng.integers(min_w, width_hi + 1))
        if height > width:
            return height, width


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    eject_dir: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a valid placement whose bbox is isolated and whose fused ejection path is clear."""
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


def _stray_candidate_mask(
    H: int, W: int, bbox: tuple[int, int, int, int], mouth_dir: str
) -> np.ndarray:
    """Cells outside the bbox on the visible mouth/short-arm side."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if mouth_dir == "R":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif mouth_dir == "L":
        mask[r0:r1 + 1, :c0] = True
    elif mouth_dir == "D":
        mask[r1 + 1:, c0:c1 + 1] = True
    else:
        mask[:r0, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    H, W = canvas_in.shape
    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["mouth_dir"]) & ~forbidden
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
    """Build one fused L cup, place it, and commit border plus output ray."""
    rotation_k = int(rng.integers(0, 4))
    height, width = _sample_cup_shape(rng, rotation_k)

    patch_canon, spike_canon, mouth_canon = _build_canonical_cup(height, width, border_color)
    patch, spike, mouth = _rotate_patch_k(patch_canon, spike_canon, mouth_canon, rotation_k)
    axis_dir = _rotate_dir("D", rotation_k)
    eject_dir = _rotate_dir("R", rotation_k)
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, eject_dir, content_count, occ_cup, occ_path)
    if placement is None:
        return None
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

    return {
        "border_color": border_color,
        "content_color": content_color,
        "axis_dir": axis_dir,
        "eject_dir": eject_dir,
        "mouth_dir": eject_dir,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(
    rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]
) -> bool:
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


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
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
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

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
