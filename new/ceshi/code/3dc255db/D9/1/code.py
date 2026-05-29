# Variant of 3dc255db
# Dimension: D9 — 杯形结构形状 (src)
# Change: Y形/漏斗形 — 两臂对角收敛到单像素尖刺，臂长可不对称，底部开口... → L形框架

"""
design_analysis:
  concretization:
    - Each cup is now a 1-pixel-thick axis-aligned L component with exactly one turn.
    - The spike is the endpoint of the shorter leg. The tail ejects straight outward by
      continuing that short leg past its endpoint.
    - The other endpoint is the mouth end. Content sits in the open rectangle spanned by the
      two legs on the concave side of the L, inside the cup's bbox.
    - Canonical example (tail=U): border = left column + bottom row, spike = top-left,
      mouth end = bottom-right, content region = rows [top..bottom-1] x cols [left+1..right].
  behavior_trace:
    - Original solver looked for a bbox edge with exactly one non-corner border cell, then
      flooded from the opposite bbox edge. That only works because Y/funnel cups encode the
      spike on a non-corner edge and place the mouth opposite the tail.
    - In this variant the spike is a corner endpoint of an L, and the mouth is at the other
      endpoint after a 90-degree turn. So "single non-corner edge cell" is no longer a valid
      cue, and "mouth opposite tail" is false.
    - New solver therefore extracts the unique 4-connected path of the border, compares the two
      leg lengths to locate the shorter endpoint, and then reconstructs the content rectangle
      from that oriented path.
  complexity_calibration:
    - The L keeps only one salient turn and two endpoints, so the rule is still legible from a
      few ARC examples.
    - Leg lengths vary but are forced to be unequal, so the short-leg spike cue is always
      visible and never ambiguous.
  implicit_assumption_audit:
    - Old code assumed the spike is on a bbox edge but not a corner: invalid here.
    - Old code assumed flooding from the side opposite the spike reaches the content: invalid
      because the mouth direction is now orthogonal to the tail.
    - Ejection itself remains compatible once the spike endpoint and outward leg direction are
      recovered.
  placement_legality:
    - Generator still enforces non-overlapping cup bboxes with a 1-cell gap and keeps the
      ejection path clear to the grid edge.
    - Content is sampled only from the reconstructed concave rectangle, so solver/generator
      agree exactly on which pixels must be erased and counted.
  observability:
    - The visual change is explicit: all cups are cornered L frames rather than funnels.
    - Tail direction is inferred from the endpoint opposite the content-filled concave rectangle,
      not from color ids or hidden ordering.
  cleanliness:
    - Each cup owns a clean bbox with sparse interior content and a clear output ray.
    - Optional stray pixels stay outside cup bboxes and outside ejection paths.
  color_reasoning:
    - Colors keep the original structure: each cup has one border color and one content color.
    - No new color mapping rule is introduced.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
VEC_TO_DIR = {v: k for k, v in DIR4.items()}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _neighbors4_in_set(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIR4.values():
        nxt = (r + dr, c + dc)
        if nxt in cell_set:
            out.append(nxt)
    return out


def _extract_l_path(border_set: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Return the ordered path of a simple 4-connected L component, or None."""
    if not border_set:
        return None

    neighbors = {cell: _neighbors4_in_set(cell, border_set) for cell in border_set}
    endpoints = [cell for cell, ns in neighbors.items() if len(ns) == 1]
    if len(endpoints) != 2:
        return None
    if any(len(ns) not in (1, 2) for ns in neighbors.values()):
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

    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    if not dirs or any(d not in VEC_TO_DIR for d in dirs):
        return None

    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    if len(runs) != 2:
        return None
    d1, d2 = runs[0][0], runs[1][0]
    if d1[0] * d2[0] + d1[1] * d2[1] != 0:
        return None
    return path


def _axis_values_excluding_end(a: int, b: int) -> list[int]:
    """Inclusive of a, exclusive of b, following the sign from a to b."""
    step = 1 if b > a else -1
    return list(range(a, b, step))


def _axis_values_excluding_start(a: int, b: int) -> list[int]:
    """Exclusive of a, inclusive of b, following the sign from a to b."""
    step = 1 if b > a else -1
    return list(range(a + step, b + step, step))


def _candidate_from_path(
    path: list[tuple[int, int]],
) -> tuple[str, str, tuple[int, int], list[tuple[int, int]]] | None:
    """Interpret ``path[0]`` as the spike endpoint and reconstruct its content rectangle."""
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
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
    tail_dir = OPPOSITE[VEC_TO_DIR[d1]]
    mouth_dir = VEC_TO_DIR[d2]
    return tail_dir, mouth_dir, spike, interior


def _order_path_from_short_leg(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Orient the L path so ``path[0]`` is the endpoint of the strictly shorter leg."""
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    if len(runs) != 2:
        return None

    first_len = runs[0][1]
    second_len = runs[1][1]
    if first_len == second_len:
        return None
    if first_len < second_len:
        return path
    return list(reversed(path))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from L-shaped cup frames outward along each cup's spike direction.

    Input structure:
    - bg = background.
    - Each cup border is a 1-pixel-thick L of a single color.
    - One endpoint is the spike; the content sits in the concave rectangle on the inner side
      of the L, nearer the other endpoint.

    Rule:
    - Recover the L path and test both endpoints as spike candidates.
    - The correct spike candidate is the one whose implied concave rectangle contains content.
    - Erase those content pixels and emit a straight line from the spike outward along the tail.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            border_set = {(int(r), int(c)) for r, c in border_coords}
            path = _extract_l_path(border_set)
            if path is None:
                continue

            ordered_path = _order_path_from_short_leg(path)
            if ordered_path is None:
                continue
            candidate = _candidate_from_path(ordered_path)
            if candidate is None:
                continue
            tail_dir, _mouth_dir, spike, interior = candidate

            cnt = Counter()
            for r, c in interior:
                v = int(g[r, c])
                if v != bg and v != int(border_col):
                    cnt[v] += 1
            if not cnt:
                continue

            top = cnt.most_common(2)
            if len(top) > 1 and top[0][1] == top[1][1]:
                continue
            content_col = int(top[0][0])
            content_cells = [p for p in interior if int(g[p[0], p[1]]) == content_col]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            sr, sc = spike
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
CUP_BBOX_MAX_UD = (8, 7)  # (max_h, max_w) when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # (max_h, max_w) when tail is L or R
CUP_BBOX_MIN = (3, 4)  # canonical short-leg x long-leg minimum
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
MOUTH_DIR_FOR_TAIL = {"U": "R", "D": "R", "L": "U", "R": "D"}
# ==================================


def _build_canonical_cup(height: int, width: int, color: int) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Build a canonical L with tail pointing UP: left wall + bottom wall."""
    assert height >= 3 and width >= 3
    g = np.zeros((height, width), dtype=int)
    g[:, 0] = color
    g[-1, :] = color
    mouth = [(r, c) for r in range(height - 1) for c in range(1, width)]
    return g, (0, 0), mouth


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Transform the canonical tail=U patch to the target tail direction."""
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


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int]:
    """Pick canonical bbox dimensions, enforcing short-leg spike asymmetry."""
    if direction in ("U", "D"):
        max_h, max_w = CUP_BBOX_MAX_UD
    else:
        max_h, max_w = CUP_BBOX_MAX_LR

    min_h, min_w = CUP_BBOX_MIN
    height = int(rng.integers(min_h, max_h + 1))
    width_lo = max(min_w, height + 1)
    if width_lo > max_w:
        height = int(rng.integers(min_h, max_w))
        width_lo = max(min_w, height + 1)
    width = int(rng.integers(width_lo, max_w + 1))
    return height, width


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
    """Find a valid placement whose bbox is isolated and whose ejection path is clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
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


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], mouth_dir: str) -> np.ndarray:
    """Cells outside the bbox on the mouth-opening side, restricted to the perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if mouth_dir == "R":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif mouth_dir == "L":
        mask[r0:r1 + 1, :c0] = True
    elif mouth_dir == "D":
        mask[r1 + 1:, c0:c1 + 1] = True
    else:  # "U"
        mask[:r0, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated reused-color pixel beyond a cup's mouth side."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for (r, c) in m["path_cells"]:
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
    """Build one L cup, place it, and commit its border plus output ejection line."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    height, width = _sample_cup_shape(rng, direction)

    patch_canon, spike_canon, mouth_canon = _build_canonical_cup(height, width, border_color)
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement
    ph, pw = patch.shape

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for (r, c) in ejection_cells:
        canvas_out[r, c] = content_color
    for (r, c) in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "mouth_dir": MOUTH_DIR_FOR_TAIL[direction],
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside that cup's concave rectangle."""
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
    """Output-first generation with oracle verification against ``main``."""
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
            rng,
            H,
            W,
            border_color,
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
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
