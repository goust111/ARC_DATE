# Variant of 3dc255db
# Dimension: D1 — 路径类型 (path)
# Change: straight_ray — 内容像素从尖刺沿尾部方向发射为一条直线... → 对角线弹射

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


# Design analysis for the diagonal-ray variant
# --------------------------------------------
# spec:
#   core_change:
#     original: the tail spike sits on the middle of one bbox edge; content fires orthogonally.
#     variant: the spike sits on one bbox corner; content fires diagonally away from that corner.
#   concrete_geometry:
#     cup_component:
#       - still a single-color 8-connected hollow frame component.
#       - canonical shape is a diagonal funnel whose two border walls both start at the same corner
#         (the spike) and diverge toward the two opposite-adjacent edges.
#       - the mouth is the open corner-region opposite the spike, so the content still lies inside
#         a visually obvious "cup", but the cup converges toward a diagonal tip instead of a side tip.
#     directions:
#       - exactly four tail directions are used: NW, NE, SW, SE.
#       - the spike corner directly determines the ray direction: NW corner -> (-1, -1), etc.
#   behavior_trace:
#     original_small_example:
#       - a cup with spike on top edge floods from bottom edge, finds 3 content cells, erases them,
#         then paints 3 cells straight upward from the spike.
#     variant_small_example:
#       - a cup with spike at the NW bbox corner floods from bottom+right edges, finds 3 content
#         cells, erases them, then paints 3 cells along NW diagonal offsets (-1,-1), (-2,-2), (-3,-3).
#       - the only changed stage is tail interpretation / path drawing; content counting, erasure,
#         and overflow behavior remain identical.
#   complexity_calibration:
#     - keeping four diagonal directions preserves the original "multiple orientations" burden.
#     - using a corner spike prevents accidental regression to the old rule: a human can see that the
#       emitted line does not share the spike's row or column.
#     - shape freedom is limited to a small family of staircase funnels so cups stay recognizable and
#       the solver remains deterministic.
#   implicit_assumption_audit:
#     old_solver_dependencies_that_break:
#       - candidate spikes were searched only on bbox edges excluding corners.
#       - the mouth flood was seeded from one opposite edge only.
#       - the ejection direction used a 4-neighbor vector.
#     replacements:
#       - candidate spikes are bbox corners that belong to the border component.
#       - the mouth flood is seeded from the two bbox edges meeting at the corner opposite the spike.
#       - the ejection direction uses one of four diagonal vectors.
#     preserved_assumptions:
#       - each usable cup has exactly one tail interpretation whose flood reaches a single dominant
#         content color.
#       - content cells of a cup are all inside the mouth region and share one color.
#   placement_legality:
#     - the generator still uses output-first construction and validates `main(input) == output`.
#     - cups keep a 1-cell bbox gap from each other via dilated occupancy checks.
#     - each diagonal ray path from the spike to the grid edge must be clear of other cups / paths.
#     - stray pixels are moved to the opposite diagonal quadrant so they remain visually related to a
#       cup but do not interfere with mouth detection inside the bbox.
#   observability_and_cleanliness:
#     - the spike always occupies exactly one bbox corner and no other bbox corner.
#     - the mouth always opens toward the opposite corner quadrant, so 2-4 examples are enough to
#       show the changed rule.
#     - diagonal rays are long whenever possible; even when clipped by the grid edge, they are still
#       unmistakably diagonal rather than orthogonal.
#   new_solver_assumptions_checked_against_generator:
#     - solver assumes candidate cups have a border-colored spike corner and the opposite corner edges
#       contain at least one non-border seed into the mouth. Generator enforces this by choosing
#       staircase endpoints away from the other three corners.
#     - solver assumes only one corner candidate's opposite-edge flood reaches content. Generator
#       enforces this by constructing cups with exactly one border corner and by keeping stray pixels
#       outside the bbox.


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_H_RANGE = (4, 7)
CUP_W_RANGE = (4, 7)
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIAG_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
MOUTH_SIDES = {
    "NW": ("D", "R"),
    "NE": ("D", "L"),
    "SW": ("U", "R"),
    "SE": ("U", "L"),
}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _corner_coord(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "NW":
        return r0, c0
    if corner == "NE":
        return r0, c1
    if corner == "SW":
        return r1, c0
    if corner == "SE":
        return r1, c1
    raise ValueError(corner)


def _edge_seeds(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    if side == "D":
        return [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    if side == "L":
        return [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    if side == "R":
        return [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]
    raise ValueError(side)


def _flood_from_bbox_sides(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], sides: tuple[str, ...]
) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from the union of selected bbox edges."""
    r0, c0, r1, c1 = bbox
    seeds: list[tuple[int, int]] = []
    for side in sides:
        seeds.extend(_edge_seeds(border_set, bbox, side))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from diagonal cup-frames outward along a 45-degree tail direction.

    Input structure:
    - bg = background.
    - Each cup is a hollow 8-connected border component whose frame converges to exactly one bbox
      corner (the spike) and opens toward the opposite corner region.
    - Content pixels of one color lie inside the mouth interior.

    Rule:
    - Detect the spike corner and thus the diagonal tail direction.
    - Flood the mouth from the two edges meeting at the corner opposite the spike.
    - Erase the content pixels found in that mouth.
    - Emit one diagonal ray pixel per erased content pixel starting just outside the spike, clipping
      at the grid edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 4:
                continue

            bbox = tuple(int(v) for v in bbox_coords(border_coords))
            border_set = {(int(r), int(c)) for r, c in border_coords}
            candidates = [
                (corner, _corner_coord(bbox, corner))
                for corner in ("NW", "NE", "SW", "SE")
                if _corner_coord(bbox, corner) in border_set
            ]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_sides(border_set, bbox, MOUTH_SIDES[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIAG_DIRS[tail_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive integer line with 8-connected steps."""
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    pts: list[tuple[int, int]] = []
    r, c = r0, c0
    while True:
        pts.append((int(r), int(c)))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pts


def _build_canonical_cup(
    h: int, w: int, bottom_exit: int, right_exit: int, color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Build a diagonal funnel whose spike is the NW corner.

    One wall runs from NW to a non-corner point on the bottom edge; the other runs from NW to a
    non-corner point on the right edge. The mouth is the SE-side region between those walls.
    """
    assert h >= 4 and w >= 4
    assert 1 <= bottom_exit <= w - 2
    assert 1 <= right_exit <= h - 2
    patch = np.zeros((h, w), dtype=int)
    for r, c in _bresenham_line((0, 0), (h - 1, bottom_exit)):
        patch[r, c] = color
    for r, c in _bresenham_line((0, 0), (right_exit, w - 1)):
        patch[r, c] = color
    return patch, (0, 0)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, sides: tuple[str, ...]
) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_set = {(int(r), int(c)) for r, c in np.argwhere(patch == int(border_color))}
    bbox = (0, 0, h - 1, w - 1)
    return sorted(_flood_from_bbox_sides(border_set, bbox, sides))


def _flip_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "NW":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "NE":
        return np.fliplr(patch).copy(), (sr, w - 1 - sc), [(r, w - 1 - c) for r, c in mouth]
    if direction == "SW":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "SE":
        return (
            np.flipud(np.fliplr(patch)).copy(),
            (h - 1 - sr, w - 1 - sc),
            [(h - 1 - r, w - 1 - c) for r, c in mouth],
        )
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    h = int(rng.integers(CUP_H_RANGE[0], CUP_H_RANGE[1] + 1))
    w = int(rng.integers(CUP_W_RANGE[0], CUP_W_RANGE[1] + 1))

    max_bottom_exit = min(w - 2, max(1, (w - 1) // 2))
    max_right_exit = min(h - 2, max(1, (h - 1) // 2))
    if max_bottom_exit < 1 or max_right_exit < 1:
        return None

    bottom_exit = int(rng.integers(1, max_bottom_exit + 1))
    right_exit = int(rng.integers(1, max_right_exit + 1))
    return h, w, bottom_exit, right_exit


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
    ph, pw = patch.shape
    dr, dc = DIAG_DIRS[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells: list[tuple[int, int]] = []
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


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "NW":
        mask[r1 + 1 :, c1 + 1 :] = True
    elif tail_dir == "NE":
        mask[r1 + 1 :, :c0] = True
    elif tail_dir == "SW":
        mask[:r0, c1 + 1 :] = True
    else:  # "SE"
        mask[:r0, :c0] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
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
    direction = str(rng.choice(["NW", "NE", "SW", "SE"]))
    shape = _sample_cup_shape(rng)
    if shape is None:
        return None
    h, w, bottom_exit, right_exit = shape

    patch_canon, spike_canon = _build_canonical_cup(h, w, bottom_exit, right_exit, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, sides=("D", "R"))
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _flip_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
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
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
