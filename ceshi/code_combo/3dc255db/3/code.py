# Combo variant of 3dc255db (index 3)
# combo_spec: D1/0+D4/0
# Name: 对角弹射+障碍物停止

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


# Design analysis for combo D1/0 + D4/0
# --------------------------------------
# a) Concrete behavior and trace:
#    - Relative to the source task, two stages change together:
#      1. Cup geometry/tail interpretation becomes diagonal. A usable cup is now a hollow
#         8-connected funnel converging to exactly one bbox corner, and the tail direction is one
#         of NW/NE/SW/SE determined by that corner.
#      2. Ejection no longer runs until the grid edge. It advances cell-by-cell on that diagonal
#         and stops at the first non-background pixel already present in the input. The blocker is
#         preserved, and any remaining content is dropped.
#    - Small trace:
#      input cup has spike at NW corner, mouth toward SE, and 4 content pixels in the mouth.
#      The diagonal cells after the spike are [bg, bg, obstacle, bg, bg].
#      Original task: it would search for a side spike, so the geometry would not even parse.
#      D1-only: it would parse the corner spike and paint 4 diagonal cells, overwriting the
#      obstacle position.
#      Combo: it parses the same diagonal cup, erases the 4 mouth pixels, paints the first 2
#      diagonal bg cells, sees the obstacle on the third step, stops, and discards the last 2.
# b) Complexity calibration:
#    - The generator gives every cup one diagonal obstacle on its own tail ray, and forces
#      content_count > free_run_before_obstacle. That makes the stopping rule active in every cup
#      instead of appearing only as a rare corner case.
#    - Cup orientation is limited to the four diagonal directions and the obstacle is always on the
#      same ray that the emitted line follows, so the rule is varied but still legible from 2-4
#      examples.
# c) Implicit-assumption audit of the original main():
#    - Original assumption "spike is the unique non-corner singleton on a bbox edge" breaks.
#      Replacement: spike candidates are bbox corners occupied by the border.
#    - Original assumption "flood from exactly one opposite edge" breaks.
#      Replacement: for a diagonal spike, flood from the two bbox edges meeting at the opposite
#      corner quadrant.
#    - Original assumption "tail vector is 4-neighbor orthogonal" breaks.
#      Replacement: use one of the four 45-degree diagonal vectors.
#    - Original assumption "every in-bounds tail cell is writable" breaks.
#      Replacement: consult the original input grid `g` at each diagonal step and stop at the first
#      non-bg blocker. Reading `g` rather than `out` avoids order-dependent false blockers.
# d) Placement legality enforced by the generator:
#    - Cup bboxes keep an 8-neighbor gap via dilated occupancy checks.
#    - The full diagonal ray for each cup is reserved, so no other cup or obstacle can land on it.
#    - Obstacles are also kept 8-neighbor separated from cup patches and from each other, preventing
#      accidental merging into border components.
# e) Visual observability:
#    - The corner spike makes the path visibly diagonal rather than orthogonal.
#    - A lone obstacle pixel is visibly sitting ahead of the spike, and the output line ends just
#      before it. Because obstacle color never equals the content color, the stop is readable.
# f) Visual cleanliness:
#    - The original optional mouth-side stray noise is removed in this combo. The diagonal cup and
#      the ray obstacle are already the only intended clues; removing extra noise keeps examples
#      sparse and readable.
# g) Color inferability:
#    - No hidden color ordering is used. Cup border color identifies the frame, one distinct content
#      color fills the mouth, and the obstacle may be any non-content color. Its role is conveyed by
#      position on the ray, not by numeric color id.
# h) Quantitative anti-degeneration target:
#    - For every generated cup, free_run_before_obstacle is at least 2 and content_count is strictly
#      larger, so the combo output differs from the diagonal-no-obstacle baseline on every cup.
# i) New solver assumptions checked against the generator:
#    - Solver assumes usable cups have exactly one border corner whose opposite-edge flood reaches a
#      dominant content color. The generator enforces a single spike corner and keeps the obstacle
#      outside the bbox.
#    - Solver assumes any blocker is already present in the input and lies on the same diagonal ray.
#      The generator explicitly places exactly one obstacle on each reserved ray.


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_H_RANGE = (4, 7)
CUP_W_RANGE = (4, 7)
CONTENT_COUNT_MAX = 7

# Obstacle must be at least 3 diagonal steps away from the spike, so the visible free prefix is >= 2.
OBSTACLE_DISTANCE_MIN = 3

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
STRUCT8 = np.ones((3, 3), dtype=bool)


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


def _edge_seeds(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
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

    Combo rule:
    - Detect a corner spike and infer one of four diagonal tail directions.
    - Flood the mouth from the two bbox edges adjacent to the opposite corner.
    - Erase the mouth content.
    - Emit content color diagonally from the spike, but stop before the first non-bg obstacle
      already present on that ray in the input. The obstacle cell is not overwritten.
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
                if int(g[rr, cc]) != bg:
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


def _choose_obstacle_color(rng: np.random.Generator, content_color: int) -> int:
    choices = [c for c in range(1, 10) if c != content_color]
    return int(rng.choice(choices))


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    max_content_count: int,
    occ_bbox: np.ndarray,
    occ_path: np.ndarray,
    occ_obstacle: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], int] | None:
    """
    Find a placement with a reserved diagonal ray and one obstacle on that ray.

    Returns:
      (r0, c0, full_path, obstacle_cell, ejection_cells, content_count)
    """
    ph, pw = patch.shape
    dr, dc = DIAG_DIRS[direction]
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=STRUCT8)
    occ_obstacle_dilated = ndimage.binary_dilation(occ_obstacle, structure=STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
        if occ_bbox_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_obstacle_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        full_path: list[tuple[int, int]] = []
        rr, cc = sr_abs + dr, sc_abs + dc
        while 0 <= rr < H and 0 <= cc < W:
            full_path.append((rr, cc))
            rr += dr
            cc += dc

        if len(full_path) < OBSTACLE_DISTANCE_MIN:
            continue
        if any(occ_bbox[r, c] or occ_path[r, c] for r, c in full_path):
            continue
        if any(occ_obstacle_dilated[r, c] for r, c in full_path):
            continue

        feasible_obstacle_indices = [
            idx for idx in range(OBSTACLE_DISTANCE_MIN - 1, len(full_path))
            if idx < max_content_count
        ]
        if not feasible_obstacle_indices:
            continue

        obstacle_idx = int(rng.choice(feasible_obstacle_indices))
        free_run = obstacle_idx
        min_content_count = free_run + 1
        if min_content_count > max_content_count:
            continue

        content_count = int(rng.integers(min_content_count, max_content_count + 1))
        obstacle_cell = full_path[obstacle_idx]
        ejection_cells = full_path[:free_run]
        return r0, c0, full_path, obstacle_cell, ejection_cells, content_count

    return None


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_bbox: np.ndarray,
    occ_path: np.ndarray,
    occ_obstacle: np.ndarray,
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
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))
    if max_content_count < OBSTACLE_DISTANCE_MIN:
        return None

    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        direction,
        max_content_count,
        occ_bbox,
        occ_path,
        occ_obstacle,
    )
    if placement is None:
        return None
    r0, c0, path_cells, obstacle_cell, ejection_cells, content_count = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_bbox[r0:r0 + ph, c0:c0 + pw] = True

    obstacle_color = _choose_obstacle_color(rng, content_color)
    orow, ocol = obstacle_cell
    canvas_in[orow, ocol] = obstacle_color
    canvas_out[orow, ocol] = obstacle_color
    occ_obstacle[orow, ocol] = True

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
        "obstacle_cell": obstacle_cell,
        "free_run": len(ejection_cells),
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

    occ_bbox = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    occ_obstacle = np.zeros((H, W), dtype=bool)

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
            occ_bbox,
            occ_path,
            occ_obstacle,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

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
