# Combo variant of 3dc255db (index 15)
# combo_spec: D4/1+D5/1+D6/1
# Name: 目标标记+渐变轨迹+边框擦除
#
# Design analysis
# ===============
# a) Concrete behavior and flow trace
# - Base task: each hollow cup frame has a single-pixel spike on one bbox edge and
#   scattered same-color content inside the mouth. The original output erases the
#   mouth content and paints a straight content-colored ray from the spike, with
#   length determined by content count.
# - Fused input here: keep the cup frame, keep the scattered mouth content, and add
#   one aligned target marker on the outward spike ray. That marker uses the same
#   color as the mouth content, so the solver must first recover the cup, then find
#   the matching marker on the correct ray.
# - Fused output here: erase both the mouth content and the whole cup frame. Paint
#   only the ray segment from the spike to the marker, inclusive. Because ARC uses a
#   discrete palette, the "gradient" is implemented as a coarse three-level ramp:
#   nearest cell = content color, farthest cell (the marker location) = border color,
#   and any strict interior cells = a fixed visible midpoint color 5.
# - Small trace:
#   * Original: 4 mouth pixels, clear ray to edge, so 4 content-colored pixels are
#     painted from the spike.
#   * D4 alone: if the marker is 3 cells away, stop after 3 cells instead of 4.
#   * Fused combo: if the marker is 3 cells away and content count is 5, output is
#     `[content, midpoint, border]` on those 3 cells, while the whole border frame
#     disappears. The endpoint is fixed by the marker location, but its color changes
#     to the border color because the gradient reaches the border color at maximum
#     distance.
#
# b) Complexity calibration
# - The endpoint mechanism uses exactly one marker per cup on the unique spike ray.
#   That keeps the D4 signal strong without introducing ambiguous marker selection.
# - The gradient uses exactly one global midpoint color (`5`) instead of many
#   pair-specific ramps. This is deliberate: ARC examples must support human
#   inference from only a few samples. A single repeated midpoint is visually
#   learnable; arbitrary unseen intermediate colors would not be.
# - Generator enforces `target_distance >= 3` so the gradient is actually visible.
#   It also enforces `content_count not in {target_distance, target_distance - 1}` so
#   the endpoint is not accidentally explainable by the original count-based rule.
# - Border erasure is visually obvious because the frame is present in the input and
#   absent in the output, leaving only the ray.
#
# c) Implicit-assumption audit for the original main()
# - `_flood_from_bbox_side(...)` still works: the marker is outside the bbox, so the
#   mouth interior remains recoverable by flooding from the opposite side.
# - Counting non-bg/non-border colors inside the mouth still recovers the content
#   color, because the marker is outside the cup.
# - The original ejection loop assumed "ray length == content count". That assumption
#   is invalid now and is replaced with "find the first same-color marker on the
#   outward spike ray and stop there".
# - The original solver preserved the border. D6 invalidates that: after solving a
#   cup, every border pixel of that cup must be erased.
# - The original solver painted a uniform color. D5 invalidates that: the ray color
#   now depends on distance-from-spike, and the endpoint must switch to the border
#   color.
# - New assumption introduced by the fused solver: the first outward same-color pixel
#   on the spike ray is the intended target marker. The generator guarantees the full
#   spike ray is reserved and contains no other pixels before the marker.
#
# d) Placement legality
# - Cup bboxes still keep a 1-cell gap via dilated occupancy.
# - The entire spike ray to the grid edge is reserved in `occ_path`; this prevents
#   another cup, marker, or future ray from entering the same line and breaking
#   target discovery.
# - Targets are placed strictly before the grid edge so the stop condition is visible.
# - Input uses no stray pixels in this combo, because D6 requires a minimal output
#   containing only rays; unrelated survivors would blur the transform.
#
# e) Visual observability
# - In the input, the only content-colored pixel outside the mouth is the aligned
#   target marker.
# - In the output, three things change together in a human-visible way:
#   1. the frame disappears,
#   2. the ray stops exactly at the aligned marker location instead of at the edge,
#   3. the ray changes from content color near the spike to border color at the end,
#      with a repeated middle color in between.
# - Across 2-4 examples this is inferable because the same midpoint color recurs and
#   the endpoint sits exactly where the input marker was.
#
# f) Visual cleanliness
# - No strays are generated.
# - Rays never overlap other cups or other rays because the full path is reserved.
# - The target marker is a single isolated pixel on a blank ray, so it is easy to
#   distinguish from mouth content.
#
# g) Color inferability
# - The output never requires solving a hidden numeric color order. The only special
#   color rule is a single repeated midpoint color `5`, plus the endpoint adopting
#   the cup's visible border color from the input. Both are structurally supported by
#   position: start near spike, middle interior, end at marker.
#
# h) New solver assumptions audited against the generator
# - Assumption: each actionable cup has exactly one candidate spike whose opposite-side
#   flood reaches mouth content. Generator preserves the original cup geometry, so
#   this remains true.
# - Assumption: marker color equals mouth content color. Generator enforces that by
#   placing the target with `content_color`.
# - Assumption: border color is distinct from content and midpoint colors. Generator
#   excludes midpoint color 5 from border/content sampling and also enforces
#   `border_color != content_color`.
# - Assumption: target distance is at least 3 so the gradient has a start, middle,
#   and end. Generator enforces this explicitly.
#
# i) Degeneracy guard
# - The combo must not collapse to "border erase only" or "marker stop only". The
#   generator therefore validates via oracle and also samples only cases where the
#   count-based original endpoint differs from the marker endpoint. This makes D4 and
#   D5 operational, while D6 is always operational because the frame is erased.

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


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
MID_COLOR = 5

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox from the chosen edge, blocked by border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
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


def _first_color_on_ray(
    g: np.ndarray, spike: tuple[int, int], direction: str, color: int,
) -> tuple[int, int] | None:
    """Return the first outward ray cell whose value equals ``color``."""
    H, W = g.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < H and 0 <= cc < W:
        if int(g[rr, cc]) == color:
            return int(rr), int(cc)
        rr += dr
        cc += dc
    return None


def _ray_cells_to_target(spike: tuple[int, int], target: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """Enumerate outward ray cells from the spike up to and including the target."""
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((int(rr), int(cc)))
        if (rr, cc) == target:
            return cells
        rr += dr
        cc += dc


def _trajectory_colors(length: int, content_color: int, border_color: int) -> list[int]:
    """Discrete gradient: start=content, end=border, strict interior=global midpoint."""
    if length <= 0:
        return []
    if length == 1:
        return [int(border_color)]
    if length == 2:
        return [int(content_color), int(border_color)]
    return [int(content_color)] + [MID_COLOR] * (length - 2) + [int(border_color)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused variant:
    - detect each cup from its border frame,
    - recover mouth content color,
    - find the aligned same-color target marker on the spike ray,
    - erase both the mouth content and the whole border frame,
    - paint only the spike-to-target segment with the discrete gradient.
    """
    g = np.array(grid, dtype=int)
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
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing: list[tuple[str, tuple[int, int], tuple[int, int], int, list[tuple[int, int]]]] = []
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
                target = _first_color_on_ray(g, spike, d, content_col)
                if target is None:
                    continue

                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, target, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, target, content_col, content_cells = passing[0]

            for r, c in border_set:
                out[r, c] = bg
            for r, c in content_cells:
                out[r, c] = bg

            ray_cells = _ray_cells_to_target(spike, target, tail_dir)
            ray_colors = _trajectory_colors(len(ray_cells), content_col, int(border_col))
            for (r, c), color in zip(ray_cells, ray_colors):
                out[r, c] = int(color)

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
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
    """Flood-fill the cup interior from the mouth side within the patch bbox."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
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
    """Rotate a canonical up-pointing cup to the requested tail direction."""
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
    """Choose canonical cup parameters that still fit the spec bbox after rotation."""
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
    direction: str, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Place a cup whose entire outward spike ray is clear and long enough."""
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
        path_cells: list[tuple[int, int]] = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        if len(path_cells) < 4:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells

    return None


def _choose_count_and_target(
    rng: np.random.Generator, mouth_size: int, ray_len: int,
) -> tuple[int, int] | None:
    """Pick `(content_count, target_distance)` with visible gradient and non-original endpoint."""
    max_count = min(CONTENT_COUNT_MAX, mouth_size)
    if max_count < 1 or ray_len < 4:
        return None

    target_choices = list(range(3, ray_len))
    pairs = [
        (count, target_dist)
        for count in range(1, max_count + 1)
        for target_dist in target_choices
        if count not in {target_dist, target_dist - 1}
    ]
    if not pairs:
        return None
    return pairs[int(rng.integers(0, len(pairs)))]


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_color: int,
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, place its marker, and commit its minimal-output gradient ray."""
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

    placement = _try_place_cup(rng, H, W, patch, spike, direction, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells = placement

    chosen = _choose_count_and_target(rng, len(mouth), len(path_cells))
    if chosen is None:
        return None
    content_count, target_distance = chosen
    target_cell = path_cells[target_distance - 1]
    ejection_cells = path_cells[:target_distance]

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    canvas_in[target_cell] = content_color
    for (r, c), color in zip(ejection_cells, _trajectory_colors(len(ejection_cells), content_color, border_color)):
        canvas_out[r, c] = int(color)
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "target_cell": target_cell,
        "target_distance": target_distance,
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
    """Output-first generator: plan final gradient rays, then reconstruct the input cups."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
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
