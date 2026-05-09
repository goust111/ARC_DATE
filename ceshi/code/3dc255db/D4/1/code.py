# Variant of 3dc255db
# Dimension: D4 — 终点/停止条件 (dst)
# Change: stacking — 内容像素从尖刺向外堆叠，超出网格边界的部分被丢弃... → 到达目标标记停止
#
# Design analysis
# ===============
# Reference-task check
# - The prompt cites 581f7754 as the inspiration for "marker_alignment".
# - In the provided workspace, only
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/581f7754.json`
#   exists; no solver file for 581f7754 is present. So the implementation below
#   only borrows the observable idea "an aligned same-color marker fixes the end
#   point", without importing any other rule from that task.
#
# a) Concrete behavior and flow trace
# - The cup frame detection stays unchanged: a hollow Y/funnel frame has a unique
#   spike pixel on one bbox side, and the mouth is on the opposite side.
# - New visible object: for each cup, there is now a single same-color marker on
#   the outward ray starting at the spike and going along the tail direction.
#   The marker is outside the cup and is already present in the input.
# - Input: border frame + scattered content inside the mouth + one isolated
#   content-colored marker on the spike ray.
# - Output: mouth content is erased, and the whole segment from the spike to that
#   marker is filled with the content color, inclusive of the marker cell.
# - Small trace:
#   * Original logic: if a cup has 3 content pixels and 5 empty cells to the edge,
#     the solver paints the first 3 ray cells.
#   * Variant logic: if the marker is 2 cells away, the solver paints exactly 2
#     cells, even if there were 3 or 6 content pixels inside.
#   * Difference: line length is no longer determined by content count or grid
#     boundary; it is determined by the marker position.
#
# b) Complexity calibration
# - The marker is only one pixel and reuses the content color, so the user must
#   jointly infer color matching and directional alignment; this is enough to make
#   the D4 change observable.
# - To avoid over-complication, each cup has exactly one relevant marker and it
#   lies on the unique spike ray. No ranking by color value or multiple marker
#   types is introduced.
# - To avoid degeneration back to the original task, generation enforces
#   `content_count` to be different from both `target_distance` and
#   `target_distance - 1` for every cup. The second exclusion matters because the
#   target marker already exists in the input; otherwise the old solver could stop
#   one cell early and still match the variant output. The marker is also never
#   placed on the last ray cell at the grid edge.
#
# c) Implicit-assumption audit for original main()
# - `_flood_from_bbox_side(...)`: assumes the mouth can still be recovered by a
#   flood inside the bbox from the side opposite a candidate spike. This remains
#   valid because the D4 change only adds an external marker.
# - `cnt[v] += 1` over flooded mouth cells: assumes the content color is visible
#   inside the mouth. Still valid; the marker is outside the bbox and does not
#   affect color counting inside the cup.
# - `len(passing) != 1`: assumes only one candidate spike sees mouth content.
#   Still valid because external markers are outside the bbox flood.
# - Original ejection loop `for t in range(1, len(content_cells) + 1)` relied on
#   the old invariant "line length equals content count, clipped by boundary".
#   That invariant no longer holds and must be replaced by "find first same-color
#   marker on the spike ray and stop there".
# - Original code also harmlessly overwrote any encountered cell on the ray. Under
#   the new rule, that would be unsafe if other objects could lie before the
#   marker. The generator therefore guarantees the spike ray is reserved and clear
#   up to the marker.
#
# d) Placement legality
# - Cup bboxes still keep a 1-cell gap using dilated occupancy.
# - The whole spike ray to the edge is still reserved in `occ_path`, even though
#   only a prefix is used in the output. This prevents collisions with other cups
#   and keeps marker detection unambiguous.
# - The marker is chosen from a strict interior prefix of the ray, never outside
#   the grid and never on the final edge cell.
#
# e) Visual observability
# - In the input, the only same-color pixel outside the mouth is the aligned
#   marker. In the output, the new line visibly reaches that marker and stops
#   there, often before the border of the canvas. Across a few examples, this is
#   a direct, human-visible clue that the endpoint is marker-based.
#
# f) Visual cleanliness
# - The reserved ray avoids overlaps between markers, cups, and other output
#   strokes.
# - Optional stray pixels stay on the mouth side, never on the spike ray, so they
#   do not compete with the target marker.
#
# g) Color inferability
# - No hidden color ordering is used. The marker color always matches the cup's
#   mouth content color, and that same color becomes the ejected line color.

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


def _first_color_on_ray(
    g: np.ndarray, spike: tuple[int, int], direction: str, color: int,
) -> tuple[int, int] | None:
    """Return the first cell on the outward spike ray whose value equals ``color``."""
    H, W = g.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < H and 0 <= cc < W:
        if int(g[rr, cc]) == color:
            return rr, cc
        rr += dr
        cc += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject cup content toward a same-color target marker on the spike ray.

    Compared with the original task, the only rule change is the endpoint:
    instead of painting one cell per content pixel until the boundary, the solver
    paints from the spike outward until it reaches the aligned marker of the same
    color as the cup's content.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
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
                target = _first_color_on_ray(g, spike, d, content_col)
                if target is None:
                    continue
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, target, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, target, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            rr, cc = spike[0] + dr, spike[1] + dc
            while True:
                out[rr, cc] = content_col
                if (rr, cc) == target:
                    break
                rr += dr
                cc += dc

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

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
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


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the interior of the cup patch from the mouth side."""
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
    """Pick canonical cup parameters that fit the bbox bounds after rotation."""
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
    """Find a placement with a clear full spike ray. The final ray cell is reserved too."""
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
        if len(path_cells) < 2:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells
    return None


def _choose_count_and_target(
    rng: np.random.Generator, mouth_size: int, ray_len: int,
) -> tuple[int, int] | None:
    """Pick `(content_count, target_distance)` with marker strictly before the edge and
    deliberately different from the original count-based endpoint."""
    max_count = min(CONTENT_COUNT_MAX, mouth_size)
    if max_count < 1 or ray_len < 2:
        return None

    target_choices = list(range(1, ray_len))
    if len(target_choices) >= 2:
        target_choices = target_choices[1:]

    pairs = [
        (count, target_dist)
        for count in range(1, max_count + 1)
        for target_dist in target_choices
        if count not in {target_dist, target_dist - 1}
    ]
    if not pairs:
        return None
    idx = int(rng.integers(0, len(pairs)))
    return pairs[idx]


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean (H, W) mask of cells on the mouth side of ``bbox``."""
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
    """Optionally place a stray pixel on the mouth side of a cup."""
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
    """Build one cup, place its target marker, and commit its expected output segment."""
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
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    canvas_in[target_cell] = content_color
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
        "target_cell": target_cell,
        "target_distance": target_distance,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content pixels at random inside its mouth interior."""
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
