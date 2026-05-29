# Variant of 3dc255db
# Dimension: D4 — 终点/停止条件 (dst)
# Change: stacking — 内容像素从尖刺向外堆叠，超出网格边界的部分被丢弃... → 遇障碍停止
#
# Design analysis for this variant
# a) Concrete behavior and trace:
#    - The cup objects stay the same: hollow Y/funnel frames with one single-pixel spike on a bbox edge.
#    - The new visible ingredient is an isolated non-bg obstacle pixel placed directly on the tail ray,
#      outside the cup, with at least one empty cell between spike and obstacle.
#    - Input: content pixels are still scattered inside the cup mouth; obstacle pixels already exist on
#      the outward ray. Output: mouth content is erased; a straight ray of the content color is emitted
#      from the spike, but it stops immediately before the first obstacle cell. The obstacle itself stays
#      unchanged; any leftover content pixels are discarded.
#    - Small trace: suppose a right-pointing cup has 4 content pixels, and cells after the spike are
#      [bg, bg, obstacle, bg, bg]. Original logic would write on the first 4 in-bounds cells and would
#      overwrite the obstacle. Variant logic writes only the first 2 bg cells, sees the obstacle on the
#      third step, stops there, and discards the remaining 2 content pixels.
# b) Complexity calibration:
#    - Every generated cup gets one obstacle on its tail ray, and the generator forces content_count to
#      exceed the free run before that obstacle. This avoids degeneration back to the original rule.
#    - Obstacle distance varies per cup, so the solver must compute the actual free prefix length rather
#      than memorize a fixed offset.
# c) Implicit-assumption audit of the original main():
#    - Cup detection via color components, bbox-edge singleton spike search, and mouth flood-fill does
#      not depend on the old dst rule, so it remains valid.
#    - The original ejection loop assumes every in-bounds tail cell is writable until the grid edge.
#      That is no longer safe: non-bg pixels on the ray must act as blockers and must not be overwritten.
#    - Because multiple cups may exist, blocker detection must read from the original input grid `g`,
#      not the progressively modified `out`; otherwise processing order would create fake blockers.
# d) Placement legality:
#    - Cups still have non-overlapping bboxes with a 1-cell gap.
#    - Full tail rays are reserved so different cups/obstacles do not share a ray cell.
#    - Obstacle pixels are also kept 8-neighbor separated from later cup patches, preventing accidental
#      same-color merging into a larger 8-connected border component.
# e) Visual observability:
#    - In each cup, an obvious lone pixel sits ahead of the spike on the ray. The output line ends right
#      before that pixel instead of extending to the border. This is legible from a few examples.
# f) Visual cleanliness:
#    - Unrelated mouth-side stray noise from the original generator is removed here; the new obstacle is
#      already the key non-cup marker. This keeps the grids readable.
# g) Color inferability:
#    - Obstacle pixels can reuse arbitrary non-bg colors, but they are always spatially identified by
#      being isolated and sitting on the ejection ray. No hidden color ordering is used.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: blockers are any non-bg pixels already present on the tail ray in the input.
#      Guarantee: generator explicitly places one such obstacle per cup on that ray.
#    - Assumption: the first blocker, if any, is the stopping point.
#      Guarantee: generator reserves the whole ray, so no nearer accidental blockers appear.
#    - Assumption: cups remain uniquely identifiable by the original mouth-flood disambiguation rule.
#      Guarantee: obstacles are outside cup bboxes and are isolated from cup borders/components.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Variant rule:
    - The cup/content identification is unchanged from the source task.
    - When ejecting from the spike, the ray now stops at the first non-bg obstacle already
      present in the input grid along the tail direction.
    - The obstacle cell itself is not overwritten; leftover content pixels are discarded.
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

            dr, dc = DIR4[tail_dir]
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


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
OBSTACLE_DISTANCE_MIN = 2
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

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
    """Pick canonical cup parameters that fit the bbox constraints after rotation."""
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
    """Find a placement with a clear ray prefix and one obstacle on that ray."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)
    occ_obstacle_dilated = ndimage.binary_dilation(occ_obstacle, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_bbox_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_obstacle_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        full_path = []
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
    """Build one cup, place one obstacle on its ray, and commit both input and output state."""
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
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))
    if max_content_count < 2:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, max_content_count, occ_bbox, occ_path, occ_obstacle,
    )
    if placement is None:
        return None
    r0, c0, path_cells, obstacle_cell, ejection_cells, content_count = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
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
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "obstacle_cell": obstacle_cell,
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
    """Single output-first generation attempt, validated by the solver."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
