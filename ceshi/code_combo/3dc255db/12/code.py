# Combo variant of 3dc255db (index 12)
# combo_spec: D4/0+D5/0
# Name: 障碍物停止+间歇式轨迹
#
# Design analysis
# a) Concrete behavior and trace:
#    - Cups are still hollow Y/funnel frames with one spike on a bbox edge and scattered payload pixels
#      in the mouth interior.
#    - The launched ray now advances cell-by-cell away from the spike. Odd offsets are paint positions,
#      even offsets are forced gaps, but any non-bg cell on the ray is a blocker regardless of parity.
#    - Example: tail points right, 4 payload pixels, ray cells after the spike are
#      [bg, bg, bg, obstacle, bg, bg, ...].
#      Original task would paint offsets 1,2,3,4 and overwrite the obstacle.
#      D5 alone would paint offsets 1,3,5,7.
#      Combo paints offset 1, leaves offset 2 blank, paints offset 3, then sees the obstacle at offset 4
#      and stops. Offsets 5+ remain untouched and the remaining payload is discarded.
# b) Complexity calibration:
#    - The generator forces every cup to show both effects at once: at least two painted odd offsets
#      before the blocker and at least one extra payload pixel discarded by the blocker.
#    - Obstacle parity is allowed to vary, so the stopping point is not encoded by a fixed gap length.
# c) Implicit-assumption audit of the original main():
#    - Border component discovery, singleton-spike search, and mouth flood-fill remain valid.
#    - The original ejection loop assumed every in-bounds ray cell was writable and every step consumed
#      one payload pixel. Both assumptions fail here: even offsets are blanks, and blockers can appear on
#      either painted or skipped cells.
#    - Blocker checks must read the original input grid `g`, not `out`, otherwise previously emitted
#      pixels from another cup would become fake obstacles.
# d) Placement legality:
#    - Cup bboxes still keep a 1-cell gap via dilated occupancy.
#    - The full ray corridor is reserved in `occ_path`, including the intentional blank cells, so no
#      other cup or obstacle can occupy a skipped position and create ambiguity.
#    - Obstacles are also kept 8-neighbor separated from cup patches and from one another.
# e) Visual observability:
#    - Each generated cup shows a dash-gap-dash pattern that ends early at an isolated blocker pixel.
#      Across 2-4 examples, a human can see both the alternating trajectory and the stop-before-obstacle rule.
# f) Visual cleanliness:
#    - Mouth-side stray noise is removed in this combo. The only isolated tail-side pixel is the blocker,
#      which keeps the launch corridor easy to read.
# g) Color inferability:
#    - Colors keep their original roles: border marks the cup, content marks emitted payload.
#      Obstacle color is just "some other non-bg color" and is identified structurally by its location
#      on the ray, not by any hidden color ordering.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: every intended blocker is an existing non-bg pixel on the ray outside the cup bbox.
#      Guarantee: generator explicitly places one obstacle on each reserved ray.
#    - Assumption: skipped even cells stay empty unless a blocker is reached there.
#      Guarantee: generator reserves the whole corridor and paints only the odd-offset output cells.
#    - Assumption: each showcased cup has enough room to reveal the combo visually.
#      Guarantee: generator requires at least two painted cells before the blocker and at least one
#      discarded payload pixel after truncation.

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
CONTENT_COUNT_MIN = 3
MIN_VISIBLE_TRAIL_PIXELS = 2
OBSTACLE_OFFSET_MIN = 4

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, seeded from one bbox side and blocked by border cells."""
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


def _eject_cells_with_blockers(
    grid: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    payload_count: int,
    bg: int,
) -> list[tuple[int, int]]:
    """Advance cell-by-cell. Paint only odd offsets; stop at edge or first blocker on any parity."""
    H, W = grid.shape
    dr, dc = DIR4[direction]
    sr, sc = spike
    painted: list[tuple[int, int]] = []
    used = 0
    offset = 1
    while used < payload_count:
        rr = sr + dr * offset
        cc = sc + dc * offset
        if not (0 <= rr < H and 0 <= cc < W):
            break
        if int(grid[rr, cc]) != bg:
            break
        if offset % 2 == 1:
            painted.append((rr, cc))
            used += 1
        offset += 1
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule: payload launches outward in an odd-offset dashed ray, but the ray is scanned
    one cell at a time and stops immediately before the first non-bg obstacle on any offset.
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

            for rr, cc in _eject_cells_with_blockers(g, spike, tail_dir, len(content_cells), bg):
                out[rr, cc] = content_col

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical Y/funnel with tail pointing up."""
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


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the mouth interior of a cup patch from the open side."""
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
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical up-pointing cup to the target tail direction."""
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
    """Sample a canonical cup that still fits the original bbox constraints after rotation."""
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


def _choose_obstacle_color(rng: np.random.Generator, border_color: int, content_color: int) -> int:
    choices = [c for c in range(1, 10) if c not in {border_color, content_color}]
    return int(rng.choice(choices))


def _odd_cells_before_index(path_cells: list[tuple[int, int]], stop_idx: int) -> list[tuple[int, int]]:
    """Visible dashed cells before a blocker at `path_cells[stop_idx]`."""
    return path_cells[:stop_idx:2]


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
    """Find a placement whose ray can show both dashed trajectory and blocker truncation."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)
    occ_obstacle_dilated = ndimage.binary_dilation(occ_obstacle, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_bbox_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_obstacle_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        path_cells: list[tuple[int, int]] = []
        rr, cc = sr_abs + dr, sc_abs + dc
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if len(path_cells) < OBSTACLE_OFFSET_MIN:
            continue
        if any(occ_bbox[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        if any(occ_obstacle_dilated[r, c] for r, c in path_cells):
            continue

        feasible: list[tuple[int, list[tuple[int, int]], int]] = []
        odd_total = len(path_cells[::2])
        for obstacle_idx in range(OBSTACLE_OFFSET_MIN - 1, len(path_cells)):
            visible_cells = _odd_cells_before_index(path_cells, obstacle_idx)
            visible_count = len(visible_cells)
            if visible_count < MIN_VISIBLE_TRAIL_PIXELS:
                continue
            min_content_count = visible_count + 1
            if min_content_count > max_content_count:
                continue
            # Ensure that if the blocker were ignored, the intermittent-only rule would paint
            # at least one additional odd-offset cell. This prevents degeneration to D5-only.
            if odd_total < min_content_count:
                continue
            feasible.append((obstacle_idx, visible_cells, min_content_count))
        if not feasible:
            continue

        obstacle_idx, ejection_cells, min_content_count = feasible[int(rng.integers(0, len(feasible)))]
        content_count = int(rng.integers(min_content_count, max_content_count + 1))
        obstacle_cell = path_cells[obstacle_idx]
        return r0, c0, path_cells, obstacle_cell, ejection_cells, content_count

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
    """Place one cup and one blocker so the combo behavior is forced to appear."""
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
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))
    if max_content_count < CONTENT_COUNT_MIN:
        return None

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, max_content_count, occ_bbox, occ_path, occ_obstacle,
    )
    if placement is None:
        return None
    r0, c0, path_cells, obstacle_cell, ejection_cells, content_count = placement

    patch_mask = patch != 0
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_bbox[r0 : r0 + ph, c0 : c0 + pw] = True

    obstacle_color = _choose_obstacle_color(rng, border_color, content_color)
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
        "visible_trail_count": len(ejection_cells),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's payload inside its mouth interior."""
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
    """Output-first generation, validated against the combo solver."""
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
